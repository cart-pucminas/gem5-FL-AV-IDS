"""
Valida os headers C gerados por exporta_para_c.py: PARSEIA cada .h, executa a
MESMA logica de inferencia do C em Python, e compara com a predicao do sklearn
(.joblib) nas amostras do dataset.h.

Se a concordancia for ~100%, a traducao para C esta correta e podemos confiar nos
.h no gem5. Se divergir, ha bug na exportacao (ordem de pesos, mapeamento de nos,
pre-proc) - melhor descobrir AQUI do que medir o modelo errado no simulador.

Nao precisa de compilador C: replica a aritmetica do .h fielmente (o C roda
exatamente essa logica). Testa o ARTEFATO real (le os arrays escritos no .h).

Uso (a partir da RAIZ, apos exporta_para_c.py):
    python experimentos/workload/valida_export_c.py

Dependencias: numpy scikit-learn xgboost joblib
"""

import os
import re
import numpy as np
import joblib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_ROOT, "experimentos", "modelos")
C_DIR = os.path.join(_ROOT, "experimentos", "workload", "c_export")


# ------------------------------------------------------------------ util parse
def _ler(nome):
    with open(os.path.join(C_DIR, nome)) as f:
        return f.read()

def _floats(txt, var):
    """Extrai o array 'float var[...] = {...};' como np.array."""
    m = re.search(rf"float\s+{re.escape(var)}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};",
                  txt, re.DOTALL)
    if not m:
        return None
    corpo = m.group(1).replace("f", "").replace("\n", " ")
    return np.array([float(x) for x in corpo.split(",") if x.strip()])

def _ints(txt, var):
    m = re.search(rf"int\s+{re.escape(var)}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};",
                  txt, re.DOTALL)
    if not m:
        return None
    corpo = m.group(1).replace("\n", " ")
    return np.array([int(x) for x in corpo.split(",") if x.strip()])

def _define(txt, nome):
    m = re.search(rf"#define\s+{nome}\s+([0-9.eE+\-]+)f?", txt)
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------ dataset.h
def ler_dataset_h():
    txt = _ler("dataset.h")
    n = int(_define(txt, "NUM_SAMPLES"))
    nf = int(_define(txt, "NUM_FEATURES"))
    m = re.search(r"can_data\[.*?\]\[.*?\]\s*=\s*\{(.*)\};", txt, re.DOTALL)
    linhas = re.findall(r"\{([^}]*)\}", m.group(1))
    X = np.array([[float(v) for v in l.split(",")] for l in linhas])
    assert X.shape == (n, nf), f"dataset.h shape {X.shape} != ({n},{nf})"
    return X


# ------------------------------------------------------------------ RF (.h)
def predict_rf_h(X):
    txt = _ler("rf_model.h")
    NC = int(_define(txt, "NUM_CLASSES"))
    roots = _ints(txt, "root_indices")
    # forest: structs {left,right,feat,thr,leaf,cls}
    m = re.search(r"TreeNode\s+forest\[[^\]]*\]\s*=\s*\{(.*)\};", txt, re.DOTALL)
    tuples = re.findall(r"\{([^}]*)\}", m.group(1))
    forest = []
    for t in tuples:
        p = [x.strip() for x in t.split(",")]
        forest.append((int(p[0]), int(p[1]), int(p[2]),
                       float(p[3]), int(p[4]), int(p[5])))
    out = []
    for x in X:
        votes = np.zeros(NC, dtype=int)
        for r in roots:
            n = r
            while not forest[n][4]:            # is_leaf
                n = forest[n][0] if x[forest[n][2]] <= forest[n][3] else forest[n][1]
            c = forest[n][5]
            if 0 <= c < NC:
                votes[c] += 1
        out.append(int(np.argmax(votes)))
    return np.array(out)


# ------------------------------------------------------------------ XGB (.h)
def predict_xgb_h(X):
    txt = _ler("xgb_model.h")
    NC = int(_define(txt, "NUM_CLASSES"))
    NT = int(_define(txt, "NUM_TREES"))
    BASE = _define(txt, "XGB_BASE")
    roots = _ints(txt, "tree_roots")
    m = re.search(r"XNode\s+xnodes\[[^\]]*\]\s*=\s*\{(.*)\};", txt, re.DOTALL)
    tuples = re.findall(r"\{([^}]*)\}", m.group(1))
    xn = []
    for t in tuples:
        p = [x.strip() for x in t.split(",")]
        xn.append((int(p[0]), float(p[1]), int(p[2]), int(p[3]),
                   float(p[4]), int(p[5])))
    out = []
    for x in X:
        margin = np.full(NC, BASE, dtype=float)
        for t in range(NT):
            base = roots[t]; n = base
            while not xn[n][5]:                # is_leaf
                n = base + (xn[n][2] if x[xn[n][0]] < xn[n][1] else xn[n][3])
            margin[t % NC] += xn[n][4]
        out.append(int(np.argmax(margin)))
    return np.array(out)


# ------------------------------------------------------------------ MLP (.h)
def predict_mlp_h(X):
    txt = _ler("mlp_model.h")
    NC = int(_define(txt, "NUM_CLASSES"))
    NL = int(_define(txt, "N_LAYERS"))
    mean = _floats(txt, "sc_mean"); scale = _floats(txt, "sc_scale")
    dims = _ints(txt, "layer_dims")
    Ws = [_floats(txt, f"W{i}") for i in range(NL)]
    bs = [_floats(txt, f"b{i}") for i in range(NL)]
    out = []
    for raw in X:
        a = (raw - mean) / scale
        for l in range(NL):
            din, dout = int(dims[l]), int(dims[l + 1])
            W = Ws[l].reshape(din, dout)       # C usa W[i*dout + j] = row-major (din x dout)
            z = bs[l] + a @ W
            if l < NL - 1:
                z = np.maximum(z, 0)           # ReLU exceto saida
            a = z
        out.append(int(np.argmax(a)))
    return np.array(out)


# ------------------------------------------------------------------ SVM (.h)
def predict_svm_h(X):
    txt = _ler("svm_model.h")
    NC = int(_define(txt, "NUM_CLASSES"))
    mean = _floats(txt, "sc_mean"); scale = _floats(txt, "sc_scale")
    B = _floats(txt, "B")
    m = re.search(r"float\s+W\[[^\]]*\]\[[^\]]*\]\s*=\s*\{(.*)\};", txt, re.DOTALL)
    linhas = re.findall(r"\{([^}]*)\}", m.group(1))
    W = np.array([[float(v.replace("f", "")) for v in l.split(",") if v.strip()]
                  for l in linhas])
    out = []
    for raw in X:
        x = (raw - mean) / scale
        scores = B + W @ x
        out.append(int(np.argmax(scores)))
    return np.array(out)


# ------------------------------------------------------------------ sklearn ref
def predict_sklearn(key, X):
    b = joblib.load(os.path.join(MODELS_DIR, f"{key}.joblib"))
    clf, sc = b["clf"], b["scaler"]
    Xin = sc.transform(X) if b["scaled"] else X
    return clf.predict(Xin)


def comparar(key, fn_h, X):
    ref = predict_sklearn(key, X)
    got = fn_h(X)
    conc = float(np.mean(ref == got))
    div = int(np.sum(ref != got))
    status = "OK" if conc > 0.99 else ("ATENCAO" if conc > 0.90 else "FALHA")
    print(f"  {key.upper():<5} concordancia C-vs-sklearn: {conc:6.2%}  "
          f"({div}/{len(X)} divergem)  [{status}]")
    return conc


def main():
    if not os.path.exists(os.path.join(C_DIR, "dataset.h")):
        print("[ERRO] c_export/dataset.h nao existe. Rode exporta_para_c.py antes.")
        return
    X = ler_dataset_h()
    print(f"Validando com {len(X)} amostras do dataset.h ({X.shape[1]} features)\n")

    checks = [("rf", predict_rf_h), ("xgb", predict_xgb_h),
              ("mlp", predict_mlp_h), ("svm", predict_svm_h)]
    concs = []
    for key, fn in checks:
        hpath = os.path.join(C_DIR, f"{key}_model.h")
        if not os.path.exists(hpath):
            print(f"  {key.upper():<5} (sem {key}_model.h - pulando)")
            continue
        try:
            concs.append(comparar(key, fn, X))
        except Exception as e:
            print(f"  {key.upper():<5} ERRO ao validar: {e}")

    print()
    if concs and all(c > 0.99 for c in concs):
        print("[OK] Todos os headers batem com o sklearn (>99%). Pode ir pro gem5.")
    else:
        print("[!] Algum header divergiu. NAO usar no gem5 antes de corrigir a exportacao.")


if __name__ == "__main__":
    main()
