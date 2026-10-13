"""
Exporta os 4 modelos treinados (.joblib) para headers C (.h) que rodam no gem5,
e regenera o dataset.h (amostras de teste, 9 features cruas).

Cada header define sua propria funcao predict(float x[NUM_FEATURES]) -> int classe,
com a logica de inferencia especifica do algoritmo:
  - rf_model.h   : travessia de floresta de arvores (INT, pointer-chasing)
  - xgb_model.h  : soma de folhas por classe + argmax (arvores rasas)
  - mlp_model.h  : forward pass (matmul + ReLU + argmax) - padroniza x internamente
  - svm_model.h  : argmax_c (w_c . x + b_c) - padroniza x internamente

RF/XGBoost usam features CRUAS; MLP/SVM padronizam (StandardScaler embutido).
Por isso dataset.h guarda os valores crus e cada header aplica seu pre-proc.

Uso (a partir da RAIZ do projeto, apos treina_modelos.py):
    python experimentos/workload/exporta_para_c.py

Saidas em experimentos/workload/c_export/:
    rf_model.h  xgb_model.h  mlp_model.h  svm_model.h  dataset.h  main.c

Dependencias: numpy pandas scikit-learn xgboost joblib
"""

import os
import numpy as np
import pandas as pd
import joblib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(_ROOT, "experimentos", "modelos")
DATASET_PATH = os.path.join(_ROOT, "experimentos", "AV-IDS-Dataset", "data",
                            "combined_can_dataset.csv.gz")
OUT_DIR = os.path.join(_ROOT, "experimentos", "workload", "c_export")
os.makedirs(OUT_DIR, exist_ok=True)

# USE_CAN_ID DEVE bater com o treino/tuning. True -> 10 features (arbitration_id
# primeiro), regime de alta acuracia. Ordem tem que ser IDENTICA a do treino, pois
# o dataset.h alimenta o predict() dos modelos. Na pratica o export ja usa a ordem
# salva no .joblib (bundle["features"]); esta lista e o fallback + base do dataset.h.
USE_CAN_ID = True
BASE_FEATURES = ['dlc'] + [f'data_{i}' for i in range(8)]
FEATURE_COLS = (['arbitration_id'] + BASE_FEATURES) if USE_CAN_ID else BASE_FEATURES
NUM_FEATURES = len(FEATURE_COLS)
N_TEST_SAMPLES = 5000   # amostras de teste no dataset.h (ROI). 5k = regime estavel
                        # ate no SVM (~325us), ~17h p/ as 102 sims (teste: RF/O3 ~22min).
RANDOM_STATE = 42


def carregar(key):
    path = os.path.join(MODELS_DIR, f"{key}.joblib")
    if not os.path.exists(path):
        print(f"[!] {path} nao encontrado - pulando {key}.")
        return None
    return joblib.load(path)


# ==============================================================================
# dataset.h - amostras de teste (features CRUAS, inteiras)
# ==============================================================================
def gerar_dataset_h(class_names, feature_cols):
    # feature_cols vem do modelo (bundle["features"]) -> a ORDEM das colunas do
    # dataset.h casa exatamente com o que o predict() espera. Isso evita mismatch
    # silencioso (ex.: CAN ID no indice errado) que produziria predicoes falsas no gem5.
    print(f"[*] Gerando dataset.h ({len(feature_cols)} features: {feature_cols})...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df.dropna(subset=feature_cols)
    # amostra estratificada por classe p/ ter todas representadas na ROI
    sub = df.sample(n=min(N_TEST_SAMPLES, len(df)), random_state=RANDOM_STATE)
    X = sub[feature_cols].astype(int).values
    n = len(X)

    with open(os.path.join(OUT_DIR, "dataset.h"), "w") as f:
        f.write(f"// dataset.h - amostras de teste AV-IDS ({len(feature_cols)} features: "
                f"{','.join(feature_cols)})\n")
        f.write("#ifndef DATASET_H\n#define DATASET_H\n\n")
        f.write(f"#define NUM_SAMPLES {n}\n")
        f.write(f"#define NUM_FEATURES {len(feature_cols)}\n\n")
        f.write("float can_data[NUM_SAMPLES][NUM_FEATURES] = {\n")
        for row in X:
            f.write("    {" + ", ".join(f"{v:.1f}" for v in row) + "},\n")
        f.write("};\n\n#endif\n")
    print(f"    dataset.h: {n} amostras")


# ==============================================================================
# RF -> travessia de floresta (mesma estrutura do main.c antigo)
# ==============================================================================
def exportar_rf(bundle):
    clf = bundle["clf"]
    nc = len(bundle["class_names"])
    all_nodes, root_indices, off = [], [], 0
    for est in clf.estimators_:
        t = est.tree_
        root_indices.append(off)
        for i in range(t.node_count):
            leaf = t.children_left[i] == -1
            cls = int(np.argmax(t.value[i])) if leaf else -1
            left = (t.children_left[i] + off) if not leaf else -1
            right = (t.children_right[i] + off) if not leaf else -1
            all_nodes.append((left, right, int(t.feature[i]),
                              float(t.threshold[i]), 1 if leaf else 0, cls))
        off += t.node_count

    with open(os.path.join(OUT_DIR, "rf_model.h"), "w") as f:
        f.write("// rf_model.h - Random Forest (travessia de arvores, features cruas)\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write(f"#define NUM_CLASSES {nc}\n")
        f.write(f"#define NUM_TREES {len(clf.estimators_)}\n")
        f.write(f"#define TOTAL_NODES {len(all_nodes)}\n\n")
        f.write("typedef struct { int left_child, right_child, feature_idx; "
                "float threshold; int is_leaf, class_pred; } TreeNode;\n\n")
        f.write("int root_indices[NUM_TREES] = {" +
                ", ".join(map(str, root_indices)) + "};\n\n")
        f.write("TreeNode forest[TOTAL_NODES] = {\n")
        for nd in all_nodes:
            f.write(f"    {{{nd[0]}, {nd[1]}, {nd[2]}, {nd[3]:.6f}, {nd[4]}, {nd[5]}}},\n")
        f.write("};\n\n")
        f.write("int predict(float f[NUM_FEATURES]) {\n")
        f.write("    int votes[NUM_CLASSES] = {0};\n")
        f.write("    for (int t = 0; t < NUM_TREES; t++) {\n")
        f.write("        int n = root_indices[t];\n")
        f.write("        while (!forest[n].is_leaf)\n")
        f.write("            n = (f[forest[n].feature_idx] <= forest[n].threshold)\n")
        f.write("                ? forest[n].left_child : forest[n].right_child;\n")
        f.write("        int c = forest[n].class_pred;\n")
        f.write("        if (c >= 0 && c < NUM_CLASSES) votes[c]++;\n")
        f.write("    }\n")
        f.write("    int best = 0; for (int c = 1; c < NUM_CLASSES; c++)\n")
        f.write("        if (votes[c] > votes[best]) best = c;\n")
        f.write("    return best;\n}\n\n#endif\n")
    print(f"    rf_model.h: {len(clf.estimators_)} arvores, {len(all_nodes)} nos")


# ==============================================================================
# XGBoost -> soma de folhas por classe (multiclasse: arvore i pertence a classe i%nc)
# ==============================================================================
def exportar_xgb(bundle):
    clf = bundle["clf"]
    nc = len(bundle["class_names"])
    feats = bundle.get("features", FEATURE_COLS)
    booster = clf.get_booster()
    dfz = booster.trees_to_dataframe()

    # Mapeia nome da feature -> indice. O XGBoost guarda os nomes reais das colunas
    # (dlc, data_0, ...) quando treinado com DataFrame; ou 'f0','f1' se sem nomes.
    name2idx = {name: i for i, name in enumerate(feats)}

    def fidx_de(nome):
        if nome in name2idx:
            return name2idx[nome]
        if nome.startswith("f") and nome[1:].isdigit():  # fallback 'f0','f1'
            return int(nome[1:])
        raise ValueError(f"feature desconhecida no XGBoost: {nome!r}")

    # empacota nos por arvore: cada no vira (feature_idx, split, yes, no, leaf_val, is_leaf)
    # indices 'Yes'/'No' sao globais 'tree-node'; mapeamos p/ offset linear.
    trees = []
    for tid, g in dfz.groupby("Tree"):
        idmap = {row["ID"]: i for i, (_, row) in enumerate(g.iterrows())}
        nodes = []
        for _, row in g.iterrows():
            if row["Feature"] == "Leaf":
                nodes.append((-1, 0.0, -1, -1, float(row["Gain"]), 1))
            else:
                fidx = fidx_de(row["Feature"])
                nodes.append((fidx, float(row["Split"]),
                              idmap[row["Yes"]], idmap[row["No"]], 0.0, 0))
        trees.append(nodes)

    base = 0.5
    try:
        base = float(booster.attr("base_score") or 0.5)
    except Exception:
        pass

    with open(os.path.join(OUT_DIR, "xgb_model.h"), "w") as f:
        f.write("// xgb_model.h - XGBoost (soma de folhas por classe, features cruas)\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write(f"#define NUM_CLASSES {nc}\n")
        f.write(f"#define NUM_TREES {len(trees)}\n")
        f.write(f"#define XGB_BASE {base:.6f}f\n\n")
        f.write("typedef struct { int feature_idx; float split; "
                "int yes, no; float leaf_val; int is_leaf; } XNode;\n\n")
        # achata todas as arvores num array + offsets
        flat, offs, o = [], [], 0
        for nodes in trees:
            offs.append(o)
            flat.extend(nodes)
            o += len(nodes)
        f.write(f"#define TOTAL_NODES {len(flat)}\n")
        f.write("int tree_roots[NUM_TREES] = {" + ", ".join(map(str, offs)) + "};\n\n")
        f.write("XNode xnodes[TOTAL_NODES] = {\n")
        for nd in flat:
            f.write(f"    {{{nd[0]}, {nd[1]:.6f}, {nd[2]}, {nd[3]}, {nd[4]:.6f}, {nd[5]}}},\n")
        f.write("};\n\n")
        f.write("int predict(float f[NUM_FEATURES]) {\n")
        f.write("    float margin[NUM_CLASSES];\n")
        f.write("    for (int c = 0; c < NUM_CLASSES; c++) margin[c] = XGB_BASE;\n")
        f.write("    for (int t = 0; t < NUM_TREES; t++) {\n")
        f.write("        int n = tree_roots[t];\n")
        f.write("        int base = tree_roots[t];\n")
        f.write("        while (!xnodes[n].is_leaf)\n")
        f.write("            n = base + ((f[xnodes[n].feature_idx] < xnodes[n].split)\n")
        f.write("                ? xnodes[n].yes : xnodes[n].no);\n")
        f.write("        margin[t % NUM_CLASSES] += xnodes[n].leaf_val;\n")
        f.write("    }\n")
        f.write("    int best = 0; for (int c = 1; c < NUM_CLASSES; c++)\n")
        f.write("        if (margin[c] > margin[best]) best = c;\n")
        f.write("    return best;\n}\n\n#endif\n")
    print(f"    xgb_model.h: {len(trees)} arvores, {len(flat)} nos")


# ==============================================================================
# MLP -> forward pass (padroniza x internamente com o scaler embutido)
# ==============================================================================
def exportar_mlp(bundle):
    clf = bundle["clf"]
    sc = bundle["scaler"]
    nc = len(bundle["class_names"])
    Ws = clf.coefs_          # lista de matrizes (in x out)
    bs = clf.intercepts_     # lista de vetores

    def carr(name, arr):
        flat = np.asarray(arr).flatten()
        return (f"float {name}[{flat.size}] = {{" +
                ", ".join(f"{v:.6f}f" for v in flat) + "};\n")

    with open(os.path.join(OUT_DIR, "mlp_model.h"), "w") as f:
        f.write("// mlp_model.h - MLP (forward: (x-mean)/scale -> ReLU camadas -> argmax)\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write(f"#define NUM_CLASSES {nc}\n")
        f.write(f"#define N_LAYERS {len(Ws)}\n\n")
        # scaler
        f.write(carr("sc_mean", sc.mean_))
        f.write(carr("sc_scale", sc.scale_))
        # dimensoes de cada camada
        dims = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
        f.write(f"int layer_dims[{len(dims)}] = {{" + ", ".join(map(str, dims)) + "};\n")
        maxd = max(dims)
        f.write(f"#define MAX_DIM {maxd}\n\n")
        for i, (W, b) in enumerate(zip(Ws, bs)):
            f.write(carr(f"W{i}", W))
            f.write(carr(f"b{i}", b))
        f.write("\nint predict(float raw[NUM_FEATURES]) {\n")
        f.write("    float a[MAX_DIM], z[MAX_DIM];\n")
        f.write("    for (int i = 0; i < NUM_FEATURES; i++) a[i] = (raw[i]-sc_mean[i])/sc_scale[i];\n")
        # ponteiros para W/b por camada
        f.write("    float* Ws[N_LAYERS] = {" + ", ".join(f"W{i}" for i in range(len(Ws))) + "};\n")
        f.write("    float* bs[N_LAYERS] = {" + ", ".join(f"b{i}" for i in range(len(bs))) + "};\n")
        f.write("    for (int l = 0; l < N_LAYERS; l++) {\n")
        f.write("        int din = layer_dims[l], dout = layer_dims[l+1];\n")
        f.write("        for (int j = 0; j < dout; j++) {\n")
        f.write("            float s = bs[l][j];\n")
        f.write("            for (int i = 0; i < din; i++) s += a[i] * Ws[l][i*dout + j];\n")
        f.write("            z[j] = (l < N_LAYERS-1 && s < 0) ? 0 : s;\n")  # ReLU exceto saida
        f.write("        }\n")
        f.write("        for (int j = 0; j < dout; j++) a[j] = z[j];\n")
        f.write("    }\n")
        f.write("    int best = 0; for (int c = 1; c < NUM_CLASSES; c++)\n")
        f.write("        if (a[c] > a[best]) best = c;\n")
        f.write("    return best;\n}\n\n#endif\n")
    print(f"    mlp_model.h: camadas {dims}")


# ==============================================================================
# SVM linear -> argmax_c (w_c . x + b_c), padroniza x internamente
# ==============================================================================
def exportar_svm(bundle):
    clf = bundle["clf"]
    sc = bundle["scaler"]
    nc = len(bundle["class_names"])
    W = np.atleast_2d(clf.coef_)      # (n_classes x n_features) p/ multiclasse OvR
    b = np.atleast_1d(clf.intercept_)

    with open(os.path.join(OUT_DIR, "svm_model.h"), "w") as f:
        f.write("// svm_model.h - SVM linear (argmax_c w_c.x+b_c, padroniza x)\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write(f"#define NUM_CLASSES {nc}\n\n")
        flat_m = sc.mean_.flatten()
        flat_s = sc.scale_.flatten()
        f.write("float sc_mean[NUM_FEATURES] = {" + ", ".join(f"{v:.6f}f" for v in flat_m) + "};\n")
        f.write("float sc_scale[NUM_FEATURES] = {" + ", ".join(f"{v:.6f}f" for v in flat_s) + "};\n\n")
        f.write("float W[NUM_CLASSES][NUM_FEATURES] = {\n")
        for c in range(W.shape[0]):
            f.write("    {" + ", ".join(f"{v:.6f}f" for v in W[c]) + "},\n")
        f.write("};\n")
        f.write("float B[NUM_CLASSES] = {" + ", ".join(f"{v:.6f}f" for v in b) + "};\n\n")
        f.write("int predict(float raw[NUM_FEATURES]) {\n")
        f.write("    float x[NUM_FEATURES];\n")
        f.write("    for (int i = 0; i < NUM_FEATURES; i++) x[i] = (raw[i]-sc_mean[i])/sc_scale[i];\n")
        f.write("    int best = 0; float bestv = -1e30f;\n")
        f.write("    for (int c = 0; c < NUM_CLASSES; c++) {\n")
        f.write("        float s = B[c];\n")
        f.write("        for (int i = 0; i < NUM_FEATURES; i++) s += W[c][i]*x[i];\n")
        f.write("        if (s > bestv) { bestv = s; best = c; }\n")
        f.write("    }\n")
        f.write("    return best;\n}\n\n#endif\n")
    print(f"    svm_model.h: W {W.shape}")


# ==============================================================================
# main.c generico (inclui MODEL_FILE que define predict(), + dataset.h)
# ==============================================================================
def gerar_main_c():
    with open(os.path.join(OUT_DIR, "main.c"), "w") as f:
        f.write("""// main.c - driver generico de inferencia p/ gem5 (ROI via m5_dump_stats)
// Compilar definindo MODEL_FILE, ex.:
//   gcc -O2 -static -DMODEL_FILE='"rf_model.h"' main.c ... -o rf_bin
#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"   // define NUM_FEATURES (usado pelo predict do MODEL_FILE)
#include MODEL_FILE     // define predict() e NUM_CLASSES
#include <gem5/m5ops.h>

int main(int argc, char *argv[]) {
    int slice_id = (argc > 1) ? atoi(argv[1]) : 0;
    int num_cpus = (argc > 2) ? atoi(argv[2]) : 1;

    int per_core = NUM_SAMPLES / num_cpus;
    int start = slice_id * per_core;
    int end = (slice_id == num_cpus - 1) ? NUM_SAMPLES : start + per_core;

    int dummy = 0;

    // warmup (fora da ROI)
    int warm = 100; if (warm > NUM_SAMPLES) warm = NUM_SAMPLES;
    for (int i = 0; i < warm; i++) dummy += predict(can_data[i]);

    if (slice_id == 0) m5_dump_reset_stats(0, 0);

    // REGIAO DE INTERESSE
    for (int i = start; i < end; i++) dummy += predict(can_data[i]);

    if (slice_id == num_cpus - 1) m5_dump_stats(0, 0);

    printf("[Core %d] %d..%d dummy=%d\\n", slice_id, start, end - 1, dummy);
    return 0;
}
""")
    print("    main.c (generico)")


def main():
    print(f"Exportando modelos de {MODELS_DIR} para {OUT_DIR}\n")
    bundles = {k: carregar(k) for k in ["rf", "xgb", "mlp", "svm"]}
    bundles = {k: v for k, v in bundles.items() if v is not None}
    if not bundles:
        print("[ERRO] nenhum modelo .joblib encontrado. Rode treina_modelos.py antes.")
        return

    # class_names e ORDEM DE FEATURES de qualquer bundle (sao iguais entre modelos).
    # Usa as features salvas no .joblib -> dataset.h fica na ordem que o predict() espera.
    ref_bundle = next(iter(bundles.values()))
    class_names = ref_bundle["class_names"]
    feature_cols = ref_bundle.get("features", FEATURE_COLS)
    gerar_dataset_h(class_names, feature_cols)

    if "rf" in bundles:  print("[*] RF...");   exportar_rf(bundles["rf"])
    if "xgb" in bundles: print("[*] XGB...");  exportar_xgb(bundles["xgb"])
    if "mlp" in bundles: print("[*] MLP...");  exportar_mlp(bundles["mlp"])
    if "svm" in bundles: print("[*] SVM...");  exportar_svm(bundles["svm"])
    gerar_main_c()

    print(f"\nOK. Headers em {OUT_DIR}/")
    print("Cada modelo tem seu proprio predict(); compile 1 binario por modelo com")
    print("  -DMODEL_FILE='\"<modelo>_model.h\"'  (rf/xgb/mlp/svm)")
    print("\n[!] IMPORTANTE: valide os .h contra o sklearn antes do gem5 (script de conferencia).")


if __name__ == "__main__":
    main()
