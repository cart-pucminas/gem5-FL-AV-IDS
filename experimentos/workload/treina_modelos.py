"""
Treino unificado dos 4 algoritmos do benchmark SSCAD (borda veicular / IDS CAN).

Dataset: AV-IDS-Dataset (dissertacao Isabela / CArT-PUC), gerado no CARLA.
  experimentos/AV-IDS-Dataset/data/combined_can_dataset.csv.gz

Adaptado do 4_train_models.py da Isabela, mas com DECISOES PROPRIAS:
  - 9 FEATURES: DLC + DATA[0-7]. NAO usamos arbitration_id (=CAN ID) -> evita o
    data leakage corrigido no ERAD/disciplina. (A Isabela usava arbitration_id.)
  - 4 algoritmos do nosso benchmark arquitetural: RF, XGBoost, MLP, SVM LINEAR
    (em vez de DT/RF/XGBoost/CNN dela). Cada um tem perfil de CPU distinto.
    Nota: usamos SVM LINEAR (nao RBF) para que os 4 usem a base inteira com o
    MESMO metodo (o RBF nao escala: ~45h e ~2M support vectors no dataset full).
  - Split por cenario (via x clima x ataque) para evitar leakage - ideia dela.
  - Salva modelos em experimentos/modelos/*.joblib para o export-para-C posterior.

Uso (a partir da RAIZ do projeto):
    python experimentos/workload/treina_modelos.py

Dependencias: pandas numpy scikit-learn imbalanced-learn xgboost joblib
  (Optuna e tensorflow NAO sao necessarios nesta versao.)
"""

import gc
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost nao instalado - XGBoost sera pulado. (pip install xgboost)")

# ==============================================================================
# CONFIGURACAO
# ==============================================================================
# Caminhos resolvidos a partir da localizacao DESTE arquivo (funciona rodando de
# qualquer pasta). Raiz do projeto = dois niveis acima de experimentos/workload/.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(_ROOT, "experimentos", "AV-IDS-Dataset", "data",
                            "combined_can_dataset.csv.gz")
OUTPUT_DIR   = os.path.join(_ROOT, "experimentos", "modelos")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
SUBSAMPLE_FRAC = None   # None = dataset completo (treino FINAL). Ex.: 0.05 p/ teste rapido.

# USE_CAN_ID=True -> inclui arbitration_id (CAN ID) = 10 features, regime de ALTA
# acuracia (decisao com Henrique: foco e impacto ARQUITETURAL, CAN ID e realista e
# sobe o F1, blindando contra revisor). False = 9 features sem CAN ID (anti-leakage,
# guardado p/ versao estendida). DEVE bater com o USE_CAN_ID do tuning e do exporta.
USE_CAN_ID = True
BASE_FEATURES = ['dlc'] + [f'data_{i}' for i in range(8)]
FEATURE_COLS = (['arbitration_id'] + BASE_FEATURES) if USE_CAN_ID else BASE_FEATURES
TARGET_COL   = 'attack_type'
# "Cenario" = contexto operacional (via x clima), SEM attack_type. Incluir
# attack_type deixaria uma classe inteira (ex.: Normal) fora do teste -> F1=0
# artificial nessa classe. Agrupar so por via/clima garante todas as classes em
# treino E teste, preservando o anti-leakage (contextos separados).
SCENARIO_COLS = ['road_type', 'climate']

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. CARGA E PRE-PROCESSAMENTO (9 features, sem CAN ID)
# ==============================================================================
def carregar_dados():
    print(f"[1] Carregando {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"    registros: {len(df):,}")

    if SUBSAMPLE_FRAC:
        df = df.sample(frac=SUBSAMPLE_FRAC, random_state=RANDOM_STATE)
        print(f"    subamostrado para {len(df):,} ({SUBSAMPLE_FRAC:.0%})")

    df = df.dropna(subset=FEATURE_COLS + SCENARIO_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].astype(np.int32)
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    class_names = list(le.classes_)

    print(f"    features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"    classes: {class_names}")
    return X, y, df, class_names


# ==============================================================================
# 2. SPLIT POR CENARIO (evita data leakage - ideia adaptada da Isabela)
#    Cenarios inteiros (via x clima x ataque) vao para treino OU teste.
# ==============================================================================
def split_por_cenario(X, y, df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    cenarios = df[SCENARIO_COLS].drop_duplicates()
    n_test = max(1, int(len(cenarios) * test_size))
    rng = np.random.default_rng(random_state)
    idx_test = rng.choice(len(cenarios), size=n_test, replace=False)
    cen_test = cenarios.iloc[idx_test]

    mask_test = pd.Series(False, index=df.index)
    for _, row in cen_test.iterrows():
        cond = pd.Series(True, index=df.index)
        for col in SCENARIO_COLS:
            cond &= (df[col] == row[col])
        mask_test |= cond

    mask_test = mask_test.values
    print(f"[2] Split por cenario - treino: {(~mask_test).sum():,} | teste: {mask_test.sum():,}")
    return X[~mask_test], X[mask_test], y[~mask_test], y[mask_test]


# ==============================================================================
# 3. BALANCEAMENTO HIBRIDO (SMOTE + undersampling) - so no treino
# ==============================================================================
def balancear(X_tr, y_tr, random_state=RANDOM_STATE):
    print("[3] Balanceando treino (SMOTE + RandomUnderSampler)...")
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('under', RandomUnderSampler(random_state=random_state)),
    ])
    X_bal, y_bal = pipe.fit_resample(X_tr, y_tr)
    print(f"    treino balanceado: {len(X_bal):,}")
    return X_bal, y_bal


# ==============================================================================
# 4. OS 4 ALGORITMOS DO BENCHMARK ('scaled': precisa padronizar? SVM/MLP sim)
# ==============================================================================
def carregar_best_params():
    """Le os hiperparametros otimizados pelo Optuna (best_params.json), se existir.
    Sem o arquivo, retorna {} e usa os defaults (fallback seguro)."""
    import json
    path = os.path.join(OUTPUT_DIR, "best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            bp = {k: v.get("best_params", {}) for k, v in json.load(f).items()}
        print(f"[*] Usando hiperparametros do Optuna: {path}")
        return bp
    print("[!] best_params.json ausente - usando hiperparametros DEFAULT "
          "(rode tuning_optuna.py p/ otimizar).")
    return {}


def construir_modelos():
    bp = carregar_best_params()

    # RF - defaults; sobrescritos pelo Optuna se disponivel
    rf_p = {"n_estimators": 500, "max_depth": 15, "bootstrap": True}
    rf_p.update(bp.get("rf", {}))
    # MLP - Optuna otimiza h1/h2/alpha/lr; traduz p/ hidden_layer_sizes
    mlp_o = bp.get("mlp", {})
    mlp_hidden = (mlp_o.get("h1", 64), mlp_o.get("h2", 32))
    mlp_extra = {k: mlp_o[k] for k in ("alpha",) if k in mlp_o}
    if "lr" in mlp_o:
        mlp_extra["learning_rate_init"] = mlp_o["lr"]
    # SVM linear
    svm_p = {"C": 1.0}
    svm_p.update(bp.get("svm", {}))
    # XGBoost
    xgb_p = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.3}
    xgb_p.update(bp.get("xgb", {}))

    modelos = {
        "rf": {"nome": "Random Forest", "scaled": False,
               "clf": RandomForestClassifier(
                   random_state=RANDOM_STATE, n_jobs=-1, **rf_p)},
        "mlp": {"nome": "MLP (rede rasa)", "scaled": True,
                "clf": MLPClassifier(
                    hidden_layer_sizes=mlp_hidden, activation='relu',
                    max_iter=300, random_state=RANDOM_STATE, **mlp_extra)},
        # SVM LINEAR (LinearSVC): escala p/ milhoes de amostras em segundos
        # (treino O(n)), permitindo usar a base inteira como os demais - mesma
        # metodologia p/ os 4. Inferencia = 1 produto interno w.x+b (muito leve),
        # o "piso" de custo do benchmark, contrastando com o RF (arvores, "teto").
        "svm": {"nome": "SVM (linear)", "scaled": True,
                "clf": LinearSVC(max_iter=5000, dual='auto',
                                 random_state=RANDOM_STATE, **svm_p)},
    }
    if HAS_XGB:
        modelos["xgb"] = {"nome": "XGBoost", "scaled": False,
                          "clf": XGBClassifier(
                              tree_method='hist', n_jobs=-1,
                              random_state=RANDOM_STATE, **xgb_p)}
    return modelos


# ==============================================================================
# 5. TREINO + VALIDACAO
# ==============================================================================
def treinar_e_validar(dados, modelos):
    resultados = []
    for key, m in modelos.items():
        print(f"\n{'='*60}\n[*] Treinando: {m['nome']}\n{'='*60}")
        Xtr = dados["X_tr_scaled"] if m["scaled"] else dados["X_tr"]
        Xte = dados["X_te_scaled"] if m["scaled"] else dados["X_te"]

        t0 = time.time()
        m["clf"].fit(Xtr, dados["y_tr"])
        t_train = time.time() - t0

        y_pred = m["clf"].predict(Xte)
        acc = accuracy_score(dados["y_te"], y_pred)
        f1m = f1_score(dados["y_te"], y_pred, average='macro')
        print(f"    treino: {t_train:.1f}s | acc: {acc:.4f} | F1-macro: {f1m:.4f}")
        print(classification_report(dados["y_te"], y_pred,
                                    target_names=dados["class_names"],
                                    digits=4, zero_division=0))

        joblib.dump({"clf": m["clf"], "scaled": m["scaled"],
                     "scaler": dados["scaler"], "features": FEATURE_COLS,
                     "class_names": dados["class_names"]},
                    f"{OUTPUT_DIR}/{key}.joblib")
        resultados.append((m["nome"], acc, f1m, t_train))
    return resultados


def main():
    X, y, df, class_names = carregar_dados()
    X_tr_raw, X_te, y_tr_raw, y_te = split_por_cenario(X, y, df)
    del df
    gc.collect()

    X_tr, y_tr = balancear(X_tr_raw, y_tr_raw)
    del X_tr_raw, y_tr_raw
    gc.collect()

    # StandardScaler p/ SVM e MLP (arvores usam X cru)
    scaler = StandardScaler().fit(X_tr)
    dados = {
        "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
        "X_tr_scaled": scaler.transform(X_tr),
        "X_te_scaled": scaler.transform(X_te),
        "scaler": scaler, "class_names": class_names,
    }

    resultados = treinar_e_validar(dados, construir_modelos())

    print(f"\n{'='*60}\nRESUMO (AV-IDS, 9 features sem CAN ID)\n{'='*60}")
    print(f"{'Algoritmo':<20}{'Acuracia':>12}{'F1-macro':>12}{'Treino(s)':>12}")
    for nome, acc, f1m, tt in resultados:
        print(f"{nome:<20}{acc:>12.4f}{f1m:>12.4f}{tt:>12.1f}")

    # Salva o resumo em CSV (o treino final demora - nao perder os numeros)
    resumo_path = os.path.join(OUTPUT_DIR, "resumo_treino.csv")
    pd.DataFrame(resultados,
                 columns=["algoritmo", "acuracia", "f1_macro", "treino_s"]
                 ).to_csv(resumo_path, index=False)
    print(f"\nResumo salvo em: {resumo_path}")

    # Alerta anti-leakage: SO faz sentido no regime SEM CAN ID. Com CAN ID (regime
    # de alta acuracia, intencional), acuracia alta e esperada e nao e suspeita.
    if not USE_CAN_ID and any(acc > 0.99 for _, acc, _, _ in resultados):
        print("\n[!] ATENCAO: acuracia > 0,99 SEM CAN ID e suspeito de leakage - "
              "conferir se 'arbitration_id' nao entrou nas features.")

    print(f"\nModelos salvos em '{OUTPUT_DIR}/'. Proximo: export-para-C.")


if __name__ == "__main__":
    main()
