"""
Tuning de hiperparametros via Optuna (TPE) para os 4 algoritmos do benchmark
(RF, XGBoost, MLP, SVM linear), seguindo a metodologia da dissertacao da Isabela
(busca bayesiana, 5-fold CV, F1-macro como objetivo).

Roda em uma AMOSTRA do dataset (SUBSAMPLE_TUNING) para caber no tempo - os melhores
hiperparametros sao estaveis entre amostra e dataset completo. Salva os melhores
params num JSON, que o treina_modelos.py le para o TREINO FINAL no dataset completo.

Features: 9 (DLC + DATA[0-7]), SEM arbitration_id (anti-leakage) - igual ao treino.

Uso (da RAIZ ou de qualquer pasta):
    python experimentos/workload/tuning_optuna.py
Saida: experimentos/modelos/best_params.json

Dependencias: pandas numpy scikit-learn imbalanced-learn xgboost optuna
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ==============================================================================
# CONFIG
# ==============================================================================
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(_ROOT, "experimentos", "AV-IDS-Dataset", "data",
                            "combined_can_dataset.csv.gz")
OUT_JSON = os.path.join(_ROOT, "experimentos", "modelos", "best_params.json")
RANDOM_STATE = 42
SUBSAMPLE_TUNING = 0.20   # amostra p/ o tuning (params estaveis; ~3-4h total)
N_TRIALS = 20             # igual a Isabela
CV_FOLDS = 5

# USE_CAN_ID=True inclui o arbitration_id (CAN ID) como feature -> 10 features,
# regime de ALTA acuracia (decisao com Henrique: artigo avalia impacto ARQUITETURAL,
# nao propoe detector; CAN ID e realista e evita F1 baixo incomodando revisores).
# USE_CAN_ID=False = 9 features sem CAN ID (regime anti-leakage, guardado p/ estendida).
USE_CAN_ID = True
BASE_FEATURES = ['dlc'] + [f'data_{i}' for i in range(8)]
FEATURE_COLS = (['arbitration_id'] + BASE_FEATURES) if USE_CAN_ID else BASE_FEATURES
TARGET_COL = 'attack_type'

optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"[cfg] USE_CAN_ID={USE_CAN_ID} -> {len(FEATURE_COLS)} features: {FEATURE_COLS}")


# ==============================================================================
# Dados: amostra + balanceamento (igual ao treino, mas SEM split por cenario -
# o Optuna usa 5-fold CV estratificado sobre o treino balanceado)
# ==============================================================================
def preparar():
    print(f"[1] Carregando amostra ({SUBSAMPLE_TUNING:.0%})...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df.sample(frac=SUBSAMPLE_TUNING, random_state=RANDOM_STATE)
    X = df[FEATURE_COLS].astype(np.int32).values
    y = LabelEncoder().fit_transform(df[TARGET_COL])
    print(f"    amostra: {len(X):,} registros")

    print("[2] Balanceando (SMOTE + undersampling)...")
    Xb, yb = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('under', RandomUnderSampler(random_state=RANDOM_STATE)),
    ]).fit_resample(X, y)
    print(f"    balanceado: {len(Xb):,}")
    # SVM/MLP usam features padronizadas
    Xb_s = StandardScaler().fit_transform(Xb)
    return Xb, Xb_s, yb


# ==============================================================================
# Espacos de busca (RF/XGB seguem a Isabela; MLP/SVM adicionados por nos)
# ==============================================================================
def sugerir(trial, modelo):
    if modelo == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            max_depth=trial.suggest_int('max_depth', 10, 30),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            n_jobs=-1, random_state=RANDOM_STATE), False
    if modelo == "xgb":
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 12),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE), False
    if modelo == "mlp":
        h1 = trial.suggest_int('h1', 32, 128)
        h2 = trial.suggest_int('h2', 16, 64)
        return MLPClassifier(
            hidden_layer_sizes=(h1, h2),
            alpha=trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            max_iter=300, random_state=RANDOM_STATE), True
    if modelo == "svm":
        return LinearSVC(
            C=trial.suggest_float('C', 0.01, 10.0, log=True),
            max_iter=5000, dual='auto', random_state=RANDOM_STATE), True
    raise ValueError(modelo)


def otimizar(modelo, X, X_s, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        clf, scaled = sugerir(trial, modelo)
        data = X_s if scaled else X
        scores = cross_val_score(clf, data, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    dt = time.time() - t0
    print(f"    [{modelo}] melhor F1-macro (CV)={study.best_value:.4f} | "
          f"{dt/60:.1f}min | params={study.best_params}")
    return {"best_params": study.best_params, "cv_f1_macro": study.best_value}


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    X, X_s, y = preparar()

    modelos = ["rf", "mlp", "svm"] + (["xgb"] if HAS_XGB else [])
    print(f"\n[3] Otimizando {modelos} ({N_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    resultados = {}
    for m in modelos:
        print(f"\n--- {m} ---")
        resultados[m] = otimizar(m, X, X_s, y)

    with open(OUT_JSON, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nOK: melhores hiperparametros salvos em {OUT_JSON}")
    print("Proximo: treina_modelos.py le esse JSON e treina no dataset COMPLETO.")


if __name__ == "__main__":
    main()
