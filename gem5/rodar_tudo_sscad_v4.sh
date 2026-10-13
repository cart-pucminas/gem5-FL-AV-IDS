#!/bin/bash
# ============================================================================
# Pipeline COMPLETO do SSCAD, versao final do calculo de potencia (v4 / opcao B):
#   1. DSE inferencia 1-core  (dse_sscad.py)                 -> 80 sims
#   2. DSE multicore          (dse_sscad_multicore.py)       -> 16 sims
#   3. DSE treino MLP (FL)    (dse_sscad_treino.py)          -> 6 sims
#   4. Re-parse com contagens (reparse_completo_mcpat_v4.py)
#   5. McPAT batch            (mcpat_batch_runner_sscad_v4.py)  [roda da RAIZ]
#   6. EDP                    (reparse_mcpat_edp_v4.py)         [roda da RAIZ]
# As etapas 1-3 sao as mesmas do rodar_tudo_sscad.sh; 4-6 usam contagens de atividade
# medidas pelo gem5 e P_chip = nucleos + caches + barramento.
#
# Rodar de dentro de gem5/, com a venv (pandas/tqdm) ativa:
#   cd <raiz do projeto>/gem5
#   source ../venv/bin/activate
#   nohup bash rodar_tudo_sscad_v4.sh > pipeline_sscad_v4.log 2>&1 &
#
# SO_POTENCIA=1 pula as etapas 1-3 (usa os m5out_* ja existentes).
# Para se qualquer etapa falhar (set -e). ~17h com 5k amostras (etapas 1-3).
# ============================================================================
set -e

GEM5_DIR="$(pwd)"                 # deve ser .../Arquitetura/gem5
RAIZ="$(cd .. && pwd)"            # .../Arquitetura
PY="${PY:-python3}"

marca() { echo ""; echo "============================================================"; echo ">>> $1  [$(date '+%H:%M:%S')]"; echo "============================================================"; }

if [ "${SO_POTENCIA:-0}" != "1" ]; then
    if [ ! -f "$RAIZ/experimentos/workload/c_export/bin/rf_arm" ]; then
        echo "[ERRO] binarios ARM nao encontrados em experimentos/workload/c_export/bin/"
        echo "       rode antes: cd experimentos/workload && bash compila_arm.sh && bash compila_train_arm.sh"
        exit 1
    fi
    marca "ETAPA 1/6 - DSE inferencia single-core (80 sims)"
    $PY dse_sscad.py
    marca "ETAPA 2/6 - DSE multicore (16 sims)"
    $PY dse_sscad_multicore.py
    marca "ETAPA 3/6 - DSE treino MLP / FL (6 sims)"
    $PY dse_sscad_treino.py
fi

marca "ETAPA 4/6 - Re-parse com contagens de atividade (102 sims -> resultados_gem5_completo_v4.csv)"
$PY reparse_completo_mcpat_v4.py

marca "ETAPA 5/6 - McPAT batch [rodando da raiz]"
cd "$RAIZ"
$PY experimentos/scripts/mcpat_batch_runner_sscad_v4.py

marca "ETAPA 6/6 - EDP (P_chip = nucleos + caches + barramento)"
$PY experimentos/scripts/reparse_mcpat_edp_v4.py

marca "PIPELINE COMPLETO"
echo "Saidas:"
echo "  gem5/resultados_sscad.csv / _multicore / _treino"
echo "  gem5/resultados_gem5_completo_v4.csv (102 sims)"
echo "  mcpat_xmls_sscad_v4/, mcpat_results_sscad_v4/"
echo "  experimentos/dados/resultados_edp_sscad_v4.csv (FINAL: perf + energia + EDP)"
echo "Fim: $(date '+%Y-%m-%d %H:%M:%S')"
