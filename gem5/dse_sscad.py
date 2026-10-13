"""
DSE para o artigo SSCAD (benchmark de cargas de IDS veicular).
NAO altera dse.py / dse_minor.py / dse_multi.py (ERAD/disciplina) - este e novo.

Eixo do SSCAD: comparar 4 ALGORITMOS (RF, XGBoost, MLP, SVM linear) x microarquitetura
(Minor in-order vs O3 out-of-order), em desempenho E energia (via McPAT depois).

Matriz (single-core; a comparacao principal do SSCAD e 1 nucleo):
  P) PRINCIPAL: 4 modelos x {Minor,O3}, memoria de REFERENCIA (L2=2MB, achado do ERAD).
  M) DSE MEMORIA: varre L2, L1D, prefetcher e BP - nos 4 modelos x {Minor,O3}.
     (decisao 20/jul: rodar nos 4 hoje p/ medir comportamento e tempo real.)

Binarios esperados em BIN_DIR: rf_arm, xgb_arm, mlp_arm, svm_arm
  (gerados por experimentos/workload/compila_arm.sh - aarch64 estatico).

Rodar de dentro de gem5/ (junto de build/ e edge_config_mc.py):
    python dse_sscad.py
Ajuste WORKERS conforme a maquina. Saida: resultados_sscad.csv
"""

import os
import re
import subprocess
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

# ============================================================
# MODELOS (binarios ARM dos 4 algoritmos)
# ============================================================
# O edge_config_mc.py faz bin_path = f'../{args.binary}' (prefixa '../').
# Como o gem5 roda de dentro de gem5/, o '../' resolve p/ a RAIZ do projeto.
# Entao o --binary deve ser o caminho relativo A RAIZ (sem o ../ inicial).
BIN_DIR = "experimentos/workload/c_export/bin"   # relativo a raiz do projeto
# (a checagem de existencia usa os.path.join("..", b), pois roda de dentro de gem5/)
MODELOS = {
    "RF":  f"{BIN_DIR}/rf_arm",
    "XGB": f"{BIN_DIR}/xgb_arm",
    "MLP": f"{BIN_DIR}/mlp_arm",
    "SVM": f"{BIN_DIR}/svm_arm",
}

# CPUs: o edge_config_mc.py precisa suportar --cpu_type. Confirmar flag.
CPUS = ["Minor", "O3"]

# Referencia de memoria (sweet spot L2=2MB descoberto no ERAD)
REF = {"l1d_size": "64kB", "l1d_assoc": 4, "l2_size": "2MB",
       "l3_size": "16MB", "prefetcher": "none", "bp": "LTAGE"}

# ============================================================
# CONSTRUCAO DA MATRIZ DE CONFIGS
# ============================================================
configs = []

# --- P) PRINCIPAL: 4 modelos x 2 CPUs, memoria de referencia ---
for cpu in CPUS:
    for mnome, mbin in MODELOS.items():
        configs.append({
            "nome": f"Princ_{mnome}_{cpu}",
            "grupo": "principal", "modelo_nome": mnome, "cpu": cpu,
            "cores": 1, "binary": mbin, **REF,
        })

# --- M) DSE MEMORIA: varre 1 parametro por vez (mantendo os demais em REF) ---
VARR = {
    "l2_size":    ["512kB", "1MB", "2MB", "4MB"],
    "l1d_size":   ["16kB", "32kB", "64kB", "128kB"],
    "prefetcher": ["none", "stride"],
    "bp":         ["BiModeBP", "TournamentBP", "LTAGE"],
}
for cpu in CPUS:
    for mnome, mbin in MODELOS.items():
        for param, valores in VARR.items():
            for v in valores:
                cfg = {"nome": f"DSE_{mnome}_{cpu}_{param}_{v}",
                       "grupo": f"dse_{param}", "modelo_nome": mnome, "cpu": cpu,
                       "cores": 1, "binary": mbin, **REF}
                cfg[param] = v          # sobrescreve o parametro variado
                # evita duplicar o ponto de referencia (ja esta no grupo principal)
                if cfg[param] == REF.get(param):
                    continue
                configs.append(cfg)

# de-dup por nome (caso algum ponto coincida)
_seen, _uniq = set(), []
for c in configs:
    if c["nome"] not in _seen:
        _seen.add(c["nome"]); _uniq.append(c)
configs = _uniq

# ============================================================
WORKERS = 3   # ajuste p/ sua maquina (cada worker roda 1 gem5)

# ============================================================
# ISOLA A ROI (2o dump) - mesma logica do dse_minor.py do ERAD.
# O stats.txt tem 3 dumps: 1=warmup, 2=ROI (m5_dump_reset->m5_dump_stats), 3=final.
# SEM isso, extrair_stat somaria/pegaria os 3 dumps e contaminaria a medicao.
# ============================================================
def extrair_dump_roi(lines):
    boundaries = [i for i, ln in enumerate(lines)
                  if "Begin Simulation Statistics" in ln]
    if len(boundaries) >= 2:
        start = boundaries[1]
        end = boundaries[2] if len(boundaries) > 2 else len(lines)
        return lines[start:end]
    return lines


# ============================================================
# PARSER ROBUSTO (mesmo do dse.py)
# ============================================================
def extrair_stat(lines, pattern, soma=True):
    regex = re.compile(pattern)
    valores = []
    for linha in lines:
        if regex.search(linha):
            try:
                valores.append(float(linha.split()[1]))
            except Exception:
                pass
    if not valores:
        return 0.0
    return sum(valores) if soma else valores[0]

# ============================================================
# EXECUCAO
# ============================================================
def executar_simulacao(cfg):
    nome = cfg["nome"]
    outdir = f"m5out_{nome}"
    os.makedirs(outdir, exist_ok=True)

    comando = [
        "build/ARM/gem5.opt",
        f"--outdir={outdir}",
        "edge_config_mc.py",
        f"--cpu_type={cfg['cpu']}",
        f"--num_cpus={cfg['cores']}",
        f"--l1d_size={cfg['l1d_size']}",
        f"--l1d_assoc={cfg['l1d_assoc']}",
        f"--l2_size={cfg['l2_size']}",
        f"--l3_size={cfg['l3_size']}",
        f"--prefetcher={cfg['prefetcher']}",
        f"--bp_type={cfg['bp']}",
        f"--binary={cfg['binary']}",
    ]

    tqdm.write(f"[RUNNING] {nome}")
    with open(f"{outdir}/sim_log.txt", "w") as log:
        result = subprocess.run(comando, stdout=log, stderr=log)
    if result.returncode != 0:
        tqdm.write(f"[ERRO] {nome} - falha no gem5 (ver {outdir}/sim_log.txt)")
        return None

    stats_file = f"{outdir}/stats.txt"
    if not os.path.exists(stats_file):
        tqdm.write(f"[SEM_STATS] {nome}")
        return None
    with open(stats_file) as f:
        lines = f.readlines()

    lines = extrair_dump_roi(lines)   # <-- mede SO a ROI (2o dump)

    tempo = extrair_stat(lines, r"^simSeconds\s+", soma=False)
    insts = extrair_stat(lines, r"^system\.cpu[0-9]*\.commitStats[0-9]*\.numInsts\s+")
    ipc = extrair_stat(lines, r"^system\.cpu[0-9]*\.ipc\s+", soma=True)
    l1d_misses = extrair_stat(lines, r"^system\.cpu[0-9]*\.dcache\.overallMisses::total\s+", soma=True)
    l2_misses = extrair_stat(lines, r"^system\.l2cache_?[0-9]*\.overallMisses::total\s+", soma=True)
    l3_misses = extrair_stat(lines, r"^system\.l3cache\.overallMisses::total\s+", soma=True)
    # NOTA: em single-core o gem5 usa 'system.cpu.' (SEM numero) -> [0-9]* (nao \d+).
    # branchPred.lookups/mispredicted existem em Minor E O3 (mesmo componente);
    # commit.branchMispredicts so existe no O3 (por isso o Minor zerava antes).
    branch_lookups = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.lookups_[0-9]*::total\s+", soma=True)
    branch_misses = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.mispredicted_[0-9]*::total\s+", soma=True)
    ciclos = extrair_stat(lines, r"system\.cpu[0-9]*\.numCycles\s+", soma=True)

    mpki_l1d = (l1d_misses * 1000 / insts) if insts else 0
    mpki_l2 = (l2_misses * 1000 / insts) if insts else 0
    branch_miss_rate = (branch_misses / branch_lookups) if branch_lookups else 0

    tqdm.write(f"[OK] {nome}")
    return {
        "cenario": nome, "grupo": cfg["grupo"], "modelo": cfg["modelo_nome"],
        "cpu": cfg["cpu"], "cores": cfg["cores"],
        "l1d_size": cfg["l1d_size"], "l2_size": cfg["l2_size"],
        "prefetcher": cfg["prefetcher"], "bp": cfg["bp"],
        "tempo_s": tempo, "ciclos_totais": ciclos, "ipc_medio": ipc,
        "insts_totais": insts, "mpki_l1d": mpki_l1d, "mpki_l2": mpki_l2,
        "l1d_misses": l1d_misses, "l2_misses": l2_misses, "l3_misses": l3_misses,
        "branch_miss_rate": branch_miss_rate,
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 56)
    print(f" DSE SSCAD - {len(configs)} simulacoes ({len(MODELOS)} modelos x "
          f"{len(CPUS)} CPUs + varredura)")
    print(f" Workers: {WORKERS}")
    print("=" * 56)

    # checagem: binarios existem? (este script roda em gem5/, entao usa CHECK_DIR)
    faltando = [n for n, b in MODELOS.items()
                if not os.path.exists(os.path.join("..", b))]
    if faltando:
        print(f"[ERRO] binarios ausentes: {faltando}")
        print(f"       compile antes com experimentos/workload/compila_arm.sh")
        raise SystemExit(1)

    resultados = []
    with mp.Pool(WORKERS) as pool:
        for r in tqdm(pool.imap_unordered(executar_simulacao, configs, chunksize=1),
                      total=len(configs), desc="DSE SSCAD"):
            if r:
                resultados.append(r)

    df = pd.DataFrame(resultados).sort_values(by=["grupo", "modelo", "cpu"])
    df.to_csv("resultados_sscad.csv", index=False)
    print("\n" + "=" * 56)
    print(f" CSV salvo: resultados_sscad.csv ({len(df)} linhas)")
    print("=" * 56)
