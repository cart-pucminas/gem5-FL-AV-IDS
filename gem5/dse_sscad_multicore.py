"""
DSE MULTICORE para o SSCAD (2a leva) - escalabilidade dos 4 algoritmos.
NAO altera dse.py/dse_minor.py/dse_multi.py (ERAD) nem dse_sscad.py (1-core).

Mede escalabilidade: 4 modelos x 2 CPUs (Minor,O3) x {2,4} cores, na config de
REFERENCIA (L2=2MB). O 1-core ja foi medido em dse_sscad.py (resultados_sscad.csv)
-> a curva de speedup sai cruzando este CSV com aquele.

= 4 x 2 x 2 = 16 simulacoes novas.

Multicore: cada core processa uma FATIA de amostras (main.c ja faz o slice via
argv). Core 0 reseta stats, ultimo core faz o dump -> stats do 2o dump (ROI)
agrega os N cores. Por isso insts/ciclos/misses sao SOMADOS entre cores; IPC
agregado = soma (throughput do sistema).

Rodar de dentro de gem5/:
    python dse_sscad_multicore.py
Saida: resultados_sscad_multicore.csv
"""
import os
import re
import subprocess
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

BIN_DIR = "experimentos/workload/c_export/bin"   # relativo a raiz (config prefixa ../)
MODELOS = {
    "RF":  f"{BIN_DIR}/rf_arm",
    "XGB": f"{BIN_DIR}/xgb_arm",
    "MLP": f"{BIN_DIR}/mlp_arm",
    "SVM": f"{BIN_DIR}/svm_arm",
}
CPUS = ["Minor", "O3"]
CORES = [2, 4]     # 1-core ja esta em resultados_sscad.csv
REF = {"l1d_size": "64kB", "l1d_assoc": 4, "l2_size": "2MB",
       "l3_size": "16MB", "prefetcher": "none", "bp": "LTAGE"}

WORKERS = 2   # multicore usa mais recursos por sim -> menos workers que o 1-core

configs = []
for cpu in CPUS:
    for mnome, mbin in MODELOS.items():
        for nc in CORES:
            configs.append({
                "nome": f"MC{nc}_{mnome}_{cpu}",
                "grupo": f"multicore_{nc}c", "modelo_nome": mnome, "cpu": cpu,
                "cores": nc, "binary": mbin, **REF,
            })


def extrair_dump_roi(lines):
    b = [i for i, ln in enumerate(lines) if "Begin Simulation Statistics" in ln]
    if len(b) >= 2:
        return lines[b[1]:(b[2] if len(b) > 2 else len(lines))]
    return lines


def extrair_stat(lines, pattern, soma=True):
    rx = re.compile(pattern)
    v = []
    for ln in lines:
        if rx.search(ln):
            try:
                v.append(float(ln.split()[1]))
            except Exception:
                pass
    if not v:
        return 0.0
    return sum(v) if soma else v[0]


def executar_simulacao(cfg):
    nome = cfg["nome"]
    outdir = f"m5out_{nome}"
    os.makedirs(outdir, exist_ok=True)
    comando = [
        "build/ARM/gem5.opt", f"--outdir={outdir}", "edge_config_mc.py",
        f"--cpu_type={cfg['cpu']}", f"--num_cpus={cfg['cores']}",
        f"--l1d_size={cfg['l1d_size']}", f"--l1d_assoc={cfg['l1d_assoc']}",
        f"--l2_size={cfg['l2_size']}", f"--l3_size={cfg['l3_size']}",
        f"--prefetcher={cfg['prefetcher']}", f"--bp_type={cfg['bp']}",
        f"--binary={cfg['binary']}",
    ]
    tqdm.write(f"[RUNNING] {nome}")
    with open(f"{outdir}/sim_log.txt", "w") as log:
        result = subprocess.run(comando, stdout=log, stderr=log)
    if result.returncode != 0:
        tqdm.write(f"[ERRO] {nome} (ver {outdir}/sim_log.txt)")
        return None
    stats_file = f"{outdir}/stats.txt"
    if not os.path.exists(stats_file):
        tqdm.write(f"[SEM_STATS] {nome}")
        return None
    with open(stats_file) as f:
        lines = extrair_dump_roi(f.readlines())

    tempo = extrair_stat(lines, r"^simSeconds\s+", soma=False)   # tempo e o mesmo p/ todos os cores
    # SOMA entre cores (cada core faz uma fatia):
    insts = extrair_stat(lines, r"^system\.cpu[0-9]*\.commitStats[0-9]*\.numInsts\s+", soma=True)
    ipc = extrair_stat(lines, r"^system\.cpu[0-9]*\.ipc\s+", soma=True)   # throughput agregado
    ciclos = extrair_stat(lines, r"system\.cpu[0-9]*\.numCycles\s+", soma=True)
    l1d = extrair_stat(lines, r"^system\.cpu[0-9]*\.dcache\.overallMisses::total\s+", soma=True)
    l2 = extrair_stat(lines, r"^system\.l2cache_?[0-9]*\.overallMisses::total\s+", soma=True)
    l3 = extrair_stat(lines, r"^system\.l3cache\.overallMisses::total\s+", soma=True)
    br_look = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.lookups_[0-9]*::total\s+", soma=True)
    br_miss = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.mispredicted_[0-9]*::total\s+", soma=True)

    tqdm.write(f"[OK] {nome}")
    return {
        "cenario": nome, "grupo": cfg["grupo"], "modelo": cfg["modelo_nome"],
        "cpu": cfg["cpu"], "cores": cfg["cores"],
        "l1d_size": cfg["l1d_size"], "l2_size": cfg["l2_size"],
        "prefetcher": cfg["prefetcher"], "bp": cfg["bp"],
        "tempo_s": tempo, "ciclos_totais": ciclos, "ipc_agregado": ipc,
        "insts_totais": insts,
        "mpki_l1d": (l1d * 1000 / insts) if insts else 0,
        "mpki_l2": (l2 * 1000 / insts) if insts else 0,
        "l1d_misses": l1d, "l2_misses": l2, "l3_misses": l3,
        "branch_miss_rate": (br_miss / br_look) if br_look else 0,
    }


if __name__ == "__main__":
    print("=" * 56)
    print(f" DSE MULTICORE SSCAD - {len(configs)} sims (cores {CORES})")
    print(f" Workers: {WORKERS}")
    print("=" * 56)
    faltando = [n for n, b in MODELOS.items()
                if not os.path.exists(os.path.join("..", b))]
    if faltando:
        print(f"[ERRO] binarios ausentes: {faltando}")
        raise SystemExit(1)

    resultados = []
    with mp.Pool(WORKERS) as pool:
        for r in tqdm(pool.imap_unordered(executar_simulacao, configs, chunksize=1),
                      total=len(configs), desc="DSE Multicore"):
            if r:
                resultados.append(r)

    df = pd.DataFrame(resultados).sort_values(["cores", "modelo", "cpu"])
    df.to_csv("resultados_sscad_multicore.csv", index=False)
    print(f"\n CSV salvo: resultados_sscad_multicore.csv ({len(df)} linhas)")
