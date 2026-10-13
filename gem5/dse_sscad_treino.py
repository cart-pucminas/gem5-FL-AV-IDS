"""
DSE do TREINO LOCAL do MLP (parte Federated Learning) para o SSCAD.
NAO altera os outros scripts DSE.

Caracteriza o custo (perf+energia via McPAT depois) de 1 round de treino local
do MLP on-device - a carga FedAvg-nativa. 3 tamanhos de batch (32/64/128) x
2 CPUs (Minor,O3) = 6 simulacoes, config de referencia (L2=2MB), single-core
(cada cliente FL = 1 core).

Binarios: c_export/bin/train_mlp_b{32,64,128} (compila_train_arm.sh).
Rodar de dentro de gem5/. Saida: resultados_sscad_treino.csv
"""
import os
import re
import subprocess
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

BIN_DIR = "experimentos/workload/c_export/bin"   # relativo a raiz (config prefixa ../)
BATCHES = [32, 64, 128]
CPUS = ["Minor", "O3"]
REF = {"l1d_size": "64kB", "l1d_assoc": 4, "l2_size": "2MB",
       "l3_size": "16MB", "prefetcher": "none", "bp": "LTAGE"}
WORKERS = 2

configs = []
for cpu in CPUS:
    for bsz in BATCHES:
        configs.append({
            "nome": f"Treino_MLP_b{bsz}_{cpu}",
            "grupo": "treino_mlp", "batch": bsz, "cpu": cpu,
            "cores": 1, "binary": f"{BIN_DIR}/train_mlp_b{bsz}", **REF,
        })


def extrair_dump_roi(lines):
    b = [i for i, ln in enumerate(lines) if "Begin Simulation Statistics" in ln]
    if len(b) >= 2:
        return lines[b[1]:(b[2] if len(b) > 2 else len(lines))]
    return lines


def extrair_stat(lines, pattern, soma=True):
    rx = re.compile(pattern)
    v = [float(ln.split()[1]) for ln in lines if rx.search(ln)
         and _num(ln.split()[1])]
    if not v:
        return 0.0
    return sum(v) if soma else v[0]


def _num(s):
    try:
        float(s); return True
    except Exception:
        return False


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
        r = subprocess.run(comando, stdout=log, stderr=log)
    if r.returncode != 0:
        tqdm.write(f"[ERRO] {nome} (ver {outdir}/sim_log.txt)")
        return None
    sf = f"{outdir}/stats.txt"
    if not os.path.exists(sf):
        tqdm.write(f"[SEM_STATS] {nome}")
        return None
    with open(sf) as f:
        lines = extrair_dump_roi(f.readlines())

    tempo = extrair_stat(lines, r"^simSeconds\s+", soma=False)
    insts = extrair_stat(lines, r"^system\.cpu[0-9]*\.commitStats[0-9]*\.numInsts\s+")
    ipc = extrair_stat(lines, r"^system\.cpu[0-9]*\.ipc\s+", soma=True)
    ciclos = extrair_stat(lines, r"system\.cpu[0-9]*\.numCycles\s+", soma=True)
    l1d = extrair_stat(lines, r"^system\.cpu[0-9]*\.dcache\.overallMisses::total\s+", soma=True)
    l2 = extrair_stat(lines, r"^system\.l2cache_?[0-9]*\.overallMisses::total\s+", soma=True)
    l3 = extrair_stat(lines, r"^system\.l3cache\.overallMisses::total\s+", soma=True)
    br_look = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.lookups_[0-9]*::total\s+", soma=True)
    br_miss = extrair_stat(lines, r"system\.cpu[0-9]*\.branchPred\.mispredicted_[0-9]*::total\s+", soma=True)

    tqdm.write(f"[OK] {nome}")
    return {
        "cenario": nome, "grupo": cfg["grupo"], "batch": cfg["batch"],
        "cpu": cfg["cpu"], "cores": cfg["cores"], "l2_size": cfg["l2_size"],
        "bp": cfg["bp"], "tempo_s": tempo, "ciclos_totais": ciclos,
        "ipc_medio": ipc, "insts_totais": insts,
        "mpki_l1d": (l1d * 1000 / insts) if insts else 0,
        "mpki_l2": (l2 * 1000 / insts) if insts else 0,
        "l1d_misses": l1d, "l2_misses": l2, "l3_misses": l3,
        "branch_miss_rate": (br_miss / br_look) if br_look else 0,
    }


if __name__ == "__main__":
    print("=" * 56)
    print(f" DSE TREINO MLP (FL) - {len(configs)} sims (batches {BATCHES})")
    print("=" * 56)
    faltando = [c["nome"] for c in configs
                if not os.path.exists(os.path.join("..", c["binary"]))]
    if faltando:
        print(f"[ERRO] binarios ausentes p/: {faltando}")
        print("       compile antes com experimentos/workload/compila_train_arm.sh")
        raise SystemExit(1)

    resultados = []
    with mp.Pool(WORKERS) as pool:
        for r in tqdm(pool.imap_unordered(executar_simulacao, configs, chunksize=1),
                      total=len(configs), desc="DSE Treino MLP"):
            if r:
                resultados.append(r)

    df = pd.DataFrame(resultados).sort_values(["batch", "cpu"])
    df.to_csv("resultados_sscad_treino.csv", index=False)
    print(f"\n CSV salvo: resultados_sscad_treino.csv ({len(df)} linhas)")
