"""
Re-parser COMPLETO: reprocessa TODAS as pastas m5out_* (inferencia 1-core,
multicore e treino MLP) e gera um CSV unico com TODAS as colunas que o pipeline
McPAT (mcpat_batch_runner.py / gen_mcpat_edge_config.py) precisa.

Motivo: os CSVs anteriores (resultados_sscad*.csv) nao tinham accesses de cache,
idle_cycles nem mpki_l3 - que o McPAT exige. Todos esses campos ESTAO nos
stats.txt; aqui reextraimos sem re-simular.

Colunas de saida (formato esperado pelo McPAT):
  cenario, grupo, cpu, cores, bp, l1d_size, l2_size,
  insts_totais, ciclos_totais, idle_cycles,
  l1d_accesses, l1i_accesses, l2_accesses, l3_accesses,
  mpki_l1d, mpki_l2, mpki_l3, branch_miss_rate,
  tempo_s, ipc_medio   (extras uteis)

Rodar de dentro de gem5/. Saida: resultados_gem5_completo.csv
"""
import os
import re
import glob
import pandas as pd

# defaults de config (para as pastas cujo nome nao codifica o parametro)
REF = {"l1d_size": "64kB", "l2_size": "2MB", "bp": "LTAGE", "prefetcher": "none"}


def extrair_dump_roi(lines):
    b = [i for i, ln in enumerate(lines) if "Begin Simulation Statistics" in ln]
    if len(b) >= 2:
        return lines[b[1]:(b[2] if len(b) > 2 else len(lines))]
    return lines


def stat(lines, pattern, soma=True):
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


def meta_do_nome(nome):
    """Reconstroi cpu/cores/bp/l1d/l2/modelo/grupo do nome da pasta."""
    cen = nome.replace("m5out_", "")
    m = {"cenario": cen, "cores": 1, "modelo": "?", **REF}
    p = cen.split("_")

    if p[0] == "Princ":          # Princ_<MOD>_<CPU>
        m.update(grupo="principal", modelo=p[1], cpu=p[2])
    elif p[0] == "DSE":          # DSE_<MOD>_<CPU>_<param>_<valor>
        m["modelo"] = p[1]; m["cpu"] = p[2]; resto = p[3:]
        if resto[:2] == ["l2", "size"]:
            m.update(grupo="dse_l2_size"); m["l2_size"] = resto[2]
        elif resto[:2] == ["l1d", "size"]:
            m.update(grupo="dse_l1d_size"); m["l1d_size"] = resto[2]
        elif resto[0] == "prefetcher":
            m.update(grupo="dse_prefetcher"); m["prefetcher"] = resto[1]
        elif resto[0] == "bp":
            m.update(grupo="dse_bp"); m["bp"] = resto[1]
    elif p[0].startswith("MC"):  # MC<n>_<MOD>_<CPU>
        nc = int(p[0][2:]); m.update(grupo=f"multicore_{nc}c", cores=nc,
                                     modelo=p[1], cpu=p[2])
    elif p[0] == "Treino":       # Treino_MLP_b<batch>_<CPU>
        m.update(grupo="treino_mlp", modelo="MLP", cpu=p[-1])
        m["batch"] = p[2].replace("b", "")
    return m


def processar(pasta):
    sf = os.path.join(pasta, "stats.txt")
    if not os.path.exists(sf):
        return None
    with open(sf) as f:
        L = extrair_dump_roi(f.readlines())

    insts = stat(L, r"^system\.cpu[0-9]*\.commitStats[0-9]*\.numInsts\s+", soma=True)
    ciclos = stat(L, r"system\.cpu[0-9]*\.numCycles\s+", soma=True)
    idle = stat(L, r"system\.cpu[0-9]*\.idleCycles\s+", soma=True)
    ipc = stat(L, r"^system\.cpu[0-9]*\.ipc\s+", soma=True)
    tempo = stat(L, r"^simSeconds\s+", soma=False)

    l1d_acc = stat(L, r"^system\.cpu[0-9]*\.dcache\.overallAccesses::total\s+", soma=True)
    l1i_acc = stat(L, r"^system\.cpu[0-9]*\.icache\.overallAccesses::total\s+", soma=True)
    l2_acc = stat(L, r"^system\.l2cache_?[0-9]*\.overallAccesses::total\s+", soma=True)
    l3_acc = stat(L, r"^system\.l3cache\.overallAccesses::total\s+", soma=True)

    l1d_m = stat(L, r"^system\.cpu[0-9]*\.dcache\.overallMisses::total\s+", soma=True)
    l2_m = stat(L, r"^system\.l2cache_?[0-9]*\.overallMisses::total\s+", soma=True)
    l3_m = stat(L, r"^system\.l3cache\.overallMisses::total\s+", soma=True)

    br_look = stat(L, r"system\.cpu[0-9]*\.branchPred\.lookups_[0-9]*::total\s+", soma=True)
    br_miss = stat(L, r"system\.cpu[0-9]*\.branchPred\.mispredicted_[0-9]*::total\s+", soma=True)

    m = meta_do_nome(os.path.basename(pasta.rstrip("/")))
    m.update({
        "insts_totais": insts, "ciclos_totais": ciclos, "idle_cycles": idle,
        "ipc_medio": ipc, "tempo_s": tempo,
        "l1d_accesses": l1d_acc, "l1i_accesses": l1i_acc,
        "l2_accesses": l2_acc, "l3_accesses": l3_acc,
        "mpki_l1d": (l1d_m * 1000 / insts) if insts else 0,
        "mpki_l2": (l2_m * 1000 / insts) if insts else 0,
        "mpki_l3": (l3_m * 1000 / insts) if insts else 0,
        "branch_miss_rate": (br_miss / br_look) if br_look else 0,
    })
    return m


def main():
    pastas = sorted(glob.glob("m5out_*"))
    print(f"Reprocessando {len(pastas)} pastas m5out_*...")
    rows = [r for p in pastas if (r := processar(p))]
    cols = ["cenario", "grupo", "modelo", "cpu", "cores", "bp", "prefetcher",
            "l1d_size",
            "l2_size", "insts_totais", "ciclos_totais", "idle_cycles",
            "l1d_accesses", "l1i_accesses", "l2_accesses", "l3_accesses",
            "mpki_l1d", "mpki_l2", "mpki_l3", "branch_miss_rate",
            "tempo_s", "ipc_medio"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols].sort_values(["grupo", "modelo", "cpu"])
    df.to_csv("resultados_gem5_completo.csv", index=False)
    print(f"OK: resultados_gem5_completo.csv ({len(df)} linhas)")
    # sanity: nenhuma coluna critica zerada onde nao deveria
    for c in ["insts_totais", "ciclos_totais", "l1d_accesses", "l2_accesses"]:
        z = (df[c] == 0).sum()
        print(f"  {c}: {z} zerados")


if __name__ == "__main__":
    main()
