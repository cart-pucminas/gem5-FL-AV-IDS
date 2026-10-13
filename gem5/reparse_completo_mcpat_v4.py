"""
Re-parser v4 (opcao B): estende o reparse_completo_mcpat.py com TODAS as contagens de
atividade que o McPAT usa, lidas da ROI (2o dump) dos stats.txt ja em disco. Nao re-simula.

Motivo: na opcao A (mcpat_batch_runner_sscad.py) as contagens de acesso por estrutura eram
fracoes fixas do numero de instrucoes. Aqui elas passam a vir do gem5.

As colunas da opcao A sao geradas pelas MESMAS funcoes do reparse_completo_mcpat.py
(importadas), de modo que so as colunas novas diferem. Contagens = soma de todos os nucleos.

Colunas novas:
  int_insts, fp_insts (FP + SIMD/vetor: no ARM o gem5 conta FP escalar como vetor),
  branch_insts, load_insts, store_insts, ialu_acc, fpu_acc, mul_acc, btb_acc,
  l1d_read_acc, l1d_write_acc, l1d_read_miss, l1d_write_miss, l1i_miss,
  l2_read_acc, l2_write_acc, l2_read_miss, l2_write_miss,
  l3_read_acc, l3_write_acc, l3_read_miss, l3_write_miss, mem_reads, mem_writes

Uso (de dentro de gem5/):
    python reparse_completo_mcpat_v4.py                       # geracao v3 (m5out_* aqui)
    python reparse_completo_mcpat_v4.py \
        --pasta ../experimentos/dados/arquivo_5k_v2_semcanid/m5out \
        --saida ../experimentos/dados/arquivo_5k_v2_semcanid/csv/resultados_gem5_completo_v4.csv
"""
import argparse
import glob
import os

import pandas as pd

from reparse_completo_mcpat import extrair_dump_roi, processar, stat

CPU = r"^system\.cpu[0-9]*\."
L2 = r"^system\.l2cache_?[0-9]*\."
L3 = r"^system\.l3cache\."

# coluna -> lista de regex somadas (todas terminam no valor da estatistica)
NOVAS = {
    "int_insts":      [CPU + r"commitStats[0-9]*\.numIntInsts\s+"],
    "fp_insts":       [CPU + r"commitStats[0-9]*\.numFpInsts\s+",
                       CPU + r"commitStats[0-9]*\.numVecInsts\s+"],
    "branch_insts":   [CPU + r"commitStats[0-9]*\.committedControl::IsControl\s+"],
    "load_insts":     [CPU + r"commitStats[0-9]*\.numLoadInsts\s+"],
    "store_insts":    [CPU + r"commitStats[0-9]*\.numStoreInsts\s+"],
    "ialu_acc":       [CPU + r"executeStats[0-9]*\.numIntAluAccesses\s+"],
    "fpu_acc":        [CPU + r"executeStats[0-9]*\.numFpAluAccesses\s+",
                       CPU + r"executeStats[0-9]*\.numVecAluAccesses\s+"],
    "mul_acc":        [CPU + r"commitStats[0-9]*\.committedInstType::IntMult\s+",
                       CPU + r"commitStats[0-9]*\.committedInstType::IntDiv\s+"],
    "btb_acc":        [CPU + r"branchPred\.BTBLookups\s+"],
    "l1d_read_acc":   [CPU + r"dcache\.ReadReq\.accesses::total\s+"],
    "l1d_write_acc":  [CPU + r"dcache\.WriteReq\.accesses::total\s+"],
    "l1d_read_miss":  [CPU + r"dcache\.ReadReq\.misses::total\s+"],
    "l1d_write_miss": [CPU + r"dcache\.WriteReq\.misses::total\s+"],
    "l1i_miss":       [CPU + r"icache\.overallMisses::total\s+"],
    # L2/L3: leituras = acessos de demanda+prefetch; escritas = writebacks vindos de cima
    "l2_read_acc":    [L2 + r"overallAccesses::total\s+"],
    "l2_write_acc":   [L2 + r"WritebackDirty\.accesses::total\s+",
                       L2 + r"WritebackClean\.accesses::total\s+"],
    "l2_read_miss":   [L2 + r"overallMisses::total\s+"],
    "l2_write_miss":  [L2 + r"WritebackDirty\.misses::total\s+",
                       L2 + r"WritebackClean\.misses::total\s+"],
    "l3_read_acc":    [L3 + r"overallAccesses::total\s+"],
    "l3_write_acc":   [L3 + r"WritebackDirty\.accesses::total\s+",
                       L3 + r"WritebackClean\.accesses::total\s+"],
    "l3_read_miss":   [L3 + r"overallMisses::total\s+"],
    "l3_write_miss":  [L3 + r"WritebackDirty\.misses::total\s+",
                       L3 + r"WritebackClean\.misses::total\s+"],
    "mem_reads":      [r"^system\.mem_ctrl\.readReqs\s+"],
    "mem_writes":     [r"^system\.mem_ctrl\.writeReqs\s+"],
}

COLS_A = ["cenario", "grupo", "modelo", "cpu", "cores", "bp", "prefetcher", "l1d_size",
          "l2_size", "insts_totais", "ciclos_totais", "idle_cycles",
          "l1d_accesses", "l1i_accesses", "l2_accesses", "l3_accesses",
          "mpki_l1d", "mpki_l2", "mpki_l3", "branch_miss_rate", "tempo_s", "ipc_medio"]


def processar_v4(pasta):
    m = processar(pasta)          # colunas da opcao A, logica identica
    if m is None:
        return None
    with open(os.path.join(pasta, "stats.txt")) as f:
        L = extrair_dump_roi(f.readlines())
    for col, regexes in NOVAS.items():
        m[col] = sum(stat(L, rx, soma=True) for rx in regexes)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=".", help="pasta que contem os m5out_*")
    ap.add_argument("--saida", default="resultados_gem5_completo_v4.csv")
    a = ap.parse_args()

    pastas = sorted(glob.glob(os.path.join(a.pasta, "m5out_*")))
    print(f"Reprocessando {len(pastas)} pastas m5out_* em {a.pasta} ...")
    rows = [r for p in pastas if (r := processar_v4(p))]
    df = pd.DataFrame(rows)
    cols = COLS_A + list(NOVAS)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols].sort_values(["grupo", "modelo", "cpu"])
    df.to_csv(a.saida, index=False)
    print(f"OK: {a.saida} ({len(df)} linhas)")
    for c in ["insts_totais", "int_insts", "ialu_acc", "l1d_read_acc", "l1i_accesses",
              "l2_read_acc", "l3_read_acc", "mem_reads", "btb_acc"]:
        print(f"  {c}: {(df[c] == 0).sum()} zerados")


if __name__ == "__main__":
    main()
