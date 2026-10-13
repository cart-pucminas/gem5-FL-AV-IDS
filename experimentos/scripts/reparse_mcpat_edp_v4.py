"""
EDP v4 (opcao B): extrai a potencia das saidas do McPAT geradas por
mcpat_batch_runner_sscad_v4.py e monta o CSV final, no mesmo formato do
resultados_edp_sscad.csv (opcao A) mais tres colunas de transparencia.

Diferenca em relacao a opcao A (reparse_mcpat_edp.py):
  P_chip = nucleos (com L1 e L2 privada) + L3 + barramento, somando Runtime Dynamic e
  leakage (subthreshold + gate) de cada bloco do resumo 'Processor:'.
  Ficam FORA o controlador de memoria e os blocos de I/O (NIU, PCIe, flash), que a opcao A
  incluia por usar o total do bloco 'Processor:'. EDP = P_chip x T^2.

Colunas extras: mc_w (controlador de memoria excluido), io_w (NIU+PCIe+flash excluidos),
mcpat_total_w (total do bloco Processor, para comparar com a opcao A).

Opcao B' (--com-mc): mesmas saidas do McPAT da opcao B, mas P_chip = total do bloco
'Processor:' (inclui controlador de memoria integrado e I/O), ou seja, a MESMA fronteira de
potencia da opcao A, so com as entradas corrigidas. Saida com sufixo _v4mc.

Rodar da RAIZ:
    python experimentos/scripts/reparse_mcpat_edp_v4.py              # opcao B,  geracao v3
    python experimentos/scripts/reparse_mcpat_edp_v4.py --v2         # opcao B,  sem CAN ID
    python experimentos/scripts/reparse_mcpat_edp_v4.py --com-mc     # opcao B', geracao v3
    python experimentos/scripts/reparse_mcpat_edp_v4.py --com-mc --v2
Saida: experimentos/dados/resultados_edp_sscad_v4{,mc}.csv
       (ou experimentos/dados/arquivo_5k_v2_semcanid/csv/resultados_edp_sscad_v4{,mc}.csv)
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

V2 = "experimentos/dados/arquivo_5k_v2_semcanid"
PERFIS = {
    "v3": dict(gem5_csv="gem5/resultados_gem5_completo_v4.csv",
               results_dir="mcpat_results_sscad_v4",
               saida="experimentos/dados/resultados_edp_sscad_v4.csv"),
    "v2": dict(gem5_csv=f"{V2}/csv/resultados_gem5_completo_v4.csv",
               results_dir=f"{V2}/mcpat_results_sscad_v4",
               saida=f"{V2}/csv/resultados_edp_sscad_v4.csv"),
}
# colunas do gem5 na mesma ordem do CSV final da opcao A
COLS_GEM5 = ["cenario", "grupo", "modelo", "cpu", "cores", "bp", "prefetcher", "l1d_size",
             "l2_size", "insts_totais", "ciclos_totais", "idle_cycles",
             "l1d_accesses", "l1i_accesses", "l2_accesses", "l3_accesses",
             "mpki_l1d", "mpki_l2", "mpki_l3", "branch_miss_rate", "ipc_medio"]
BLOCOS = {"Total Cores": "cores", "Total L3s": "l3", "Total NoCs": "noc", "Total MCs": "mc",
          "Total Flash": "flash", "Total NIUs": "niu", "Total PCIes": "pcie"}
RX = re.compile(r"(Subthreshold Leakage|Gate Leakage|Runtime Dynamic) = ([\d.e+-]+) W")


def blocos_processor(txt):
    """Leakage e Runtime Dynamic de cada bloco do resumo 'Processor:' (topo do arquivo)."""
    d, cur = {}, None
    for ln in txt.split("\n"):
        s = ln.strip()
        if s.startswith("Core:"):          # fim do resumo; comeca o detalhamento por nucleo
            break
        hit = next((v for k, v in BLOCOS.items() if s.startswith(k)), None)
        if hit:
            cur = hit
            d[cur] = {"leak": 0.0, "dyn": 0.0}
            continue
        m = RX.match(s)
        if m and cur:
            d[cur]["dyn" if m.group(1) == "Runtime Dynamic" else "leak"] += float(m.group(2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="processa a geracao sem CAN ID")
    ap.add_argument("--com-mc", action="store_true",
                    help="opcao B': P_chip = total do Processor (com controlador de memoria e I/O)")
    args = ap.parse_args()
    p = dict(PERFIS["v2" if args.v2 else "v3"])
    chip = list(BLOCOS.values()) if args.com_mc else ["cores", "l3", "noc"]
    if args.com_mc:
        p["saida"] = p["saida"].replace("_v4.csv", "_v4mc.csv")

    df = pd.read_csv(p["gem5_csv"])
    tempo = dict(zip(df["cenario"], df["tempo_s"].astype(float)))
    arquivos = sorted(glob.glob(f"{p['results_dir']}/*.txt"))
    if not arquivos:
        sys.exit(f"ERRO: nenhuma saida do McPAT em {p['results_dir']}/")

    linhas = []
    for path in arquivos:
        cen = os.path.basename(path)[:-4]
        d = blocos_processor(open(path).read())
        ok = all(b in d for b in ("cores", "l3", "noc")) and d["cores"]["dyn"] > 0
        soma = lambda bs, k: sum(d.get(b, {}).get(k, 0.0) for b in bs)
        dyn, leak = soma(chip, "dyn"), soma(chip, "leak")
        tot = lambda bs: soma(bs, "dyn") + soma(bs, "leak")
        t = tempo.get(cen, 0.0)
        w = dyn + leak
        linhas.append({
            "cenario": cen,
            "runtime_dynamic_w": round(dyn, 4),
            "total_leakage_w": round(leak, 4),
            "total_power_w": round(w, 4),
            "tempo_s": round(t, 6),
            "energy_mj": round(w * t * 1000, 4),
            "edp": round(w * t ** 2, 12),
            "mcpat_success": ok,
            "mc_w": round(tot(["mc"]), 4),
            "io_w": round(tot(["niu", "pcie", "flash"]), 4),
            "mcpat_total_w": round(tot(list(BLOCOS.values())), 4),
        })

    final = df[COLS_GEM5].merge(pd.DataFrame(linhas), on="cenario", how="left")
    final.to_csv(p["saida"], index=False)
    n_ok = int(final["mcpat_success"].fillna(False).sum())
    print(f"OK: {p['saida']} ({len(final)} linhas, {n_ok} com sucesso)")
    print(f"  P_chip (W): min={final['total_power_w'].min():.3f} max={final['total_power_w'].max():.3f}")
    print(f"  {'incluido' if args.com_mc else 'excluido'} -> controlador de memoria: "
          f"{final['mc_w'].min():.3f}-{final['mc_w'].max():.3f} W;"
          f" I/O: {final['io_w'].min():.3f}-{final['io_w'].max():.3f} W")
    if n_ok < len(final):
        print("  ATENCAO, sem saida valida:", sorted(final.loc[~final['mcpat_success'].fillna(False), 'cenario']))


if __name__ == "__main__":
    main()
