#!/usr/bin/env python3
"""
Figuras do artigo SSCAD 2026 - benchmark de 4 algoritmos de ML na borda veicular
(RF, XGBoost, MLP, SVM linear) caracterizados em gem5 + McPAT + CACTI (22nm),
comparando in-order (MinorCPU) vs out-of-order (O3CPU).

Fonte de dados (OPCAO B, McPAT com contagens medidas): experimentos/dados/resultados_edp_sscad_v4.csv (102 sims, pipeline v3 COM CAN ID).
Estilo: padrao conferencias de arquitetura (ISCA/MICRO) - Tufte, serif, PDF vetorial.
Paleta checada para daltonismo (CVD) e contraste (ver comentario abaixo).

Uso (da RAIZ do projeto, no venv com pandas+matplotlib):
    python experimentos/scripts/graficos_sscad_v4.py

Saida: graficos_sscad_v4/*.pdf (+ .png 300dpi)
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# ESTILO GLOBAL (Tufte / arquitetura)
# ============================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

OUTPUT_DIR = 'graficos_sscad_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = 'experimentos/dados/resultados_edp_sscad_v4.csv'
CSV_V1   = 'experimentos/dados/arquivo_5k_v1_f1baixo/csv/resultados_edp_sscad.csv'
# v2 = SEM CAN ID (F1~0,72), arquivada. Comparada com a atual (v3, COM CAN ID, F1~0,83)
# para mostrar o CUSTO arquitetural de elevar a qualidade de deteccao.
CSV_V2_SEMCANID = 'experimentos/dados/arquivo_5k_v2_semcanid/csv/resultados_edp_sscad_v4.csv'

# ------------------------------------------------------------
# PALETA - checada para daltonismo (CVD) e contraste.
#   2-cor CPU (Minor vs O3): #2a78d6 / #e34948  -> ALL CHECKS PASS (light)
#   4-cor algoritmos:        #2a78d6 #eb6834 #1baf7a #4a3aa7 -> ALL CHECKS PASS
#   (verde #1baf7a fica <3:1 de contraste -> sempre com legenda + rotulo direto)
# ------------------------------------------------------------
C_MINOR = '#2a78d6'   # azul - In-Order
C_O3    = '#e34948'   # vermelho - Out-of-Order
C_INK   = '#333333'   # texto/rotulos (nunca a cor da serie)
C_MUTE  = '#8a8a8a'   # eixos/grade secundaria
C_LIGHT = '#e5e5e5'   # barra de fundo (leakage)

# ordem categorica fixa dos algoritmos (nunca reciclada)
ALGOS = ['RF', 'XGB', 'MLP', 'SVM']
ALGO_NOME = {'RF': 'Random Forest', 'XGB': 'XGBoost',
             'MLP': 'MLP', 'SVM': 'SVM linear'}
ALGO_COR = {'RF': '#2a78d6', 'XGB': '#eb6834', 'MLP': '#1baf7a', 'SVM': '#4a3aa7'}
ALGO_MARK = {'RF': 'o', 'XGB': 's', 'MLP': '^', 'SVM': 'D'}

HATCH_O3 = '////'     # textura p/ separar O3 em grayscale/impressao


# ============================================================
# CARGA DE DADOS
# ============================================================
def carregar(path):
    with open(path) as f:
        return list(csv.DictReader(f))


ROWS = carregar(CSV_PATH)


def f(row, col):
    """float seguro."""
    try:
        return float(row[col])
    except (TypeError, ValueError, KeyError):
        return float('nan')


def pega(rows, grupo, modelo, cpu, **kw):
    """1a linha que casa grupo/modelo/cpu (+ filtros extras tipo l2_size=...)."""
    for r in rows:
        if r['grupo'] != grupo:
            continue
        if r['modelo'].upper() != modelo.upper():
            continue
        if r['cpu'] != cpu:
            continue
        if all(str(r.get(k, '')) == str(v) for k, v in kw.items()):
            return r
    return None


def save_fig(fig, name):
    fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', format='pdf')
    fig.savefig(f'{OUTPUT_DIR}/{name}.png', format='png', dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.pdf / .png")


# ============================================================
# FIGURA 1 (MANCHETE): EDP por algoritmo, Minor vs O3 (escala log)
#   EDP varia ~7 ordens de grandeza -> log e obrigatorio.
# ============================================================
def fig1_edp_algoritmos():
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    minor = [f(pega(ROWS, 'principal', a, 'Minor'), 'edp') for a in ALGOS]
    o3    = [f(pega(ROWS, 'principal', a, 'O3'), 'edp') for a in ALGOS]

    x = np.arange(len(ALGOS))
    w = 0.36

    ax.bar(x - w/2, minor, w, color=C_MINOR, label='In-Order (Minor)', zorder=3)
    ax.bar(x + w/2, o3, w, color=C_O3, label='Out-of-Order (O3)', zorder=3)

    ax.set_yscale('log')
    ax.tick_params(axis='y', which='minor', length=0)  # remove minor ticks (poluicao)
    ax.set_ylabel('EDP (J·s)  —  escala log')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NOME[a] for a in ALGOS], fontsize=8)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2,
              handlelength=1.4, columnspacing=1.2)

    # rotulo do ganho O3 sobre cada par
    for i, (m, o) in enumerate(zip(minor, o3)):
        ax.text(x[i], max(m, o) * 1.6, f'{m/o:.1f}×', ha='center', va='bottom',
                fontsize=7.5, color=C_INK, fontweight='bold')

    save_fig(fig, 'fig1_edp_algoritmos')


# ============================================================
# FIGURA 2 (TESE): ganho O3/Minor - speedup e EDP por algoritmo
#   Barra agrupada: mostra que o O3 sempre vence, mas MENOS no RF.
# ============================================================
def fig2_ganho_o3():
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    speedup, ganho_edp = [], []
    for a in ALGOS:
        m = pega(ROWS, 'principal', a, 'Minor')
        o = pega(ROWS, 'principal', a, 'O3')
        speedup.append(f(m, 'tempo_s') / f(o, 'tempo_s'))
        ganho_edp.append(f(m, 'edp') / f(o, 'edp'))

    x = np.arange(len(ALGOS))
    w = 0.36

    ax.bar(x - w/2, speedup, w, color=C_MUTE, label='Speedup (tempo)', zorder=3)
    ax.bar(x + w/2, ganho_edp, w, color=C_O3, label='Ganho de EDP', zorder=3)

    # "x" direto nos numeros das barras (auto-explicativo, sem unidade no eixo)
    for i, (s, g) in enumerate(zip(speedup, ganho_edp)):
        ax.text(x[i]-w/2, s+0.08, f'{s:.1f}×', ha='center', va='bottom',
                fontsize=7, color=C_INK)
        ax.text(x[i]+w/2, g+0.08, f'{g:.1f}×', ha='center', va='bottom',
                fontsize=7, color=C_O3, fontweight='bold')

    ax.set_ylabel('Ganho do O3 sobre o Minor')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NOME[a] for a in ALGOS], fontsize=8)
    ax.set_ylim(0, max(ganho_edp) * 1.15)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', handlelength=1.4)

    save_fig(fig, 'fig2_ganho_o3')


# ============================================================
# FIGURA 3: escalabilidade multicore - paineis (a) Minor / (b) O3 lado a lado.
#   Eixo Y compartilhado. Mostra RF/XGB saturando vs MLP ~linear.
#   SVM EXCLUIDO: sua ROI (<10^6 instrucoes) e curta demais para que a janela de
#   medicao seja consistente entre configuracoes. A execucao de 4 nucleos no O3
#   registra 637.652 instrucoes contra 856.883 em 1 nucleo para o MESMO trabalho
#   (-26%), o que invalida a razao de tempos como speedup (dava 4,9x, superlinear).
#   Contagem de instrucoes commitadas e arquitetural: nao pode diferir entre
#   Minor e O3, e de fato nao difere com 1 nucleo (856.882 vs 856.883).
#   RF/XGB/MLP conferidos e limpos (desvio de -2,5%, -0,4% e +0,003%).
# ============================================================
MC_ALGOS = ['RF', 'XGB', 'MLP']   # SVM excluido (ver nota acima)


def fig3_multicore():
    fig, (axm, axo) = plt.subplots(1, 2, figsize=(5.6, 2.8), sharey=True)
    cores = [1, 2, 4]

    for ax, cpu, titulo in [(axm, 'Minor', '(a) In-Order (Minor)'),
                            (axo, 'O3', '(b) Out-of-Order (O3)')]:
        ends = []
        for a in MC_ALGOS:
            t1 = f(pega(ROWS, 'principal', a, cpu), 'tempo_s')
            t2 = f(pega(ROWS, 'multicore_2c', a, cpu), 'tempo_s')
            t4 = f(pega(ROWS, 'multicore_4c', a, cpu), 'tempo_s')
            sp = [1.0, t1/t2, t1/t4]
            ax.plot(cores, sp, '-', marker=ALGO_MARK[a], color=ALGO_COR[a],
                    markersize=5, linewidth=1.8, markeredgecolor='white',
                    markeredgewidth=0.8, label=ALGO_NOME[a], zorder=4)
            ends.append((a, sp[-1]))

        # rotulos finais, desempilhando colisoes
        ends.sort(key=lambda e: e[1])
        ultimo_y = -99
        for a, y in ends:
            y_lbl = y if y - ultimo_y >= 0.24 else ultimo_y + 0.24
            ax.text(4.10, y_lbl, f'{y:.1f}×', fontsize=6.8, color=ALGO_COR[a],
                    va='center', ha='left', fontweight='bold')
            ultimo_y = y_lbl

        ax.set_title(titulo, fontsize=8.5, pad=4)
        ax.set_xlabel('Número de núcleos')
        ax.set_xticks(cores)
        ax.set_xlim(0.9, 4.8)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
        ax.set_axisbelow(True)

    axm.set_ylabel('Speedup vs 1 núcleo')
    axo.legend(loc='upper left', handlelength=1.6, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, 'fig3_multicore')


# ============================================================
# FIGURA 4A: sensibilidade a L2 - SMALL MULTIPLES (padrao Co-design, Figs.12-14).
#   Um painel por algoritmo, EDP NORMALIZADO ao baseline L2=1MB, linha em 1,0.
#   Mensagem: so o XGBoost desce abaixo de 1,0 (mais L2 ajuda); os outros 3
#   sobem (mais L2 atrapalha). Eixos identicos -> padrao "1 desce, 3 sobem".
#   Sobrepoe Minor (linha cheia) e O3 (tracejada) em cada painel.
# ============================================================
L2_ORD = ['512kB', '1MB', '2MB', '4MB']
L2_LBL = ['512K', '1M', '2M', '4M']


def _edp_l2(a, cpu, size):
    """EDP para um tamanho de L2. O ponto de 2MB e a configuracao de referencia
    do artigo e esta registrado no grupo 'principal', nao no sweep de L2."""
    if size == '2MB':
        return f(pega(ROWS, 'principal', a, cpu), 'edp')
    return f(pega(ROWS, 'dse_l2_size', a, cpu, l2_size=size), 'edp')


def _edp_norm_l2(a, cpu):
    edp = [_edp_l2(a, cpu, s) for s in L2_ORD]
    base = edp[L2_ORD.index('2MB')]  # referencia = 2MB (config principal)
    return [e / base for e in edp]


def fig4a_l2_small_multiples():
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.8), sharex=True, sharey=True)
    x = np.arange(len(L2_ORD))

    for ax, a in zip(axes.flat, ALGOS):
        nm = _edp_norm_l2(a, 'Minor')
        no = _edp_norm_l2(a, 'O3')
        ax.plot(x, nm, '-o', color=C_MINOR, markersize=4, linewidth=1.6,
                markeredgecolor='white', markeredgewidth=0.7, label='Minor', zorder=4)
        ax.plot(x, no, '--s', color=C_O3, markersize=4, linewidth=1.6,
                markeredgecolor='white', markeredgewidth=0.7, label='O3', zorder=4)
        ax.axhline(1.0, color=C_INK, linewidth=0.7, alpha=0.5, zorder=2)

        ax.set_title(ALGO_NOME[a], fontsize=8.5, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(L2_LBL)
        ax.grid(True, axis='y', linestyle='-', linewidth=0.35, alpha=0.22, color=C_MUTE)
        ax.set_axisbelow(True)

    axes[0, 0].legend(loc='upper right', fontsize=6.5, handlelength=1.4)
    fig.supxlabel('Tamanho da cache L2', fontsize=9)
    fig.supylabel('EDP normalizado (L2 = 2MB → 1,0)', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'fig4a_l2_small_multiples')


# ============================================================
# FIGURA 4B: L2 importa POUCO vs microarquitetura importa MUITO.
#   Paineis (a) Minor / (b) O3 (padrao Gem5-X), curvas EDP absoluto (log) vs L2,
#   MESMA escala Y nos dois. As curvas sao quase planas (L2 mal move o EDP),
#   mas o painel O3 inteiro fica deslocado p/ baixo -> "trocar de CPU vale mais
#   que dobrar a cache". Anota o gap medio Minor->O3.
# ============================================================
def fig4b_l2_vs_micro():
    fig, (axm, axo) = plt.subplots(1, 2, figsize=(5.4, 2.9), sharey=True)
    x = np.arange(len(L2_ORD))

    for ax, cpu, titulo in [(axm, 'Minor', '(a) In-Order (Minor)'),
                            (axo, 'O3', '(b) Out-of-Order (O3)')]:
        for a in ALGOS:
            edp = [_edp_l2(a, cpu, s) for s in L2_ORD]
            ax.plot(x, edp, '-', marker=ALGO_MARK[a], color=ALGO_COR[a],
                    markersize=4.5, linewidth=1.7, markeredgecolor='white',
                    markeredgewidth=0.7, label=ALGO_NOME[a], zorder=4)
        ax.set_yscale('log')
        ax.set_title(titulo, fontsize=8.5, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(L2_LBL)
        ax.set_xlabel('Tamanho da L2')
        ax.grid(True, axis='y', linestyle='-', linewidth=0.35, alpha=0.22, color=C_MUTE)
        ax.set_axisbelow(True)

    axm.set_ylabel('EDP (J·s) — escala log')
    axo.legend(loc='center right', fontsize=6.6, handlelength=1.5)

    # anota o "degrau" medio de EDP ao trocar Minor->O3 (media geometrica dos ratios)
    ratios = []
    for a in ALGOS:
        em = _edp_l2(a, 'Minor', '2MB')
        eo = _edp_l2(a, 'O3', '2MB')
        ratios.append(em / eo)
    gm = np.exp(np.mean(np.log(ratios)))
    fig.suptitle(f'Variar L2 mal move o EDP; trocar Minor→O3 o reduz ~{gm:.1f}× (média)',
                 fontsize=8, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig4b_l2_vs_micro')


# ============================================================
# FIGURA 5: custo do round de treino do MLP (Federated Learning)
#   EDP e energia vs batch size (32/64/128), Minor vs O3.
# ============================================================
def fig5_treino_fl():
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    batches = [32, 64, 128]
    minor = [f(pega(ROWS, 'treino_mlp', 'MLP', 'Minor', cenario=f'Treino_MLP_b{b}_Minor'), 'edp')
             for b in batches]
    o3 = [f(pega(ROWS, 'treino_mlp', 'MLP', 'O3', cenario=f'Treino_MLP_b{b}_O3'), 'edp')
          for b in batches]

    x = np.arange(len(batches))
    w = 0.36
    ax.bar(x - w/2, minor, w, color=C_MINOR, label='In-Order (Minor)', zorder=3)
    ax.bar(x + w/2, o3, w, color=C_O3, hatch=HATCH_O3, edgecolor='white',
           linewidth=0.4, label='Out-of-Order (O3)', zorder=3)

    ax.set_ylim(0, max(minor) * 1.15)  # folga p/ o rotulo do batch 128
    for i, (m, o) in enumerate(zip(minor, o3)):
        ax.text(x[i], max(m, o) + max(minor)*0.015, f'{m/o:.1f}×', ha='center',
                va='bottom', fontsize=7.5, color=C_INK, fontweight='bold')

    ax.set_xlabel('Tamanho do minibatch (round de FL)')
    ax.set_ylabel('EDP do treino local (J·s)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', handlelength=1.4)

    save_fig(fig, 'fig5_treino_fl')


# ============================================================
# FIGURA 6: decomposicao de potencia (dinamica vs leakage) por modelo/cpu
#   No 22nm o leakage domina - barra empilhada mostra isso.
# ============================================================
def fig6_power_breakdown():
    fig, ax = plt.subplots(figsize=(4.2, 2.7))

    scen, labels = [], []
    for a in ALGOS:
        for cpu in ['Minor', 'O3']:
            r = pega(ROWS, 'principal', a, cpu)
            if r:
                scen.append(r)
                labels.append(f'{a}\n{cpu}')

    dyn = [f(r, 'runtime_dynamic_w') for r in scen]
    leak = [f(r, 'total_leakage_w') for r in scen]
    x = np.arange(len(scen))
    w = 0.62

    ax.bar(x, dyn, w, color=C_O3, edgecolor='white', linewidth=0.5,
           label='Dinâmica', zorder=3)
    ax.bar(x, leak, w, bottom=dyn, color=C_LIGHT, edgecolor='white',
           linewidth=0.5, label='Leakage (estática)', zorder=3)

    for i, d in enumerate(dyn):
        frac = leak[i] / (dyn[i] + leak[i]) * 100
        ax.text(i, dyn[i] + leak[i] + 0.04, f'{frac:.0f}%', ha='center',
                va='bottom', fontsize=6.5, color=C_INK)

    ax.set_ylabel('Potência (W)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, max(np.array(dyn)+np.array(leak)) * 1.18)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', ncol=2, handlelength=1.4)
    ax.text(len(scen)-0.5, max(np.array(dyn)+np.array(leak))*1.1,
            '% = fração de leakage', fontsize=6, color=C_INK, style='italic',
            ha='right')

    save_fig(fig, 'fig6_power_breakdown')


# ============================================================
# FIGURA 7: efeito do preditor de desvios (BiMode vs Tournament)
#   Onde o BP importa? XGB/RF (muito branch) vs MLP (quase nada).
# ============================================================
def fig7_branch_predictor(cpu='O3'):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    x = np.arange(len(ALGOS))
    w = 0.36
    bi = [f(pega(ROWS, 'dse_bp', a, cpu, bp='BiModeBP'), 'branch_miss_rate')*100 for a in ALGOS]
    to = [f(pega(ROWS, 'dse_bp', a, cpu, bp='TournamentBP'), 'branch_miss_rate')*100 for a in ALGOS]

    ax.bar(x - w/2, bi, w, color=C_MUTE, label='BiMode', zorder=3)
    ax.bar(x + w/2, to, w, color=C_O3, label='Tournament', zorder=3)

    ax.set_ylabel(f'Taxa de erro de desvio (%) — {cpu}')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NOME[a] for a in ALGOS], fontsize=8)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', handlelength=1.4)

    save_fig(fig, f'fig7_branch_predictor_{cpu.lower()}')


# ============================================================
# FIGURA 8 (CUSTO DA DETECCAO): impacto de elevar o F1 no custo arquitetural.
#   Responde ao Henrique: "melhorar a predicao pode aumentar o uso de recursos".
#   EDP absoluto (Minor, escala log) por algoritmo: SEM CAN ID (F1~0,72) vs
#   COM CAN ID (F1~0,83). O RF encarece ~4,7x ao subir o F1; SVM/MLP quase nao mudam.
# ============================================================
def fig8_custo_deteccao():
    if not os.path.exists(CSV_V2_SEMCANID):
        print("  [skip] fig8 - CSV v2 (sem CAN ID) nao encontrado")
        return
    rows_sem = carregar(CSV_V2_SEMCANID)   # SEM CAN ID (F1~0,72)
    # ROWS = atual = COM CAN ID (F1~0,83)

    def edp_minor(rows, a):
        return f(pega(rows, 'principal', a, 'Minor'), 'edp')

    sem = [edp_minor(rows_sem, a) for a in ALGOS]
    com = [edp_minor(ROWS, a) for a in ALGOS]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    x = np.arange(len(ALGOS))
    w = 0.36
    # NOTA: o termo segue o do artigo - "arbitration ID", nao "CAN ID" (mesmo campo).
    ax.bar(x - w/2, sem, w, color=C_MUTE, label='sem arbitration ID (F1≈0,72)', zorder=3)
    ax.bar(x + w/2, com, w, color=C_O3, label='com arbitration ID (F1≈0,83)', zorder=3)

    ax.set_yscale('log')
    ax.tick_params(axis='y', which='minor', length=0)
    # fator de aumento do EDP anotado sobre cada par
    for i, (s, c) in enumerate(zip(sem, com)):
        fator = c / s
        txt = f'{fator:.1f}×' if fator >= 1 else f'{fator:.2f}×'
        ax.text(x[i], max(s, c) * 1.5, txt, ha='center', va='bottom',
                fontsize=7.5, color=C_INK, fontweight='bold')

    ax.set_ylabel('EDP no núcleo Minor (J·s) — escala log')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_NOME[a] for a in ALGOS], fontsize=8)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color=C_MUTE)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2,
              handlelength=1.4, columnspacing=1.2)

    save_fig(fig, 'fig8_custo_deteccao')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Gerando figuras SSCAD (4 algoritmos, Minor vs O3)...")
    print(f"Fonte: {CSV_PATH}\nOutput: {OUTPUT_DIR}/\n")

    fig1_edp_algoritmos()
    fig2_ganho_o3()
    fig3_multicore()
    fig4a_l2_small_multiples()
    fig4b_l2_vs_micro()
    fig5_treino_fl()
    fig6_power_breakdown()
    fig7_branch_predictor('O3')
    fig8_custo_deteccao()

    print(f"\n{'='*52}")
    print("Figuras geradas em graficos_sscad_v4/ (PDF vetorial + PNG 300dpi)")
    print(f"{'='*52}")
