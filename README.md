# Reprodutibilidade: Eficiência Energética de Arquiteturas *In-Order* e *Out-of-Order* para Aprendizado Federado de Borda na Detecção de Intrusão em CAN de Veículos Autônomos

Este repositório contém os *scripts*, as configurações e os resultados necessários para
reproduzir os experimentos do artigo homônimo (SSCAD 2026). O estudo caracteriza quatro cargas
de detecção de intrusão em redes CAN (*Random Forest*, XGBoost, MLP e SVM linear) em núcleos
ARM *in-order* (MinorCPU) e *out-of-order* (O3CPU), com o gem5 (modo SE) acoplado ao
McPAT e ao CACTI no nó de 22 nm. Ao todo são 102 simulações: configuração de
referência, varreduras de L2, L1D, *prefetcher* e preditor de desvios, escalabilidade em 1, 2
e 4 núcleos e o *round* de treino local do MLP.

Os *scripts* com sufixo `_v4` correspondem à versão do cálculo de potência usada no artigo:
contagens de atividade medidas pelo gem5 e núcleos descritos no McPAT como foram simulados
(Seção 6). Nos comentários desses *scripts*, "opção A" designa a versão anterior do cálculo,
em que as contagens de atividade eram estimadas por frações fixas do número de instruções e o
O3 do McPAT não correspondia ao simulado; ela não faz parte deste pacote. A opção `--com-mc` de
`reparse_mcpat_edp_v4.py` é uma variante de análise (P_chip incluindo o controlador de memória)
que não foi usada no artigo.

---

## 1. Ambiente

| Componente | Versão / detalhe |
|---|---|
| Simulador | gem5 v25.1.0.1, ISA ARM (`build/ARM/gem5.opt`), modo SE |
| Potência e área | McPAT v1.3, nó de 22 nm |
| Latências de *cache* | CACTI v7.0, nó de 22 nm (tabelas fixas em `gem5/edge_config_mc.py`) |
| Compilador cruzado | GCC 15.2.0 (`aarch64-linux-gnu-gcc`), `-O2 -static` |
| Python | 3.14 (versões das bibliotecas em `requirements.txt`) |
| Sistema operacional | Linux x86-64 |
| *Dataset* | AV-IDS-Dataset, *commit* `67ff53c` |

```bash
pip install -r requirements.txt
```

O gem5, o McPAT, o CACTI e o *dataset* não estão incluídos (são projetos de terceiros ou
arquivos volumosos). As fontes oficiais são:

- gem5: <https://github.com/gem5/gem5> (*tag* `v25.1.0.1`)
- McPAT: <https://github.com/HewlettPackard/mcpat>
- CACTI: <https://github.com/HewlettPackard/cacti>
- AV-IDS-Dataset: <https://github.com/cart-pucminas/AV-IDS-Dataset>

---

## 2. Estrutura do repositório

A estrutura de pastas é a mesma do projeto original, de modo que os *scripts* rodam sem
alteração de caminhos.

```
reproducibility_sscad/
├── gem5/                         # nossos arquivos; o gem5 é clonado por cima (Seção 3)
│   ├── edge_config_mc.py             # CPU e hierarquia de memória de borda (SE mode)
│   ├── dse_sscad.py                  # 80 sims 1 núcleo: referência + varreduras
│   ├── dse_sscad_multicore.py        # 16 sims com 2 e 4 núcleos
│   ├── dse_sscad_treino.py           # 6 sims do round de treino do MLP
│   ├── reparse_completo_mcpat.py     # extração básica dos stats.txt (usada pelo _v4)
│   ├── reparse_completo_mcpat_v4.py  # + contagens de atividade por estrutura (entrada do McPAT)
│   ├── rodar_tudo_sscad_v4.sh        # orquestra as 6 etapas do pipeline
│   └── resultados_*.csv              # CSVs intermediários do gem5 (pré-computados)
├── experimentos/
│   ├── workload/
│   │   ├── tuning_optuna.py          # busca de hiperparâmetros (Optuna/TPE)
│   │   ├── treina_modelos.py         # treino dos 4 modelos
│   │   ├── exporta_para_c.py         # exporta os modelos para headers C + dataset.h
│   │   ├── valida_export_c.py        # confere C × scikit-learn (99–100% de concordância)
│   │   ├── compila_arm.sh            # compila os 4 binários de inferência (aarch64)
│   │   ├── compila_train_arm.sh      # compila o micro-benchmark de treino (minibatch 32/64/128)
│   │   ├── train_mlp.c               # micro-benchmark do round de treino local do MLP
│   │   └── c_export/main.c           # driver de inferência com a ROI (m5ops)
│   ├── modelos/
│   │   ├── best_params.json          # hiperparâmetros do Optuna usados no artigo
│   │   └── resumo_treino.csv         # acurácia, F1-macro e tempo de treino de cada modelo
│   ├── scripts/
│   │   ├── mcpat_batch_runner_sscad_v4.py   # gera os XMLs do McPAT e executa em lote
│   │   ├── reparse_mcpat_edp_v4.py          # extrai P_chip das saídas do McPAT -> CSV final
│   │   └── graficos_sscad_v4.py             # gera as figuras do artigo
│   └── dados/
│       ├── resultados_edp_sscad_v4.csv           # CSV final: desempenho + energia + EDP
│       └── arquivo_5k_v2_semcanid/csv/           # geração sem arbitration ID (figura de
│                                                 #   custo de detecção)
├── resultados_brutos/
│   ├── m5out_sscad.tar.gz                  # stats.txt + config.ini das 102 simulações
│   ├── m5out_sscad_v2_semcanid.tar.gz      # idem, geração sem arbitration ID
│   └── mcpat_sscad_v4.tar.gz               # 102 XMLs de entrada e 102 saídas do McPAT
├── graficos_sscad_v4/            # figuras em PDF
├── requirements.txt
└── README.md
```

---

## 3. Preparação

Todos os comandos partem da raiz deste repositório.

```bash
# 1) Dataset
git clone https://github.com/cart-pucminas/AV-IDS-Dataset experimentos/AV-IDS-Dataset
git -C experimentos/AV-IDS-Dataset checkout 67ff53c

# 2) gem5 v25.1.0.1, mesclado à pasta gem5/ (o -n preserva os nossos arquivos)
git clone --branch v25.1.0.1 https://github.com/gem5/gem5 /tmp/gem5
cp -rn /tmp/gem5/. gem5/
(cd gem5 && scons build/ARM/gem5.opt -j"$(nproc)")
(cd gem5/util/m5 && scons build/arm64/out/m5)      # libm5.a usada pelos binários

# 3) McPAT v1.3 na pasta mcpat/ (o runner chama ./mcpat/mcpat)
git clone https://github.com/HewlettPackard/mcpat mcpat
(cd mcpat && make)
```

---

## 4. Pipeline completo

Os comandos rodam da raiz, exceto quando há `cd` explícito.

```bash
# 0) (opcional, ~3–4 h) Tuning Optuna em 20% do dataset -> experimentos/modelos/best_params.json
#    O best_params.json usado no artigo já está incluído; pule esta etapa para reproduzi-lo.
python experimentos/workload/tuning_optuna.py

# 1) Treino dos 4 modelos -> experimentos/modelos/*.joblib
python experimentos/workload/treina_modelos.py

# 2) Export para C e validação contra o scikit-learn
python experimentos/workload/exporta_para_c.py
python experimentos/workload/valida_export_c.py

# 3) Binários aarch64 -> experimentos/workload/c_export/bin/
(cd experimentos/workload && bash compila_arm.sh && bash compila_train_arm.sh)

# 4) DSE no gem5 + McPAT + EDP (~17 h; ajuste WORKERS nos dse_sscad*.py)
(cd gem5 && nohup bash rodar_tudo_sscad_v4.sh > pipeline_sscad_v4.log 2>&1 &)

# 5) Figuras -> graficos_sscad_v4/
python experimentos/scripts/graficos_sscad_v4.py
```

O `rodar_tudo_sscad_v4.sh` executa seis etapas, que também podem ser rodadas isoladamente
(`SO_POTENCIA=1` pula as simulações e refaz só as etapas 4–6):

| # | Comando | Pasta | Saída |
|---|---|---|---|
| 1 | `python dse_sscad.py` | `gem5/` | 80 sims 1 núcleo: `resultados_sscad.csv` |
| 2 | `python dse_sscad_multicore.py` | `gem5/` | 16 sims: `resultados_sscad_multicore.csv` |
| 3 | `python dse_sscad_treino.py` | `gem5/` | 6 sims: `resultados_sscad_treino.csv` |
| 4 | `python reparse_completo_mcpat_v4.py` | `gem5/` | `resultados_gem5_completo_v4.csv` (102 linhas) |
| 5 | `python experimentos/scripts/mcpat_batch_runner_sscad_v4.py` | raiz | `mcpat_xmls_sscad_v4/`, `mcpat_results_sscad_v4/` |
| 6 | `python experimentos/scripts/reparse_mcpat_edp_v4.py` | raiz | `experimentos/dados/resultados_edp_sscad_v4.csv` |

A geração sem *arbitration ID* (figura de custo de detecção) usa as mesmas etapas 4–6 com
`--v2`; ver a Seção 5.

---

## 5. Atalhos com os resultados pré-computados

Não é preciso refazer as simulações para conferir os números do artigo.

| Objetivo | O que rodar | Precisa de |
|---|---|---|
| Regerar as figuras | `python experimentos/scripts/graficos_sscad_v4.py` | só Python |
| Extrair os XMLs e as saídas do McPAT | `tar xzf resultados_brutos/mcpat_sscad_v4.tar.gz` (na raiz) | nada |
| Recalcular P_chip e EDP a partir das saídas do McPAT | depois de extraí-las, `python experimentos/scripts/reparse_mcpat_edp_v4.py` | só Python |
| Reextrair as contagens do gem5 | `tar xzf resultados_brutos/m5out_sscad.tar.gz -C gem5/` e depois `cd gem5 && python reparse_completo_mcpat_v4.py` | só Python |
| Idem, geração sem *arbitration ID* | `mkdir -p experimentos/dados/arquivo_5k_v2_semcanid/m5out && tar xzf resultados_brutos/m5out_sscad_v2_semcanid.tar.gz -C experimentos/dados/arquivo_5k_v2_semcanid/m5out` e depois, em `gem5/`, `python reparse_completo_mcpat_v4.py --pasta ../experimentos/dados/arquivo_5k_v2_semcanid/m5out --saida ../experimentos/dados/arquivo_5k_v2_semcanid/csv/resultados_gem5_completo_v4.csv` | só Python |
| Regerar os XMLs do McPAT sem executá-lo | `python experimentos/scripts/mcpat_batch_runner_sscad_v4.py --so-xml` (e `--v2`) | só Python |
| Recalcular a potência da geração sem *arbitration ID* | reextrair as contagens dessa geração (linha "Idem, geração sem *arbitration ID*"), depois `python experimentos/scripts/mcpat_batch_runner_sscad_v4.py --v2` e `python experimentos/scripts/reparse_mcpat_edp_v4.py --v2` | McPAT |

As saídas do McPAT incluídas são as da geração com *arbitration ID*, usada em todo o artigo. Da
geração sem *arbitration ID* estão incluídos o CSV final (suficiente para regerar as figuras) e
as estatísticas brutas do gem5.

---

## 6. Notas metodológicas

- Região de interesse (ROI). O `main.c` executa 100 inferências de aquecimento,
  `m5_dump_reset_stats`, o laço de 5.000 inferências e `m5_dump_stats`. O `stats.txt` tem
  três *dumps*; a ROI é o segundo.
- Mapeamento do gem5 para o McPAT. `gem5/reparse_completo_mcpat_v4.py` extrai da ROI as contagens
  de atividade: instruções por tipo (inteiras, FP/SIMD, desvios, *loads*, *stores*), acessos
  às ALUs e FPUs, consultas à BTB, acessos e faltas em L1I, L1D, L2 e L3 e requisições à
  memória. `generate_mcpat_xml()` (em `mcpat_batch_runner_sscad_v4.py`) escreve essas
  contagens no XML do McPAT, com `pipeline_duty_cycle` = IPC / largura de emissão. Os núcleos
  são descritos como no `config.ini` do gem5:
  - O3: largura 8 (*fetch*/*decode*/*issue*/*commit*), fila de instruções 64, ROB 192,
    LQ/SQ 32, 256 registradores físicos inteiros e FP, 6 ALUs, 2 MUL/DIV, 4 FPUs;
  - Minor: largura 2, *buffer* de *stores* 5, 2 ALUs, 1 MUL, 1 FPU.

  O gem5 não registra acessos às TLBs no modo SE; usa-se acessos à ITLB = acessos à L1I e
  acessos à DTLB = *loads* + *stores*. Os demais parâmetros (organização das *caches*,
  preditor, `core_tech_node` = 22, clock de 2,76 GHz) vêm da configuração simulada.
- P_chip e EDP. `reparse_mcpat_edp_v4.py` soma, do resumo `Processor:` da saída do McPAT,
  a potência dinâmica (`Runtime Dynamic`) e o *leakage* (*subthreshold* + *gate*) dos núcleos
  (com L1 e L2 privada), da L3 e do barramento. Ficam fora o controlador de memória e as
  interfaces de E/S modelados pelo McPAT. EDP = P_chip × T², com T o tempo da ROI;
  `energy_mj` = P_chip × T × 1000. O `Peak Dynamic` é um máximo teórico e não é usado.
- Leakage. Nas configurações de referência, o *leakage* responde por 90% a 98% de P_chip,
  dominado pela L3 de 16 MiB (61% a 76% de P_chip). As *caches* L2 e L3 são modeladas no McPAT
  com dispositivos ITRS de alto desempenho (`device_type` = 0), o que pesa nessa proporção.
- DRAM. O modelo `LPDDR5_6400_1x16_BG_BL16` do gem5 não define as correntes de alimentação,
  e as estatísticas de energia da memória externa saem zeradas; por isso a DRAM não entra no
  EDP. O tráfego com a DRAM é medido e está nos `stats.txt` (`mem_ctrl.dram.bytesRead`).
- Multinúcleo. As 5.000 inferências são divididas em fatias disjuntas entre os núcleos. O
  SVM é excluído da análise: sua ROI (< 10⁶ instruções) é curta demais para uma janela de
  medição consistente.
- Atributos. `USE_CAN_ID = True` (10 atributos, com `arbitration_id`) deve ser igual em
  `tuning_optuna.py`, `treina_modelos.py` e `exporta_para_c.py`. A geração sem CAN ID
  (`USE_CAN_ID = False`, 9 atributos), usada só na figura de custo de detecção, está em
  `experimentos/dados/arquivo_5k_v2_semcanid/`.
- Sementes. `RANDOM_STATE = 42` no tuning, no treino e na amostragem do `dataset.h`.

### O que não está incluído

| Item | Motivo | Como obter |
|---|---|---|
| Modelos `.joblib` | `rf.joblib` tem ~1,4 GB | etapa 1 |
| Headers C (`*_model.h`, `dataset.h`) | `rf_model.h` tem ~558 MB | etapa 2 |
| Binários ARM | dependem do compilador local | etapa 3 |
| Pastas `m5out_*` completas | volume; os `stats.txt` e `config.ini` estão nos `.tar.gz` | etapas 1–3 |

---

## 7. Como citar

Pereira, L. F. M. C., Souza, M. A. e Freitas, H. C. (2026). Eficiência Energética de
Arquiteturas *In-Order* e *Out-of-Order* para Aprendizado Federado de Borda na Detecção de
Intrusão em CAN de Veículos Autônomos. In: *Anais do XXVII Simpósio em Sistemas Computacionais
de Alto Desempenho (SSCAD)*. SBC.
