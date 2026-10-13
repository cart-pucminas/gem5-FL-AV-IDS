#!/usr/bin/env python3
"""
McPAT Batch Runner v4 (opcao B) - SSCAD
=======================================

Mesmo XML do mcpat_batch_runner_sscad.py (opcao A), com tres correcoes:

1. Contagens de atividade MEDIDAS pelo gem5 (gem5/resultados_gem5_completo_v4.csv, gerado por
   gem5/reparse_completo_mcpat_v4.py), no lugar das fracoes fixas do numero de instrucoes:
   instrucoes int/FP/desvio, ALU/FPU/MUL, BTB, L1I, L1D (leituras/escritas/faltas), L2, L3,
   barramento e memoria. pipeline_duty_cycle = IPC / largura de emissao.
2. Nucleos descritos como no gem5 (valores do config.ini das simulacoes):
   - O3:    largura 8 (fetch/decode/issue/commit), fila de instrucoes 64, ROB 192,
            LQ/SQ 32, 256 registradores fisicos int/FP, 6 ALU, 2 MUL/DIV, 4 FPU
   - Minor: largura 2, buffer de stores 5, 2 ALU, 1 MUL, 1 FPU
3. A potencia NAO e calculada aqui: o P_chip (nucleos + caches + barramento, sem controlador
   de memoria e I/O) e extraido por reparse_mcpat_edp_v4.py.

Sem medicao no gem5 (modo SE nao registra TLB): acessos a ITLB = acessos a L1I; acessos a
DTLB = loads + stores; faltas de TLB mantidas como na opcao A.
O restante do XML (caches, preditor, tecnologia, clock, duty cycles de L2/L3/NoC/MC, NIU,
PCIe e flash) e identico ao da opcao A.

Rodar da RAIZ do projeto (o McPAT e ./mcpat/mcpat, binario Linux x86-64):
    python experimentos/scripts/mcpat_batch_runner_sscad_v4.py            # geracao v3
    python experimentos/scripts/mcpat_batch_runner_sscad_v4.py --v2       # geracao sem CAN ID
    ... --so-xml   # so gera os XMLs (nao precisa do McPAT)
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import pandas as pd

MCPAT_BIN = "./mcpat/mcpat"
V2 = "experimentos/dados/arquivo_5k_v2_semcanid"
PERFIS = {
    "v3": dict(gem5_csv="gem5/resultados_gem5_completo_v4.csv",
               xml_dir="mcpat_xmls_sscad_v4", results_dir="mcpat_results_sscad_v4"),
    "v2": dict(gem5_csv=f"{V2}/csv/resultados_gem5_completo_v4.csv",
               xml_dir=f"{V2}/mcpat_xmls_sscad_v4", results_dir=f"{V2}/mcpat_results_sscad_v4"),
}
WORKERS = 2

# Parametros estruturais dos nucleos (config.ini do gem5)
NUCLEO = {
    "O3":    dict(machine_type=0, machine_desc="Out-of-Order (O3)", fetch_width=8,
                  decode_width=8, issue_width=8, commit_width=8, iw_size=64, rob_size=192,
                  phy_irf=256, phy_frf=256, alu=6, mul=2, fpu=4, load_buf=32, store_buf=32),
    "Minor": dict(machine_type=1, machine_desc="In-Order (Minor)", fetch_width=2,
                  decode_width=2, issue_width=2, commit_width=2, iw_size=20, rob_size=0,
                  phy_irf=64, phy_frf=64, alu=2, mul=1, fpu=1, load_buf=16, store_buf=5),
}


def convert_size_to_bytes(size_str):
    """Converte tamanhos como '64KiB' para bytes (igual a opcao A)"""
    multipliers = {'kB': 1000, 'KiB': 1024, 'MB': 1000**2, 'MiB': 1024**2, 'GB': 1000**3, 'GiB': 1024**3}
    for unit, mult in multipliers.items():
        if unit in size_str:
            num = float(size_str.replace(unit, '').replace(' ', ''))
            return int(num * mult)
    try:
        return int(size_str)
    except Exception:
        return 0


def generate_mcpat_xml(r):
    """Gera o XML do McPAT a partir de uma linha do CSV v4 do gem5"""
    cpu_type = r['cpu']
    n = int(r['cores'])
    nuc = NUCLEO[cpu_type]
    clock_ghz = 2.76
    clock_mhz = int(clock_ghz * 1000)
    l1d_bytes = convert_size_to_bytes(r['l1d_size'])
    l2_bytes = convert_size_to_bytes(r['l2_size'])
    l3_bytes = convert_size_to_bytes('16MiB')
    l3_assoc, l1d_assoc, l2_assoc = 16, 4, 8

    bp_params = {'BiModeBP': (1024, 2), 'TournamentBP': (4096, 2), 'LTAGE': (16384, 3)}
    bp_entries, bp_bits = bp_params.get(r['bp'], (4096, 2))

    l1d_lat = 2 if l1d_bytes <= 128*1024 else 3
    l2_lat = 3 if l2_bytes <= 512*1024 else 4 if l2_bytes <= 1024*1024 else 5 if l2_bytes <= 2*1024*1024 else 6
    l3_lat = 10

    total_cycles = int(float(r['ciclos_totais']))
    idle_cycles = int(float(r['idle_cycles']))
    busy_cycles = total_cycles - idle_cycles

    pc = lambda col: int(float(r[col]) // n)          # contagem media por nucleo
    insts_pc = pc('insts_totais')
    cyc_pc = total_cycles // n
    idle_pc = idle_cycles // n
    busy_pc = busy_cycles // n
    duty = min(1.0, insts_pc / cyc_pc / nuc['issue_width']) if cyc_pc else 0.0
    mc_duty_cycle = max(0.05, busy_cycles / total_cycles) if total_cycles > 0 else 0.1

    core_xml = '''        <component id="system.core{core_id}" name="core{core_id}">
            <!-- Core: {cpu_type} -->
            <param name="clock_rate" value="{clock_mhz}"/>
            <param name="opt_local" value="0"/>
            <param name="instruction_length" value="32"/>
            <param name="opcode_width" value="7"/>
            <param name="x86" value="0"/>
            <param name="micro_opcode_width" value="8"/>
            <param name="machine_type" value="{machine_type}"/><!-- {machine_desc} -->
            <param name="number_hardware_threads" value="1"/>
            <param name="fetch_width" value="{fetch_width}"/>
            <param name="number_instruction_fetch_ports" value="1"/>
            <param name="decode_width" value="{decode_width}"/>
            <param name="issue_width" value="{issue_width}"/>
            <param name="peak_issue_width" value="{issue_width}"/>
            <param name="commit_width" value="{commit_width}"/>
            <param name="fp_issue_width" value="1"/>
            <param name="prediction_width" value="1"/>
            <param name="pipelines_per_core" value="1,1"/>
            <param name="pipeline_depth" value="8,8"/>
            <param name="ALU_per_core" value="{alu}"/>
            <param name="MUL_per_core" value="{mul}"/>
            <param name="FPU_per_core" value="{fpu}"/>
            <param name="instruction_buffer_size" value="32"/>
            <param name="decoded_stream_buffer_size" value="16"/>
            <param name="instruction_window_scheme" value="0"/>
            <param name="instruction_window_size" value="{iw_size}"/>
            <param name="fp_instruction_window_size" value="15"/>
            <param name="ROB_size" value="{rob_size}"/>
            <param name="archi_Regs_IRF_size" value="32"/>
            <param name="archi_Regs_FRF_size" value="32"/>
            <param name="phy_Regs_IRF_size" value="{phy_irf}"/>
            <param name="phy_Regs_FRF_size" value="{phy_frf}"/>
            <param name="rename_scheme" value="0"/>
            <param name="checkpoint_depth" value="1"/>
            <param name="LSU_order" value="inorder"/>
            <param name="store_buffer_size" value="{store_buf}"/>
            <param name="load_buffer_size" value="{load_buf}"/>
            <param name="memory_ports" value="2"/>
            <param name="RAS_size" value="32"/>

            <!-- Core Statistics (medidas no gem5) -->
            <stat name="total_instructions" value="{insts_pc}"/>
            <stat name="int_instructions" value="{int_insts}"/>
            <stat name="fp_instructions" value="{fp_insts}"/>
            <stat name="branch_instructions" value="{branch_insts}"/>
            <stat name="committed_instructions" value="{insts_pc}"/>
            <stat name="pipeline_duty_cycle" value="{duty:.4f}"/>
            <stat name="total_cycles" value="{cyc_pc}"/>
            <stat name="idle_cycles" value="{idle_pc}"/>
            <stat name="busy_cycles" value="{busy_pc}"/>

            <!-- Functional Unit Stats -->
            <stat name="ialu_accesses" value="{ialu_acc}"/>
            <stat name="fpu_accesses" value="{fpu_acc}"/>
            <stat name="mul_accesses" value="{mul_acc}"/>

            <!-- Branch Predictor -->
            <param name="number_of_BPT" value="1"/>
            <component id="system.core{core_id}.predictor" name="BPT">
                <param name="global_predictor_entries" value="{bp_entries}"/>
                <param name="global_predictor_bits" value="{bp_bits}"/>
                <param name="chooser_predictor_entries" value="{bp_entries}"/>
                <param name="chooser_predictor_bits" value="{bp_bits}"/>
            </component>

            <!-- TLBs -->
            <component id="system.core{core_id}.itlb" name="itlb">
                <param name="number_entries" value="64"/>
                <stat name="total_accesses" value="{itlb_acc}"/>
                <stat name="total_misses" value="10"/>
            </component>

            <!-- L1I Cache -->
            <component id="system.core{core_id}.icache" name="icache">
                <param name="icache_config" value="65536,32,4,1,2,2,32,0"/>
                <param name="buffer_sizes" value="4,4,4,0"/>
                <stat name="read_accesses" value="{l1i_acc}"/>
                <stat name="read_misses" value="{l1i_miss}"/>
            </component>

            <!-- Data TLB -->
            <component id="system.core{core_id}.dtlb" name="dtlb">
                <param name="number_entries" value="64"/>
                <stat name="total_accesses" value="{dtlb_acc}"/>
                <stat name="total_misses" value="10"/>
            </component>

            <!-- L1D Cache -->
            <component id="system.core{core_id}.dcache" name="dcache">
                <param name="dcache_config" value="{l1d_bytes},32,{l1d_assoc},1,{l1d_lat},{l1d_lat},32,1"/>
                <param name="buffer_sizes" value="4,4,4,4"/>
                <stat name="read_accesses" value="{l1d_read_acc}"/>
                <stat name="write_accesses" value="{l1d_write_acc}"/>
                <stat name="read_misses" value="{l1d_read_miss}"/>
                <stat name="write_misses" value="{l1d_write_miss}"/>
            </component>

            <!-- BTB -->
            <param name="number_of_BTB" value="1"/>
            <component id="system.core{core_id}.BTB" name="BTB">
                <param name="BTB_config" value="4096,4,2,2,1,1"/>
                <stat name="read_accesses" value="{btb_acc}"/>
            </component>
        </component>
'''

    xml = '''<?xml version="1.0" ?>
<!-- McPAT configuration file for NVIDIA Thor Edge Device DSE (v4: contagens medidas no gem5) -->
<!-- Generated from gem5 results: {cenario} -->
<!-- CPU Type: {cpu_type}, Cores: {num_cores}, Clock: {clock_ghz} GHz -->
<component id="root" name="root">
    <component id="system" name="system">
        <!-- System parameters -->
        <param name="number_of_cores" value="{num_cores}"/>
        <param name="number_of_L1Directories" value="0"/>
        <param name="number_of_L2Directories" value="0"/>
        <param name="Private_L2" value="1"/>
        <param name="number_of_L2s" value="{num_cores}"/>
        <param name="number_of_L3s" value="1"/>
        <param name="homogeneous_L1Directories" value="0"/>
        <param name="homogeneous_L2Directories" value="0"/>
        <param name="homogeneous_cores" value="0"/>
        <param name="homogeneous_L2s" value="0"/>
        <param name="homogeneous_L3s" value="1"/>
        <param name="number_of_NoCs" value="1"/>
        <param name="homogeneous_NoCs" value="1"/>
        <param name="core_tech_node" value="22"/><!-- nm -->
        <param name="target_core_clockrate" value="{clock_mhz}"/><!-- MHz -->
        <param name="temperature" value="350"/><!-- Kelvin -->
        <param name="number_cache_levels" value="3"/>
        <param name="interconnect_projection_type" value="1"/><!-- conservative -->
        <param name="device_type" value="2"/><!-- LOP: Low Operating Power -->
        <param name="longer_channel_device" value="1"/>
        <param name="Embedded" value="1"/><!-- ARM embedded processor -->
        <param name="machine_bits" value="32"/>
        <param name="virtual_address_width" value="32"/>
        <param name="physical_address_width" value="40"/>
        <param name="virtual_memory_page_size" value="4096"/>

        <!-- Global statistics -->
        <stat name="total_cycles" value="{total_cycles}"/>
        <stat name="idle_cycles" value="{idle_cycles}"/>
        <stat name="busy_cycles" value="{busy_cycles}"/>

        <!-- ========== CORES ========== -->
'''.format(cenario=r.get('cenario', 'unknown'), cpu_type=cpu_type, num_cores=n,
           clock_ghz=clock_ghz, clock_mhz=clock_mhz, total_cycles=total_cycles,
           idle_cycles=idle_cycles, busy_cycles=busy_cycles)

    for core_id in range(n):
        xml += core_xml.format(
            core_id=core_id, cpu_type=cpu_type, clock_mhz=clock_mhz, **nuc,
            insts_pc=insts_pc, int_insts=pc('int_insts'), fp_insts=pc('fp_insts'),
            branch_insts=pc('branch_insts'), duty=duty, cyc_pc=cyc_pc, idle_pc=idle_pc,
            busy_pc=busy_pc, ialu_acc=pc('ialu_acc'), fpu_acc=pc('fpu_acc'),
            mul_acc=pc('mul_acc'), bp_entries=bp_entries, bp_bits=bp_bits,
            itlb_acc=pc('l1i_accesses'), dtlb_acc=pc('load_insts') + pc('store_insts'),
            l1i_acc=pc('l1i_accesses'), l1i_miss=pc('l1i_miss'),
            l1d_bytes=l1d_bytes, l1d_assoc=l1d_assoc, l1d_lat=l1d_lat,
            l1d_read_acc=pc('l1d_read_acc'), l1d_write_acc=pc('l1d_write_acc'),
            l1d_read_miss=pc('l1d_read_miss'), l1d_write_miss=pc('l1d_write_miss'),
            btb_acc=pc('btb_acc'))

    xml += '        <!-- ========== L2 PRIVATE CACHES ========== -->\n'
    for core_id in range(n):
        xml += '''        <component id="system.L2{core_id}" name="L2{core_id}">
            <param name="L2_config" value="{l2_bytes},32,{l2_assoc},8,8,{l2_lat},32,1"/>
            <param name="buffer_sizes" value="16,16,16,16"/>
            <param name="clockrate" value="{clock_mhz}"/>
            <param name="ports" value="1,1,1"/>
            <param name="device_type" value="0"/>
            <stat name="read_accesses" value="{l2_read_acc}"/>
            <stat name="write_accesses" value="{l2_write_acc}"/>
            <stat name="read_misses" value="{l2_read_miss}"/>
            <stat name="write_misses" value="{l2_write_miss}"/>
            <stat name="duty_cycle" value="1.0"/>
        </component>
'''.format(core_id=core_id, l2_bytes=l2_bytes, l2_assoc=l2_assoc, l2_lat=l2_lat,
           clock_mhz=clock_mhz, l2_read_acc=pc('l2_read_acc'), l2_write_acc=pc('l2_write_acc'),
           l2_read_miss=pc('l2_read_miss'), l2_write_miss=pc('l2_write_miss'))

    l3_total = int(float(r['l3_read_acc']) + float(r['l3_write_acc']))
    xml += '''        <!-- ========== L3 SHARED CACHE ========== -->
        <component id="system.L30" name="L30">
            <param name="L3_config" value="{l3_bytes},64,{l3_assoc},16,16,{l3_lat},64,1"/>
            <param name="buffer_sizes" value="16,16,16,16"/>
            <param name="clockrate" value="{clock_mhz}"/>
            <param name="ports" value="1,1,1"/>
            <param name="device_type" value="0"/>
            <stat name="read_accesses" value="{l3_read_acc}"/>
            <stat name="write_accesses" value="{l3_write_acc}"/>
            <stat name="read_misses" value="{l3_read_miss}"/>
            <stat name="write_misses" value="{l3_write_miss}"/>
            <stat name="duty_cycle" value="0.5"/>
        </component>

        <!-- ========== NoC ========== -->
        <component id="system.NoC0" name="noc0">
            <param name="clockrate" value="{clock_mhz}"/>
            <param name="type" value="0"/><!-- Bus -->
            <param name="topology" value="1"/>
            <param name="horizontal_nodes" value="{num_cores}"/>
            <param name="vertical_nodes" value="1"/>
            <param name="has_global_link" value="0"/>
            <param name="link_throughput" value="1"/>
            <param name="link_latency" value="1"/>
            <param name="input_ports" value="1"/>
            <param name="output_ports" value="1"/>
            <param name="virtual_channel_per_port" value="2"/>
            <param name="flit_bits" value="128"/>
            <param name="input_buffer_entries_per_vc" value="8"/>
            <param name="chip_coverage" value="1"/>
            <param name="link_routing_over_percentage" value="0.5"/>
            <stat name="total_accesses" value="{noc_acc}"/>
            <stat name="duty_cycle" value="0.1"/>
        </component>

        <!-- ========== Memory Controller ========== -->
        <component id="system.mc" name="mc">
            <param name="type" value="1"/>
            <param name="mc_clock" value="3200"/><!-- LPDDR5_6400: 6400 MT/s / 2 = 3200 MHz -->
            <param name="peak_transfer_rate" value="6400"/>
            <param name="block_size" value="64"/>
            <param name="number_mcs" value="1"/>
            <param name="memory_channels_per_mc" value="1"/>
            <param name="number_ranks" value="0"/>
            <param name="req_window_size_per_channel" value="32"/>
            <param name="IO_buffer_size_per_channel" value="32"/>
            <param name="databus_width" value="128"/>
            <param name="addressbus_width" value="51"/>
            <stat name="memory_accesses" value="{memory_accesses}"/>
            <stat name="memory_reads" value="{memory_reads}"/>
            <stat name="memory_writes" value="{memory_writes}"/>
            <stat name="duty_cycle" value="{mc_duty_cycle:.4f}"/>
            <param name="withPHY" value="1"/>
        </component>

        <!-- ========== NIU ========== -->
        <component id="system.niu" name="niu">
            <param name="clockrate" value="350"/>
            <param name="number_units" value="1"/>
            <param name="type" value="1"/>
            <stat name="duty_cycle" value="0.1"/>
            <stat name="total_load_perc" value="0.1"/>
        </component>

        <!-- ========== PCIe ========== -->
        <component id="system.pcie" name="pcie">
            <param name="type" value="1"/>
            <param name="withPHY" value="1"/>
            <param name="clockrate" value="350"/>
            <param name="number_units" value="1"/>
            <param name="num_channels" value="8"/>
            <stat name="duty_cycle" value="0.1"/>
            <stat name="total_load_perc" value="0.1"/>
        </component>

        <!-- ========== Flash Controller ========== -->
        <component id="system.flashc" name="flashc">
            <param name="number_flashcs" value="1"/>
            <param name="type" value="1"/>
            <param name="withPHY" value="1"/>
            <param name="peak_transfer_rate" value="200"/>
            <stat name="duty_cycle" value="0.1"/>
            <stat name="total_load_perc" value="0.1"/>
        </component>

    </component>
</component>
'''.format(l3_bytes=l3_bytes, l3_assoc=l3_assoc, l3_lat=l3_lat, clock_mhz=clock_mhz,
           l3_read_acc=int(float(r['l3_read_acc'])), l3_write_acc=int(float(r['l3_write_acc'])),
           l3_read_miss=int(float(r['l3_read_miss'])), l3_write_miss=int(float(r['l3_write_miss'])),
           num_cores=n, noc_acc=l3_total,
           memory_accesses=int(float(r['mem_reads']) + float(r['mem_writes'])),
           memory_reads=int(float(r['mem_reads'])), memory_writes=int(float(r['mem_writes'])),
           mc_duty_cycle=mc_duty_cycle)
    return xml


def run_one(args):
    row, xml_dir, results_dir, so_xml = args
    cen = row['cenario']
    xml_path = Path(xml_dir) / f"{cen}.xml"
    xml_path.write_text(generate_mcpat_xml(row))
    if so_xml:
        return cen, True
    res = subprocess.run([MCPAT_BIN, "-infile", str(xml_path)],
                         capture_output=True, text=True, timeout=120)
    out = res.stdout + res.stderr
    (Path(results_dir) / f"{cen}.txt").write_text(out)
    ok = "Runtime Dynamic" in out
    print(f"[{'OK' if ok else 'ERRO'}] {cen}")
    return cen, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="processa a geracao sem CAN ID")
    ap.add_argument("--so-xml", action="store_true", help="so gera os XMLs, sem rodar o McPAT")
    a = ap.parse_args()
    p = PERFIS["v2" if a.v2 else "v3"]

    if not Path(p["gem5_csv"]).exists():
        sys.exit(f"ERRO: {p['gem5_csv']} nao encontrado (rode gem5/reparse_completo_mcpat_v4.py)")
    if not a.so_xml and not Path(MCPAT_BIN).exists():
        sys.exit(f"ERRO: {MCPAT_BIN} nao encontrado (rode da raiz do projeto)")
    Path(p["xml_dir"]).mkdir(parents=True, exist_ok=True)
    Path(p["results_dir"]).mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(p["gem5_csv"]).to_dict("records")
    print(f"{len(rows)} cenarios de {p['gem5_csv']} -> {p['xml_dir']}/"
          + ("" if a.so_xml else f" e {p['results_dir']}/"))
    jobs = [(r, p["xml_dir"], p["results_dir"], a.so_xml) for r in rows]
    with mp.Pool(WORKERS) as pool:
        res = list(pool.imap_unordered(run_one, jobs, chunksize=1))
    ok = sum(1 for _, s in res if s)
    print(f"\nConcluido: {ok}/{len(rows)} com sucesso")
    if ok < len(rows):
        print("Falharam:", sorted(c for c, s in res if not s))
        sys.exit(1)


if __name__ == "__main__":
    main()
