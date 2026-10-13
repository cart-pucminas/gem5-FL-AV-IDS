import m5
from m5.objects import *
import argparse

# ============================================================
# PARSER DE ARGUMENTOS
# ============================================================
parser = argparse.ArgumentParser(
    description='NVIDIA Thor Multicore DSE'
)

# OBRIGATÓRIO PARA O SSCAD: Toggle de Arquitetura
parser.add_argument(
    '--cpu_type',
    type=str,
    default='O3',
    choices=['O3', 'Minor'],
    help='Define a microarquitetura: O3 (Out-of-Order) ou Minor (In-Order)'
)

parser.add_argument('--num_cpus', type=int, default=4)
parser.add_argument('--prefetcher', type=str, default='none', choices=['none', 'stride'])
parser.add_argument('--l3_assoc', type=int, default=16)
parser.add_argument('--l2_size', type=str, default='1MiB')
parser.add_argument('--l3_size', type=str, default='16MiB')
parser.add_argument('--l1d_size', type=str, default='64KiB')
parser.add_argument('--l1d_assoc', type=int, default=4)
parser.add_argument('--bp_type', type=str, default='TournamentBP', choices=['BiModeBP', 'TournamentBP', 'LTAGE'])
parser.add_argument('--binary', type=str, default='rf_edge_arm_d15')

args, unknown = parser.parse_known_args()

# ============================================================
# LATÊNCIAS FÍSICAS REAIS (HP CACTI 22nm - 2.76 GHz)
# ============================================================

L1D_LATENCIES_SIZE = {
    '16KiB': 1,
    '32KiB': 2,
    '64KiB': 2,
    '128KiB': 2
}

L2_LATENCIES = {
    '512KiB': 3,
    '1MiB': 4,
    '2MiB': 5,
    '4MiB': 6
}

# ============================================================
# CLASSES DE CACHE
# ============================================================

class L1ICache(Cache):
    # Física extraída do CACTI para 64KiB, 4-way, 22nm
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 1  

    mshrs = 4
    tgts_per_mshr = 20

class L1DCache(Cache):
    mshrs = 4
    tgts_per_mshr = 20

    def __init__(self, size, assoc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.assoc = assoc
        
        lat_size = L1D_LATENCIES_SIZE.get(size, 2)
        
        # Se for 16KiB e 8-way, força 2 ciclos 
        if size == '16KiB' and assoc == 8:
            lat_size = 2
            
        self.tag_latency = lat_size
        self.data_latency = lat_size
        self.response_latency = 1

class L2PrivateCache(Cache):
    assoc = 8
    mshrs = 20
    tgts_per_mshr = 12

    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        
        lat_cycles = L2_LATENCIES.get(size, 4)
        self.tag_latency = lat_cycles
        self.data_latency = lat_cycles
        self.response_latency = 2

class L3SharedCache(Cache):
    # Fixado conforme CACTI
    tag_latency = 10
    data_latency = 10
    response_latency = 5
    mshrs = 20
    tgts_per_mshr = 12

# ============================================================
# SISTEMA
# ============================================================

system = System()
system.clk_domain = SrcClockDomain(clock='2.76GHz', voltage_domain=VoltageDomain())
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('4GiB')]

system.l3bus = L2XBar()
system.membus = SystemXBar()

# Snoop filter dimensionado p/ multicore com caches grandes (L2>=2MB).
# O default (131072 blocos) estoura com 4 cores + L2 2MB + L3 16MB.
# So AUMENTA o limite - nao altera resultados de configs que ja cabiam (ex. ERAD 1MB).
if args.num_cpus > 1:
    system.l3bus.snoop_filter.max_capacity = '32MiB'

# ============================================================
# CPUs
# ============================================================

cpu_list = []

for i in range(args.num_cpus):

    # Branch Predictor
    if args.bp_type == 'BiModeBP':
        bp = BiModeBP()
    elif args.bp_type == 'LTAGE':
        bp = LTAGE()
    else:
        bp = TournamentBP()

    # MÁGICA DA ARQUITETURA
    if args.cpu_type == 'Minor':
        cpu = MinorCPU()
    else:
        try:
            cpu = ArmO3CPU()
        except:
            cpu = DerivO3CPU()

    cpu.branchPred = BranchPredictor(conditionalBranchPred=bp)

    icache = L1ICache(size='64KiB')
    dcache = L1DCache(size=args.l1d_size, assoc=args.l1d_assoc)

    cpu.addPrivateSplitL1Caches(icache, dcache)
    cpu.createInterruptController()

    cpu_list.append(cpu)

system.cpu = cpu_list

# ============================================================
# HIERARQUIA DE CACHE E MEMÓRIA
# ============================================================

for i in range(args.num_cpus):
    cpu = system.cpu[i]
    l2bus = L2XBar()
    setattr(system, f'l2bus_{i}', l2bus)
    cpu.connectCachedPorts(l2bus.cpu_side_ports)

    l2cache = L2PrivateCache(size=args.l2_size)
    if args.prefetcher == 'stride':
        l2cache.prefetcher = StridePrefetcher(degree=8, latency=1)

    setattr(system, f'l2cache_{i}', l2cache)
    l2cache.cpu_side = l2bus.mem_side_ports
    l2cache.mem_side = system.l3bus.cpu_side_ports

system.l3cache = L3SharedCache(size=args.l3_size, assoc=args.l3_assoc)
system.l3cache.cpu_side = system.l3bus.mem_side_ports
system.l3cache.mem_side = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = LPDDR5_6400_1x16_BG_BL16()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

# ============================================================
# WORKLOAD MULTIPROCESS
# ============================================================

bin_path = f'../{args.binary}'
process_list = []

for i in range(args.num_cpus):
    process = Process()
    process.executable = bin_path
    process.cmd = [bin_path, str(i), str(args.num_cpus)]
    process.pid = 100 + i
    process_list.append(process)

system.workload = SEWorkload.init_compatible(bin_path)

for i in range(args.num_cpus):
    system.cpu[i].workload = process_list[i]
    system.cpu[i].createThreads()

# ============================================================
# ROOT + INSTANTIATE
# ============================================================

root = Root(full_system=False, system=system)

print("=" * 60)
print("NVIDIA THOR GEM5 MULTICORE (CACTI + CPU TOGGLE)")
print("=" * 60)
print(f"CPU Type     : {args.cpu_type}")
print(f"Binary       : {args.binary}")
print(f"Cores        : {args.num_cpus}")
print(f"L1D Size     : {args.l1d_size} (Assoc: {args.l1d_assoc})")
print(f"L2 Size      : {args.l2_size}")
print(f"BranchPred   : {args.bp_type}")
print("=" * 60)

m5.instantiate()

print("[INFO] Iniciando simulação...")
exit_event = m5.simulate()

print("=" * 60)
print("[INFO] Simulação finalizada")
print(f"Tick final : {m5.curTick()}")
print(f"Causa      : {exit_event.getCause()}")
print("=" * 60)