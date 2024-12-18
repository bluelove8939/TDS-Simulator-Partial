import math
import numpy as np

from tds_sim.simulation.module import *
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.spm_cache import SPMCacheModule, CacheReplacePolicy
from tds_sim.simulation.modeling.gcn_accelerator_core import GCNAcceleratorCore


class DebugModule(Module):
    def __init__(self, name):
        super().__init__(name)
        
    @callback_method
    def debug_callback(self):
        print(f"[CALLBACK] timestamp: {self.context.timestamp:<3d}  response: {self.child_session.response}")
    
    @property
    def is_idle(self) -> bool:
        return len(self.session.child_session_dir.keys()) == 0
    

if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    
    memory   = MainMemoryModule  (name="Memory",  access_latency=40, access_granularity=32)
    cache    = SPMCacheModule    (name="Cache",   nxt_level_memory=memory, capacity=4096, way_num=4, cacheline_size=32, replace_policy=CacheReplacePolicy.LRU, bank_num=32, access_latency=1)
    gcn_core = GCNAcceleratorCore(name="GCNCore", unified_cache=cache, main_memory=memory, pe_num=32, mac_per_pe=32, word_dtype=np.dtype(np.int8), ptr_dtype=np.dtype(np.int32), acc_dtype=np.dtype(np.int32), verbose=True)
    debugger = DebugModule       (name="Debugger")
    
    context = Context()
    context.register_module(cache, memory, gcn_core, debugger)
    
    #############################################
    # Create Workload
    #############################################
    
    M = 8
    N = 32
    K = 128
    
    a = np.arange(0, M*K, 1, dtype=gcn_core.word_dtype).reshape((M, K))
    b = np.arange(0, K*N, 1, dtype=gcn_core.word_dtype).reshape((K, N))
    
    print(f"Tensor A: {M}x{K}")
    print(f"data: ", end="")
    print(a)
    
    print(f"\nTensor B: {K}x{N}")
    print(b)
    
    print(f"\nCreating Memory Dump")
    addr_align = cache.cacheline_size
    
    addr_data_a, size_data_a = 0,                                                                a.size     * a.itemsize
    addr_data_b, size_data_b = math.ceil((addr_data_a + size_data_a) / addr_align) * addr_align, b.size     * b.itemsize
    addr_data_c              = math.ceil((addr_data_b + size_data_b) / addr_align) * addr_align
    
    print(f"data A address: {addr_data_a}")
    print(f"data B address: {addr_data_b}")
    
    memory.memory.set_data(addr_data_a, a.flatten())
    memory.memory.set_data(addr_data_b, b.flatten())
    
    print("\n=== SIMULATION PROCESS ===")   
    subsession = gcn_core.gemm_tile(tiled_dimM=M, tiled_dimN=N, tiled_dimK=K)
    debugger.session.submit_child_session(subsession=subsession, callbacks=[debugger.debug_callback])

    while not debugger.is_idle:
        context.increase_timestamp()
