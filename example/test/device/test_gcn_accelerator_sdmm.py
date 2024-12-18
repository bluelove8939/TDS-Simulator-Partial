import math
import numpy as np

from tds_sim.simulation.device import *
from tds_sim.simulation.program import Program
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.spm_cache import SPMCacheModule, CacheReplacePolicy
from tds_sim.simulation.modeling.gcn_accelerator_core import GCNAcceleratorCore


class GCNAccelerator(Device):
    def __init__(self, print_debug_info=False):
        super().__init__(print_debug_info)
        
        self.memory = MainMemoryModule(
            name="Memory", access_latency=40, access_granularity=32)
        
        self.cache = SPMCacheModule(
            name="Cache", nxt_level_memory=self.memory, capacity=4096, way_num=4, cacheline_size=32, replace_policy=CacheReplacePolicy.LRU, bank_num=32, access_latency=1)
        
        self.gcn_core = GCNAcceleratorCore(
            name="GCNCore", unified_cache=self.cache, main_memory=self.memory, pe_num=32, mac_per_pe=32,
            word_dtype=np.dtype(np.int8), ptr_dtype=np.dtype(np.int32), acc_dtype=np.dtype(np.int32), verbose=True)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    
    #############################################
    # Create Workload
    #############################################
    
    device = GCNAccelerator(print_debug_info=True)
    device.initialize()
    
    M = 64
    N = 64
    K = 64
    
    a = np.arange(0, M*K, 1, dtype=device.gcn_core.word_dtype).reshape((M, K))
    b = np.arange(0, K*N, 1, dtype=device.gcn_core.word_dtype).reshape((K, N))
    
    ptr_a = np.arange(0, M*K+1, K, dtype=device.gcn_core.ptr_dtype)
    idx_a = np.tile(np.arange(0, K, 1, dtype=device.gcn_core.ptr_dtype), M)
    
    print(f"Tensor A: {M}x{K}")
    print(f"data: ", end="")
    print(a.flatten())
    print(f"ptr:  ", end="")
    print(ptr_a)
    print(f"idx:  ", end="")
    print(idx_a)
    
    print(f"\nTensor B: {K}x{N}")
    print(b)
    
    print(f"\nCreating Memory Dump")
    addr_align = device.cache.cacheline_size
    
    rowA_addr, rowA_size = 0,                                                            a.size     * a.itemsize
    rowB_addr, rowB_size = math.ceil((rowA_addr + rowA_size) / addr_align) * addr_align, b.size     * b.itemsize
    ptrA_addr, ptrA_size = math.ceil((rowB_addr + rowB_size) / addr_align) * addr_align, ptr_a.size * ptr_a.itemsize
    idxA_addr, idxA_size = math.ceil((ptrA_addr + ptrA_size) / addr_align) * addr_align, idx_a.size * idx_a.itemsize
    rowC_addr            = math.ceil((idxA_addr + idxA_size) / addr_align) * addr_align
    
    print(f"data A address: {rowA_addr}")
    print(f"data B address: {rowB_addr}")
    print(f"ptr  A address: {ptrA_addr}" )
    print(f"idx  A address: {idxA_addr}" )
    
    device.memory.memory.set_data(rowA_addr, a.flatten())
    device.memory.memory.set_data(rowB_addr, b.flatten())
    device.memory.memory.set_data(ptrA_addr, ptr_a.flatten())
    device.memory.memory.set_data(idxA_addr, idx_a.flatten())
    
    print("\n=== SIMULATION PROCESS ===")  
    
    default_tiled_dimM = device.gcn_core.pe_num
    default_tiled_dimN = device.gcn_core.mac_per_pe
    
    tile_numM = math.ceil(M / default_tiled_dimM)
    tile_numN = math.ceil(N / default_tiled_dimN)
    
    program = Program()
    
    thread = program.create_thread()
    thread.add_session(device.cache.cache_flush())
    
    program.merge_thread()
    
    for tile_idxN in range(tile_numN):
        tiled_dimN = min(N - tile_idxN * default_tiled_dimN, default_tiled_dimN)
        
        for tile_idxM in range(tile_numM):
            thread = program.create_thread()
            
            tiled_dimM = min(M - tile_idxM * default_tiled_dimM, default_tiled_dimM)
            
            tile_rowA_addr = rowA_addr
            tile_rowB_addr = rowB_addr + tile_idxN * default_tiled_dimN * device.gcn_core.word_dtype.itemsize
            tile_ptrA_addr = ptrA_addr + tile_idxM * default_tiled_dimM * device.gcn_core.ptr_dtype.itemsize
            tile_idxA_addr = idxA_addr
            
            subsession = device.gcn_core.sdmm_tile(
                rowA_addr=tile_rowA_addr, rowB_addr=tile_rowB_addr, ptrA_addr=tile_ptrA_addr, idxA_addr=tile_idxA_addr,
                tiled_dimM=tiled_dimM, dimN=N, dimK=K)
            
            thread.add_session(subsession)
            
            program.merge_thread()
            
            thread = program.create_concurrent_thread()
            
            for pe_idx in range(device.gcn_core.pe_num):
                tile_offsetM = tile_idxM * default_tiled_dimM + pe_idx
                tile_offsetN = tile_idxN * default_tiled_dimN
                
                addr = rowC_addr + (tile_offsetM * N + tile_offsetN) * device.gcn_core.acc_dtype.itemsize
                numel = tiled_dimN
                
                thread.add_session(device.gcn_core.flush_single_pe(addr=addr, pe_idx=pe_idx, numel=numel))
                
            program.merge_thread()

    program.run_with_device(device=device, progress_bar=False)
    
    print("\n=== SIMULATION RESULT ===")
    reference = np.matmul(a.astype(device.gcn_core.acc_dtype), b.astype(device.gcn_core.acc_dtype))
    simulated = device.memory.memory.get_data(
        addr=rowC_addr, size=N*M*device.gcn_core.acc_dtype.itemsize, dtype=device.gcn_core.acc_dtype
    ).reshape((M, N))
    
    print("\nReference Result")
    print(reference)
    print("\nSimulated Result")
    print(simulated)
    
    print(f"\ntest terminated with {'succeed' if np.array_equal(reference, simulated) else 'failed'}")