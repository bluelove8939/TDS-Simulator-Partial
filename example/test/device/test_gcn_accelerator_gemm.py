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
            name="Cache", nxt_level_memory=self.memory, capacity=4096, way_num=4, cacheline_size=32, replace_policy=CacheReplacePolicy.LRU, bank_num=8, access_latency=1)
        
        self.gcn_core = GCNAcceleratorCore(
            name="GCNCore", unified_cache=self.cache, main_memory=self.memory, pe_num=8, mac_per_pe=32,
            word_dtype=np.dtype(np.int8), ptr_dtype=np.dtype(np.int32), acc_dtype=np.dtype(np.int32), verbose=True)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    
    #############################################
    # Create Workload
    #############################################
    
    device = GCNAccelerator(print_debug_info=True)
    device.initialize()
    
    M = 16
    N = 33
    K = 4
    
    a = np.arange(0, M*K, 1, dtype=device.gcn_core.word_dtype).reshape((M, K))
    b = np.arange(0, K*N, 1, dtype=device.gcn_core.word_dtype).reshape((K, N))
    
    print(f"Tensor A: {M}x{K}")
    print(a)
    print(f"\nTensor B: {K}x{N}")
    print(b)
    
    print("\n=== TILING PROCESS ===")
    tiled_dimM = device.gcn_core.pe_num
    tiled_dimN = device.gcn_core.mac_per_pe
    tiled_dimK = min(math.floor(device.cache.spm_bank_capacity / (math.ceil(device.gcn_core.pe_num / device.cache.bank_num) + math.ceil(device.gcn_core.mac_per_pe / device.cache.bank_num))), K)
    
    tile_numM = math.ceil(M / tiled_dimM)
    tile_numN = math.ceil(N / tiled_dimN)
    tile_numK = math.ceil(K / tiled_dimK)
    
    pad_dimM = tile_numM * tiled_dimM - M
    pad_dimN = tile_numN * tiled_dimN - N
    pad_dimK = tile_numK * tiled_dimK - K
    
    a = np.pad(a, ((0, pad_dimM), (0, pad_dimK)), 'constant', constant_values=0)
    b = np.pad(b, ((0, pad_dimK), (0, pad_dimN)), 'constant', constant_values=0)
    
    print(f"pad M: {pad_dimM}")
    print(f"pad N: {pad_dimN}")
    print(f"pad K: {pad_dimK}")
    
    padded_dimM = M + pad_dimM
    padded_dimN = N + pad_dimN
    padded_dimK = K + pad_dimK
    
    # a = a.reshape((tile_numM, tiled_dimM, tile_numK, tiled_dimK)).transpose((2, 0, 3, 1))  # transposed! (M, K) -> (K, M)
    a_tiled = a.reshape((tile_numM, tiled_dimM, tile_numK, tiled_dimK)).transpose((0, 2, 1, 3))
    b_tiled = b.reshape((tile_numK, tiled_dimK, tile_numN, tiled_dimN)).transpose((0, 2, 1, 3))
    
    print(f"\nCreating Memory Dump")
    addr_align = device.cache.cacheline_size
    
    rowA_addr, rowA_size = 0,                                                            a_tiled.size     * a_tiled.itemsize
    rowB_addr, rowB_size = math.ceil((rowA_addr + rowA_size) / addr_align) * addr_align, b_tiled.size     * b_tiled.itemsize
    rowC_addr            = math.ceil((rowB_addr + rowB_size) / addr_align) * addr_align
    
    device.memory.memory.set_data(rowA_addr, a_tiled.flatten())
    device.memory.memory.set_data(rowB_addr, b_tiled.flatten())
    
    print(f"data A address: {rowA_addr}")
    print(f"data B address: {rowB_addr}")
    
    print(f"\nCreating On-chip Memory Map")
    rowA_bank_offset = 0
    rowB_bank_offset = tiled_dimK * math.ceil(device.gcn_core.pe_num / device.cache.bank_num)
    
    print(f"data A on-chip bank offset: {rowA_bank_offset}")
    print(f"data B on-chip bank offset: {rowB_bank_offset}")
    
    print("\n=== SIMULATION PROCESS ===")   
    program = Program()
    
    thread = program.create_thread()
    thread.add_session(device.cache.cache_flush())
    
    program.merge_thread()
    
    word_dtype = device.gcn_core.word_dtype
    acc_dtype  = device.gcn_core.acc_dtype
    
    dataA_tile_size = tiled_dimM * tiled_dimK
    dataB_tile_size = tiled_dimK * tiled_dimN
    
    for tile_idxM in range(tile_numM):
        for tile_idxN in range(tile_numN):    
            for tile_idxK in range(tile_numK):
                load_a_thread = program.create_thread()
                load_b_thread = program.create_thread()
                
                # load matrix A
                for bank_idx in range(min(device.cache.bank_num, tiled_dimM)):
                    load_a_thread.add_session(device.cache.spm_load_data(
                        ofc_addr=rowA_addr + ((tile_idxM *  tile_numK + tile_idxK) * dataA_tile_size + bank_idx * padded_dimK) * word_dtype.itemsize,
                        size=tiled_dimK * word_dtype.itemsize,
                        bank_offset=rowA_bank_offset,
                        used_bank_num=1,
                        start_bank_idx=bank_idx,
                    ))
                
                # load matrix B
                load_b_thread.add_session(device.cache.spm_load_data(
                    ofc_addr=rowB_addr + (tile_idxK *  tile_numN + tile_idxN) * dataB_tile_size * word_dtype.itemsize,
                    size=tiled_dimK * tiled_dimN * word_dtype.itemsize,
                    bank_offset=rowB_bank_offset,
                    used_bank_num=min(device.cache.bank_num, tiled_dimN),
                ))
                
                program.merge_thread()
                
                thread = program.create_thread()
                thread.add_session(device.cache.spm_exchange_buffering_context())
                
                program.merge_thread()
                
                thread = program.create_thread()
                
                # compute
                thread.add_session(device.gcn_core.gemm_tile(
                    dataA_bank_offset=rowA_bank_offset,
                    dataB_bank_offset=rowB_bank_offset,
                    tiled_dimM=tiled_dimM,
                    tiled_dimN=tiled_dimN,
                    tiled_dimK=tiled_dimK,
                ))
                
                tile_osetM = tile_idxM * tiled_dimM
                tile_osetN = tile_idxN * tiled_dimN
                
                for pe_idx in range(min(device.gcn_core.pe_num, tiled_dimM)):
                    thread.add_session(device.gcn_core.flush_single_pe(
                        addr=rowC_addr + ((tile_osetM + pe_idx) * padded_dimN + tile_osetN) * acc_dtype.itemsize,
                        pe_idx=pe_idx,
                        numel=tiled_dimN,
                    ))
                    
    program.merge_thread()

    program.run_with_device(device=device, progress_bar=False)
    
    print("\n=== SIMULATION RESULT ===")
    reference = np.matmul(a.astype(device.gcn_core.acc_dtype), b.astype(device.gcn_core.acc_dtype))
    simulated = device.memory.memory.get_data(
        addr=rowC_addr, size=padded_dimN*padded_dimM*device.gcn_core.acc_dtype.itemsize, dtype=device.gcn_core.acc_dtype
    ).reshape((padded_dimM, padded_dimN))
    
    print("\nReference Result")
    print(reference[0:M, 0:N])
    print("\nSimulated Result")
    print(simulated[0:M, 0:N])
    
    print(f"\ntest terminated with {'succeed' if np.array_equal(reference, simulated) else 'failed'}")