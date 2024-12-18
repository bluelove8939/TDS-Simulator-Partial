import math
import torch
import numpy as np

import scipy
from scipy.sparse import coo_matrix

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

from tds_sim.simulation.device import *
from tds_sim.simulation.program import Program
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.spm_cache import SPMCacheModule, CacheReplacePolicy
from tds_sim.simulation.modeling.gcn_accelerator_core import GCNAcceleratorCore

from tds_sim.compiler.device_config import DeviceConfig, runtime_method
from tds_sim.common.custom_exception import CustomException
from tds_sim.common.quantize import quantize, dequantize
from tds_sim.common.sdmm_reordering import sdmm_reordering


class GCNAccelerator(Device):
    def __init__(
        self, 
        pe_num:         int, 
        mac_per_pe:     int, 
        cache_capacity: int, 
        cache_way_num:  int,
        word_dtype:     np.dtype=np.dtype(np.int8), 
        ptr_dtype:      np.dtype=np.dtype(np.int32), 
        acc_dtype:      np.dtype=np.dtype(np.int32), 
        mem_access_latency:   int=100,
        cache_access_latency: int=1,
        print_debug_info=False
    ):
        super().__init__(print_debug_info)
        
        cacheline_size = mac_per_pe * word_dtype.itemsize

        self.memory = MainMemoryModule(
            name="Memory", access_latency=mem_access_latency, access_granularity=cacheline_size)
        
        self.cache = SPMCacheModule(
            name="Cache", nxt_level_memory=self.memory, capacity=cache_capacity, way_num=cache_way_num, cacheline_size=cacheline_size, 
            replace_policy=CacheReplacePolicy.LRU, bank_num=pe_num, access_latency=cache_access_latency)
        
        self.gcn_core = GCNAcceleratorCore(
            name="GCNCore", unified_cache=self.cache, main_memory=self.memory, pe_num=pe_num, mac_per_pe=mac_per_pe,
            word_dtype=word_dtype, ptr_dtype=ptr_dtype, acc_dtype=acc_dtype, verbose=True)


class GCNAcceleratorConfig(DeviceConfig):
    def __init__(self, device: GCNAccelerator, verbose: bool=True, apply_reordering: bool=False, reordering_tile_size: int=512, simulate_sdmm: bool=True, simulate_gemm: bool=True, validate_output: bool=False):
        super().__init__(device)
        
        self.device = device
        self.verbose = verbose
        self.apply_reordering = apply_reordering
        self.reordering_tile_size = reordering_tile_size
        self.simulate_sdmm = simulate_sdmm
        self.simulate_gemm = simulate_gemm
        self.validate_output = validate_output
        
        if not self.device.is_initialized:
            self.device.initialize()
        
    def run_gemm(self, a: torch.Tensor, b: torch.Tensor):
        a = a.detach().clone().numpy()
        b = b.detach().clone().numpy()
        
        if a.shape[1] != b.shape[0]:
            raise CustomException(self, f"run_gemm failed because of the shape mismatch")
        
        if a.dtype != b.dtype:
            raise CustomException(self, f"run_gemm failed because of the dtype mismatch")
        
        M, K = a.shape
        K, N = b.shape
        
        if self.verbose:
            print(f"execute: run_gemm")
            print(f"  * matrix A shape:  {a.shape}")
            print(f"  * matrix B shape:  {b.shape}")
        
        #####################################################
        # STEP 1: Quantize the input tensors
        #####################################################
        
        a, scale_a = quantize(a, dtype=self.device.gcn_core.word_dtype)
        b, scale_b = quantize(b, dtype=self.device.gcn_core.word_dtype)
        
        #####################################################
        # STEP 2: Apply Tiling
        #####################################################
        
        tiled_dimM = self.device.gcn_core.pe_num
        tiled_dimN = self.device.gcn_core.mac_per_pe
        tiled_dimK = min(math.floor(self.device.cache.spm_bank_capacity / (math.ceil(self.device.gcn_core.pe_num / self.device.cache.bank_num) + math.ceil(self.device.gcn_core.mac_per_pe / self.device.cache.bank_num))), K)
        
        tile_numM = math.ceil(M / tiled_dimM)
        tile_numN = math.ceil(N / tiled_dimN)
        tile_numK = math.ceil(K / tiled_dimK)
        
        pad_dimM = tile_numM * tiled_dimM - M
        pad_dimN = tile_numN * tiled_dimN - N
        pad_dimK = tile_numK * tiled_dimK - K
        
        a = np.pad(a, ((0, pad_dimM), (0, pad_dimK)), 'constant', constant_values=0)
        b = np.pad(b, ((0, pad_dimK), (0, pad_dimN)), 'constant', constant_values=0)
        
        padded_dimM = M + pad_dimM
        padded_dimN = N + pad_dimN
        padded_dimK = K + pad_dimK
        
        # a = a.reshape((tile_numM, tiled_dimM, tile_numK, tiled_dimK)).transpose((2, 0, 3, 1))  # transposed! (M, K) -> (K, M)
        a_tiled = a.reshape((tile_numM, tiled_dimM, tile_numK, tiled_dimK)).transpose((0, 2, 1, 3))
        b_tiled = b.reshape((tile_numK, tiled_dimK, tile_numN, tiled_dimN)).transpose((0, 2, 1, 3))
        
        if self.verbose:
            print(f"  * tiled M dimension: {tiled_dimM}")
            print(f"  * tiled N dimension: {tiled_dimN}")
            print(f"  * tiled K dimension: {tiled_dimK}")
            
            print(f"  * pad M: {pad_dimM}")
            print(f"  * pad N: {pad_dimN}")
            print(f"  * pad K: {pad_dimK}")
        
        #####################################################
        # STEP 3: Create Off-chip Memory Map
        #####################################################
        
        addr_align = self.device.cache.cacheline_size
    
        rowA_addr, rowA_size = 0,                                                            a_tiled.size     * a_tiled.itemsize
        rowB_addr, rowB_size = math.ceil((rowA_addr + rowA_size) / addr_align) * addr_align, b_tiled.size     * b_tiled.itemsize
        rowC_addr            = math.ceil((rowB_addr + rowB_size) / addr_align) * addr_align
        
        self.device.memory.memory.set_data(rowA_addr, a_tiled.flatten())
        self.device.memory.memory.set_data(rowB_addr, b_tiled.flatten())
        
        #####################################################
        # STEP 4: Create On-chip Memory Map
        #####################################################
        
        rowA_bank_offset = 0
        rowB_bank_offset = tiled_dimK * math.ceil(self.device.gcn_core.pe_num / self.device.cache.bank_num)
        
        #####################################################
        # STEP 5: Compute GEMM
        #####################################################
        
        program = Program()
        
        thread = program.create_thread()
    
        thread.add_session(self.device.cache.cache_flush())
        program.merge_thread()
        
        word_dtype = self.device.gcn_core.word_dtype
        acc_dtype  = self.device.gcn_core.acc_dtype
        
        dataA_tile_size = tiled_dimM * tiled_dimK
        dataB_tile_size = tiled_dimK * tiled_dimN
        
        for tile_idxM in range(tile_numM):
            for tile_idxN in range(tile_numN):    
                for tile_idxK in range(tile_numK):
                    load_a_thread = program.create_thread()
                    load_b_thread = program.create_thread()
                    
                    # load matrix A
                    for bank_idx in range(min(self.device.cache.bank_num, tiled_dimM)):
                        load_a_thread.add_session(self.device.cache.spm_load_data(
                            ofc_addr=rowA_addr + ((tile_idxM *  tile_numK + tile_idxK) * dataA_tile_size + bank_idx * padded_dimK) * word_dtype.itemsize,
                            size=tiled_dimK * word_dtype.itemsize,
                            bank_offset=rowA_bank_offset,
                            used_bank_num=1,
                            start_bank_idx=bank_idx,
                        ))
                    
                    # load matrix B
                    load_b_thread.add_session(self.device.cache.spm_load_data(
                        ofc_addr=rowB_addr + (tile_idxK *  tile_numN + tile_idxN) * dataB_tile_size * word_dtype.itemsize,
                        size=tiled_dimK * tiled_dimN * word_dtype.itemsize,
                        bank_offset=rowB_bank_offset,
                        used_bank_num=min(self.device.cache.bank_num, tiled_dimN),
                    ))
                    
                    program.merge_thread()
                    
                    thread = program.create_thread()
                    thread.add_session(self.device.cache.spm_exchange_buffering_context())
                    
                    program.merge_thread()
                    
                    thread = program.create_thread()
                    
                    # compute
                    thread.add_session(self.device.gcn_core.gemm_tile(
                        dataA_bank_offset=rowA_bank_offset,
                        dataB_bank_offset=rowB_bank_offset,
                        tiled_dimM=tiled_dimM,
                        tiled_dimN=tiled_dimN,
                        tiled_dimK=tiled_dimK,
                    ))
                    
                    tile_osetM = tile_idxM * tiled_dimM
                    tile_osetN = tile_idxN * tiled_dimN
                    
                    for pe_idx in range(min(self.device.gcn_core.pe_num, tiled_dimM)):
                        thread.add_session(self.device.gcn_core.flush_single_pe(
                            addr=rowC_addr + ((tile_osetM + pe_idx) * padded_dimN + tile_osetN) * acc_dtype.itemsize,
                            pe_idx=pe_idx,
                            numel=tiled_dimN,
                        ))
                        
        program.merge_thread()
        
        st_time = self.device.context.timestamp     
        program.run_with_device(device=self.device, progress_bar=not self.device.print_debug_info, progress_bar_header="GEMM")
        ed_time = self.device.context.timestamp
        
        if self.verbose:
            print(f"summary: run_gemm")
            print(f"  * execution cycle: {ed_time - st_time}")
        
        if self.validate_output:
            reference = np.matmul(a.astype(self.device.gcn_core.acc_dtype), b.astype(self.device.gcn_core.acc_dtype))
            simulated = self.device.memory.memory.get_data(
                addr=rowC_addr, size=padded_dimN*padded_dimM*self.device.gcn_core.acc_dtype.itemsize, dtype=self.device.gcn_core.acc_dtype
            ).reshape((padded_dimM, padded_dimN))

            print(f"  * output validated: {np.array_equal(reference, simulated)}")
        
        self.create_log(action="GEMM", start_time=st_time, end_time=ed_time)
        
    def run_sdmm(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor):
        edge_index = edge_index.detach().clone().numpy()
        edge_weight = edge_weight.detach().clone().numpy()
        x = x.detach().clone().numpy()
        
        row, col = edge_index
        N = x.shape[0]
        
        ########################################################################
        # STEP 1: Preprocess adjacency matrix (COO -> CSR)
        ########################################################################
        
        adj = coo_matrix((edge_weight, (row, col)), shape=(N, N)).tocsr()
        
        if self.apply_reordering:
            adj, x = sdmm_reordering(adj, x, tile_size=self.reordering_tile_size, leave_pbar=False, only_dimM=True)
        
        a = adj.data
        ptr_a = adj.indptr.astype(self.device.gcn_core.ptr_dtype)
        idx_a = adj.indices.astype(self.device.gcn_core.ptr_dtype)
        
        ########################################################################
        # STEP 2: Preprocess node features
        ########################################################################
        # Note
        #   * We should apply zero padding to align the matrix into the cacheline
        #   * The number of node feautures should be the multiple of 'self.device.gcn_core.mac_per_pe'
        
        nMAC = self.device.gcn_core.mac_per_pe
        if x.shape[1] % nMAC != 0:
            pad_size = nMAC - (x.shape[1] % nMAC)
            b = np.pad(x, ((0, 0), (0, pad_size)), 'constant', constant_values=0)
        else:
            b = x
        
        ########################################################################
        # STEP 3: Quantize the input tensors
        ########################################################################
        
        a, scale_a = quantize(a, dtype=self.device.gcn_core.word_dtype)
        b, scale_b = quantize(b, dtype=self.device.gcn_core.word_dtype)
        
        ########################################################################
        # STEP 4: Create address map and push data to the memory
        ########################################################################
        
        addr_align = self.device.cache.cacheline_size
        
        rowA_addr, rowA_size = 0,                                                            a.size     * a.itemsize
        rowB_addr, rowB_size = math.ceil((rowA_addr + rowA_size) / addr_align) * addr_align, b.size     * b.itemsize
        ptrA_addr, ptrA_size = math.ceil((rowB_addr + rowB_size) / addr_align) * addr_align, ptr_a.size * ptr_a.itemsize
        idxA_addr, idxA_size = math.ceil((ptrA_addr + ptrA_size) / addr_align) * addr_align, idx_a.size * idx_a.itemsize
        rowC_addr            = math.ceil((idxA_addr + idxA_size) / addr_align) * addr_align
        
        self.device.memory.memory.set_data(rowA_addr, a.flatten())
        self.device.memory.memory.set_data(rowB_addr, b.flatten())
        self.device.memory.memory.set_data(ptrA_addr, ptr_a.flatten())
        self.device.memory.memory.set_data(idxA_addr, idx_a.flatten())
        
        ########################################################################
        # STEP 5: Compute SDMM
        ########################################################################
        
        dimM = N
        dimN = b.shape[1]
        dimK = N
        
        default_tiled_dimM = self.device.gcn_core.pe_num
        default_tiled_dimN = self.device.gcn_core.mac_per_pe
        
        tile_numM = math.ceil(dimM / default_tiled_dimM)
        tile_numN = math.ceil(dimN / default_tiled_dimN)
        
        program = Program()
        
        thread = program.create_thread()
        thread.add_session(self.device.cache.cache_flush())
        program.merge_thread()
        
        for tile_idxN in range(tile_numN):
            tiled_dimN = min(dimN - tile_idxN * default_tiled_dimN, default_tiled_dimN)
            
            for tile_idxM in range(tile_numM):
                thread = program.create_thread()

                tiled_dimM = min(dimM - tile_idxM * default_tiled_dimM, default_tiled_dimM)
                
                tile_rowA_addr = rowA_addr
                tile_rowB_addr = rowB_addr + tile_idxN * default_tiled_dimN * self.device.gcn_core.word_dtype.itemsize
                tile_ptrA_addr = ptrA_addr + tile_idxM * default_tiled_dimM * self.device.gcn_core.ptr_dtype.itemsize
                tile_idxA_addr = idxA_addr
                
                subsession = self.device.gcn_core.sdmm_tile(
                    rowA_addr=tile_rowA_addr, rowB_addr=tile_rowB_addr, ptrA_addr=tile_ptrA_addr, idxA_addr=tile_idxA_addr,
                    tiled_dimM=tiled_dimM, dimN=dimN, dimK=dimK)

                thread.add_session(subsession)
                
                for pe_idx in range(self.device.gcn_core.pe_num):
                    tile_offsetM = tile_idxM * default_tiled_dimM + pe_idx
                    tile_offsetN = tile_idxN * default_tiled_dimN
                    
                    addr = rowC_addr + (tile_offsetM * dimN + tile_offsetN) * self.device.gcn_core.acc_dtype.itemsize
                    numel = tiled_dimN
                    
                    thread.add_session(self.device.gcn_core.flush_single_pe(addr=addr, pe_idx=pe_idx, numel=numel))
                    
                program.merge_thread()
        
        if self.verbose:
            print(f"execution: run_sdmm")
            print(f"  * # of nodes:         {b.shape[0]}")
            print(f"  * # of node features: {b.shape[1]}")
            print(f"  * # of edges:         {edge_index.shape[1]}")
            print(f"  * matrix A sparsity:  {edge_index.shape[1] / (N * N) * 100:.4f}%")
            print(f"  * matrix A size:      {edge_index.shape[1] * (self.device.gcn_core.word_dtype.itemsize + self.device.gcn_core.ptr_dtype.itemsize)}Bytes")
            print(f"  * matrix B size:      {b.size * self.device.gcn_core.word_dtype.itemsize}Bytes")

        st_time = self.device.context.timestamp
        program.run_with_device(device=self.device, progress_bar=not self.device.print_debug_info, progress_bar_header="SDMM")
        ed_time = self.device.context.timestamp
        
        if self.verbose:
            print(f"summary: run_sdmm")
            print(f"  * execution cycle:  {ed_time - st_time}")
        
        if self.validate_output:
            adj_quant, scale_adj_quant = quantize(adj.todense(), dtype=self.device.gcn_core.word_dtype)
            
            acc_dtype = self.device.gcn_core.acc_dtype
            
            reference = np.matmul(adj_quant.astype(acc_dtype), b.astype(acc_dtype))
            simulated = self.device.memory.memory.get_data(addr=rowC_addr, size=dimN*dimM*acc_dtype.itemsize, dtype=acc_dtype).reshape((dimM, dimN))
            
            print(f"  * output validated: {np.array_equal(reference, simulated)}")
        
        self.create_log(action="SDMM", start_time=st_time, end_time=ed_time)
        
    @runtime_method
    def GCNConv(self, module: GCNConv, x: torch.Tensor, edge_index: torch.Tensor):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step #1: Preprocess adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step #2: GEMM with weight and input features (linear combination)
        weight = module.lin.weight.data
        bias = 0 if module.lin.bias is None else module.lin.bias.data
        aggr_x = torch.matmul(x, weight.T) + bias

        # Step #3: SDMM with adjacency matrix and input features (message passing)
        out = module.propagate(edge_index, x=aggr_x, edge_weight=norm)
        out = out + module.bias
        
        # Step #4: Simulate GEMM and SDMM with device
        if self.simulate_gemm:
            self.run_gemm(x, weight.T)
        if self.simulate_sdmm:
            self.run_sdmm(edge_index=edge_index, edge_weight=norm, x=aggr_x)

        return out
    
