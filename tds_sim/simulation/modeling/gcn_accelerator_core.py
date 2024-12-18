import math
import numpy as np

from tds_sim.simulation.module import Module, Session, rpc_method, callback_method
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.spm_cache import SPMCacheModule

from tds_sim.common.custom_exception import CustomException
from tds_sim.common.data_tools import cast2anytype, cast2bytearr


__all__ = ['GCNAcceleratorCore']


class GCNAcceleratorCore(Module):
    def __init__(
        self, name, unified_cache: SPMCacheModule, main_memory: MainMemoryModule, 
        pe_num: int, mac_per_pe: int,
        word_dtype: np.dtype, ptr_dtype: np.dtype, acc_dtype: np.dtype,
        verbose: bool=False
    ):
        super().__init__(name)
        
        # connected memory modules
        self.unified_cache = unified_cache  # cache that stores A and B (SDMM: B | GEMM: A and B) -> can be reconfigured to cache and SPM
        self.main_memory   = main_memory    # fetch A, C and store D
        
        # module parameters
        self.pe_num     = pe_num      # number of PEs
        self.mac_per_pe = mac_per_pe  # number of MACs per each PE
        
        self.word_dtype = word_dtype  # datatype of the operand (A, B)
        self.ptr_dtype  = ptr_dtype   # datatype of the pointers (A row pointer and column index)
        self.acc_dtype  = acc_dtype   # datatype of the accumulated result (C, D)
        
        self.verbose = verbose  # verbose option for debugging
        
        # check compatibility
        if self.unified_cache.bank_num != self.pe_num:
            raise CustomException(self, f"the bank number of the unified cache '{self.unified_cache.bank_num}' is not the same with the pe number '{self.pe_num}'")
        
        if self.unified_cache.cacheline_size < self.mac_per_pe * self.word_dtype.itemsize:
            raise CustomException(self, f"the cacheline size of the unified cache '{self.unified_cache.cacheline_size}' is smaller than the tiled rowB size '{self.mac_per_pe * self.word_dtype.itemsize}'")
        
        # registers
        self.acc_registers = np.zeros(shape=(self.pe_num, self.mac_per_pe), dtype=self.acc_dtype)
        self.ptrA_buffer   = np.zeros(shape=(self.pe_num+1,), dtype=self.ptr_dtype)
        self.idxA_buffer   = np.zeros(shape=(self.pe_num,),   dtype=self.ptr_dtype)
        self.rowA_buffer   = np.zeros(shape=(self.pe_num,),   dtype=self.word_dtype)
        self.rowA_cursors  = np.zeros(shape=(self.pe_num,),   dtype=np.dtype(np.int32))
    
    #############################################
    # GEMM Operations
    #############################################
    
    @rpc_method
    def gemm_tile(self, dataA_bank_offset: int, dataB_bank_offset: int, tiled_dimM: int, tiled_dimN: int, tiled_dimK: int):
        self.rowA_cursors[0] = 0
        self.acc_registers[:, :] = 0
        
        self.gemm_tile_read_A_callback(dataA_bank_offset=dataA_bank_offset, dataB_bank_offset=dataB_bank_offset, tiled_dimM=tiled_dimM, tiled_dimN=tiled_dimN, tiled_dimK=tiled_dimK)
        
        return 1
    
    @callback_method
    def gemm_tile_read_A_callback(self, dataA_bank_offset: int, dataB_bank_offset: int, tiled_dimM: int, tiled_dimN: int, tiled_dimK: int):
        if self.rowA_cursors[0] >= tiled_dimK:
            return
        
        bank_offset = dataA_bank_offset + self.rowA_cursors[0] * math.ceil(tiled_dimM / self.unified_cache.bank_num) * self.word_dtype.itemsize
        banked_size = math.ceil(tiled_dimM / self.unified_cache.bank_num) * self.word_dtype.itemsize
        used_bank_num = min(self.unified_cache.bank_num, tiled_dimM)

        subsession = self.unified_cache.spm_read_data(bank_offset=bank_offset, banked_size=banked_size, used_bank_num=used_bank_num)
        self.session.submit_child_session(subsession=subsession, callbacks=[self.gemm_tile_read_B_callback])
        
    @callback_method
    def gemm_tile_read_B_callback(self, dataA_bank_offset: int, dataB_bank_offset: int, tiled_dimM: int, tiled_dimN: int, tiled_dimK: int):
        self.rowA_buffer[:] = cast2anytype(self.child_session.response, self.word_dtype)
        
        bank_offset = dataB_bank_offset + self.rowA_cursors[0] * math.ceil((tiled_dimN / self.unified_cache.bank_num)) * self.word_dtype.itemsize
        banked_size = math.ceil((tiled_dimN / self.unified_cache.bank_num)) * self.word_dtype.itemsize
        used_bank_num = min(self.unified_cache.bank_num, tiled_dimN)

        subsession = self.unified_cache.spm_read_data(bank_offset=bank_offset, banked_size=banked_size, used_bank_num=used_bank_num)
        self.session.submit_child_session(subsession=subsession, callbacks=[self.gemm_tile_compute_callback])
        
    @callback_method
    def gemm_tile_compute_callback(self, dataA_bank_offset: int, dataB_bank_offset: int, tiled_dimM: int, tiled_dimN: int, tiled_dimK: int):
        rowB_elements = cast2anytype(self.child_session.response, self.word_dtype)
        
        a_rep = np.repeat(self.rowA_buffer.reshape((tiled_dimM, 1)), tiled_dimN, axis=1)
        b_rep = np.repeat(rowB_elements.reshape((1, tiled_dimN)),    tiled_dimM, axis=0)
        
        self.acc_registers[0:tiled_dimM, 0:tiled_dimN] = self.acc_registers[0:tiled_dimM, 0:tiled_dimN] + a_rep.astype(self.acc_dtype) * b_rep.astype(self.acc_dtype)
        self.rowA_cursors[0] = self.rowA_cursors[0] + 1
        
        self.gemm_tile_read_A_callback(dataA_bank_offset=dataA_bank_offset, dataB_bank_offset=dataB_bank_offset, tiled_dimM=tiled_dimM, tiled_dimN=tiled_dimN, tiled_dimK=tiled_dimK)
    
    #############################################
    # SDMM Operations: Tiled SDMM
    #############################################
    
    @rpc_method
    def sdmm_tile(self, rowA_addr: int, rowB_addr: int, ptrA_addr: int, idxA_addr: int, tiled_dimM: int, dimN: int, dimK: int):
        size = (self.pe_num + 1) * self.ptr_dtype.itemsize
        subsession = self.main_memory.access_memory(addr=ptrA_addr, size=size, req_type=0, )
        self.session.submit_child_session(subsession=subsession, callbacks=[self.sdmm_tile_ptrA_read_callback])
        
        return 1

    @callback_method
    def sdmm_tile_ptrA_read_callback(self, rowA_addr: int, rowB_addr: int, ptrA_addr: int, idxA_addr: int, tiled_dimM: int, dimN: int, dimK: int):
        self.ptrA_buffer[:] = cast2anytype(self.child_session.response, dtype=self.ptr_dtype)
        self.session.response = self.ptrA_buffer[-1]  # return next rowA offset to continue tiled SDMM operation
        
        self.rowA_cursors[:] = 0
        
        # create single PE computation session for each PE
        for pe_idx in range(min(tiled_dimM, self.pe_num)):
            if self.rowA_cursors[pe_idx] < (self.ptrA_buffer[pe_idx+1] - self.ptrA_buffer[pe_idx]):
                pe_rowA_addr = rowA_addr + self.ptrA_buffer[pe_idx] * self.word_dtype.itemsize
                pe_rowB_addr = rowB_addr
                pe_idxA_addr = idxA_addr + self.ptrA_buffer[pe_idx] * self.ptr_dtype.itemsize

                subsession = self.sdmm_tile_compute_single_pe(pe_idx=pe_idx, pe_rowA_addr=pe_rowA_addr, pe_rowB_addr=pe_rowB_addr, pe_idxA_addr=pe_idxA_addr, dimN=dimN, )
                self.session.submit_child_session(subsession=subsession, callbacks=[])
        
    @rpc_method
    def sdmm_tile_compute_single_pe(self, pe_idx: int, pe_rowA_addr: int, pe_rowB_addr: int, pe_idxA_addr: int, dimN: int):
        addr = pe_idxA_addr + self.rowA_cursors[pe_idx] * self.ptr_dtype.itemsize
        size = self.ptr_dtype.itemsize
        subsession = self.unified_cache.access_memory(addr=addr, size=size, req_type=0, )
        self.session.submit_child_session(subsession=subsession, callbacks=[self.sdmm_tile_compute_single_pe_idxA_read_callback])
        
        return 1
        
    @callback_method
    def sdmm_tile_compute_single_pe_idxA_read_callback(self, pe_idx: int, pe_rowA_addr: int, pe_rowB_addr: int, pe_idxA_addr: int, dimN: int):
        self.idxA_buffer[pe_idx] = cast2anytype(self.child_session.response, dtype=self.ptr_dtype)
        
        addr = pe_rowA_addr + self.rowA_cursors[pe_idx] * self.word_dtype.itemsize
        size = self.word_dtype.itemsize
        subsession = self.unified_cache.access_memory(addr=addr, size=size, req_type=0, )
        
        self.session.submit_child_session(subsession=subsession, callbacks=[self.sdmm_tile_compute_single_pe_rowA_read_callback])
        
    @callback_method
    def sdmm_tile_compute_single_pe_rowA_read_callback(self, pe_idx: int, pe_rowA_addr: int, pe_rowB_addr: int, pe_idxA_addr: int, dimN: int):
        self.rowA_buffer[pe_idx] = cast2anytype(self.child_session.response, dtype=self.word_dtype)
        
        addr = pe_rowB_addr + self.idxA_buffer[pe_idx] * dimN * self.word_dtype.itemsize
        size = min(self.mac_per_pe, dimN) * self.word_dtype.itemsize
        subsession = self.unified_cache.access_memory(addr=addr, size=size, req_type=0, )

        self.session.submit_child_session(subsession=subsession, callbacks=[self.sdmm_tile_compute_single_pe_rowB_read_callback])
        
    @callback_method
    def sdmm_tile_compute_single_pe_rowB_read_callback(self, pe_idx: int, pe_rowA_addr: int, pe_rowB_addr: int, pe_idxA_addr: int, dimN: int):
        rowB_elements = cast2anytype(self.child_session.response, dtype=self.word_dtype)
        rowB_numel = min(self.mac_per_pe, dimN)
        self.acc_registers[pe_idx][:rowB_numel] = self.acc_registers[pe_idx][:rowB_numel] + rowB_elements.astype(self.acc_dtype) * self.rowA_buffer[pe_idx].astype(self.acc_dtype)
        
        self.rowA_cursors[pe_idx] = self.rowA_cursors[pe_idx] + 1
        
        if self.rowA_cursors[pe_idx] < (self.ptrA_buffer[pe_idx+1] - self.ptrA_buffer[pe_idx]):
            subsession = self.sdmm_tile_compute_single_pe(pe_idx=pe_idx, pe_rowA_addr=pe_rowA_addr, pe_rowB_addr=pe_rowB_addr, pe_idxA_addr=pe_idxA_addr, dimN=dimN, )
            self.session.submit_child_session(subsession=subsession, callbacks=[])
    
    #############################################
    # SDMM Operations: Preload Accumulator
    #############################################
    
    @rpc_method
    def preload_single_pe(self, addr: int, pe_idx: int, numel: int):
        if numel > self.mac_per_pe:
            raise CustomException(self, f"numel '{numel}' cannot exceed the number of MAC units per PE '{self.mac_per_pe}'")

        size = numel * self.acc_dtype.itemsize
        subsession = self.main_memory.access_memory(addr=addr, size=size, req_type=0, )
        self.session.submit_child_session(subsession=subsession, callbacks=[self.preload_single_pe_callback])
        
        return 1
    
    @callback_method
    def preload_single_pe_callback(self, addr: int, pe_idx: int, numel: int):
        self.acc_registers[pe_idx][0:numel] = cast2anytype(self.child_session.response, dtype=self.acc_dtype)
    
    #############################################
    # SDMM Operations: Flush Accumulator
    #############################################
    
    @rpc_method
    def flush_single_pe(self, addr: int, pe_idx: int, numel: int):
        if numel > self.mac_per_pe:
            raise CustomException(self, f"numel '{numel}' cannot exceed the number of MAC units per PE '{self.mac_per_pe}'")
        
        size = numel * self.acc_dtype.itemsize
        data = cast2bytearr(self.acc_registers[pe_idx][:numel])
        subsession = self.main_memory.access_memory(addr=addr, size=size, req_type=1, data=data, )
        self.session.submit_child_session(subsession=subsession, callbacks=[self.flush_single_pe_callback])
        
        return 1
    
    @callback_method
    def flush_single_pe_callback(self, addr: int, pe_idx: int, numel: int):
        self.acc_registers[pe_idx] = np.zeros(shape=(self.mac_per_pe), dtype=self.acc_dtype)
    
    #############################################
    # Common methods
    #############################################
    
    def grant_received_sessions(self):
        for session_type, session_queue in self.received_sessions.items():
            if len(session_queue) and len(self.session.child_session_dir.keys()) == 0:
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
