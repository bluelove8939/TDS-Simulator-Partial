import enum
import math
import random
import numpy as np
from typing import List

from tds_sim.simulation.module import Module, Session, rpc_method, callback_method
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.cache import CacheModule, TagEntry, CacheReplacePolicy
from tds_sim.common.custom_exception import CustomException
from tds_sim.common.data_tools import cast2bytearr


__all__ = ['TagEntry', 'CacheReplacePolicy', 'SPMCacheModule']


class SPMCacheModule(CacheModule):
    def __init__(self, name, nxt_level_memory, capacity, way_num, cacheline_size, replace_policy, bank_num = 1, access_latency = 1):
        super().__init__(name, nxt_level_memory, capacity, way_num, cacheline_size, replace_policy, bank_num, access_latency)
        
        self.spm_bank_capacity = self.capacity // self.bank_num // 2  # double buffering
        
        self.spm_layout = self.data_entries.reshape((self.bank_num, 2, self.spm_bank_capacity))
        self.spm_ld_st_layout = self.spm_layout[:, 0, :].reshape((self.bank_num, self.spm_bank_capacity))  # load-store layout
        self.spm_rd_wr_layout = self.spm_layout[:, 1, :].reshape((self.bank_num, self.spm_bank_capacity))  # read-write layout
        
        self.spm_store_status  = False
        self.spm_load_status = False
        
    ##################################
    # SPM Load Data Methods
    ##################################

    @rpc_method
    def spm_load_data(self, ofc_addr: int, size: int, bank_offset: int, used_bank_num: int, start_bank_idx: int=0):
        if size % used_bank_num != 0:
            raise CustomException(self, f"the data load size '{size}' should be the multiple of the bank number '{self.bank_num}'")
        
        if used_bank_num > self.bank_num:
            raise CustomException(self, f"the used_bank_num '{used_bank_num}' cannot exceed the bank number '{self.bank_num}'")
        
        subsession = self.nxt_level_memory.access_memory(addr=ofc_addr, size=size, req_type=0)
        self.session.submit_child_session(subsession=subsession, callbacks=[self.spm_load_data_callback])
        
        self.spm_load_status = True
        
        return 1
    
    @callback_method
    def spm_load_data_callback(self, ofc_addr: int, size: int, bank_offset: int, used_bank_num: int, start_bank_idx: int=0):
        bank_occupied_size = size // used_bank_num
        
        data = self.child_session.response
        self.spm_ld_st_layout[start_bank_idx:start_bank_idx+used_bank_num, bank_offset:bank_offset+bank_occupied_size] = data.reshape((bank_occupied_size, used_bank_num)).T
        
        self.spm_load_status = False
        
    ##################################
    # SPM Store Data Methods
    ##################################
    
    @rpc_method
    def spm_store_data(self, ofc_addr: int, size: int, bank_offset: int, used_bank_num: int, start_bank_idx: int=0):
        if size % used_bank_num != 0:
            raise CustomException(self, f"the data load size '{size}' should be the multiple of the bank number '{self.bank_num}'")
        
        if used_bank_num > self.bank_num:
            raise CustomException(self, f"the used_bank_num '{used_bank_num}' cannot exceed the bank number '{self.bank_num}'")
        
        bank_occupied_size = size // used_bank_num
        data = np.copy(self.spm_ld_st_layout[start_bank_idx:start_bank_idx+used_bank_num, bank_offset:bank_offset+bank_occupied_size].T.flatten())
        
        subsession = self.nxt_level_memory.access_memory(addr=ofc_addr, size=size, req_type=1, data=data)
        self.session.submit_child_session(subsession=subsession, callbacks=[])
        
        self.spm_store_status = True
        
        return 1
    
    @callback_method
    def spm_store_data_callback(self, ofc_addr: int, size: int, bank_offset: int, used_bank_num: int, start_bank_idx: int=0):
        self.spm_store_status = False
        
    ##################################
    # SPM Read/Write Data Methods
    ##################################
        
    @rpc_method
    def spm_read_data(self, bank_offset: int, banked_size: int, used_bank_num: int, start_bank_idx: int=0):
        self.session.response = np.copy(self.spm_rd_wr_layout[start_bank_idx:start_bank_idx+used_bank_num, bank_offset:bank_offset+banked_size].T.flatten())
        
        return 1
    
    @rpc_method
    def spm_write_data(self, bank_offset: int, banked_size: int, data: np.ndarray, used_bank_num: int, start_bank_idx: int=0):
        self.spm_rd_wr_layout[start_bank_idx:start_bank_idx+used_bank_num, bank_offset:bank_offset+banked_size] = cast2bytearr(data.flatten()).reshape((banked_size, used_bank_num)).T
        
        return 1
    
    ##################################
    # Switch Double Buffering
    ##################################
    
    @rpc_method
    def spm_exchange_buffering_context(self):
        tmp = self.spm_ld_st_layout
        self.spm_ld_st_layout = self.spm_rd_wr_layout
        self.spm_rd_wr_layout = tmp
        
        return 1
    
    ##################################
    # Common Methods
    ##################################
    
    def grant_received_sessions(self):
        for session_type, session_queue in self.received_sessions.items():
            if session_type == "spm_load_data" and len(session_queue) and not self.spm_load_status:
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
                
            elif session_type == "spm_store_data" and len(session_queue) and not self.spm_store_status:
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
                
            elif session_type == "spm_read_data" and len(session_queue):
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
                
            elif session_type == "spm_write_data" and len(session_queue):
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
                
            elif session_type == "spm_exchange_buffering_context" and len(session_queue):
                while len(session_queue):
                    session = session_queue.pop(0)
                    self.context.suspend_session(session=session)
        
        return super().grant_received_sessions()
