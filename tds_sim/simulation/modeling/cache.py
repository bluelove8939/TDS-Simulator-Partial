import enum
import math
import random
import numpy as np
from typing import List

from tds_sim.simulation.module import Module, Session, rpc_method, callback_method
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.common.custom_exception import CustomException
from tds_sim.common.data_tools import cast2bytearr


__all__ = ['CacheModule', 'TagEntry', 'CacheReplacePolicy', 'CacheModule']


class CacheReplacePolicy(enum.Enum):
    LRU    = enum.auto()
    RANDOM = enum.auto()
    
    
class TagEntry(object):
    def __init__(self, lru_cnt: int) -> None:
        self.valid_bit  = 0
        self.dirty_bit  = 0
        self.tag_bits   = 0
        self.lru_cnt    = lru_cnt
        
        
class CacheModule(Module):
    def __init__(self, name: str, nxt_level_memory, capacity: int, way_num: int, cacheline_size: int, replace_policy: CacheReplacePolicy, bank_num: int=1, access_latency: int=1):
        super().__init__(name=name)
        
        self.capacity       = capacity
        self.way_num        = way_num
        self.set_num        = capacity // way_num // cacheline_size
        self.cacheline_size = cacheline_size
        self.replace_policy = replace_policy
        self.bank_num       = bank_num
        self.set_per_bank   = math.ceil(self.set_num / self.bank_num)
        self.access_latency = access_latency
        
        if self.bank_num > self.set_num:
            raise CustomException(self, f"the set number '{self.set_num}' should be larger than or equals to the bank number '{self.bank_num}'")
        
        self.tag_entries:  List[List[List[TagEntry]]] = [[[TagEntry(lru_cnt=way_idx) for way_idx in range(self.way_num)] for _ in range(self.set_per_bank)] for _ in range(self.bank_num)]
        self.data_entries: np.ndarray                 = np.zeros(shape=(self.bank_num, self.set_per_bank, self.way_num, self.cacheline_size), dtype=np.uint8)
        
        self.mshr_container: dict[int, Session]  = {}
        
        self.nxt_level_memory: CacheModule | MainMemoryModule = nxt_level_memory
    
    def parsing_set_idx(self, set_idx: int) -> tuple[int, int]:
        bank_idx  = set_idx % self.bank_num
        bank_oset = set_idx // self.bank_num
        return bank_idx, bank_oset
    
    def generate_set_idx(self, bank_idx: int, bank_oset: int) -> int:
        return bank_oset * self.bank_num + bank_idx
        
    def generate_address(self, tag: int, set_idx: int) -> int:
        return (tag << math.floor(math.log2(self.set_num * self.cacheline_size))) + (set_idx << math.floor(math.log2(self.cacheline_size)))
    
    def parsing_address(self, addr: int) -> tuple[int, int, int]:
        block_filtered_addr = addr >> math.floor(math.log2(self.cacheline_size))
        
        block_offset = addr % self.cacheline_size
        set_idx = block_filtered_addr % self.set_num
        tag     = block_filtered_addr >> math.floor(math.log2(self.set_num))
        
        return tag, set_idx, block_offset
        
    def update_lru_cnt(self, bank_idx: int, bank_oset: int, target_way_idx: int):
        threshold = self.tag_entries[bank_idx][bank_oset][target_way_idx].lru_cnt
        
        for way_idx in range(self.way_num):
            if way_idx == target_way_idx:
                self.tag_entries[bank_idx][bank_oset][way_idx].lru_cnt = 0
            elif self.tag_entries[bank_idx][bank_oset][way_idx].lru_cnt < threshold:
                self.tag_entries[bank_idx][bank_oset][way_idx].lru_cnt += 1
                
    def push_tag_entry(self, way_idx: int, bank_idx: int, bank_oset: int, tag: int, is_write: int):
        self.tag_entries[bank_idx][bank_oset][way_idx].valid_bit  = 1
        self.tag_entries[bank_idx][bank_oset][way_idx].dirty_bit  = is_write
        self.tag_entries[bank_idx][bank_oset][way_idx].tag_bits   = tag
        
        if self.replace_policy == CacheReplacePolicy.LRU:
            self.update_lru_cnt(bank_idx=bank_idx, bank_oset=bank_oset, target_way_idx=way_idx)
        
    def push_data_entry(self, way_idx: int, bank_idx: int, bank_oset: int, block_oset: int, data: np.ndarray):
        st = block_oset
        ed = block_oset + data.size
        self.data_entries[bank_idx][bank_oset][way_idx][st:ed] = data
        
    def read_data_entry(self, way_idx: int, bank_idx: int, bank_oset: int, block_oset: int, size: int) -> np.ndarray:
        st = block_oset
        ed = block_oset + size
        return np.copy(self.data_entries[bank_idx][bank_oset][way_idx][st:ed])

    def tag_matching(self, addr: int) -> tuple[bool, int]:
        tag, set_idx, _ = self.parsing_address(addr=addr)
        bank_idx, bank_oset = self.parsing_set_idx(set_idx=set_idx)
        
        for way_idx in range(self.way_num):
            tag_entry  = self.tag_entries[bank_idx][bank_oset][way_idx]
            
            if tag == tag_entry.tag_bits and tag_entry.valid_bit:
                return True, way_idx

        return False, 0
    
    def get_victim(self, bank_idx: int, bank_oset: int) -> tuple[int, int, int]:
        victim_tag = -1
        victim_way_idx = -1
        victim_is_dirty = 0
        
        if self.replace_policy == CacheReplacePolicy.LRU:
            for way_idx in range(self.way_num):
                if not self.tag_entries[bank_idx][bank_oset][way_idx].valid_bit:
                    victim_way_idx = way_idx
                    break
                
            target_lru_level = self.way_num-1
            
            while victim_way_idx == -1 and target_lru_level >= 0:
                for way_idx in range(self.way_num):
                    if self.tag_entries[bank_idx][bank_oset][way_idx].lru_cnt == target_lru_level:
                        victim_way_idx = way_idx
                        break
                
                target_lru_level -= 1
            
            # if victim_way_idx == -1:
            #     for way_idx in range(self.way_num):
            #         if self.tag_entries[bank_idx][bank_oset][way_idx].lru_cnt == (self.way_num-1):
            #             victim_way_idx = way_idx
            #             break
            
            victim_tag = self.tag_entries[bank_idx][bank_oset][victim_way_idx].tag_bits
            victim_is_dirty = self.tag_entries[bank_idx][bank_oset][victim_way_idx].dirty_bit
                    
        elif self.replace_policy == CacheReplacePolicy.RANDOM:
            victim_way_idx = random.randint(0, self.way_num-1)
            
            victim_tag = self.tag_entries[bank_idx][bank_oset][victim_way_idx].tag_bits
            victim_is_dirty = self.tag_entries[bank_idx][bank_oset][victim_way_idx].dirty_bit
        else:
            raise Exception(f"[ERROR] Unknown replacement policy: {self.replace_policy}")
        
        return victim_is_dirty, victim_tag, victim_way_idx
    
    @rpc_method
    def cache_flush(self):
        self.tag_entries = [[[TagEntry(lru_cnt=way_idx) for way_idx in range(self.way_num)] for _ in range(self.set_per_bank)] for _ in range(self.bank_num)]
        self.mshr_container: dict[int, tuple[Session, int, int, int]]  = {}
        
        return 1
    
    @rpc_method
    def access_memory(self, addr: int, size: int, req_type: int, data: np.ndarray=None):
        self.access_memory_callback(addr=addr, size=size, req_type=req_type, data=data)
                
        return self.access_latency
    
    @callback_method
    def access_memory_callback(self, addr: int, size: int, req_type: int, data: np.ndarray=None):
        if addr % size != 0:
            raise CustomException(self, f"cache request with the address '{addr}' and size '{size}' is invalid since the lsb {math.ceil(math.log2(size))}bit of address is not zero")
        
        if data is not None:
            data = cast2bytearr(data)[:size]
        
        tag, set_idx, block_oset = self.parsing_address(addr=addr)
        bank_idx, bank_oset = self.parsing_set_idx(set_idx=set_idx)
        is_hit, way_idx = self.tag_matching(addr=addr)
                    
        if is_hit == 1:
            if req_type == 1:
                self.push_tag_entry(way_idx=way_idx, bank_idx=bank_idx, bank_oset=bank_oset, tag=tag, is_write=1)
                self.push_data_entry(way_idx=way_idx, bank_idx=bank_idx, bank_oset=bank_oset, block_oset=block_oset, data=data)
                self.session.response = True
            else:
                self.update_lru_cnt(bank_idx=bank_idx, bank_oset=bank_oset, target_way_idx=way_idx)
                read_data = self.read_data_entry(way_idx=way_idx, bank_idx=bank_idx, bank_oset=bank_oset, block_oset=block_oset, size=size)
                self.session.response = read_data
        else:
            target_addr = self.generate_address(tag=tag, set_idx=set_idx)
            
            if target_addr not in self.mshr_container.keys():
                mshr_session = self.mshr_request(target_addr=target_addr)
                self.mshr_container[target_addr] = mshr_session
                register_to_target_module = True
            else:
                mshr_session = self.mshr_container[target_addr]
                register_to_target_module = False
                
            self.session.submit_child_session(
                subsession=mshr_session, callbacks=[self.access_memory_callback],  # execute 'access_memory' again after the 'mshr_session'
                register_to_target_module=register_to_target_module)      # if mshr entry is newly created, register the 'mshr_session' to the target module
                
    @rpc_method
    def mshr_request(self, target_addr: int):
        subsession = self.nxt_level_memory.access_memory(addr=target_addr, size=self.cacheline_size, req_type=0)
        self.session.submit_child_session(subsession=subsession, callbacks=[self.mshr_nl_wait_callback])
            
        return 1
        
    @callback_method
    def mshr_nl_wait_callback(self, target_addr: int):
        tag, set_idx, block_oset = self.parsing_address(addr=target_addr)
        bank_idx, bank_oset = self.parsing_set_idx(set_idx=set_idx)
        
        victim_is_dirty, victim_tag, victim_way_idx = self.get_victim(bank_idx=bank_idx, bank_oset=bank_oset)
        
        if victim_is_dirty:
            victim_addr = self.generate_address(tag=victim_tag, set_idx=set_idx)
            victim_data = self.read_data_entry(way_idx=victim_way_idx, bank_idx=bank_idx, bank_oset=bank_oset, block_oset=0, size=self.cacheline_size)
            
            subsession = self.nxt_level_memory.access_memory(addr=victim_addr, size=self.cacheline_size, req_type=1, data=victim_data)
            self.session.submit_child_session(subsession=subsession, callbacks=[])
        
        self.push_tag_entry(way_idx=victim_way_idx, bank_idx=bank_idx, bank_oset=bank_oset, tag=tag, is_write=0)  # update tag entry first
        self.push_data_entry(way_idx=victim_way_idx, bank_idx=bank_idx, bank_oset=bank_oset, block_oset=block_oset, data=self.child_session.response)
        self.mshr_container.pop(target_addr)
    
    def grant_received_sessions(self):
        for session_type, session_queue in self.received_sessions.items():
            if session_type == "access_memory" and len(session_queue):
                selected_bank = []
                skipped_sessions = []
                
                while len(session_queue):
                    session = session_queue.pop(0)
                    
                    addr = session.kwargs['addr']
                    tag, set_idx, block_oset = self.parsing_address(addr=addr)
                    bank_idx, bank_oset = self.parsing_set_idx(set_idx=set_idx)
                    
                    if bank_idx not in selected_bank:
                        self.context.suspend_session(session=session)
                        selected_bank.append(bank_idx)
                    else:
                        skipped_sessions.append(session)
                
                self.received_sessions["access_memory"] = skipped_sessions
            
            elif session_type == "mshr_request" and len(session_queue):
                while len(session_queue):
                    session = session_queue.pop(0)
                    self.context.suspend_session(session=session)
                    
            else:
                while len(session_queue):
                    session = session_queue.pop(0)
                    self.context.suspend_session(session=session)
