import numpy as np

from tds_sim.simulation.module import *
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.spm_cache import SPMCacheModule
from tds_sim.simulation.modeling.cache import CacheReplacePolicy


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
    
    memory = MainMemoryModule(name="Memory", access_latency=40, access_granularity=32)
    cache = SPMCacheModule(name="Cache", nxt_level_memory=memory, capacity=4096, way_num=4, cacheline_size=32, replace_policy=CacheReplacePolicy.LRU, bank_num=4, access_latency=1)
    debugger = DebugModule(name="Debugger")
    
    memory.memory.set_data(addr=0, data=np.arange(32, dtype=np.dtype(np.int32)))
    
    debug_subsessions = [
        cache.spm_load_data(ofc_addr=0,  bank_idx=0, bank_oset=0,  size=32),
        cache.spm_load_data(ofc_addr=32, bank_idx=1, bank_oset=32, size=32),
        cache.spm_load_data(ofc_addr=64, bank_idx=2, bank_oset=0,  size=32),
        cache.spm_load_data(ofc_addr=96, bank_idx=3, bank_oset=32, size=32),
        cache.spm_exchange_buffering_context(),
        cache.spm_read_data(bank_idx=0, bank_oset=0,  size=32),
        cache.spm_read_data(bank_idx=1, bank_oset=32, size=32),
        cache.spm_read_data(bank_idx=2, bank_oset=0,  size=32),
        cache.spm_read_data(bank_idx=3, bank_oset=32, size=32),
        cache.spm_write_data(bank_idx=0, bank_oset=0,  size=32, data=np.ones(8, dtype=np.dtype(np.int32))),
        cache.spm_write_data(bank_idx=1, bank_oset=32, size=32, data=np.ones(8, dtype=np.dtype(np.int32))),
        cache.spm_write_data(bank_idx=2, bank_oset=0,  size=32, data=np.ones(8, dtype=np.dtype(np.int32))),
        cache.spm_write_data(bank_idx=3, bank_oset=32, size=32, data=np.ones(8, dtype=np.dtype(np.int32))),
        cache.spm_exchange_buffering_context(),
        cache.spm_store_data(ofc_addr=0,  bank_idx=0, bank_oset=0,  size=32),
        cache.spm_store_data(ofc_addr=32, bank_idx=1, bank_oset=32, size=32),
        cache.spm_store_data(ofc_addr=64, bank_idx=2, bank_oset=0,  size=32),
        cache.spm_store_data(ofc_addr=96, bank_idx=3, bank_oset=32, size=32),
    ]
    
    context = Context()
    context.register_module(cache, memory, debugger)

    for subsession in debug_subsessions:
        debugger.session.submit_child_session(subsession=subsession, callbacks=[debugger.debug_callback])
        
        while not debugger.is_idle:
            context.increase_timestamp()
            
    print(f"memory dump: {memory.memory.get_data(addr=0, size=128, dtype=np.dtype(np.int32))}")