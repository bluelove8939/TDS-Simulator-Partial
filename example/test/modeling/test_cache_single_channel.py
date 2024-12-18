import numpy as np

from tds_sim.simulation.module import *
from tds_sim.simulation.modeling.main_memory import MainMemoryModule
from tds_sim.simulation.modeling.cache import CacheModule, CacheReplacePolicy


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
    memory = MainMemoryModule(name="Memory", access_latency=40, access_granularity=32)
    cache = CacheModule(name="Cache", nxt_level_memory=memory, capacity=4096, way_num=4, cacheline_size=32, replace_policy=CacheReplacePolicy.LRU, bank_num=1, access_latency=1)
    debugger = DebugModule(name="Debugger")
    
    debug_subsessions = [
        cache.access_memory(addr=0,  size=16, req_type=0, data=None, ),
        cache.access_memory(addr=16, size=16, req_type=0, data=None, ),
        cache.access_memory(addr=32, size=16, req_type=0, data=None, ),
        cache.access_memory(addr=0,  size=16, req_type=1, data=np.array([1, 2, 3, 4], dtype=np.dtype(np.int32)), ),
        cache.access_memory(addr=16, size=16, req_type=1, data=np.array([4, 5, 6, 7], dtype=np.dtype(np.int32)), ),
        cache.access_memory(addr=32, size=16, req_type=1, data=np.array([2, 4, 6, 8], dtype=np.dtype(np.int32)), ),
        cache.access_memory(addr=0,  size=16, req_type=0, data=None, ),
        cache.access_memory(addr=16, size=16, req_type=0, data=None, ),
        cache.access_memory(addr=32, size=16, req_type=0, data=None, ),
    ]
    
    context = Context()
    context.register_module(cache, memory, debugger)

    for subsession in debug_subsessions:
        debugger.session.submit_child_session(subsession=subsession, callbacks=[debugger.debug_callback])
        
        while not debugger.is_idle:
            context.increase_timestamp()