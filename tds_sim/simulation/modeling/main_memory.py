import math
import numpy as np

from tds_sim.simulation.module import Module, rpc_method

from tds_sim.simulation.tools.data_buffer import OffchipMainMemory
from tds_sim.common.data_tools import cast2bytearr


__all__ = ['MainMemoryModule']


class MainMemoryModule(Module):
    def __init__(self, name: str, access_latency: int, access_granularity: int):
        super().__init__(name=name)
        
        self.memory = OffchipMainMemory()
        
        self.access_latency     = access_latency  # access latency
        self.access_granularity = access_granularity
    
    @rpc_method
    def access_memory(self, addr: int, size: int, req_type: int, data: np.ndarray=None):
        if req_type == 0:
            self.session.response = self.memory.get_data(addr=addr, size=size, dtype=np.uint8)[:size]
        else:
            self.memory.set_data(addr=addr, data=cast2bytearr(data)[0:size])
            self.session.response = True
             
        return self.access_latency + math.ceil(size / self.access_granularity)  # TODO: currently, it assumes that the memory request is fully overlapped
    
    def grant_received_sessions(self):
        for session_type, session_queue in self.received_sessions.items():
            if session_type == "access_memory" and len(session_queue):  # TODO: currently, it assumes that the memory request is fully overlapped
                session = session_queue.pop(0)
                self.context.suspend_session(session=session)
