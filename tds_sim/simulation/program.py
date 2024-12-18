import tqdm

from tds_sim.simulation.module import Session
from tds_sim.simulation.device import Device


class ProgramThread(object):
    def __init__(self):
        self.sequence: list[Session] = []
        self.cursor: int = 0
        
    def initialize(self):
        self.cursor = 0
        
    def add_session(self, session: Session):
        self.sequence.append(session)
        
    def pop_top_session(self) -> Session:
        if self.is_finished:
            return None

        if self.cursor > 0 and (not self.sequence[self.cursor - 1].is_finished):
            return None
        
        top_sequence = self.sequence[self.cursor]
        self.cursor += 1
        
        return top_sequence
    
    @property
    def is_finished(self) -> bool:
        return self.cursor >= len(self.sequence)
    

class ConcurrentThread(ProgramThread):
    def __init__(self):
        super().__init__()
        
    def pop_top_session(self) -> list[Session]:
        self.cursor = len(self.sequence)
        return self.sequence


class Program(object):
    def __init__(self):
        self.thread_cursors: list[ProgramThread|ConcurrentThread] = []
        self.registered_thread: list[list[ProgramThread|ConcurrentThread]] = []
        
    def create_thread(self) -> ProgramThread:
        new_thread = ProgramThread()
        self.thread_cursors.append(new_thread)
        return new_thread
    
    def create_concurrent_thread(self) -> ConcurrentThread:
        new_thread = ConcurrentThread()
        self.thread_cursors.append(new_thread)
        return new_thread
    
    def merge_thread(self):
        self.registered_thread.append(self.thread_cursors)
        self.thread_cursors = []
        
    def run_with_device(self, device: Device, max_clock_cycle: int=-1, progress_bar=True, progress_bar_header="execute"):
        if not device.is_initialized:
            device.initialize()
        
        st_time = device.context.timestamp
        
        for thread_cursors in tqdm.tqdm(self.registered_thread, ncols=100, leave=False, disable=not progress_bar, desc=f"{progress_bar_header} "):
            while True:
                nxt_thread_group_flag = True
                
                for thread in thread_cursors:
                    subsession = thread.pop_top_session() 
                    
                    if isinstance(subsession, Session):
                        device.debugger.session.submit_child_session(subsession=subsession, callbacks=[device.debugger.debug_callback])
                    elif isinstance(subsession, list):
                        for ss in subsession:
                            device.debugger.session.submit_child_session(subsession=ss, callbacks=[device.debugger.debug_callback])
                    
                    if not thread.is_finished:
                        nxt_thread_group_flag = False
                        
                if nxt_thread_group_flag:
                    break
                
                device.context.increase_timestamp()
                
                if device.context.timestamp - st_time > max_clock_cycle and max_clock_cycle >= 0:
                    return
                    
            while not device.debugger.is_idle:
                device.context.increase_timestamp()
                
                if device.context.timestamp - st_time > max_clock_cycle and max_clock_cycle >= 0:
                    return