import sys
import tqdm
from typing import Any

from tds_sim.common.custom_exception import CustomException
from tds_sim.simulation.module import Module, Context, Session, callback_method


__all__ = ['DebugModule', 'Device']
        

class DebugModule(Module):
    def __init__(self, name, verbose: bool):
        super().__init__(name)
        
        self.verbose = verbose
        
    @callback_method
    def debug_callback(self):
        if self.verbose:
            sys.stdout.write(f"[DEBUG] session: {self.child_session}  timestamp: {self.context.timestamp:<3d}  response: {self.child_session.response}\n")
    
    @property
    def is_idle(self) -> bool:
        return len(self.session.child_session_dir.keys()) == 0


class Device(object):
    def __init__(self, print_debug_info: bool=False):
        self.print_debug_info = print_debug_info
        
        self.context  = Context()
        self.debugger = DebugModule(name="Debugger", verbose=print_debug_info)
        
        self._is_initialized = False
        
    def _register_components(self, component: Any):
        if isinstance(component, Module):
            self.context.register_module(component)
        elif isinstance(component, (list, tuple)):
            for elem in component:
                self._register_components(component=elem)
        
    def initialize(self) -> 'Device':
        for name, component in self.__dict__.items():
            self._register_components(component=component)
            
        self._is_initialized = True
                
        return self
    
    # def run_program(self, program: Program, max_clock_cycle: int=-1):
    #     if not self.is_initialized:
    #         raise CustomException(self, f"cannot run the program since the device is not initialized, call 'initialize()' before executing the program")
        
    #     st_time = self.context.timestamp
        
    #     for session_list in tqdm.tqdm(program.session_sequence, ncols=100, leave=False, disable=self.print_debug_info):
    #         for subsession in session_list:
    #             self.debugger.session.submit_child_session(subsession=subsession, callbacks=[self.debugger.debug_callback])
                
    #         while not self.debugger.is_idle:
    #             self.context.increase_timestamp()
                
    #             if self.context.timestamp - st_time > max_clock_cycle and max_clock_cycle >= 0:
    #                 return
                
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized