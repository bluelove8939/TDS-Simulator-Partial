import time
import sys
import tqdm

from tds_sim.simulation.module import *


class DebugModule(Module):
    def __init__(self, name):
        super().__init__(name)
        
    @callback_method
    def debug_callback(self, child_session: Session):
        print(f"{type(self).__name__}: callback called (child session response: {child_session.response})")
        print(f"  * current session: {self.session}")
        print(f"  * child session:   {child_session}")
    
    @property
    def is_idle(self) -> bool:
        return len(self.session.child_session_dir.keys()) == 0


class ModuleA(Module):
    def __init__(self, name: str):
        super().__init__(name=name)
        
        self.target_module: Module = None
        
    @rpc_method
    def rpc_method(self):
        print(f"{type(self).__name__}: rpc method called with context: {self.session}")
        subsession = self.target_module.rpc_method()
        self.session.submit_child_session(subsession=subsession, callbacks=[a.rpc_method_callback])
        
        return 1
    
    @callback_method
    def rpc_method_callback(self, child_session: Session):
        print(f"{type(self).__name__}: callback called (child session response: {child_session.response})")
        print(f"  * current session: {self.session}")
        print(f"  * child session:   {child_session}")
        
        self.session.response = child_session.response + 10
        
        return
        
    def grant_received_sessions(self):
        for session_type in self.received_sessions.keys():
            while len(self.received_sessions[session_type]):
                session = self.received_sessions[session_type].pop(-1)
                self.context.suspend_session(session=session)
                
class ModuleB(Module):
    def __init__(self, name: str):
        super().__init__(name=name)
        
        self.target_module: Module = None
    
    @rpc_method
    def rpc_method(self) -> int:
        print(f"{type(self).__name__}: rpc method called with context: {self.session}")
        subsession = self.target_module.rpc_method()
        self.session.submit_child_session(subsession=subsession, callbacks=[b.rpc_method_callback])
        
        return 1
    
    @callback_method
    def rpc_method_callback(self, child_session: 'Session'):
        print(f"{type(self).__name__}: callback called (child session response: {child_session.response})")
        print(f"  * current session: {self.session}")
        print(f"  * child session:   {child_session}")
        
        self.session.response = child_session.response * 2  # current channel response is same with the (child * 2)
        
        return
        
    def grant_received_sessions(self):
        for session_type in self.received_sessions.keys():
            while len(self.received_sessions[session_type]):
                session = self.received_sessions[session_type].pop(-1)
                self.context.suspend_session(session=session)
                
class ModuleC(Module):
    def __init__(self, name: str):
        super().__init__(name=name)
        
        self.target_module: Module = None
    
    @rpc_method
    def rpc_method(self) -> int:
        print(f"{type(self).__name__}: rpc method called with context: {self.session}")
        self.session.response = 100
        return 1
        
    def grant_received_sessions(self):
        for session_type in self.received_sessions.keys():
            while len(self.received_sessions[session_type]):
                session = self.received_sessions[session_type].pop(-1)
                self.context.suspend_session(session=session)


if __name__ == "__main__":
    a = ModuleA(name="A")
    b = ModuleB(name="B")
    c = ModuleC(name="C")
    debugger = DebugModule(name="Debugger")
    a.target_module = b
    b.target_module = c
    
    context = Context()
    context.register_module(a, b, c, debugger)
    
    subsession = a.rpc_method()
    debugger.session.submit_child_session(subsession=subsession, callbacks=[debugger.debug_callback])

    for _ in range(4):
        print(f"\n=== TIMESTAMP: {context.timestamp} ===")
        context.increase_timestamp()