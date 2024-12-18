from typing import Callable, TypeVar, Generic, Any

T = TypeVar('T')


__all__ = ['Module', 'Session', 'Context', 'rpc_method', 'callback_method']


def rpc_method(func: Callable):
    def __rpc_method_wrapper(self: 'Module', create_session: bool=True, **kwargs):
        if not isinstance(self, Module):
            raise Exception(f"[ERROR] rpc_method: the rpc_method '{func.__name__}' should be the method of 'Module' but the first positional argument is '{type(self).__name__}'")
            
        if create_session:
            return Session(module=self, func=func, kwargs=kwargs)
        
        cycles = func(self, **kwargs)
        
        return cycles
    return __rpc_method_wrapper


def callback_method(func: Callable):
    def __callback_method_wrapper(self: 'Module', **kwargs):
        if not isinstance(self, Module):
            raise Exception(f"[ERROR] callback_method: the callback_method '{func.__name__}' should be the method of 'Module' but the first positional argument is '{type(self).__name__}'")
        
        return func(self, **kwargs)
    return __callback_method_wrapper


class Module:
    def __init__(self, name: str):
        self.name = name
        
        self.context: Context = None
        self.session: Session = Session.create_main_session(module=self)
        self.child_session: Session = Session.create_main_session(module=self)
        self.received_sessions: dict[str, list[Session]] = {}
    
    def grant_received_sessions(self):
        pass  # implement this method: executes received session (select from the 'self._received_sessions')
    
    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"
    
    
class Session:
    def __init__(self, module: Module, func: Callable, kwargs: dict[str, Any]={}):
        self.module = module
        self.func   = func
        self.kwargs = kwargs
        
        self.parent_session_list: list[Session] = []
        self.child_session_dir: dict[Session, list[Callable]] = {}
        
        self.response: Any = None
        
        self._release_time: int = None
        self._finish_time:  int = None
        
    @property
    def is_finished(self) -> bool:
        return self._finish_time is not None and len(self.child_session_dir.keys()) == 0
    
    @property
    def execution_cycles(self) -> int:
        if self.is_finished:
            return self._finish_time - self._release_time
        return 0
        
    def submit_child_session(self, subsession: 'Session', callbacks: list[Callable]=[], register_to_target_module: bool=True):
        # register the subsession to the current session
        self.child_session_dir[subsession] = callbacks
        subsession.parent_session_list.append(self)  # session is fundamentally a doubly linked list
        
        # register subsession to the target module
        if register_to_target_module:
            target_session_dir = subsession.module.received_sessions
            
            if subsession.session_type not in target_session_dir.keys():
                target_session_dir[subsession.session_type] = []
                
            target_session_dir[subsession.session_type].append(subsession)
            
        subsession._release_time = self.module.context.timestamp
        
    @classmethod
    def create_main_session(cls, module: Module) -> 'Session':
        return cls(module=module, func=None)

    @property
    def session_type(self) -> str:
        if self.func is None:
            return "main_thread"
        return self.func.__name__
    
    def run_callbacks(self):
        self._finish_time = self.module.context.timestamp
        
        for parent_session in self.parent_session_list:
            session_history = parent_session.module.session
            child_session_history = parent_session.module.child_session
            
            parent_session.module.session = parent_session
            parent_session.module.child_session = self
            
            callback_list = parent_session.child_session_dir[self]
            
            for callback in callback_list:
                callback(**parent_session.kwargs)  # execute all the callbacks related to the session
                
            parent_session.module.session = session_history
            parent_session.module.child_session = child_session_history
                
            parent_session.child_session_dir.pop(self)  # remove the session from the parent's child session directory
            
            if len(parent_session.child_session_dir.keys()) == 0:  # if there aren't any child sessions belongs to the parent session ...
                parent_session.run_callbacks()  # call 'run_callbacks' of the parent session
    
    def _execute(self):
        # change context of the Module
        session_history = self.module.session
        self.module.session = self
        
        # execute this session
        cycles = self.func(self.module, **self.kwargs)
        
        if not isinstance(cycles, int):
            raise Exception(f"[ERROR] Session: the rpc_method '{self.session_type}' should return execution cycles (integer), not {type(cycles).__name__}")
        if cycles <= 0:
            raise Exception(f"[ERROR] Session: the execution cycle of the rpc_method '{self.session_type}' should be larger than 0 but the cycles is '{cycles}'")
        
        # restore the context of the Module
        self.module.session = session_history
        
        # run callback methods if the session is terminated
        if len(self.child_session_dir.keys()) == 0:  # if there aren't any child sessions -> it implies that the current session is finally over without calling other RPC methods
            # instead of executing the callbacks directly, suspend the 'run_callbacks' method to the 'context' of the target module
            self.module.context.suspend_session_callbacks(self, cycles=cycles)
    
    def __str__(self) -> str:
        # return f"Session(id={id(self)}, module={self.module}, session_type={self.session_type})"
        return f"Session(module={self.module.name}, session_type={self.session_type}, kwargs=(" + ', '.join(map(lambda x: f'{x[0]}={x[1]}', self.kwargs.items())) + "))"
    
    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)

    def __ne__(self, other):
        return not(self == other)
    
    
class Context:
    def __init__(self):
        self._timestamp: int = 0
        self._modules: list[Module] = []
        
        # Queue that stores suspended sessions
        #   * The session is not executed directly by the Module (or the 'suspend_sessions' method)
        #   * Instead, the 'suspend_sessions' method suspends the session to the context
        #   * For each cycle, the context executes the sessions until the queue is empty.
        self._suspended_sessions: list[Session] = []
        
        # Directory that stores (release time, method list)
        #   * This directory stores session callback ('run_callbacks' method) that should be executed for each timestamp
        #   * Each session suspends 'run_callback' method with respect to the execution cycle of the RPC message
        #   * 'run_callbacks' method does not receive any arguments (as well as keyword arguments). 
        self._suspended_session_callbacks: dict[int, list[Session]] = {}
        
    def register_module(self, *modules: 'Module'):
        self._modules += modules
        
        for module in modules:
            module.context = self
            
    def suspend_session(self, session: Session):
        self._suspended_sessions.append(session)
        
    def suspend_session_callbacks(self, run_callback_method: Callable, cycles: int):
        target_timestamp = self.timestamp + cycles
        
        if target_timestamp not in self._suspended_session_callbacks.keys():
            self._suspended_session_callbacks[target_timestamp] = []
            
        self._suspended_session_callbacks[target_timestamp].append(run_callback_method)
        
    @property
    def timestamp(self) -> int:
        return self._timestamp
    
    def increase_timestamp(self):
        # STEP #1: For each module, grant received sessions
        for module in self._modules:
            module.grant_received_sessions()
            
        # STEP #2: Execute suspended sessions    
        for session in self._suspended_sessions:
            session._execute()
                
        self._suspended_sessions = []
        
        # STEP #3: Increase the timestamp
        self._timestamp += 1
        
        # STEP #4: Execute the suspended callbacks derived from the sessions
        if self.timestamp in self._suspended_session_callbacks.keys():
            for session in self._suspended_session_callbacks[self.timestamp]:
                session.run_callbacks()
                
            self._suspended_session_callbacks.pop(self.timestamp)    
