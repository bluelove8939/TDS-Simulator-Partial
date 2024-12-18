import torch
from typing import Callable, Dict

from tds_sim.simulation.device import Device


def runtime_method(func: Callable):
    def __runtime_wrapper(*args, **kwargs):
        if (len(args) + len(kwargs.keys())) < 1:
            raise Exception(f"[ERROR] runtime_method: argument given to the runtime_method should be at least 2")
        
        self: DeviceConfig = args[0]
        
        if not isinstance(self, DeviceConfig):
            raise Exception(f"[ERROR] runtime method: the runtime method should be the method of DeviceConfig, not '{type(self).__name__}'")
        
        if len(args) >= 2:
            if not isinstance(args[1], torch.nn.Module):
                raise Exception(f"[ERROR] runtime_method: the second positional argument should be 'torch.nn.Module', not '{type(args[0].__name__)}'")
        else:
            if 'module' not in kwargs.keys():
                raise Exception(f"[ERROR] runtime_method: keyword argument 'module' is required since the method receives less than 2 positional arguments")
            
            if not isinstance(kwargs['module'], torch.nn.Module):
                raise Exception(f"[ERROR] runtime_method: keyword argument should be  'torch.nn.Module', not '{type(kwargs['module'].__name__)}'")
        
        self.set_log_context(context_name=func.__name__)
        out = func(*args, **kwargs)
        self.reset_log_context()
        
        return out
    
    return __runtime_wrapper


class DeviceConfig(object):
    def __init__(self, device: Device):
        self.device = device
        
        self._supported_runtime: Dict[str, Callable] = {}
        
        self._context_queue: list[str] = []
        self.execution_logs: list[tuple[str, str, int, int]] = []  # each entry refers to the (context, action, start time, end time)
        
        for obj_name in dir(self):
            obj = self.__getattribute__(obj_name)
            
            if isinstance(obj, Callable):
                if obj.__name__ == "__runtime_wrapper":
                    self._supported_runtime[obj_name] = obj
    
    def get_runtime(self, module: torch.nn.Module) -> Callable:
        module_name = module._get_name() if isinstance(module, torch.jit.TracedModule) else type(module).__name__
        if module_name not in self._supported_runtime.keys():
            return None
        return self._supported_runtime[module_name]
    
    def set_log_context(self, context_name: str):
        self._context_queue.append(context_name)
        
    def reset_log_context(self):
        self._context_queue.pop(-1)
        
    def create_log(self, action: str, start_time: int, end_time: int):
        self.execution_logs.append((self.log_context, action, start_time, end_time))
        
    def clear_log(self):
        self.execution_logs: list[tuple[str, str, int, int]] = []
        
    def save_log_file(self, filepath: str, clear_log: bool=False):
        content = ["context,action,start time,end time,cycles"]
        for log_context, action, st_time, ed_time in self.execution_logs:
            content.append(f"{log_context},{action},{st_time},{ed_time},{ed_time-st_time}")
        
        with open(filepath, "wt") as file:
            file.write("\n".join(content))
            
        if clear_log:
            self.clear_log()
                
    @property
    def log_context(self) -> str:
        return '.'.join(self._context_queue)
