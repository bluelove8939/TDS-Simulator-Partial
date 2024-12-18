import torch

from tds_sim.compiler.device_config import DeviceConfig


class HookSession(object):
    def __init__(self, device_config: DeviceConfig, verbose: bool=False, overwrite: bool=False):
        self.device_config = device_config
        self.verbose = verbose
        self.overwrite = overwrite
        
    def create_hook(self, module_name, method):
        def __hook(module, ref_input, ref_output):
            if self.verbose:
                print(f"forward hook called for {module_name}")
                
            self.device_config.set_log_context(module_name)
            
            if isinstance(ref_input, tuple):
                sim_output = method(module, *ref_input)
            else:
                sim_output = method(module, ref_input)
                
            self.device_config.reset_log_context()

            if self.overwrite:
                return sim_output
            
        return __hook
        
    def register_hook(self, module_name: str, module: torch.nn.Module):
        method = self.device_config.get_runtime(module=module)
        
        if method is not None:
            module.register_forward_hook(hook=self.create_hook(module_name=module_name, method=method))
            
        for submodule_name, submodule in module.named_children():
            self.register_hook(submodule_name, submodule)
    
    def execute_model(self, model: torch.nn.Module, *args):
        self.register_hook(module_name="top", module=model)
        
        model = model.eval()
        
        y = model(*args)
        
        return y