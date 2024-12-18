import warnings

warnings.filterwarnings("ignore", category=UserWarning)    # TODO: suppress leaf Tensor access warning
warnings.filterwarnings("ignore", category=FutureWarning)  # TODO: suppress quantized model warning

import abc
import torch
from typing import Any

from tds_sim.compiler.device_config import DeviceConfig
from tds_sim.common.custom_exception import CustomException


def get_primitive_of_data(data: Any) -> str:
    if isinstance(data, (list, tuple)):
        return "[" + ", ".join(list(map(get_primitive_of_data, data))) + "]"
    elif isinstance(data, torch.Tensor):
        return f"Tensor(shape={tuple(data.shape)})"
    else:
        return data

def check_jit_type_compatibility(s_type, m_arg: Any) -> tuple[bool, bool, Any]:  # flag_compatible, flag_optional, converted arg
    try:
        m_type = torch._C._jit_try_infer_type(m_arg).type()

        if s_type.isSubtypeOf(m_type):
            return (True, "Optional" in s_type.kind(), m_arg)

        if s_type.kind() == "NumberType" and m_type.kind() in ("IntType", "FloatType"):
            return (True, False, m_arg)
        elif s_type.kind() == "DeviceObjType" and m_type.kind() in ("StringType"):
            return (True, False, torch.device(m_arg))
        elif s_type.kind() == "TensorType" and m_type.kind() in ("ListType"):
            return (True, False, torch.tensor(m_arg))
        elif "Optional" in s_type.kind():
            s_opt_type = s_type.getElementType()
            _flag_compatible, _, _converted_arg = check_jit_type_compatibility(s_opt_type, m_arg)
            return (_flag_compatible, True, _converted_arg)
    except:
        return (False, False, m_arg)
    
    return (False, False, m_arg)

def check_argument_compatibility(s_arg, m_arg: Any) -> tuple[bool, Any]:
    s_type = s_arg.type
    
    return check_jit_type_compatibility(s_type=s_type, m_arg=m_arg)


class InterpreterSession(object):
    def __init__(self, device_config: DeviceConfig, verbose: bool=False) -> None:
        self.device_config = device_config
        self.verbose = verbose
        
        self.var_farm = {}
        self._context_storage = []
        
    def start_new_context(self):
        self._context_storage.append(self.var_farm)
        self.var_farm = {}
        
    def restore_previous_context(self):
        self.var_farm = self._context_storage[-1]
        self._context_storage.pop(-1)
    
    @staticmethod
    def _get_attr_from_node(node: torch.Node, attr_name: str) -> any:
        for attr_types in ['f', 'fs', 'c', 's', 'ss', 'i', 'g', 'gs', 'ival', 't', 'ts', 'ty', 'tys']:
            try:
                return getattr(node, attr_types)(attr_name)
            except:
                pass
        
        return None
    
    @staticmethod
    def _find_nonprim_method(method_domain: str, method_name: str) -> callable:
        for aten_ref in [getattr(torch.ops, method_domain), torch, torch.nn.functional]:
            try:
                return getattr(aten_ref, method_name)
            except:
                pass
        
        return None
        
    def _get_ivar_from_node(self, node: torch.Node, start: int=0) -> list[any]:
        def ivar_cast(ivar: torch.Value):
            if 'bool' in ivar.type().kind().lower():
                return bool(self.var_farm[ivar.debugName()])
            return self.var_farm[ivar.debugName()]
            
        return list(map(ivar_cast, list(node.inputs())[start:]))
    
    def _get_kwargs_ivar_from_node(self, node: torch.Node, start: int=0) -> dict[str, any]:
        def ivar_cast(ivar: torch.Value):
            value = self.var_farm[ivar.debugName()]
            if 'bool' in ivar.type().kind().lower():
                value = bool(self.var_farm[ivar.debugName()])
            
            return ivar.debugName(), value
            
        return dict(map(ivar_cast, list(node.inputs())[start:]))
    
    def _set_ovar_from_node(self, node: torch.Node, outputs: any):
        ovars = list(map(lambda x: x.debugName(), node.outputs()))
        
        if len(ovars) == 1:
            self.var_farm[ovars[0]] = outputs
        else:
            for var_name, output in zip(ovars, outputs):
                self.var_farm[var_name] = output
    
    def _execute_prim_node(self, node: torch.Node):
        node_domain, node_action = node.kind().split("::")
        
        if node_domain != "prim":
            raise CustomException(self, f"node '{node.kind()}' cannot be executed by '_execute_prim_node()' method")
        
        attrs = {attr_name: self._get_attr_from_node(node, attr_name) for attr_name in node.attributeNames()}
        
        if len(list(node.outputs())) == 1:
            if node.output().type().kind() == 'NoneType':
                self.var_farm[node.output().debugName()] = None
                return  # TODO: if the output type is NoneType, do not execute the node and store the output as None instead
        
        if node_action == "GetAttr":
            ivar = node.input().debugName()
            ovar = node.output().debugName()
            attr_name = attrs['name']
            self.var_farm[ovar] = getattr(self.var_farm[ivar], attr_name)
        elif node_action == "CallMethod":
            inst_name = list(node.inputs())[0].debugName()
            inst = self.var_farm[inst_name]
            args = self._get_ivar_from_node(node, start=1)
            method_name = attrs['name']
            
            if isinstance(inst, torch.nn.Module) and "forward" in method_name:
                outputs = self.execute_model(inst, *args)
            else:
                method = getattr(inst, method_name)
                outputs = method(*args)

            self._set_ovar_from_node(node, outputs)
        elif node_action == "Constant":
            ovar = node.output().debugName()
            if 'value' in attrs.keys():
                self.var_farm[ovar] = attrs['value']
            else:
                self.var_farm[ovar] = None
        elif node_action == "ListConstruct":
            ovar = node.output().debugName()
            self.var_farm[ovar] = self._get_ivar_from_node(node, start=0)
        elif node_action == "TupleConstruct":
            ovar = node.output().debugName()
            self.var_farm[ovar] = tuple(self._get_ivar_from_node(node, start=0))
        elif node_action == "NumToTensor":
            ovar = node.output().debugName()
            self.var_farm[ovar] = torch.tensor(self._get_ivar_from_node(node, start=0)[0])
        else:
            raise CustomException(self, f"action '{node_action}' is not supported by the session in 'prim' domain\nexception occurred for the node: {node.kind()}")
        
    def _execute_nonprim_node(self, node: torch.Node):
        node_domain, node_action = node.kind().split("::")
        
        method = self._find_nonprim_method(node_domain, node_action)
        
        if method is None:
            raise CustomException(self, f"non prim method '{node_domain}::{node_action}' is not known")
        
        args = self._get_ivar_from_node(node, start=0)
        
        # check schema
        pp_args = []
        pp_kwargs = {}
        
        for overload_name in method._overload_names:
            schema = torch._C._get_schema(method._qualified_op_name, overload_name)
            
            if len(schema.arguments) < len(args):
                continue
            
            flag_schema_compatible = True
            tmp_pp_args = []
            tmp_pp_kwargs = {}
            
            for s_arg, m_arg in zip(schema.arguments, args):
                flag_compatible, flag_optional, converted_arg = check_argument_compatibility(s_arg=s_arg, m_arg=m_arg)
                
                if not flag_compatible and not flag_optional:
                    flag_schema_compatible = False
                else:
                    tmp_pp_kwargs[s_arg.name] = converted_arg
            
            if flag_schema_compatible:
                pp_args = tmp_pp_args
                pp_kwargs = tmp_pp_kwargs
            
        outputs = method(*pp_args, **pp_kwargs)
        
        self._set_ovar_from_node(node, outputs)
        
    def execute_model(self, model: torch.nn.Module | torch.jit.TracedModule, *args):
        model_trace: torch.jit.TracedModule = model if isinstance(model, torch.jit.TracedModule) else torch.jit.trace(model, args)
        graph:       torch.Graph            = model_trace.graph
        
        if self.verbose:
            print(f"== MODULE: {model._get_name()}")
            
        # STEP #0: check whether device config supports the model (or module)
        method = self.device_config.get_runtime(module=model_trace)
        
        if method is not None:
            print(f"  * running with the runtime given from the device config")
            return method(model_trace, *args)
    
        # STEP #1: initialize variable farm
        self.start_new_context()
        
        self.var_farm = {}
        
        graph_ivars: list[torch.Value] = list(graph.inputs())
        graph_ovars: list[str] = list(map(lambda x: x.debugName(), graph.outputs()))
        
        if self.verbose >= 2:
            print(f"  * initialize context")
        
        self.var_farm[graph_ivars[0].debugName()] = model_trace
        for idx, var in enumerate(graph_ivars[1:]):
            self.var_farm[var.debugName()] = args[idx]
            
            if self.verbose >= 2:
                print(f"    - %{var.debugName()} = {get_primitive_of_data(args[idx])}")
        
        # STEP #2: execute graph
        for node in graph.nodes():
            node_domain, node_action = node.kind().split("::")
            node_primitive = node.__str__().strip()
            if "# " in node_primitive:
                node_primitive = node_primitive[:node_primitive.find("# ")]
                    
            if self.verbose == 2:
                print(f"  * {', '.join(list(map(lambda x: '%' + x.debugName(), node.outputs())))} = {node_primitive.split(' = ')[-1].strip().split(', scope')[0].strip()}")
            if self.verbose == 3:
                print(f"  * {node_primitive}")
                
            if node_domain == "prim":
                self._execute_prim_node(node=node)
            else:
                self._execute_nonprim_node(node=node)
        
        # STEP #3: return output
        if len(graph_ovars) == 1:
            outputs = self.var_farm[graph_ovars[0]]
        else:
            outputs = [self.var_farm[var_name] for var_name in graph_ovars]
                    
        self.restore_previous_context()
        
        if self.verbose >= 2:
            print(f"== MODULE END")
        
        return outputs
