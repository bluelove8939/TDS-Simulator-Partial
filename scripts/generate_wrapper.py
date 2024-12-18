import os
import json


def generate_wrapper(config, verilated_root):
    templates_root = os.path.abspath(os.path.join(os.path.split(__file__)[0], "templates"))
    
    result_header_name = f"{config}_wrapper.h"
    result_source_name = f"{config}_wrapper.cpp"
    
    # read input output ports
    verilated_top_header_path = os.path.join(verilated_root, f"{config}_top.h")
    
    port_info = {}
    
    with open(verilated_top_header_path, "rt") as file:
        for line in file.readlines():
            line = line.strip()
            
            if line.startswith("VL_IN"):
                token_ptype = "input"
                line = line[5:]
            elif line.startswith("VL_OUT"):
                token_ptype = "output"
                line = line[6:]
            elif line.startswith("VL_SIG"):
                token_ptype = "signal"
                line = line[6:]
            else:
                continue
            
            step = 0
            token_size  = ""
            token_name  = ""
            token_msb   = ""
            token_lsb   = ""
            token_words = ""
            
            for lt in line:
                if step == 0:
                    if lt == "(":
                        step += 1
                    elif lt != " ":
                        token_size += lt
                elif step == 1:
                    if lt == ",":
                        step += 1
                    elif lt != " " and lt != "&":
                        token_name += lt
                elif step == 2:
                    if lt == ",":
                        step += 1
                    elif lt != " ":
                        token_msb += lt
                elif step == 3:
                    if lt == ",":
                        step += 1
                    elif lt == ")":
                        step = -1
                    elif lt != " ":
                        token_lsb += lt
                elif step == 4:
                    if lt == ")":
                        step = -1
                    elif lt != " ":
                        token_words += lt
                        
            port_info[token_name] = {
                "ptype": token_ptype,
                "size": token_size if token_size != "" else "32",
                "msb": token_msb,
                "lsb": token_lsb,
                "words": token_words
            }
            
    set_get_func_headers = []
    set_get_func_sources = []
    
    with open(os.path.join(templates_root, "get_header.template"), "rt") as file:
        get_h_fmt = file.read()
        
    with open(os.path.join(templates_root, "get_source.template"), "rt") as file:
        get_s_fmt = file.read()
        
    with open(os.path.join(templates_root, "set_header.template"), "rt") as file:
        set_h_fmt = file.read()
        
    with open(os.path.join(templates_root, "set_source.template"), "rt") as file:
        set_s_fmt = file.read()
        
    with open(os.path.join(templates_root, "getw_header.template"), "rt") as file:
        getw_h_fmt = file.read()
        
    with open(os.path.join(templates_root, "getw_source.template"), "rt") as file:
        getw_s_fmt = file.read()
        
    with open(os.path.join(templates_root, "setw_header.template"), "rt") as file:
        setw_h_fmt = file.read()
        
    with open(os.path.join(templates_root, "setw_source.template"), "rt") as file:
        setw_s_fmt = file.read()
    
    for name, info in port_info.items():
        if info["size"] != "W":
            set_get_func_headers.append(get_h_fmt.format(config=config, name=name, size=info["size"]))
            set_get_func_sources.append(get_s_fmt.format(config=config, name=name, size=info["size"]))
            if info["ptype"] != "output":
                set_get_func_headers.append(set_h_fmt.format(config=config, name=name, size=info["size"]))
                set_get_func_sources.append(set_s_fmt.format(config=config, name=name, size=info["size"]))
        else:
            total_len = int(info["msb"]) - int(info["lsb"]) + 1
            word_size = total_len // int(info["words"])
            
            set_get_func_headers.append(getw_h_fmt.format(config=config, name=name, word_size=word_size))
            set_get_func_sources.append(getw_s_fmt.format(config=config, name=name, word_size=word_size))
            if info["ptype"] != "output":
                set_get_func_headers.append(setw_h_fmt.format(config=config, name=name, word_size=word_size))
                set_get_func_sources.append(setw_s_fmt.format(config=config, name=name, word_size=word_size))
            
    with open(os.path.join(verilated_root, f"{config}_port_info.json"), "wt") as file:
        file.write(json.dumps(port_info, indent=2))
        
    # generate wrapper source and header from the template
    with open(os.path.join(templates_root, "wrapper_header.template"), "rt") as file:
        header_content = file.read().format(config=config, set_get_funcs="\n".join(set_get_func_headers))
        
    with open(os.path.join(templates_root, "wrapper_source.template"), "rt") as file:
        source_content = file.read().format(config=config, set_get_funcs="\n".join(set_get_func_sources))
    
    with open(os.path.join(verilated_root, result_header_name), "wt") as file:
        file.write(header_content)
        
    with open(os.path.join(verilated_root, result_source_name), "wt") as file:
        file.write(source_content)
    