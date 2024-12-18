import os
import argparse


def generate_config_header(config, rtl_config_root, verilated_root):
    config_path = os.path.join(rtl_config_root, f"{config}.sv")
    params = {}
    
    with open(config_path, 'rt') as file:
        for line in file.readlines():
            if line.startswith("parameter"):
                line = ' '.join(line.split(" ")[1:])
                pname, pval = line.split("=")
                pname = pname.strip()
                pval = pval.split(';')[0].strip()
                pval = pval.split(',')[0].strip()
                params[pname] = pval
            elif line.startswith("`define"):
                line = ' '.join(line.split(" ")[1:])
                pname, pval = line.split("=")
                pname = pname.strip()
                pval = pval.split(';')[0].strip()
                pval = pval.split(',')[0].strip()
                params[pname] = pval
                
    with open(os.path.join(verilated_root, f"{config}_config.h"), 'wt') as file:
        content = []
        for pname, pval in params.items():
            content.append(f"#define {pname} {pval}")
        content += [""]
        file.write('\n'.join(content))