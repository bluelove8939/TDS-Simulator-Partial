import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_list = ['Cora', 'CiteSeer', 'PubMed']

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, default="all", choices=dataset_list + ['all'], dest="dataset")
args = parser.parse_args()

for dataset_name in dataset_list:
    if args.dataset not in ("all", dataset_name):
        continue
    
    log_dirname = os.path.join(os.curdir, "logs", "design_explore", "cache_way_num", dataset_name)
    img_dirname = os.path.join(os.curdir, "logs", "design_explore", "images")
    log_filename_fmt = "{way_num}.csv"
    img_filename = f"analyze_cache_way_num_gain_{dataset_name}.png"
    
    os.makedirs(img_dirname, exist_ok=True)
    
    categories = []
    sdmm_cycles = []

    way_num_list = [1, 2, 4, 8, 16, 32]

    for way_num in way_num_list:
        log_filename = log_filename_fmt.format(way_num=way_num)

        df = pd.read_csv(os.path.join(log_dirname, log_filename))
        
        categories.append(f"{way_num}")
        sdmm_cycles.append(np.sum(df['cycles'][df['action'] == "SDMM"]))
            
    sdmm_cycles = sdmm_cycles[0] / np.array(sdmm_cycles)
            
    x = np.arange(len(categories))

    fig, ax1 = plt.subplots()

    fig.set_size_inches(6, 3)

    ax1.bar(x, sdmm_cycles, width=0.6, label="SDMM", color="#FF7F7F", edgecolor="black")

    ax1.set_xticks(x, categories)
    ax1.set_xlabel("Cache Way Number")
    ax1.set_ylabel("Performance Gain over\nDirect-mapped Cache")
    ax1.margins(x=0.05)
    ax1.set_ylim((0.9, 1.1))

    ax1.legend(loc=9, bbox_to_anchor=(0, 1.16, 1, 0), frameon=False, ncols=2)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dirname, img_filename), dpi=1000, pad_inches=0.05, bbox_inches='tight')

    print(f"image created at {os.path.join(img_dirname, img_filename)}")