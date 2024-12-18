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
    
    log_dirname = os.path.join(os.curdir, "logs", "design_explore", "pe_num", dataset_name)
    img_dirname = os.path.join(os.curdir, "logs", "design_explore", "images")
    log_filename_fmt = "{pe_num}.csv"
    img_filename = f"analyze_pe_num_{dataset_name}.png"
    
    os.makedirs(img_dirname, exist_ok=True)
    
    categories = []
    gemm_cycles = []
    sdmm_cycles = []

    pe_num_list = [4, 8, 16, 32, 64]

    for pe_num in pe_num_list:
        log_filename = log_filename_fmt.format(pe_num=pe_num)

        df = pd.read_csv(os.path.join(log_dirname, log_filename))
        
        categories.append(f"{pe_num}")
        gemm_cycles.append(np.sum(df['cycles'][df['action'] == "GEMM"]))
        sdmm_cycles.append(np.sum(df['cycles'][df['action'] == "SDMM"]))
            
    x = np.arange(len(categories))

    fig, ax1 = plt.subplots()

    fig.set_size_inches(6, 3)

    ax1.bar(x, gemm_cycles, width=0.6, label="GEMM", color="#A6E494", edgecolor="black")
    ax1.bar(x, sdmm_cycles, width=0.6, label="SDMM", color="#FF7F7F", edgecolor="black", bottom=gemm_cycles)

    ax1.set_xticks(x, categories)
    ax1.set_xlabel("Number of PEs")
    ax1.set_ylabel("Execution Cycles")
    ax1.margins(x=0.05)

    ax1.legend(loc=9, bbox_to_anchor=(0, 1.16, 1, 0), frameon=False, ncols=2)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dirname, img_filename), dpi=1000, pad_inches=0.05, bbox_inches='tight')

    print(f"image created at {os.path.join(img_dirname, img_filename)}")