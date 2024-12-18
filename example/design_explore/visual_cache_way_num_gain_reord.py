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
    img_filename = f"analyze_cache_way_num_gain_reordered_{dataset_name}.png"
    
    os.makedirs(img_dirname, exist_ok=True)
    
    way_num_list = [1, 2, 4, 8, 16, 32]
    tile_size_list = [0, 256, 512, 768, 1024]
    colors = ["#FF7F7F", "#AFD2FF", "#7FB7FF", "#4494FD", "#0D70F0"]
    
    categories = []
    sdmm_cycles: dict[int, list] = {tile_size: [] for tile_size in tile_size_list}
    
    for way_num in way_num_list:
        log_filename = log_filename_fmt.format(way_num=way_num)

        df = pd.read_csv(os.path.join(log_dirname, log_filename))
        
        categories.append(f"{way_num}")
        sdmm_cycles[0].append(np.sum(df['cycles'][df['action'] == "SDMM"]))

    for tile_size in tile_size_list[1:]:
        for way_num in way_num_list:
            log_filename = f"reordered_tile{tile_size}_" + log_filename_fmt.format(way_num=way_num)

            df = pd.read_csv(os.path.join(log_dirname, log_filename))

            sdmm_cycles[tile_size].append(np.sum(df['cycles'][df['action'] == "SDMM"]))
    
    crit = sdmm_cycles[0][0]
    for tile_size in tile_size_list:
        sdmm_cycles[tile_size] = crit / np.array(sdmm_cycles[tile_size])
            
    x = np.arange(len(categories))

    fig, ax1 = plt.subplots()

    fig.set_size_inches(7, 3)
    
    width = 0.8
    num_bars = len(tile_size_list)

    for bar_idx, tile_size in enumerate(tile_size_list):
        bar_width   = width / num_bars
        bar_x       = x + bar_width * (bar_idx - ((num_bars - 1) / 2))
        bar_label   = f"T={tile_size}" if tile_size != 0 else "not reordered"
        
        ax1.bar(bar_x, sdmm_cycles[tile_size], width=bar_width, label=bar_label, color=colors[bar_idx], edgecolor="black")

    ax1.set_xticks(x, categories)
    ax1.set_xlabel("Cache Way Number")
    ax1.set_ylabel("Performance Gain over\nDirect-mapped Cache")
    ax1.margins(x=0.02)
    ax1.set_ylim((0.9, 1.4))

    ax1.legend(loc=9, bbox_to_anchor=(0, 1.18, 1, 0), frameon=False, ncols=len(tile_size_list))
    # ax1.legend(loc=7, bbox_to_anchor=(0, 0.5, 1.3, 0), frameon=False, ncols=1)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dirname, img_filename), dpi=1000, pad_inches=0.05, bbox_inches='tight')

    print(f"image created at {os.path.join(img_dirname, img_filename)}")