import os
import math
import argparse
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.utils import add_self_loops, degree

from tds_sim.models.gcn.create_dataset import create_citation_full_dataset
from tds_sim.common.sdmm_reordering import sdmm_reordering
        

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('-om', '--only-dimM', action="store_true", required=False, dest="only_dimM")
args = parser.parse_args()


def visualize_matrix(matrix: csr_matrix, img_path: str):
    row_idx, col_idx = matrix.nonzero()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(row_idx, col_idx, marker='o', color='blue', s=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.margins(x=0, y=0)
    plt.savefig(img_path, dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    torch_device = 'cpu'
    
    dataset_name = args.dataset
    only_dimM = args.only_dimM
    
    img_dirname = os.path.join(os.curdir, "logs", "simple_reordering_matrix")
    orig_img_filename  = f"{dataset_name}_original.png"
    
    os.makedirs(img_dirname, exist_ok=True)
    
    dataset = create_citation_full_dataset(dataset_name)
    data = dataset[0].to(torch_device)  # Cora dataset only have one graph
    data = dataset[0].to(torch_device)
    
    x = data.x
    edge_index = data.edge_index
    
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    row, col = edge_index
    N = x.shape[0]

    adj_csr = coo_matrix((edge_weight, (row, col)), shape=(N, N)).tocsr()
    visualize_matrix(adj_csr, img_path=os.path.join(img_dirname, orig_img_filename))

    for tile_size in [256, 512, 768, 1024]:
        reord_img_filename = f"{dataset_name}_reordered_tile{tile_size}.png" if not only_dimM else f"{dataset_name}_reordered_tile{tile_size}_only_dimM.png"
        
        reord_adj_csr = sdmm_reordering(adj_csr, tile_size=tile_size, leave_pbar=True, only_dimM=only_dimM)
        visualize_matrix(reord_adj_csr, img_path=os.path.join(img_dirname, reord_img_filename))
        
    
