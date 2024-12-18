import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from tds_sim.models.gcn.model_presets import SimpleGCN
from tds_sim.models.gcn.create_dataset import create_planetoid_dataset
from tds_sim.compiler.hook_session import HookSession
from tds_sim.compiler.configs.gcn_accelerator import GCNAcceleratorConfig, GCNAccelerator
        

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed'], dest="dataset")
parser.add_argument('-ar', '--apply-reorder', action="store_true", required=False, dest="apply_reorder")
parser.add_argument('-ts', '--tile-size', type=int, default=512, dest="tile_size")
args = parser.parse_args()


if __name__ == "__main__":
    torch_device = 'cpu'
    
    dataset_name = args.dataset
    tile_size = args.tile_size
    apply_reorder = args.apply_reorder
    
    save_dir = os.path.join(os.curdir, "saved_models")
    save_name = f"SimpleGCN_{dataset_name}.pth"
    log_dir = os.path.join(os.curdir, "logs", "design_explore", "cache_way_num", dataset_name)
    
    os.makedirs(log_dir, exist_ok=True)
    
    dataset = create_planetoid_dataset(dataset_name)
    data = dataset[0].to(torch_device)  # Cora dataset only have one graph
    data = dataset[0].to(torch_device)
    
    x = data.x
    edge_index = data.edge_index

    for way_num in [1, 2, 4, 8, 16, 32]:
        print(f"executing with way number: {way_num}")

        model = SimpleGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes)
        state_dict = torch.load(os.path.join(save_dir, save_name), weights_only=True)
        model.load_state_dict(state_dict)
        
        model = model.to(torch_device).eval()
        
        # device_params = default_device_params[dataset_name]
        device = GCNAccelerator(pe_num=32, mac_per_pe=32, cache_capacity=32 * 1024, cache_way_num=way_num)
        device_config = GCNAcceleratorConfig(
            device=device, verbose=False, apply_reordering=apply_reorder, reordering_tile_size=tile_size, 
            simulate_gemm=False, simulate_sdmm=True)
        
        sim = HookSession(device_config=device_config, verbose=True)
        _, y = sim.execute_model(model, x, edge_index).max(dim=1)
        
        device_config.save_log_file(
            filepath=os.path.join(log_dir, f"{way_num}.csv" if not apply_reorder else f"reordered_tile{tile_size}_{way_num}.csv"), clear_log=True)
    
