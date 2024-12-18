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
args = parser.parse_args()


if __name__ == "__main__":
    torch_device = 'cpu'
    
    dataset_name = args.dataset
    apply_reorder = args.apply_reorder
    
    save_dir = os.path.join(os.curdir, "saved_models")
    save_name = f"SimpleGCN_{dataset_name}.pth"
    log_dir = os.path.join(os.curdir, "logs", "design_explore", "cache_capacity", dataset_name)
    
    os.makedirs(log_dir, exist_ok=True)
    
    dataset = create_planetoid_dataset(dataset_name)
    data = dataset[0].to(torch_device)  # Cora dataset only have one graph
    data = dataset[0].to(torch_device)
    
    x = data.x
    edge_index = data.edge_index

    cache_capacity = 8 * 1024
    
    while cache_capacity <= 2 * 1024 * 1024:
        print(f"executing with cache capacity: {cache_capacity}")

        model = SimpleGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes)
        state_dict = torch.load(os.path.join(save_dir, save_name), weights_only=True)
        model.load_state_dict(state_dict)
        
        model = model.to(torch_device).eval()
        
        # device_params = default_device_params[dataset_name]
        device = GCNAccelerator(pe_num=32, mac_per_pe=32, cache_capacity=cache_capacity, cache_way_num=8, print_debug_info=False)
        device_config = GCNAcceleratorConfig(device=device, verbose=False, apply_reordering=apply_reorder, simulate_gemm=True, simulate_sdmm=True)
        
        sim = HookSession(device_config=device_config, verbose=True)
        _, y = sim.execute_model(model, x, edge_index).max(dim=1)
        
        device_config.save_log_file(filepath=os.path.join(log_dir, f"{cache_capacity//1024}KB.csv" if not apply_reorder else f"reordered_{cache_capacity//1024}KB.csv"), clear_log=True)
        
        cache_capacity *= 2
    
