import os
import torch
from torch_geometric.datasets import Planetoid, CitationFull

DEFAULT_DATASET_PATH=os.path.join(".tds-sim", "tmp", "datasets")

def create_planetoid_dataset(name: str, root: str=DEFAULT_DATASET_PATH, train_ratio: float=0.6, val_ratio: float=0.2):
    os.makedirs(root, exist_ok=True)
    
    dataset = Planetoid(root=root, name=name)
    
    num_nodes      = dataset[0].num_nodes
    train_set_size = int(num_nodes * train_ratio)
    val_set_size   = int(num_nodes * val_ratio)
    test_set_size  = num_nodes - (val_set_size + train_set_size)
    
    train_offset = 0
    val_offset   = train_offset + train_set_size
    test_offset  = val_offset   + val_set_size
    
    dataset[0].train_mask.fill_(False)
    dataset[0].val_mask.fill_(False)
    dataset[0].test_mask.fill_(False)
    
    dataset[0].train_mask[train_offset:train_offset + train_set_size] = 1
    dataset[0].val_mask  [val_offset  :val_offset   + val_set_size  ] = 1
    dataset[0].test_mask [test_offset :test_offset  + test_set_size ] = 1
    dataset[0].val_mask[0] = 1
    
    return dataset

def create_citation_full_dataset(name: str, root: str=DEFAULT_DATASET_PATH, train_ratio: float=0.6, val_ratio: float=0.2):
    os.makedirs(root, exist_ok=True)
    
    dataset = CitationFull(root=root, name=name)
    
    # num_nodes      = dataset[0].num_nodes
    # train_set_size = int(num_nodes * train_ratio)
    # val_set_size   = int(num_nodes * val_ratio)
    # test_set_size  = num_nodes - (val_set_size + train_set_size)
    
    # train_offset = 0
    # val_offset   = train_offset + train_set_size
    # test_offset  = val_offset   + val_set_size
    
    # dataset[0].train_mask   = torch.zeros(num_nodes, dtype=torch.uint8)
    # dataset[0].val_mask     = torch.zeros(num_nodes, dtype=torch.uint8)
    # dataset[0].test_mask    = torch.zeros(num_nodes, dtype=torch.uint8)
    
    # dataset[0].train_mask[train_offset:train_offset + train_set_size] = 1
    # dataset[0].val_mask  [val_offset  :val_offset   + val_set_size  ] = 1
    # dataset[0].test_mask [test_offset :test_offset  + test_set_size ] = 1
    # dataset[0].val_mask[0] = 1
    
    return dataset