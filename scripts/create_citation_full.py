import os
import argparse

import torch
import torch.nn.functional as F

from tds_sim.models.gcn.model_presets import CitationFullGCN
from tds_sim.models.gcn.create_dataset import create_citation_full_dataset


parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed', 'DBLP'])
args = parser.parse_args()


if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_name = args.dataset
    save_dir = os.path.join(os.curdir, "saved_models")
    save_name = f"CitationFullGCN_{dataset_name}.pth"
    
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = create_citation_full_dataset(dataset_name)
    data = dataset[0].to(torch_device)  # Cora dataset only have one graph
    
    model = CitationFullGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(torch_device)
    data = dataset[0].to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_ratio: float=0.6
    val_ratio: float=0.2
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_set_size = int(data.num_nodes * train_ratio)
    val_set_size   = int(data.num_nodes * val_ratio)
    test_set_size  = data.num_nodes - (val_set_size + train_set_size)
    
    train_offset = 0
    val_offset   = train_offset + train_set_size
    test_offset  = val_offset   + val_set_size
    
    train_mask[train_offset:train_offset + train_set_size] = 1
    val_mask  [val_offset  :val_offset   + val_set_size  ] = 1
    test_mask [test_offset :test_offset  + test_set_size ] = 1
    val_mask[0] = 1

    model.train()
    for epoch in range(200):
        print(f"training epoch: {epoch}")
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    
    correct = float (pred[val_mask].eq(data.y[val_mask]).sum().item())
    acc = correct / val_mask.sum().item()
    
    print(f'VALIDATION ACCURACY:  {acc:.4f}')
    
    correct = float (pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    
    print(f'TEST ACCURACY:        {acc:.4f}')
    
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))
