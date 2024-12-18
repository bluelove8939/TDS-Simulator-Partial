import os
import argparse

import torch
import torch.nn.functional as F

from tds_sim.models.gcn.model_presets import SimpleGCN
from tds_sim.models.gcn.create_dataset import create_planetoid_dataset


parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed'])
args = parser.parse_args()


if __name__ == "__main__":
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_name = args.dataset
    save_dir = os.path.join(os.curdir, "saved_models")
    save_name = f"SimpleGCN_{dataset_name}.pth"
    
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = create_planetoid_dataset(dataset_name)
    data = dataset[0].to(torch_device)  # Cora dataset only have one graph
    
    model = SimpleGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(torch_device)
    data = dataset[0].to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data], data.y[data])
        loss.backward()
        optimizer.step()
        
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    
    correct = float (pred.eq(data.y).sum().item())
    acc = correct / data.num_nodes
    
    print(f'VALIDATION ACCURACY:  {acc:.4f}')
    
    correct = float (pred.eq(data.y).sum().item())
    acc = correct / data.num_nodes
    
    print(f'TEST ACCURACY:        {acc:.4f}')
    
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))
