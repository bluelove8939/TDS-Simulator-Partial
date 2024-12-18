import os
import argparse

import torch
import torch.nn.functional as F

from tds_sim.models.gcn.model_presets import SimpleGCN
from tds_sim.models.gcn.create_dataset import create_planetoid_dataset
from tds_sim.compiler.interpreter_session import InterpreterSession
from tds_sim.compiler.device_config import DeviceConfig


parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed'])
args = parser.parse_args()


# if __name__ == "__main__":
torch_device = 'cpu'

dataset_name = args.dataset
save_dir = os.path.join(os.curdir, "saved_models")
save_name = f"SimpleGCN_{dataset_name}.pth"

dataset = create_planetoid_dataset(dataset_name)
data = dataset[0].to(torch_device)  # Cora dataset only have one graph

model = SimpleGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes)
state_dict = torch.load(os.path.join(save_dir, save_name))
model.load_state_dict(state_dict)

model = model.to(torch_device).eval()
data = dataset[0].to(torch_device)

x = data.x
edge_index = data.edge_index

_, pred = model(x, edge_index).max(dim=1)

device_config = DeviceConfig(device=None)

sim = InterpreterSession(device_config=device_config, verbose=2)
_, y = sim.execute_model(model, x, edge_index).max(dim=1)

print(f"OUTPUT VALIDATION:  {torch.equal(pred, y)}")
print(f"NUMBER OF MISMATCH: {torch.count_nonzero(pred != y)}")