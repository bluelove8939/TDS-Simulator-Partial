import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
class CitationFullGCN(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(CitationFullGCN, self).__init__()
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)