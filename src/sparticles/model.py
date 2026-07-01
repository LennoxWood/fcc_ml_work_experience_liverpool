"""
model.py

Graph Attention Network (GAT) for signal vs background classification.
Import this in the notebook rather than defining the class there.

Usage:
    from model import GAT
    input_channels = dataset[0].data_norm.shape[1]
    model = GAT(input_channels=input_channels, hidden_channels=50, num_classes=1)
"""

import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool

MANUAL_SEED = 1234


class GAT(torch.nn.Module):
    """
    Three-layer Graph Attention Network.

    Each GATv2Conv layer lets every node attend to its neighbours with
    learned attention weights, updating its own representation based on
    which neighbours are most relevant. After three rounds of message
    passing, global mean pooling collapses the whole graph into one
    vector, and a linear head outputs a single classification score.

    A score > 0 after sigmoid means signal-like; < 0 means background-like.

    Args:
        input_channels (int): Number of input features per node.
        hidden_channels (int): Width of the hidden layers.
        num_classes (int): Output dimension (1 for binary classification).
    """

    def __init__(self, input_channels, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(MANUAL_SEED)

        self.conv1       = GATv2Conv(input_channels,   hidden_channels)
        self.activation1 = torch.nn.ReLU()
        self.conv2       = GATv2Conv(hidden_channels,  hidden_channels)
        self.activation2 = torch.nn.ReLU()
        self.conv3       = GATv2Conv(hidden_channels,  hidden_channels)
        self.aggregate   = global_mean_pool
        self.head        = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.activation1(self.conv1(x, edge_index))
        x = self.activation2(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = self.aggregate(x, batch)
        return self.head(x)