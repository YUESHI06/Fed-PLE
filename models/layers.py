import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


class GraphConvolution(nn.Module):
    """Graph convolution with normalized A^3 propagation."""
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, input_dim) node features
            adj: (B, N, N) adjacency matrices
        Returns:
            output: (B, N, output_dim)
        """
        mask = (adj.diagonal(dim1=1, dim2=2) > 0).float().unsqueeze(-1)
        adj2 = torch.bmm(adj, adj)
        adj3 = torch.bmm(adj2, adj)
        support = torch.matmul(x, self.weight)  # (B, N, output_dim)
        output = torch.bmm(adj3, support)  # (B, N, output_dim)
        if self.bias is not None:
            output = output + self.bias
        return output * mask

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})'
