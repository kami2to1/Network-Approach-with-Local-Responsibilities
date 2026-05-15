import torch
import torch.nn as nn
import numpy as np
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Code ====================
class BellLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_groups: int = 3, n_dim: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.n_dim = n_dim
        
        scale = 0.1 / np.sqrt(in_dim)
        
        self.W = nn.Parameter(torch.randn(out_dim, in_dim, n_groups) * scale)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.dim_expand = nn.Parameter(torch.randn(in_dim, n_groups, n_dim) * scale)
        self.dim_bias = nn.Parameter(torch.zeros(n_groups, n_dim))
        
        self.c = nn.Parameter(torch.randn(out_dim, n_groups, n_dim) * 0.5)
        self.p = nn.Parameter(torch.zeros(out_dim, n_groups, n_dim))
        self.k = nn.Parameter(torch.zeros(out_dim, n_groups, n_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.exp(self.p) + 1e-6
        k = torch.exp(self.k) + 1e-6
        
        x_expanded = torch.einsum('bi,ign->bgn', x, self.dim_expand) + self.dim_bias
        diff = x_expanded.unsqueeze(1) - self.c
        
        R = 1.0 / (diff.pow(2) * p.pow(2) + 1.0)
        R_weighted = R * k.pow(2)
        
        R_sum = R_weighted.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) + 1e-8
        alpha = (R_weighted / R_sum).sum(dim=-1)
        
        out = torch.einsum('bi,oig,bog->bo', x, self.W, alpha) + self.bias
        return out


class DeepBellNet(nn.Module):
    def __init__(self, layer_dims: List[int], n_groups: int = 3, n_dim: int = 4, activation: str = 'none'):
        super().__init__()
        
        act_fn = {
            'relu': nn.ReLU(), 
            'gelu': nn.GELU(), 
            'tanh': nn.Tanh(), 
            'none': nn.Identity()
        }[activation]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 2):
            self.layers.append(BellLayer(layer_dims[i], layer_dims[i+1], n_groups, n_dim))
            self.layers.append(nn.LayerNorm(layer_dims[i+1]))
            self.layers.append(act_fn)
        
        self.output = nn.Linear(layer_dims[-2], layer_dims[-1])
        self.layer_dims = layer_dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
