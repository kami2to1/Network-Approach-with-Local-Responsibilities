import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================== Code====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BellLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_groups=3, n_dim=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.n_dim = n_dim  # per-group dimension
        
        self.W = nn.Parameter(
            torch.randn(out_dim, in_dim, n_groups, device=device) * 0.1 / np.sqrt(in_dim)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device))
        
        # Per-group dimension expansion: [in_dim, n_groups * n_dim]
        self.dim_expand = nn.Parameter(
            torch.randn(in_dim, n_groups, n_dim, device=device) * 0.1 / np.sqrt(in_dim)
        )
        self.dim_bias = nn.Parameter(torch.zeros(n_groups, n_dim, device=device))
        
        # Per-group, per-dimension parameters
        self.c = nn.Parameter(torch.randn(out_dim, n_groups, n_dim, device=device) * 0.5)
        self.p = nn.Parameter(torch.zeros(out_dim, n_groups, n_dim, device=device))
        self.k = nn.Parameter(torch.zeros(out_dim, n_groups, n_dim, device=device))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        p = torch.exp(self.p) + 1e-6
        k = torch.exp(self.k) + 1e-6
        
        # Expand to per-group dimensions: [batch, n_groups, n_dim]
        x_expanded = torch.einsum('bi,igj->bgj', x, self.dim_expand) + self.dim_bias.unsqueeze(0)
        
        # diff: [batch, out_dim, n_groups, n_dim]
        diff = x_expanded.unsqueeze(1) - self.c.unsqueeze(0)
        p_exp = p.unsqueeze(0)
        
        R = 1.0 / (diff.pow(2) * p_exp.pow(2) + 1.0)
        k_exp = k.unsqueeze(0)
        R_weighted = R * k_exp.pow(2)
        
        # Sum over dimension to get group-level weights
        R_sum = R_weighted.sum(dim=-1, keepdim=True)
        alpha = R_weighted / (R_sum + 1e-8)
        alpha_reduced = alpha.sum(dim=-1)  # [batch, out_dim, n_groups]
        
        W_exp = self.W.unsqueeze(0)  # [1, out_dim, in_dim, n_groups]
        x_exp = x.unsqueeze(1).unsqueeze(-1)  # [batch, 1, in_dim, 1]
        alpha_exp = alpha_reduced.unsqueeze(2)  # [batch, out_dim, 1, n_groups]
        
        out = (x_exp * W_exp * alpha_exp).sum(dim=2).sum(dim=-1) + self.bias
        return out


class DeepBellNet(nn.Module):
    def __init__(
        self, 
        layer_dims,
        n_groups=3,
        n_dim=4,
        dropout=0.01,
        use_batch_norm=False,
        use_layer_norm=True,
        activation='relu',
        skip_connections=True
    ):
        super().__init__()
        
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        
        if isinstance(n_groups, int):
            n_groups = [n_groups] * (self.n_layers - 1)
        if isinstance(n_dim, int):
            n_dim = [n_dim] * (self.n_layers - 1)
        
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }[activation]
        
        self.bell_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.skip_connections = skip_connections
        
        for i in range(self.n_layers - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            n_g = n_groups[i]
            n_d = n_dim[i]
            
            self.bell_layers.append(BellLayer(in_dim, out_dim, n_g, n_d))
            
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(out_dim))
            elif use_layer_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            else:
                self.norms.append(nn.Identity())
            
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            self.activations.append(act_fn)
        
        self.output_layer = nn.Linear(layer_dims[-2], layer_dims[-1])
        self.to(device)
    
    def forward(self, x):
        for bell, norm, dropout, act in zip(
            self.bell_layers, self.norms, self.dropouts, self.activations
        ):
            identity = x
            x = bell(x)
            
            if self.skip_connections and identity.shape[-1] == x.shape[-1]:
                x = x + identity
            
            x = norm(x)
            x = dropout(x)
            x = act(x)
        
        x = self.output_layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
