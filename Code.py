import torch
import torch.nn as nn
import numpy as np


# ==================== Device Configuration ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Core Bell Layer ====================
class BellLayer(nn.Module):
    """
    Bell Basis Function Layer - k^2 participates in normalization as responsibility
    
    Uses bell-shaped basis functions: R = 1 / ((x-c)^2 * p^2 + 1)
    where c is the center, p is the width, and k is the amplitude parameter.
    The responsibility alpha is computed by normalizing the k^2-weighted responses.
    """
    def __init__(self, in_dim, out_dim, n_groups=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups = n_groups
        
        # Weight matrix: (out_dim, in_dim, n_groups)
        self.W = nn.Parameter(torch.randn(out_dim, in_dim, n_groups, device=device) * 0.1 / np.sqrt(in_dim))
        
        # Bias: (out_dim,)
        self.bias = nn.Parameter(torch.zeros(out_dim, device=device))
        
        # Basis function centers: (out_dim, n_groups)
        self.c = nn.Parameter(torch.randn(out_dim, n_groups, device=device) * 0.5)
        
        # Basis function widths p (log form ensures positivity): (out_dim, n_groups)
        self.p = nn.Parameter(torch.zeros(out_dim, n_groups, device=device))
        
        # Group amplitude parameters k (log form ensures positivity): (out_dim, n_groups)
        self.k = nn.Parameter(torch.zeros(out_dim, n_groups, device=device))
    
    def forward(self, x):
        # x shape: (batch, in_dim)
        p = torch.exp(self.p) + 1e-6  # Ensure p > 0
        k = torch.exp(self.k) + 1e-6  # Ensure k > 0
        
        # Compute difference from centers: (batch, in_dim, out_dim, n_groups)
        diff = x.unsqueeze(-1).unsqueeze(-1) - self.c.view(1, 1, self.out_dim, self.n_groups)
        
        # Reshape width parameter: (1, 1, out_dim, n_groups)
        p_exp = p.view(1, 1, self.out_dim, self.n_groups)
        
        # Bell basis response: R = 1 / ((x-c)^2 * p^2 + 1)
        R = 1.0 / (diff.pow(2) * p_exp.pow(2) + 1.0)
        
        # Apply k^2 weighting for responsibility computation
        k_exp = k.view(1, 1, self.out_dim, self.n_groups)
        R_weighted = R * k_exp.pow(2)  # Weight by k^2
        
        # Normalize to get responsibility alpha (soft assignment)
        alpha = R_weighted / (R_weighted.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted combination of inputs using learned weights and responsibilities
        W_exp = self.W.unsqueeze(0)              # (1, out_dim, in_dim, n_groups)
        x_exp = x.unsqueeze(1).unsqueeze(-1)     # (batch, 1, in_dim, 1)
        alpha_exp = alpha.permute(0, 2, 1, 3)    # (batch, out_dim, in_dim, n_groups)
        
        # Sum over input dimensions and groups
        out = (x_exp * W_exp * alpha_exp).sum(dim=2).sum(dim=-1) + self.bias
        return out


# ==================== Deep Configurable Bell Network ====================
class DeepBellNet(nn.Module):
    """
    Deep Bell Network with configurable architecture.
    
    Supports:
    - Variable depth and width via layer_dims
    - Configurable number of basis groups per layer
    - Multiple normalization options (BatchNorm, LayerNorm, or None)
    - Dropout regularization
    - Various activation functions (ReLU, GELU, Tanh, SiLU)
    - Optional skip connections for residual learning
    """
    def __init__(
        self, 
        layer_dims,           # List of dimensions: [in_dim, hidden_1, ..., hidden_n, out_dim]
        n_groups=3,           # Number of basis groups per layer (int or list per layer)
        dropout=0.01,          # Dropout rate
        use_batch_norm=False, # Whether to use Batch Normalization
        use_layer_norm=True,  # Whether to use Layer Normalization
        activation='relu',    # Activation function type
        skip_connections=True # Whether to use residual skip connections
    ):
        super().__init__()
        
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1
        
        # Handle uniform or per-layer group specification
        if isinstance(n_groups, int):
            n_groups = [n_groups] * (self.n_layers - 1)
        
        # Select activation function
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }[activation]
        
        # Initialize network components
        self.bell_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.skip_connections = skip_connections
        
        # Build hidden layers
        for i in range(self.n_layers - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            n_g = n_groups[i]
            
            # Bell layer as the core transformation
            self.bell_layers.append(BellLayer(in_dim, out_dim, n_g))
            
            # Normalization layer
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(out_dim))
            elif use_layer_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            else:
                self.norms.append(nn.Identity())
            
            # Dropout for regularization
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            
            # Activation function
            self.activations.append(act_fn)
        
        # Final output layer (linear projection)
        self.output_layer = nn.Linear(layer_dims[-2], layer_dims[-1])
    
    def forward(self, x):
        """Forward pass through all Bell layers with optional skip connections."""
        for bell, norm, dropout, act in zip(
            self.bell_layers, self.norms, self.dropouts, self.activations
        ):
            identity = x
            x = bell(x)
            
            # Apply skip connection if dimensions match
            if self.skip_connections and identity.shape[-1] == x.shape[-1]:
                x = x + identity
            
            # Apply normalization, dropout, and activation
            x = norm(x)
            x = dropout(x)
            x = act(x)
        
        # Final linear projection to output dimension
        x = self.output_layer(x)
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

