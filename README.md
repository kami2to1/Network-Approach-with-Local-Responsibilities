# DeepBellNet

A configurable deep neural network built on **learnable Bell-shaped basis functions**, supporting PyTorch and GPU acceleration.

## Features

- **Bell Basis Layers** – Each layer learns centers, widths, and amplitudes of bell-shaped bases, producing soft gating via normalized `k²` responsibilities
- **Flexible Architecture** – Control depth and width via a simple `layer_dims` list (e.g., `[2, 64, 64, 1]`)
- **Per-Layer Groups** – Set the number of basis groups per layer (integer or list)
- **Normalization Options** – Choose between BatchNorm, LayerNorm, or None
- **Regularization** – Built-in Dropout support
- **Activation Functions** – ReLU, GELU, Tanh, SiLU
- **Skip Connections** – Optional residual connections for deeper networks
- **Parameter Counter** – `count_parameters()` method included

## Quick Usage

```python
from model import DeepBellNet

# Define a 2-input, 2-hidden, 1-output network
model = DeepBellNet(
    layer_dims=[2, 64, 32, 1],   # [input, hidden..., output]
    n_groups=5,                   # basis groups per layer
    dropout=0.0,
    use_layer_norm=True,
    activation='gelu',
    skip_connections=True
)

# Forward pass
x = torch.randn(32, 2)           # batch of 32, 2D input
out = model(x)                   # shape: (32, 1)

# Check parameters
print(f"Trainable params: {model.count_parameters():,}")
