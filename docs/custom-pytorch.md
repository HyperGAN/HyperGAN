# Custom PyTorch Modules & Datasets

HyperGAN is designed with a **config-first, decoupled architecture**. While HNDL provides declarative neural network construction, HyperGAN allows you to drop in arbitrary PyTorch `torch.nn.Module` classes, custom datasets, custom loss functions, and custom operators without forking or modifying HyperGAN itself.

---

## 1. Config-First Architecture Philosophy

In HyperGAN, your custom Python code lives in your own application repository or package. HyperGAN discovers and instantiates your classes at runtime via standard Python module dot-notation (`"package.module:ClassName"`).

You can maintain custom networks, loss functions, and data loaders in your repository:
```text
my_project/
├── my_project/
│   ├── __init__.py
│   ├── networks.py      # Custom nn.Module generators and discriminators
│   ├── losses.py        # Custom perceptual loss functions
│   └── data.py          # Custom PyTorch Dataset loaders
├── configs/
│   └── custom_gan.toml  # Model file referencing my_project.* classes
```

---

## 2. Authoring Custom Neural Networks

Any PyTorch `torch.nn.Module` can serve as a generator, critic, encoder, or auxiliary network.

### 2.1 Custom Generator (`nn.Module`)

A generator receives keyword arguments that match the input ports declared in `inputs`:

```python
# my_project/networks.py
import torch
import torch.nn as nn

class CustomGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.fc = nn.Linear(z_dim, base_channels * 4 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: [batch_size, z_dim]
        h = self.fc(z).view(z.shape[0], -1, 4, 4)
        h = self.act(self.deconv1(h))
        h = self.act(self.deconv2(h))
        return self.tanh(self.deconv3(h))
```

Reference it in your model file:
```toml
[networks.generator]
role = "generator"
module = "my_project.networks:CustomGenerator"
inputs = { z = "latent" }

[networks.generator.args]
z_dim = 128
base_channels = 64
```

### 2.2 Custom Critic (`nn.Module`)

A discriminator/critic receives candidate images (`x = "candidate"`):

```python
# my_project/networks.py
class CustomCritic(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 3, H, W]
        return self.net(x)
```

Reference it in your model file:
```toml
[networks.critic]
role = "critic"
module = "my_project.networks:CustomCritic"
inputs = { x = "candidate" }

[networks.critic.args]
in_channels = 3
base_channels = 64
```

---

## 3. Registering Custom HNDL Operators

You can extend HNDL's 85 built-in operators by registering custom operator classes:

```python
# my_project/ops.py
import torch
import torch.nn as nn
from hypergan.hndl.registry import operator, registry

@operator("custom_squash", category="activation")
class CustomSquash(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        scale = (norm ** 2) / (1.0 + norm ** 2)
        return scale * (x / (norm + self.eps))

# Register into the global HNDL operator registry
registry.add(CustomSquash)
```

Once registered, the operator can be used directly inside any HNDL source or file:
```hndl
linear(128)
custom_squash(dim=-1)
linear(10)
```

---

## 4. Hybrid Architectures

HyperGAN natively supports hybrid networks combining declarative HNDL with PyTorch modules.

For example, pair a declarative HNDL generator with a frozen foundation vision transformer critic:

```toml
[networks.generator]
role = "generator"
inputs = { z = "latent" }
source = """
linear(256 * 4 * 4)
reshape(['B', 256, 4, 4])
group_norm(32)
relu()
deconv(128, kernel=4, stride=2, padding=1)
group_norm(32)
relu()
deconv(3, kernel=4, stride=2, padding=1)
tanh()
"""

[networks.critic]
role = "critic"
inputs = { x = "candidate" }
source = """
# Freeze pretrained ResNet-18 backbone and train lightweight head
pretrained("torchvision_resnet18", output="layer3")
group_norm(32)
relu()
global_avg_pool()
linear(1)
"""
```

---

## 5. Custom Loss Objectives

To optimize auxiliary objectives (e.g., L1 reconstruction, feature matching, or perceptual loss), define a Python callable:

```python
# my_project/losses.py
import torch
import torch.nn.functional as F

def perceptual_l1_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(input_tensor, target_tensor)
```

Bind it in the model file under `[[losses]]`:

```toml
[[losses]]
type = "objective"
id = "rec_l1"
fn = "my_project.losses:perceptual_l1_loss"
input = "generated"
target = "batch.real"
weight = 10.0
```

The metric is evaluated every step, added to the generator's optimization gradient, and logged to `events.jsonl` under `loss/objectives/rec_l1`.

---

## 6. Custom Datasets & Samplers

### 6.1 Custom PyTorch `Dataset`

HyperGAN supports standard `torch.utils.data.Dataset` implementations:

```python
# my_project/data.py
from torch.utils.data import Dataset
import torch

class SyntheticSphereDataset(Dataset):
    def __init__(self, size: int = 10000, dim: int = 16):
        self.size = size
        self.dim = dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        vec = torch.randn(self.dim)
        vec = vec / torch.norm(vec)
        return {"real": vec}
```

Reference it under `[data]`:

```toml
[data]
dataset = "my_project.data:SyntheticSphereDataset"
batch_size = 64

[data.args]
size = 20000
dim = 32
```
