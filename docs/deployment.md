# Deploying Exported Models

HyperGAN models train with full dynamic infrastructure — ParticleGAN learned latent spaces, HNDL compilers, and distributed telemetry. For deployment, HyperGAN compiles and strips these runtimes into lightweight, standalone model artifacts that run with **zero ParticleGAN and zero HNDL runtime dependency**.

---

## 1. Zero-Dependency Deployment Philosophy

When you export a trained generator:

- **Baked Particle Table**: The learned particle coordinates (`prior.z` and `prior.sigma`) are baked directly into constant buffers inside the generator module (`particle_z`, `particle_sigma`). At inference time, random draws sample directly from these frozen buffers without initializing ParticleGAN optimizers.
- **Pure PyTorch Computation**: HNDL architecture graphs are compiled into pure standard PyTorch `torch.nn.Module` subclasses. The exported model requires only standard PyTorch (or ONNX Runtime) to execute.
- **Privacy-Preserving Export**: Training batch pixels (`last_batch`) are stripped from the export payload by default. The export stores only tensor metadata (`{shape, dtype}`) unless explicitly requested via `--include-example-inputs`.

---

## 2. Export Formats & Commands

HyperGAN supports four production export formats via `hypergan export`:

```sh
# 1. Standalone PyTorch Bundle (.pt)
hypergan export runs/cifar --format bundle -o models/generator.pt

# 2. SafeTensors (.safetensors) for Rust, C++, WebAssembly, Hugging Face
hypergan export runs/cifar --format safetensors -o models/generator.safetensors

# 3. PyTorch 2.0 AOT / TorchScript (.pt2) for C++ LibTorch
hypergan export runs/cifar --format pt2 -o models/generator.pt2

# 4. ONNX (.onnx) for ONNX Runtime, TensorRT, CoreML, OpenVINO
hypergan export runs/cifar --format onnx -o models/generator.onnx
```

### Export Format Matrix

| Format | Target Runtimes | Size Footprint | Dependencies |
| :--- | :--- | :--- | :--- |
| **`bundle`** (`.pt`) | Python (`LoadedGenerator`), PyTorch | Minimal | `torch` only |
| **`safetensors`** | Hugging Face `safetensors`, Rust `candle` | Minimal | Zero Python runtime |
| **`pt2`** | C++ LibTorch, Mobile PyTorch | Optimized graph | C++ LibTorch |
| **`onnx`** | ONNX Runtime, TensorRT, Triton, Web | Graph-optimized | `onnxruntime` |

---

## 3. Python Serving with `LoadedGenerator`

The standard way to serve exported models in Python is using `hg.load_bundle()`, which returns a callable `LoadedGenerator` (`torch.nn.Module`):

```python
import hypergan as hg
import torch

# Load the standalone bundle onto GPU
generator = hg.load_bundle("models/generator.pt", device="cuda")
generator.eval()

# 1. Generate random samples from the baked particle prior
with torch.no_grad():
    # Returns Tensor [16, 3, 128, 128] normalized in [-1, 1]
    images = generator.sample(count=16, seed=42)

# 2. Sample explicit latent coordinates
latents = generator.sample_latents(count=4, seed=123)
with torch.no_grad():
    images = generator(z=latents)

# 3. Conditional generation (e.g. colorization or class conditioning)
condition = torch.randn(4, 1, 128, 128, device="cuda")
with torch.no_grad():
    colored = generator.sample(count=4, inputs={"gray": condition})
```

---

## 4. Spherical Latent Interpolation (`slerp`)

Generators trained with spherical or normalized priors support smooth latent space walks using spherical linear interpolation (`slerp`):

```python
import hypergan as hg

generator = hg.load_bundle("models/generator.pt", device="cuda")

# Generate 20 smooth transition frames between two random seeds
frames = generator.interpolate(
    seed_a=1001,
    seed_b=2002,
    steps=20,
    mode="slerp"  # or "lerp" for linear interpolation
)
# frames shape: [20, 3, H, W]
```

---

## 5. High-Throughput Batch Sampling

For offline dataset generation or synthetic data augmentation pipelines, use the `hypergan sample` CLI:

```sh
# Generate 1,000 images directly to disk using CUDA
hypergan sample runs/cifar --device cuda --output-dir ./synthetic_data/ --count 1000

# Generate with conditional input image
hypergan sample runs/conditional_gan --device cuda --output-dir ./samples/ -i gray=condition.png
```

---

## 6. Serving via ONNX Runtime

Exported `.onnx` models run with high throughput in Python, C++, Go, or Rust services without PyTorch installed:

```python
import numpy as np
import onnxruntime as ort

# Initialize ONNX Runtime session with GPU (TensorRT/CUDA) provider
session = ort.InferenceSession("models/generator.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Prepare input batch [batch_size, z_dim]
batch_size = 8
z_dim = 128
z_inputs = np.random.randn(batch_size, z_dim).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output_images = session.run([output_name], {input_name: z_inputs})[0]

# output_images shape: (8, 3, 128, 128), float32 in [-1, 1]
print("Generated batch shape:", output_images.shape)
```
