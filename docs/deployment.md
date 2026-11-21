# Deploying exported generators

Choose a format according to how the serving process will supply inputs and load the generator. A Python bundle includes prior sampling and interpolation. PT2 and ONNX take explicit latent tensors. SafeTensors exposes the weights to a consumer that implements the model. All four exports use the checkpoint's EMA generator and omit training example pixels by default.

## Produce the artifacts

Run these commands in the training environment with the `all` extra installed. The CPU `gaussian-grid` example maps four latent coordinates to a point in two dimensions. Each format has its own directory so the bundle and SafeTensors digest files can travel with their respective artifacts.

```sh
hypergan train gaussian-grid --device cpu --run-dir runs/toy --stop-after-steps 1 --checkpoint-every 1 --no-server
hypergan export runs/toy --format bundle -o models/bundle/toy.pt
hypergan export runs/toy --format pt2 -o models/pt2/toy.pt2
hypergan export runs/toy --format onnx -o models/onnx/toy.onnx
hypergan export runs/toy --format safetensors -o models/weights/toy.safetensors
```

Use a fresh Python 3.12 environment for each serving example. Run the Python snippets from the directory containing `models/`.

## Python bundle

For an HNDL generator, install CPU Torch, HyperGAN, NumPy and HNDL. HyperGAN rebuilds the component graph on load. Models with custom Python factories also need their importable modules and runtime dependencies. The bundle contains the learned prior used by `sample`.

```sh
python -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "hypergan==2.1.0" "hndl==0.11.0" "numpy>=1.26,<3"
```

```python
import hypergan as hg
import torch

generator = hg.load_bundle("models/bundle/toy.pt", device="cpu")
with torch.inference_mode():
    points = generator.sample(count=4, seed=42)
    assert points.shape == (4, 2)
    latent = generator.sample_latents(count=4, seed=42)
    torch.testing.assert_close(generator(z=latent), points)
    walk = generator.interpolate(seed_a=100, seed_b=200, steps=20, mode="slerp")
```

For conditional models, pass the model's named batch inputs to `generator.sample(count=..., inputs={...})`. A model binding `batch.gray` expects a batch under the key `gray`. For offline sampling directly from a run, see the [sampling guide](../README.md#sampling-evaluation--standalone-export).

## PyTorch exported program (PT2)

Install the same Torch version used to export the graph. HyperGAN captures an example batch of one. Use one row per call and supply any conditioning tensors after the latent in export order. When a model requires the exporter's TorchScript trace fallback, load that archive with `torch.jit.load(path)`.

```sh
python -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
```

```python
import torch

model = torch.export.load("models/pt2/toy.pt2").module()
with torch.inference_mode():
    assert model(torch.zeros(1, 4)).shape == (1, 2)
```

## ONNX Runtime

Install these packages on the CPU serving host. The ONNX graph declares a batch dimension on the latent, conditioning inputs and output. Inspect the session for the input names and shapes of your exported model.

```sh
python -m pip install onnxruntime numpy
```

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("models/onnx/toy.onnx", providers=["CPUExecutionProvider"])
latent = np.zeros((4, 4), dtype=np.float32)
points, = session.run(["output"], {"latent": latent})
assert points.shape == (4, 2)
```

## SafeTensors weights and metadata

The consumer constructs the generator from the recorded configuration and applies its named component weights. Inspect the tensor names, prior and configuration before mapping them into that implementation. This example reads the weights on CPU.

```sh
python -m pip install "safetensors>=0.8,<1" numpy
```

```python
import json
from safetensors import safe_open

with safe_open("models/weights/toy.safetensors", framework="numpy") as weights:
    config = json.loads(weights.metadata()["config"])
    for name in weights.keys():
        print(name, weights.get_tensor(name).shape)
```
