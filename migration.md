# Migrating to HyperGAN 2.0.0

## Continue a 2.x alpha run or move its generator to GA

Keep the original environment and complete run directory for exact resume. HyperGAN checks the recorded dependency versions and runtime identity as well as the training configuration. A CPU `gaussian-grid` checkpoint from 2.0.0a2 and ParticleGAN 0.9.7 resumed in that environment. With ParticleGAN 0.10.1, the same checkpoint reported a ParticleGAN version mismatch. See the [checkpoint recovery contract](recovery.md).

To keep training, activate the alpha environment and run `hypergan resume runs/my-alpha-run`. To serve its EMA generator after upgrading, export in the alpha environment first. Keep the bundle and its sibling digest together.

```sh
hypergan export runs/my-alpha-run --checkpoint latest -o models/alpha-generator.pt
```

In a separate GA environment, install the serving dependencies for that model and load the exported file. A Gaussian bundle exported from a2 reproduced the original four samples exactly for the same seed. Conditional generators also require their model-specific inputs. See [deployment requirements and examples](deployment.md).

```python
import hypergan as hg

generator = hg.load_bundle("models/alpha-generator.pt", device="cpu")
points = generator.sample(count=4, seed=42)
```

To train with GA's ParticleGAN preset, install `hypergan[all]==2.0.0` with the [documented Torch wheels](https://github.com/HyperGAN/HyperGAN/blob/73570ab17c3eb04ea843551bd7300aca34c29a38/README.md#install) in a fresh environment. Start a new run directory with your model. For example, this creates a short Gaussian run:

```sh
hypergan train gaussian-grid --device cpu --run-dir runs/ga-toy --stop-after-steps 2 --checkpoint-every 1 --no-server
```

## Access a legacy 1.x experiment

HyperGAN 2 uses TOML model files, HNDL networks and a HyperGAN-owned training loop. Legacy JSON configurations, dynamic layer strings, viewers and training checkpoints require the archived 1.x code and its original environment. Port the model and input bindings to a new configuration to train it under 2.0. [Preserved master commit](https://github.com/HyperGAN/HyperGAN/tree/291ddccda847e4f4ccb273bb26121a0a0d738164) · [Archive tags and preservation record](https://github.com/HyperGAN/HyperGAN/blob/73570ab17c3eb04ea843551bd7300aca34c29a38/reports/archive/resurrection-preservation-2026-09-18.md)
