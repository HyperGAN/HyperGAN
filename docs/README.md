# HyperGAN documentation

Start with the [README](../README.md) for installation and the CLI tour, and the [glossary](../GLOSSARY.md) for terms. Pages below are grouped by who they are for.

## User guide

Writing a model file, preparing data, training, watching and recovering runs.

- [Configuration](configuration.md): every table and field of a `hypergan-model/1` model file.
- [Image-folder data](image-data.md): image folders, preprocessing and dataset checks.
- [Image previews and samples](image-previews.md): preview grids during training and `hypergan sample`.
- [Image FID evaluation](image-fid.md): the pinned FID protocol and its `fid` extra.
- [Web Observatory](local-web.md): `hypergan view`, the dashboard, authentication and the REST/SSE API.
- [Observe a run and request a checkpoint](observation.md): events, progress and on-demand checkpoints.
- [Recover training runs](recovery.md): checkpoints, resume and compatibility rules.
- [Native CUDA training](cuda.md): single-GPU training and the CUDA test gate.
- [Public execution profiles](execution.md) and [execution profiles and preflight](execution-profiles.md): running on several devices.
- [Multi-GPU readiness before a paid run](multigpu-readiness.md): local gates, larger GPU counts, transport settings and cloud recovery limits.
- [Bounded scaling benchmark](scaling-benchmark.md): fixed-batch or weak-scaling measurements locally and on Modal, with resource estimates and rank evidence.
- [Container profile](docker.md): building and running the Docker image.
- [AMD ROCm readiness](rocm.md): ROCm support status and the preflight script; no ROCm hardware run backs it yet.
- [Migrating from historical HyperGAN](migration.md): legacy configurations and checkpoints.

## Recipes

Measured or exploratory model recipes, each with its example file under `examples/`.

- [CIFAR with a frozen pretrained critic](cifar-recipe.md): the first measured image recipe.
- [CIFAR tiny transformer with a feature-only critic](cifar-tiny-transformer.md)
- [Tiny transformer generator at 128px](tiny-transformer.md)
- [128px logos with the CIFAR feature-only recipe](logos-tiny-transformer-features.md)
- [Logo colorization](colorization.md)
- [RGB logo AE-GAN at 256×256](logos-ae-gan.md)
- [Eight-color words](color-words.md)
- [Simple StyleTransformer at 128px](simple-styletransformer.md)
- [TransGAN 128px with the section 3.4 recipe](transgan-stable.md) and its [comparison with the official TransGAN implementation](transgan-upstream-comparison.md)
- [Image recipe numerical controls](image-core.md): trainable and frozen components, and the numerical settings the image recipes share.

## Internals

Contracts for contributors working on the engine, distributed execution and storage.

- [ParticleGAN integration](particlegan.md): the boundary between ParticleGAN and HyperGAN.
- [Native HNDL integration](hndl-upstream-requirements.md): the HNDL operators the image recipes need.
- [Shared training lifecycle](run-lifecycle.md)
- [Foldlite DB integration](foldlitedb.md): event storage and shared reducers.
- [Event maps and file-backed views](event-views.md): how metric events become viewer series.
- [Distributed numerical groundwork](distributed.md), [fixed-topology CPU recovery](distributed-recovery.md) and [CPU gradient accumulation](accumulation.md)
- [Local CPU worker supervision](cpu-workers.md) and [persistent CPU worker commands](cpu-worker-service.md)
- [Replicated run service](replicated-run-service.md), [replicated CUDA training](replicated-cuda.md) and [bounded observation for replicated runs](replicated-observation.md)
- [Local two-GPU NCCL readiness](gpu-readiness.md)
- [CI and PyPI releases](releasing.md)

The test layout and how to run each tier are in [tests/README.md](../tests/README.md). Dated design notes and experiment reports are in [reports/](../reports/), with older ones under `reports/archive/`.
