# Summary

* [HyperGAN documentation](README.md)

## User guide

* [Model & Recipe Configuration (`hypergan-model/1`)](configuration.md)

* [Python API Reference (`hypergan.api`)](python-api.md)

* [HNDL (HyperGAN Network Description Language) Reference](hndl-reference.md)

* [Train on Your Own Images](custom-images.md)

* [Custom PyTorch Modules & Datasets](custom-pytorch.md)

* [Particle LoRAs and frozen-model hosts](particle-sliders.md)

* [Deploying exported generators](deployment.md)

* [Image-folder data (`image_folder`)](image-data.md)

* [Image previews and samples](image-previews.md)

* [Image FID evaluation](image-fid.md)

* [Web Observatory, Local Browser & Streaming API (`hypergan view`)](local-web.md)

* [Observe a run and request a checkpoint](observation.md)

* [Recover training runs](recovery.md)

* [Native CUDA training](cuda.md)

* [Public execution profiles](execution.md)

* [Execution profiles and preflight](execution-profiles.md)

* [Multi-GPU readiness before a paid run](multigpu-readiness.md)

* [Bounded training scaling benchmark](scaling-benchmark.md)

* [Container Profile (`Dockerfile`)](docker.md)

* [AMD ROCm Readiness & Verification in HyperGAN](rocm.md)

* [Migrating to HyperGAN 2.0.0](migration.md)

## Recipes

* [CIFAR with a frozen pretrained critic](cifar-recipe.md)

* [CIFAR tiny transformer with a feature-only critic](cifar-tiny-transformer.md)

* [Tiny transformer generator at 128px](tiny-transformer.md)

* [128px logos with the CIFAR feature-only recipe](logos-tiny-transformer-features.md)

* [Logo colorization in HyperGAN and HNDL](colorization.md)

* [RGB logo AE-GAN at 256×256](logos-ae-gan.md)

* [Eight-color words](color-words.md)

* [Simple StyleTransformer at 128px](simple-styletransformer.md)

* [TransGAN 128px with the section 3.4 recipe](transgan-stable.md)

* [Comparison with the official TransGAN implementation](transgan-upstream-comparison.md)

* [Image recipe numerical controls](image-core.md)

## Internals

* [ParticleGAN integration](particlegan.md)

* [Native HNDL integration](hndl-upstream-requirements.md)

* [Shared training lifecycle](run-lifecycle.md)

* [Foldlite DB integration](foldlitedb.md)

* [Event maps and file-backed views](event-views.md)

* [Distributed numerical groundwork](distributed.md)

* [Fixed-topology CPU recovery](distributed-recovery.md)

* [CPU gradient accumulation](accumulation.md)

* [Local CPU worker supervision](cpu-workers.md)

* [Internal persistent CPU worker commands](cpu-worker-service.md)

* [Internal replicated run service](replicated-run-service.md)

* [Internal replicated CUDA training](replicated-cuda.md)

* [Bounded observation for replicated CPU runs](replicated-observation.md)

* [Local two-GPU NCCL readiness](gpu-readiness.md)

* [CI and PyPI releases](releasing.md)

* [Versioned API docs and assets on S3](s3-releases.md)
