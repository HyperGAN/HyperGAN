# Working CIFAR recipe with logos data at 32px

Only the name and training/evaluation data bindings differ from
`cifar-transgan32-adversarial`. Generator and pretrained ResNet critic HNDLs are
byte-identical; latent/prior, optimizer, batch, seeds, EMA and schedule are unchanged.
The runner verifies this configuration equality and matches the initial parameter
hash against the completed CIFAR control before any training updates.

The research adapter `bridge_data:Logos32Data` verifies the existing logos128
manifest unchanged, then explicitly decodes its pinned train images to 32px using
the same aspect-fit white padding, Lanczos, EXIF and alpha handling. Its recovery
identity records both the source manifest policy and effective32px geometry.
It samples with replacement and applies independent horizontal flips using the
caller device generator, matching CIFAR's draw protocol. It normalizes uint8
pixels with `float()/127.5-1` on that device. Source bytes are verified per access.
The logits remain the original CIFAR critic's; its ResNet input remains64px.

This changes dataset content/count and the preprocessing needed to form32px logo
images. It does not silently adopt logos128's permutation sampler, remove flips,
change the data RNG device, or modify its source manifest. Class labels from CIFAR
are absent; both recipes' networks are unconditional and never consume them.
Evaluation data bindings also use logos, preserving their schedule without
accidentally reporting a CIFAR-reference FID. This diagnostic rollout does not
execute those FID jobs.

Use the research runner so the adapter is importable:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331 \
/home/martyn/dev/hypergan/training-runs/transgan-128-env/bin/python \
research/startup_tuning/logos_data_bridge_screen.py --device cuda:0 --output-root OUTPUT
```

Check GPU0 is available before running. GPU1 remains reserved. No seed sweep.
The runner observes512 native updates and restores state without a checkpoint.
