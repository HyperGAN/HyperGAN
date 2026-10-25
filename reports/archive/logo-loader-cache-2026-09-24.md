# Logo loader: verified resized-pixel cache

The second loader optimization retains the exact image preprocessing, shuffled
sampling, independent phase draws, model and optimizer. It adds an optional
disk cache to the manifest-backed `ColorizationData` loader and enables it in
the example and local 128px logo configs:

```toml
cache_dir = "/mnt/ml7tb/hypergan-cache/logos-128"
```

The running training process was neither stopped nor restarted. The local
configuration passes read-only `prepare_train` compatibility checks against
the existing run. The user can enable this code by restarting the same launcher.

## Changes

- Keep the old permutation reference for batch rollback, avoiding a copy of
  all 404,757 indices per batch. Epoch changes replace rather than mutate it.
- Walk source-path ancestors directly instead of repeatedly constructing and
  searching their ancestor lists. Containment and symlink checks remain.
- Start next-batch prefetch before CPU tensor assembly and normalization.
- Cache exact resized RGB uint8 bytes in source-hash-addressed files, namespaced
  by preprocessing, Pillow version and cache format. Length/checksum validation
  binds pixels to the source and namespace. Missing or damaged entries regenerate.
- Atomically publish entries so multiple workers/loaders can share the cache.
  Write errors warn and disable further writes for that loader, without treating
  a cache error as a bad training image.
- Verify original source contents and paths at every consumption, including
  warm-cache hits. Speculative cache hits defer that check until consumption,
  avoiding a redundant source read. Cold prefetch verifies before decoding too.
- Permit enabling, moving or disabling this derived cache on checkpoint resume;
  retain all other recipe checks and the existing checkpoint/data schemas.

The cache grows on demand. A complete unique-image cache has 18.53 GiB of
pixel payload, roughly 20 GiB with allocation overhead. Benchmarking populated
only about 109 MiB. The first traversal still needs decoding and cache writes;
subsequent traversals and restarts can reuse those pixels. No full-dataset
prewarm, training experiment, seed comparison or GPU benchmark was run.

## CPU loader measurements

Measured with the training Python environment, four workers, one batch of
prefetch, batch 64, and one Torch/OMP/MKL CPU thread. Each case consumed the same
32 batches from the same caller-owned RNG state. Medians exclude the first two
batches. A sleep between calls simulates time available for background work;
it is not an actual GPU workload. The baseline uses code from `380e084a`.
Cases ran sequentially while the existing logo training process continued.

| Simulated compute between calls | Baseline | Optimized, no disk cache | Optimized, warm disk cache |
| --- | ---: | ---: | ---: |
| 0 ms | 128.0 ms | 120.5 ms | 20.9 ms |
| 50 ms | 68.9 ms | 61.1 ms | 23.4 ms |
| 100 ms | 30.5 ms | 30.4 ms | 24.1 ms |

These are time spent inside the loader, not end-to-end training speedups. The
warm cache helps most when decoding cannot fit inside the compute interval.
Source verification and CPU batch assembly remain on the consumption path.
At 50 ms overlap, the warm cache cuts exposed loader time by about 66%; with
back-to-back requests it cuts it by about 84%.

Every case produced the same SHA256 over returned RGB tensor bytes:
`f15c60cad9f0744ff97083598c0a5163494e43946e2debe0b7d07d83203130d9`.
Final warm cases performed zero decodes and 2,048 source reads, one per consumed
image. Baseline cases performed roughly 4,040–4,096 reads plus decoding.

An earlier exploratory profile measured 65.9 ms at 100 ms overlap, but repeating
the baseline gave 28.4 ms before optimization. That variation is why the table
uses the later comparison sequence and does not attribute the initial drop to
the implementation. The initial cold-cache check at 100 ms overlap measured
29.9 ms versus 27.8 ms uncached; cache population has a cost.

## Validation

181 focused tests passed across image data, workers, caching, damaged-image
recovery, colorization data/metrics, image/full training recovery, checkpoint
compatibility, configuration, the data CLI and distributed restore identity.
The broad invocation passed 175 tests with one deselected; six distributed
configuration cases passed separately. New/modified worker/cache test files
and the new cache module pass Ruff; changed legacy modules pass fatal-error
checks. `git diff --check` passes.

Coverage includes RGB/grayscale/alpha/palette/EXIF exact pixels, cache reuse
across instances, serial/threaded epoch crossings, corrupt/oversized/swapped
cache entries, concurrent publication, cache write failure, changed/missing/
symlinked sources despite a warm cache, persistent bad-image exclusions,
rollback after assembly failure with outstanding prefetch, and full CPU
training-state equality after enabling then removing caching during resume.
