# Image-folder data

`image_folder` loads local images into CPU batches with an explicit shape and preprocessing policy. It supports flat folders, recursive folders and named class directories. It does not download data, provide an image GAN architecture, or qualify image quality.

Install the optional decoder in your active Python environment:

```sh
python -m pip install '.[train,image]'
```

For data inspection alone, `.[image]` is sufficient; folder discovery, decoding and identity inspection do not import torch. These source-install commands refer to this unreleased checkout.

Replace the data section of your recipe with:

```toml
[data]
factory = "image_folder"
[data.args]
root = "/absolute/path/to/images"
height = 160
width = 320
mode = "RGB"
resize = "pad"
interpolation = "bilinear"
fill = 0
recursive = true
labels = false
shuffle = true
```

The generator and discriminator still need compatible image shapes. Substituting this section into the default 2D Gaussian recipe does not create an image model. Height and width are mandatory; there is no silent 64×64 fallback. Relative roots resolve against the process working directory, so absolute paths are easiest to reproduce.

Inspect the actual files before training:

```sh
hypergan data-check path/to/config.toml
hypergan data-check path/to/config.toml --output data-manifest.json
```

The command reports the inventory, class map, preprocessing and content identity without importing the training runtime. Output files are created exclusively, so an existing manifest is not overwritten. The Python equivalent is `ImageFolder(**args).resume_identity()` plus its `inventory` attribute.

## Shape and preprocessing

The callable `data(batch_size, generator=cpu_generator)` returns `real`, a contiguous CPU float32 tensor shaped **N,C,H,W**, scaled as `pixel / 127.5 - 1` into `[-1,1]`. `mode="RGB"` produces three channels; `mode="L"` produces one grayscale channel. Conversion from RGB to grayscale or grayscale to RGB follows the explicit output mode.

| Setting | Behavior |
| --- | --- |
| `resize="none"` (default) | Require the exact declared width and height after EXIF orientation; never resize implicitly |
| `resize="stretch"` | Resize directly to width/height, changing aspect ratio |
| `resize="center_crop"` | Preserve aspect ratio while filling the requested rectangle, cropping centrally |
| `resize="pad"` | Preserve aspect ratio while fitting inside the rectangle; center with padding |
| `interpolation` | `nearest`, `bilinear` (default), `bicubic` or `lanczos`; used when resizing |
| `fill` | Padding intensity, integer 0–255 applied to every output channel; defaults to 0 and only accepts a nonzero value with `pad` |

Resize modes can enlarge small inputs. EXIF orientation is applied before resizing. JPEG, PNG, BMP, WebP and TIFF extensions are recognized case-insensitively. Sources must be single-frame RGB or grayscale; palette, alpha, CMYK, higher-bit-depth and animated/multi-frame inputs fail with conversion guidance. In particular, alpha is never silently discarded. Source mode/dimensions, decoder version and every preprocessing option enter the identity.

`max_pixels` defaults to 16,777,216 for both decoded and output images, and `max_file_bytes` to 67,108,864 per file. Override them explicitly for larger data. Pillow's own decompression limits also apply. These bounds limit each decode; batch memory still scales with the chosen batch size and output dimensions.

## Folder discovery and labels

With `recursive=true` (default), all ordinary image files beneath the root are inventoried in relative POSIX-path lexical order. `recursive=false` considers root-level files only. Unsupported extensions are reported as ignored paths and counts; they are not decoded. Symlinks are rejected so inventory membership is explicit.

With `labels=true`, every accepted image must sit beneath a named top-level class directory, such as `cats/a.png` and `zebras/nested/b.jpg`. Class directory names need not be numeric. Recursive discovery is required, and root-level images fail with layout guidance. The vocabulary is the sorted set of class names containing accepted images; empty directories do not create classes. Output adds `labels`, an int64 tensor shaped `[N]`; bind it explicitly as `batch.labels` in a compatible conditional recipe. Classification losses and multi-attribute labels are separate recipe concerns.

Preflight reads, hashes, fully decodes and validates every accepted image. A zero-byte, unreadable, truncated or incompatible image stops preflight with its relative path and corrective guidance. Nothing is silently skipped or retried. An empty dataset reports supported extensions and discovered/ignored counts. There is no quarantine mode.

## Background loading

`ImageFolder` and `ColorizationData` default to four persistent decoder threads
and one batch of lookahead. Workers read, verify, decode and resize individual
images concurrently. While the GPU trains, they prepare the next images in the
already-selected permutation. CPU tensor assembly and normalization happen once
per batch on the caller thread. Workers never use Torch or advance any RNG.

For new configurations, optional `[data.args]` settings `workers = 4` and
`prefetch_batches = 1` make these defaults explicit. `workers = 0` selects serial
loading; `prefetch_batches = 0` disables lookahead while retaining parallel batch
decoding. Both must be nonnegative integers. Existing configs that omit these
arguments pick up the defaults after restarting with the updated code; leave their
config unchanged to preserve the run's exact configuration check on resume.

Lookahead stores immutable uint8 pixels, bounded by the requested batch size times
`prefetch_batches`, plus active decoder working memory. At batch 64 and 128px RGB,
one full prefetched batch contains 3 MiB of pixels. There is no full-dataset RAM
cache or persistent disk cache. Prefetched files are read and hashed again when
consumed, so a changed file or symlink is still rejected. Lookahead stops at the epoch
boundary rather than speculatively drawing the next permutation; the first batch
and epoch transitions can therefore wait longer.

Sampler state records only consumed images. Restore discards speculative work;
worker completion order never affects batch order, pixels, labels, or RNG state.
Standalone callers can use `data.close()` to drain work and release threads; a
later call recreates the pool. Single-process training closes it at shutdown,
and discarded loaders release idle workers automatically.

## Skipping damaged training sources

The manifest-backed `hypergan.colorization_data:ColorizationData` loader supports
an opt-in policy for **shuffled training only**:

```toml
[data.args]
bad_image_policy = "skip"
max_bad_images = 100
max_consecutive_bad_images = 8
```

These are the default limits; the default policy remains `"error"`. Limits must
be positive integers. Missing, unreadable, changed-content, unsafe-path and
undecodable sources are excluded when encountered, with their path and reason
logged as a warning. Workers only report failures: the sampler consumes results
in order, records exclusions and draws subsequent usable entries to fill the
batch. Speculative failures do not exclude an image until it is requested.
Ordinary programming/runtime errors still propagate. Nothing is deleted or
rewritten, and no changed image is accepted under its original manifest hash.

Training stops on the **101st excluded image**, the **8th consecutive new source
failure**, or exhaustion of all usable entries, with the defaults above. A good
image resets the consecutive-failure counter. Fatal batches roll back sampler,
RNG and exclusions to the pre-call boundary; their warnings remain visible.

Checkpoint sampler schema 2 stores excluded indices, paths and reasons along
with the failure counter. Restoring preserves exclusions even if a source has
since been repaired. Legacy schema-1 checkpoints load with no exclusions.
Exclusions discovered after the latest durable checkpoint must be rediscovered
if the process crashes; exact continuation assumes the remaining sources have
not changed. The policy is recorded in the resolved run/checkpoint configuration.

An existing pinned training run can explicitly enable skipping by adding these
settings to its config and rerunning its launcher. This is a narrow allowed
resume change: manifest, preprocessing, model, optimizer and other recipe fields
must still match. Disabling skipping or changing its limits after enabling it
requires a new run. Healthy-source sampling and legacy fingerprints are unchanged.

Held-out and finite sequential evaluation remain strict. `ImageFolder` also
remains strict: it rebuilds its inventory during construction, so safely resuming
after deletion requires the separate pinned manifest supplied by
`ColorizationData`. Manifest preparation still validates sources and requires
explicit approval for initial rejections; this option handles later training
failures without a cleanup script.

## Exact sampling and resume

Shuffle uses only the caller-provided CPU `torch.Generator`. Each epoch is a permutation of the stable inventory. Batches crossing an epoch boundary wrap into the next epoch, so every batch has its requested size; there is no drop-last or replacement sampling within an epoch. Unshuffled mode repeats inventory order without consuming RNG. A fresh loader starts at epoch zero; the first generated permutation advances to epoch one.

Checkpoint both `data.state_dict()` and `generator.get_state()` at the same safe training boundary, then restore with `data.load_state_dict(...)` and `generator.set_state(...)`. The loader's JSON-compatible state contains its schema, identity hash, permutation, cursor and epoch. It does not own or serialize the generator. Invalid permutations and mismatched identities fail before changing the sampler. On a batch read/decode failure, both sampler and supplied RNG return to their pre-call state.

`resume_identity()` returns a JSON descriptor containing sorted relative filenames and SHA256 content hashes, original sizes/modes, the class map, discovery/shuffle options and preprocessing/decoder metadata. Instantiate a new loader before resume to re-inventory the dataset, then compare identities before restoring state. Each sampled file's hash is checked again during loading to catch edits after preflight. Additions require a new inventory; do not modify a dataset during a run.

The data identity omits the absolute root, so identical relocated folders have identical *data* identities. End-to-end exact resume also checks the resolved recipe, including its root argument; changing the configured root is not currently a supported relocation workflow. Changing bytes, filenames, labels, preprocessing or decoder version changes the identity and requires restoring the original dataset or starting a new run. The identity describes data compatibility, not permission to transfer old optimizer state to a different task.

The fixtures cover grayscale and rectangular pixels, aspect policies and EXIF orientation, named classes, corrupt inputs, content/class-map changes, optional dependency isolation, and bitwise sampler continuation across epoch boundaries. These establish the data contract on CPU; they do not establish a qualified image generator or distributed data sharding.
