# Train on Your Own Images

This guide walks through training HyperGAN on custom image collections, from raw photo folders and sprite sheets to transparent PNG datasets, paired colorization tasks, and multi-GPU production runs.

---

## 1. Quickstart: Auto-Scaffolding a Project

HyperGAN can inspect your image directory and scaffold an end-to-end project configured with appropriate image dimensions, channel counts, and architecture scales:

```sh
hypergan new --project my_art --dataset /path/to/my/images
```

The scaffolding engine performs header inspection on candidate images across the target directory, extracting spatial dimensions (`width`, `height`), channel depth (`channels = 3` for RGB or `channels = 1` for grayscale), and color modes without loading entire image arrays into system RAM.

The generated project structure:
```text
my_art/
├── configs/
│   └── dcgan.toml       # Canonical model file with tuned architecture and cadences
├── data -> /path/to/my/images
└── runs/                # Isolated run directories for checkpoints and previews
```

You can start training immediately:
```sh
hypergan train my_art/configs/dcgan.toml
```

---

## 2. Dataset Configuration (`builtin = "image_folder"`)

For custom image datasets, define the `[data]` section of your model file using `builtin = "image_folder"`:

```toml
[data]
builtin = "image_folder"
root = "./data/images"
width = 128
height = 128
channels = 3
mode = "RGB"
resize = "center_crop"
alpha = "composite_white"
follow_symlinks = true
shuffle = true
workers = 4
batch_size = 32
```

### Configuration Options Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`root`** | `string` | *(required)* | Path to directory containing training images (searched recursively). |
| **`width`** | `int` | *(required)* | Target width in pixels for neural network input. |
| **`height`** | `int` | *(required)* | Target height in pixels for neural network input. |
| **`channels`** | `int` | `3` | Number of image channels: `3` (RGB) or `1` (Grayscale / L). |
| **`mode`** | `string` | `"RGB"` | Pillow color mode: `"RGB"` or `"L"`. |
| **`resize`** | `string` | `"center_crop"` | Resizing strategy: `"center_crop"`, `"pad"`, `"stretch"`, or `"none"`. |
| **`fill`** | `int` or `list` | `0` | Background fill pixel value used when `resize = "pad"` (e.g. `0` for black, `255` for white). |
| **`interpolation`**| `string` | `"bilinear"` | Resampling filter: `"bilinear"`, `"bicubic"`, `"nearest"`, or `"lanczos"`. |
| **`alpha`** | `string` | `"composite_white"`| Handling transparent alpha channels: `"composite_white"`, `"composite_black"`, or `"error"`. |
| **`follow_symlinks`**| `bool` | `true` | Whether directory recursion follows filesystem symbolic links. |
| **`shuffle`** | `bool` | `true` | Shuffles dataset order each epoch. |
| **`workers`** | `int` | `4` | Number of PyTorch DataLoader worker subprocesses. |

---

## 3. Handling Alpha Channels, Palettes & Aspect Ratios

Real-world web scrapes and game sprite datasets frequently contain 4-channel RGBA images, paletted PNGs (mode `P`), or varying aspect ratios. HyperGAN provides native pre-processing controls to standardize inputs.

### 3.1 Transparent PNGs and Alpha Compositing

If your training images have transparent backgrounds (such as icons, logos, or sprites), training directly on raw RGBA files causes black boundary artifacts. Configure `alpha` compositing:

- **`alpha = "composite_white"`**: Composites translucent and transparent pixels over a pure white ($[255, 255, 255]$) canvas before normalisation. Recommended for logos, documents, and light-background illustrations.
- **`alpha = "composite_black"`**: Composites alpha over a black ($[0, 0, 0]$) canvas. Recommended for dark-background assets and space/astrophotography imagery.
- **`alpha = "error"`**: Raises a strict `ValueError` immediately upon encountering any image with an alpha or transparency channel, enforcing clean input pipelines.

### 3.2 Aspect Ratio Preservation (`resize = "pad"`)

When training on images with diverse aspect ratios (e.g. portrait photos mixed with landscape vistas), naive resizing distorts geometry:

- **`resize = "center_crop"`**: Scales the shorter dimension to target size and crops equal margins from the longer dimension. Preserves aspect ratio and natural textures, but discards outer borders.
- **`resize = "pad"`**: Scales the longer dimension to target size and letterboxes/pillarboxes the remaining margins using the `fill` color. Preserves 100% of the image content.
- **`resize = "stretch"`**: Resizes both axes independently to `(width, height)`. Useful only when training on intrinsically square content.

---

## 4. Preflight Dataset Verification (`hypergan data-check`)

Before starting long training jobs, verify your dataset integrity, decode speeds, and channel consistency:

```sh
hypergan data-check configs/dcgan.toml
```

To emit machine-readable JSON:
```sh
hypergan data-check configs/dcgan.toml --json
```

The data check tool:
1. Recursively traverses `root`, validating supported extensions (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`).
2. Validates image decodability and checks for corrupt headers or truncated byte streams.
3. Computes a deterministic SHA-256 dataset inventory fingerprint.
4. Reports total sample count, min/max dimensions, and estimated PyTorch batch load latency.

---

## 5. Colorization & Paired Workflows

For image-to-image translation and colorization (translating grayscale $L$ inputs into $RGB$ color), use `builtin = "image_folder"` with dual bindings:

```toml
[data]
builtin = "image_folder"
root = "./data/photos"
width = 128
height = 128
channels = 3
mode = "RGB"

[networks.generator]
role = "generator"
inputs = { gray = "batch.gray" }
source = """
# Generator receives single-channel grayscale input: ['B', 1, 128, 128]
conv(64, kernel=3, padding=1)
relu()
resblock(128, stride=1)
conv(3, kernel=3, padding=1)
tanh()
"""
```

Grayscale luminance is derived following ITU-R BT.601 standard:
$$Y = 0.299\,R + 0.587\,G + 0.114\,B$$

---

## 6. Multi-GPU & Performance Guidelines

When scaling image training across multiple GPUs or high-resolution images:

1. **DataLoader Workers**: Set `workers = 4` per GPU rank. If training stalls waiting for data, increase `workers = 8` and ensure images reside on local NVMe storage rather than network filesystems.
2. **Resolution & Batch Sizes**:
   - $128 \times 128$: Batch size $32$ to $64$ per GPU.
   - $256 \times 256$: Batch size $16$ to $32$ per GPU.
   - $512 \times 512$: Batch size $4$ to $8$ per GPU with gradient accumulation (`accumulation_steps = 2` or `4`).
3. **Normalization Stability**: Prefer `group_norm(num_groups=32)` or `rms_norm()` in HNDL image networks. GroupNorm and RMSNorm calculate statistics per sample or group, remaining perfectly invariant across distributed GPU rank counts.
