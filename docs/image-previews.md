# Image previews and samples

Image recipes producing floating `[N, C, H, W]` tensors with one grayscale or
three RGB channels save PNG grids beside their periodic numeric previews.
The browser displays them through the authenticated artifact API. No viewer is
needed to save or inspect the files; metrics can be disabled.

```sh
hypergan train project/config.toml --run-dir runs/images --preview-every 100 --preview-keep 4
hypergan resume runs/images --no-server
hypergan sample runs/images --count 16 --seed 42 --output samples.png
hypergan sample runs/images --count 16 --seed 73 --output fresh-samples.png
```

Use a configured image recipe and prepared data. These commands do not download
weights/data or qualify image quality. `--preview-every` explicitly enables
periodic previews; resume inherits its schedule and retention unless overridden.
`--no-server` keeps training headless. A sample command loads the saved EMA
generator and prior on CPU, independently of training and metrics. It works in a
fresh process and never overwrites an existing output. Select `.png` explicitly
for a grid; the default and other outputs retain generic tensor JSON. PNG
requests for non-image tensors fail with an actionable shape error.

Each periodic image generation contains `preview.json`, `grid.png` and
`manifest.json`. The directory is published atomically, then indexed, and
retention removes the whole generation. Numeric JSON retains samples and gains
a PNG descriptor recording size, SHA256 and dimensions. The same digest-checked
file is shown in the browser. Final inference bundles, final JSON samples and
explicitly requested sample files are not pruned by preview retention. Preview
filenames retain run/attempt/sequence identity across resume.

Previews use `sampling.seed` at each boundary for comparable particle selections
and random draws. Learned prior means/noise state and generator weights still
evolve; learned latent vectors are not held constant. Another `--seed` provides
fresh draws. Conditional previews record inputs from the last completed batch;
standalone samples record whether conditions were supplied or cycled from saved
examples.

Conversion clamps to `[-1,1]`, rounds `(x+1)*127.5` to uint8, and tiles images in
row-major order with `ceil(sqrt(N))` columns. Empty final cells are black; no
per-image contrast scaling occurs. PNGs embed bounded JSON provenance in a
`hypergan` text chunk: step, seed, identity, shape and grid layout. Standalone
PNGs also record the inference-bundle digest and input shapes. Generic JSON
remains the format for recording complete conditional input values.

Periodic previews retain limits of 16 samples, 65,536 output/input elements and
2 MiB JSON/renderer transport. Standalone PNG sampling permits at most 64 images
and 4,194,304 tensor elements. Every grid is limited to 4,096 pixels per side,
4,194,304 canvas pixels, 8 MiB PNG and 64 KiB provenance. Bounds reject oversized
outputs; they do not sandbox arbitrary allocations in trusted custom generators.
Rendering keeps copied EMA state and isolated RNG; the replicated renderer
remains separately supervised.

The browser reads only indexed artifact IDs. PNG framing, dimensions, byte count
and digest are checked before inline delivery; authentication, path/link checks,
same-origin policy and `nosniff` remain in force. SVG/HTML and other media are
never rendered as images. The shelf renders at most 20 entries and loads images
lazily. Older retained grids remain available through the bounded index; unknown
modalities remain downloads. There is no arbitrary file/image URL input.
