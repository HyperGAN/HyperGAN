# Image previews and samples

Image recipes producing floating `[N, C, H, W]` tensors with one grayscale or
three RGB channels save PNG grids beside their periodic numeric previews.
The browser displays them through the authenticated artifact API. No viewer is
needed to save or inspect the files; metrics can be disabled.

```sh
hypergan train project/config.toml --run-dir runs/images --preview-every 100
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

Each periodic image generation contains `preview.json`, `grid.png`,
`manifest.json` and, for image recipes, `real.png`. The directory is published
atomically and then indexed. Generations are retained under the thinning policy
below, and a prune removes the whole generation.
Numeric JSON retains samples and gains a PNG descriptor recording size, SHA256
and dimensions. The same digest-checked file is shown in the browser. Final inference bundles, final JSON samples and
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

## Named samples and their history

Every sample carries a short stable name so a viewer can index it across steps
rather than by publication digest. The EMA generator output is `g` and the real
batch it is compared against is `x`; `--preview-name NAME` renames the generated
source for a run (1-16 letters, digits, dots, colons, underscores or hyphens)
and resume inherits it. The name is recorded in the preview payload, in each PNG
provenance chunk, in the retained `previews/index.json` entries (with the
retained `names` list) and in the public artifact records, beside the unchanged
digest artifact IDs.

Image recipes publish the real grid as `real.png` in the same atomic generation
as `grid.png`, so `x` and `g` share one retention decision and one sequence. It
is the comparable real batch actually used at that boundary, not regenerated or
rescaled data, and it is skipped when the batch is not a finite RGB/grayscale
image tensor.

A run retains at most **128** published previews by default, and the history it
keeps still spans the run: the browser slider always scrubs from the first sample
to the latest. When a run outgrows the bound, the older samples are *thinned*
rather than dropped from the beginning. Every other one goes, so a run publishing
a preview every 500 steps is left with one every 1,000, then every 2,000, and so
on. Thinning happens in one chunk, which drops about half of the older samples at
once and postpones the next prune by many publications instead of deleting a
generation on every publication. The first sample of the run is never pruned, the
latest never is, and the newest 16 sequences stay at full density. Thinning only
ever removes, so a sample that survives one prune is never reordered and a sample
that is gone never comes back.

The expired generations are deleted by a background worker: the index is rewritten
first, so the viewer's picture of the run (and its `preview_count`) updates to the
retained history immediately, and the directories are then moved aside and removed
without holding up the next publication or the training loop; anything a crash
leaves behind is swept by the next publication.

`--preview-keep N` asks for a different bound and `--preview-keep all` keeps every
generation for the life of the run, tensor payload and PNGs together. A resume
inherits a bound only when it was asked for explicitly: a run whose manifest
recorded the bound simply because it was the default of the day resumes into the
current default instead of being pinned to a stale one. The run manifest records
that choice as `preview_keep_source`, `"explicit"` or `"default"`; a manifest
written before this field existed counts as a default. `hypergan train --run-dir`
on an existing run directory behaves exactly like `hypergan resume`.

Retention is a count, not a disk quota: a retained generation is bounded by 2 MiB
of JSON plus its PNGs, and the periodic element budget below keeps ordinary grids
far smaller. A 100,000-step run at `--preview-every 500` publishes 200
generations; at `--preview-every 100` it publishes 1,000. Ask for a different
bound when grids are large or the disk is small.

`previews/index.json` lists every retained generation, records the requested
`keep` (`0` meaning every preview) and a `retention` of `all` or `thinned`, and
is rewritten on each publication from the records it already holds, so a long
history does not reread every generation manifest. The run manifest repeats only
the most recent 16 records plus a `preview_count`; the index is the whole
history. The viewer accepts an index of up to 4,096 generations.

The browser reads only indexed artifact IDs. PNG framing, dimensions, byte count
and digest are checked before inline delivery; authentication, path/link checks,
same-origin policy and `nosniff` remain in force. SVG/HTML and other media are
never rendered as images. The shelf renders at most 20 named groups, one image per
group, and loads images lazily. Each name shows its most recent version; earlier
retained versions are reached with that group's history slider (keyboard
supported), which shows the step of the version being viewed. The slider spans
the whole retained history, from the earliest sample to the latest. While it
shows the latest sample it follows new publications; once it is moved to an
earlier sample it holds that exact sample, matched by its step, even as newer
samples arrive and shift every slider position. The **Latest** button resumes
following. Unknown modalities remain downloads. There is no arbitrary file/image URL input.

## The raw tensor behind a picture

Every published generation carries the numbers as well as the picture: the JSON
tensor is the EMA generator output the grid was drawn from, at the same name and
step. The API keeps both records unchanged. In the viewer the picture comes
first: when a grid exists for that name and step, the tensor is a secondary
**Download raw tensor (JSON, shape ...)** action on the image card, with a line
saying what it is, rather than a second group beside the image.

Recipes that produce no images (numerical runs) keep a card of their own for the
tensor, labelled as the raw generator output for that step, where **Preview
numbers** shows the first values and Download saves the whole tensor. The sample
saved when a run finishes is marked `final` in its provenance and is named as the
final sample instead of appearing as one more anonymous tensor.
