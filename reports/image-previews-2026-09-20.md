# Bounded PNG image previews and sampling

This is the rendering slice of [I3](image-training-plan-2026-09-19.md), alongside
the separate recipe and numerical work. Floating NCHW RGB/grayscale outputs now
produce real PNG grids in managed preview history. The browser displays them
through the existing authenticated, indexed artifact API. Headless execution and
disabled metrics retain saved previews. `hypergan sample RUN --output fresh.png`
loads the EMA inference bundle in a fresh CPU process and creates a bounded grid.
Generic tensor JSON remains supported and existing final sample JSON is unchanged.

The [user guide](../docs/image-previews.md) records commands, provenance, pixel
conversion, fixed-seed behavior and limits. Preview directories atomically publish
PNG and JSON together and are pruned together; final bundles/samples remain
immutable. Encoding uses the standard library with no Pillow/Torch import in the
viewer. PNG validation rejects non-PNG/oversized/corrupt framing before inline
delivery, preserves digest and path checks, and keeps CSP/authentication/nosniff.
No SVG, arbitrary URL loading or executable artifact renderer was added.

## Validation

A fresh wheel built through its source distribution was installed into
`/tmp/hypergan-image-preview-verify`, an independent CPU-only environment. Focused
tests exercise exact RGB/grayscale pixels and rounding, black padding, element/
count/dimension/provenance bounds, nonfinite rejection, atomic publication and
retention, malformed metadata cleanup, existing-output refusal, and fresh seed
sampling. A real isolated renderer captures/transports/publishes image previews
and preserves the parent RNG/model/prior/Adam state.

The installed CPU image fixture trains with metrics disabled and periodic PNGs,
stops, resumes, and compares complete checkpoint state exactly with an unobserved
run. Its fresh-interpreter CLI sample opens as PNG with the expected dimensions,
seed, step and bundle digest. This is a small correctness fixture, not the
selected architecture or an image-quality result.

The real ASGI/Chromium test checks authenticated PNG pixels via Canvas, receives
a later image through artifact notifications while no metrics are selected,
retains history and downloads exactly the indexed bytes. Initial test-script
evaluation encountered the strict CSP; expressing the polling predicate as a
function fixed the test without relaxing the product CSP. Source-only setup
probes also exposed missing fixture resume declarations and uninstalled package
paths; the final qualification uses an installed package outside the checkout.

Broader installed foundation/preview/recovery/web/browser regression results and
final PR/CI receipt will be recorded after completion. Durable local evidence is
under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-image-previews/`.
The frontend bundle rebuild/check is byte-for-byte reproducible with the pinned
offline npm dependencies. No GPU, paid compute, downloads of data/weights or
release publication were used for this slice.

## Integration

The independent I2 component-graph change supplies prior bindings and component
aliases to inference. Its `artifacts.py`, `preview_snapshot.py` and
`previews.py` traversal/call changes must be retained when integrating these
nonoverlapping PNG additions. The numerical/configuration/trainer modules were
not changed here. The full selected-recipe walkthrough remains the coordinator's
integration gate; this slice does not claim the entire I3 or image plan complete.
