# 256px logo colorization acceptance

The owner requested a learned grayscale encoder, 4,096 particle VAEGAN, a frozen
DINOv3 discriminator backbone with attention, multiple colorizations, independent
metrics, and a GPU-1 launcher left stopped for direct owner control. The
[recipe guide](../docs/colorization.md) describes the conditional adaptation and
its limits. No trained quality or distributed qualification is claimed.

Implementation is split between [PR #357](https://github.com/HyperGAN/HyperGAN/pull/357)
(bounded conditional image previews and routed particle provenance) and
[PR #358](https://github.com/HyperGAN/HyperGAN/pull/358) (the colorization recipe,
data, models and metrics). The source accepted by the installed run is
`165dea8e05f63394e52538620200b7379c805bf3`, built clean from an sdist-derived wheel.
All 65 installed runtime Python files byte-matched that source. After CI caught
a preview byte-limit default captured at function definition, the limit now
resolves at publication time (`377974f8` / `49f206b5`); all 42 focused preview
checks passed. The launcher was rebuilt and updated to clean combined source
`903393713d592059cdcbc924f53a0374537cf426`, again matching all 65 runtime files.
That head also incorporates the owner's develop heavy-test selection (`6ed2eafb`). The existing local
develop history at `a83d7072` was pushed under owner authorization; prior
performance PRs #355/#356 are consequently merged.

## Integration

The combined head `5a34cc76` passed all required checks, including Windows and
Foundation CI, in [run 35561553137](https://github.com/HyperGAN/HyperGAN/actions/runs/35561553137).
PR #358 merged as `48e0b34648d732d57bca1cdf5feeb3430515f352`; #357 and
[#359](https://github.com/HyperGAN/HyperGAN/pull/359) are also merged by ancestry.
The final policy runs heavy GitHub jobs only for master pushes and PRs targeting
master, as subsequently requested by the owner. Both jobs were intentionally
omitted for this develop PR; fast foundation/reference, viewer, reducer and
repository integrity checks passed. The standalone #359 run hit the existing
Windows CRLF fixture failure; the passing combined head includes its #357 fix.
Earlier superseded heavy CI runs were canceled under the new policy.

## Pinned artifacts and preparation

- DINOv3 source: `6876159a11b4df116f30f667f8c9888617df0751`.
- Owner-supplied `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` SHA256:
  `08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d`.
- Dataset root: `/mnt/ml7tb/data/logos256`.
- Prepared manifest SHA256:
  `e336ddec33364aa9300108fa9566f51a1e321d632067f553ffb97f410e3fbdb0`.
- Reviewed exclusion report SHA256:
  `e551097f63e998be73bd0bfb52af4fd2de810f40359e00175a982e7da3088282`.

The census attempted all 426,445 image paths. It accepted **426,343** images:
**404,757 training / 21,586 held-out**. Exactly **102** exclusions are recorded
with source hashes and reasons: 99 truncated, two unreadable, one animated.
Three valid one-bit PNGs initially encountered an unsupported-mode error; support
was added and those exact files were revalidated and retained. Initial and final
reports remain preserved. The `apple.tgz` archive is explicitly inventoried as
a non-image and is not expanded. No source images were changed or deleted.

## Verification

- Final combined installed fast suite: **851 passed, 185 deselected in 24.46s**.
  The deselected tests are the owner's explicit `heavy` selection, run separately
  locally as needed. Owner subsequently requested GitHub heavy jobs only for
  master pushes and PRs targeting master (`41f3cedb`). All 36 final-gate result
  combinations passed validation. Develop keeps fast, viewer, reducer and
  integrity checks; no heavy-suite pass is claimed for this combined head.

- 58 focused configuration/model/data/metric tests passed; 34 installed focused
  tests passed, then the final mode-1 data/metric suite passed 12 checks.
- Preview/API/browser validation passed 67 tests initially; one server test
  exposed missing web dependencies under isolated Python in the new environment.
  Installing the declared web extras resolved it and that exact test passed.
- Actual pinned DINO CPU and physical GPU-1 forward/image-double-backward checks
  passed. Backbone parameters stay frozen and particle gradients are nonzero.
- A real-logo fixture public CLI run trained through lazy b-cap at step 8,
  resumed to step 10, and recovered from an older step-4 snapshot to step 5.
  Exact restoration compared 1,050 tensors and 2,717 values.
- The **actual start-color.sh** launcher with the full inventory trained to
  step 8 and resumed to step 10 in a separate test directory. Complete restored
  state matched **1,050 tensors and 407,414 values**, including model, optimizer,
  EMA, sampler, last batch and RNG state. The fast backend does not promise
  bitwise-identical future training trajectories.
- All three full **512-output** held-out snapshot evaluations completed at
  source step 10: chroma distribution, grayscale edges, and four-draw chroma
  diversity. Their scores are plumbing evidence from an untrained model, not
  quality claims. The evaluation protocol records the hash-ordered held-out
  subset and repeated-condition grouping.
- Actual Chromium inspection of the full run confirmed rendered 768×768
  `g`/`x`/`gray` grids and no JavaScript or artifact-route errors. The UI identifies
  the retained preview's source step 4 separately from current run step 10.

Commands included `python -m build --sdist`, `pip wheel --no-deps`, installed
`python -m pytest`, bounded public `train`, repeated `train`, earlier-checkpoint
`resume`, all three `evaluate` calls, `stop-server`, source/installed byte
comparison and exact complete-state restoration. All CUDA commands mask only
GPU UUID `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce` (physical card 1).

Evidence, exact harness commands, failed setup logs, successful receipts, built
packages, screenshots and stopped test runs are under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-colorization/`.

## Owner handoff

`~/dev/hypergan/training-runs/start-color.sh` runs the pinned installed package
in `colorization-env`, reads `logos-colorization-256/colorization.toml`, and
creates/resumes `train-color`. Defaults: batch 16, 200,000 total updates,
1,000-update checkpoints and 100-update previews. GPU 1 is selected by UUID.
The owner run remains fresh. The verification run and its viewer are stopped.
The neighboring `README-color.md` gives the local commands and inventory details.

The remaining research step belongs to the owner's controlled training session:
inspect shape retention, particle use and repeated-color diversity while tracking
the independent held-out metrics. The hard particle bottleneck can lose detail;
changing the model is a new configuration/run decision. No paid compute or
release was launched.
