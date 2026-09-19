# Metrics backbone implementation — 2026-09-19

The [accepted research](metrics-first-research-2026-09-19.md) now has a working file-centric implementation: completed observations in strict JSONL, immutable metric definitions, explicit Python maps, bounded contribution streams, and one shared Rust/WASM reducer for server history and browser live continuation. No database is required. Live HTTP/SSE fanout never reduces metrics. Samples are separate modality-neutral artifacts, while manual evaluations have independent immutable streams and provenance.

## Use it

From the develop checkout, install the optional viewer alongside the appropriate training runtime:

```sh
python -m pip install '.[train,web]'
hypergan new demo                        # CUDA is the default
hypergan train demo --run-dir runs/demo --open
hypergan serve runs/demo --open          # inspect after training ends
hypergan metrics runs/demo
hypergan contributions runs/demo --limit 100
```

The CLI starts a loopback viewer and independent projection process when `web` is installed. Browser opening is explicit. `--server` requires startup before numerical work; `--no-server` avoids serving imports and listening sockets. The Python training API stays headless. Startup prints a private credential-file location; paste its token into the sign-in form, or use the same public API with a Bearer token. Credentials never enter run artifacts or URLs. Automatic viewing ends with its CLI command; standalone `serve` works independently. `project RUN [--follow]` explicitly catches up incomplete projection history.

Configuration defaults are individually removable, including `[metrics] preset = "none"`. Ordinary Python scalar factories bind approved detached update values with explicit cadence and failure policy. `hypergan evaluate RUN --metric ID` explicitly evaluates a configured manual snapshot metric; it pins EMA weights, its own dataset and RNG, sample count and protocol. The UI shows scalar/histogram results, failures, provenance, accessible values and raw exports. See [configuration](../docs/configuration.md), [maps and views](../docs/event-views.md), and [API/viewer usage](../docs/local-web.md).

## Integration slices and evidence

| PR | Slice | Evidence |
| --- | --- | --- |
| [#320](https://github.com/HyperGAN/HyperGAN/pull/320) | Shared reducer and Python/browser hosts | [Reducer proof](metrics-shared-reducer-2026-09-19.md) |
| [#321](https://github.com/HyperGAN/HyperGAN/pull/321) | Removable defaults, named contributions, runtime and recovery | [Publication acceptance](metrics-publication-2026-09-19.md) |
| [#322](https://github.com/HyperGAN/HyperGAN/pull/322) | Python event maps and replayable contributions | [File-view acceptance](metrics-event-views-2026-09-19.md) |
| [#323](https://github.com/HyperGAN/HyperGAN/pull/323) | Standalone API, streaming and browser | [Viewer acceptance](metrics-web-2026-09-19.md) |
| [#325](https://github.com/HyperGAN/HyperGAN/pull/325) | Custom scalar/manual snapshot evaluation and UI | [Evaluation acceptance](metrics-custom-evaluation-2026-09-19.md) |
| [#324](https://github.com/HyperGAN/HyperGAN/pull/324) | Supervised automatic CLI startup and integrated qualification | [Startup proof](metrics-autostart-proof-2026-09-19.md) |

All slices integrate through protected PRs targeting develop; each exact reviewed head requires platform CI before merge. The durable receipt directory is `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/`; it retains package hashes, commands, failed checks and fixes, PR/merge receipts, complete GPU comparison artifacts and measured source identities. Installed-package checks rebuild wheels through source distributions. Actual Chromium tests exercise shared reduction, reconnect/acknowledgement, recovery lineage, artifact/evaluation discovery and the browser's public API.

Review corrected source-boundary skipping, public cursor portability, stale history cache handling, Windows indexed paths, startup subscription ordering, bounded gap replay, redundant artifact refreshes and test-process contamination. Gaps preserve acknowledged reducer state and reconnect in bounded segments; changed history/generation explicitly resets. Slow clients cannot block the trainer. Default metrics and custom metrics preserve mandatory numerical/finite checks, even when no values are published.

Final installed integration at `a9df1100` passed **423 tests, zero failures or skips**, in 170.12 seconds, followed by a real default-auto train/resume → earlier-snapshot scalar/histogram evaluation → authenticated standalone API walkthrough. The final frontend change at `1774c01e` was rebuilt through a fresh source distribution and passed **15 actual browser tests** in 24.39 seconds, including a visible single-point canvas mark. All 44 Python modules and the WASM bytes were unchanged between these proofs. Offline JavaScript rebuilt exactly (542,872 bytes); the pinned WASM rebuild matched. The final wheel SHA256 is `112c175fbc03c78d1a61402fa119041014f298141aad098773c72ed307f78e6a`. The durable `complete-installed-acceptance.json` receipt records full commands and both artifact sets.

Actual native CUDA and accumulated two-GPU comparisons qualify metrics on/off and recovery. The installed CUDA CLI proof compares headless training against viewer-enabled stop/resume, including 71 tensors and 1,370 values across all checkpoint state sections, samples, numerical metrics, CUDA RNG, child cleanup and credential cleanup. Sixteen longer CUDA observation trials also preserved identical full state with zero/five API consumers.

## Measured limits

The [performance report](metrics-observation-performance-2026-09-19.md) gives the workload, repetitions, actual runtime/source hashes and noise. On its million-update fixture, the portable-source server used **217.53 MiB** peak RSS; warm query p95 was **2.29 ms** and five-subscriber queue delivery **19.56 ms**, with **zero live reductions**. Cold indexing took **66.38 seconds**, followed by **67.56 seconds** historical reduction. Queue delivery is not browser-render latency. Historical jobs are bounded at 180 seconds by default and share compatible fixed-watermark results.

The candidate default-metric 1% and five-viewer 2% throughput targets are **not established**. Paired estimates were +3.74% and +2.05%, respectively, with wide intervals that included zero; external GPU activity and clock changes confounded small overhead estimates. This implementation makes no sub-percent performance guarantee. Custom scalar workers have explicit startup cost (about 194 ms median in the recorded lightweight fixture), so their cadence matters.

V1 supports scalar training views, manual scalar/fixed-histogram evaluations, generic artifact downloads and bounded numeric tensor previews. Automatic snapshot scheduling, built-in FID/weight provisioning, audio/image renderers, WebSocket transport, federation, a database and real multi-host qualification remain future work. Unsupported schedules fail explicitly. No implicit dataset/weight downloads, paid compute or release publishing occurred.
