# Metrics observation performance qualification, 2026-09-19

Default metrics and five simultaneous live API consumers preserved the complete
saved numerical state in every CUDA trial. The proposed 1% default-publication
and 2% five-consumer overhead targets are **not established** by this experiment:
the paired estimates were +3.74% and +2.05%, with wide intervals on a shared,
changing host. These are measured limitations, not a performance acceptance claim.

## Reproducible CUDA experiment

The shipped `scripts/metrics_training_proof.py` uses an installed HyperGAN package,
an explicitly selected local CUDA device, and independent training, projection,
HTTP server and consumer processes. It makes no dataset download or cloud request.
Its private child commands use Python isolated mode. Install `[train,web]` and an
appropriate CUDA-enabled torch build first; then, from an unpacked source archive:

```sh
python scripts/metrics_training_proof.py run --device cuda:1 \
  --steps 1088 --warmup 64 --repetitions 4 --output /tmp/cuda-observation-proof.json
```

The output JSON records the exact installed Python source hashes, runtime versions,
each trial, GPU telemetry, full-state digests, consumer continuity and comparisons.
Its adjacent log directory records the effective recipe and subprocess logs. Use a
new output basename for another run because existing log directories are rejected.
Owned subprocesses are terminated and reaped in `finally`; temporary training runs
and checkpoint payloads are removed. This is an explicit qualification command,
not a CUDA job executed by normal package installation or CPU CI.

This run used source revision `460caef743618990d6c8c657a710444fac10abeb`, installed
into `/tmp/hypergan-observation-perf-verify`; Python 3.12.13, torch 2.14.0+cu130,
CUDA 13.0, NumPy 2.5.2, and one RTX A6000 (`cuda:1`). The fixture uses the default
particle recipe with two 512-unit hidden layers, batch size 256, 4,096 particles,
the ordinary b-cap/VIC objectives, and 1,088 updates. It has no periodic previews
or intermediate checkpoints. The last 1,024 updates are measured after 64 warmup
updates, using existing completed-update event times; no benchmark callback or
per-step profiler was added to the training loop. Initialization and final export
are excluded. Four counterbalanced blocks contain all four conditions:

| Condition | Active observation processes | Median updates/s | Trial range |
| --- | --- | ---: | ---: |
| `none` | `[metrics] preset = "none"` | 88.61 | 80.68–93.20 |
| `metrics` | Default metrics, no server/projector | 85.43 | 77.56–90.00 |
| `server_zero` | Defaults, projector, server, zero consumers | 90.20 | 85.66–91.97 |
| `server_five` | Defaults, projector, server, five HTTP SSE consumers | 87.88 | 84.98–90.04 |

The five consumers connect before training, discover its projection stream through
the actual authenticated server, and validate every projection sequence. Every
consumer received 1,090 contiguous frames through step 1,088 in each of the four
trials. This exercises steady live forwarding and does not add cold history
reduction or graphical browser rendering to each training trial. The standalone
history experiment below measures those server-side history costs separately.

| Paired elapsed-time comparison | Geometric mean overhead | Two-sided 95% interval | Candidate target |
| --- | ---: | ---: | ---: |
| Default metrics / none | +3.74% | −4.15% to +12.27% | ≤1%, not established |
| Five consumers / zero consumers | +2.05% | −2.43% to +6.74% | ≤2%, not established |
| Five consumers / metrics without server | −3.66% | −13.61% to +7.44% | ≤2%, not established |

Intervals use the Student-t distribution on four paired log elapsed ratios. They
describe this small experiment, not all workloads or an equivalence guarantee.
Negative estimates are not evidence of an observation speedup. Unrelated GPU0
work ended during the experiment (37% utilization/1,250 MiB to 0%/15 MiB); GPU1
also served the desktop, with reported utilization changing 41% to 21% and SM
clock 1,635 to 1,755 MHz. Other repository/build work continued. No external jobs
were stopped. More repetitions on a quiescent host, with sustained larger workloads,
are needed to resolve overhead near one percent.

All 16 checkpoints had identical type/shape/byte hashes covering every saved
tensor and scalar, including model, EMA, optimizer, prior, RNG and data state:
`08e5a962a98c4f7647bac988076a9a7a3dabd6406965581a9fdeeaa2bc6e9490`.
Observed, durable and checkpoint steps were 1,088. This qualifies numerical
noninterference for this fixture. It does not replace image/audio, multihost or
two-GPU qualification. The coordinator's M5 proof separately exercises the product
CLI's automatic server lifecycle; this script deliberately uses explicit processes.

## Million-event proof with portable source cursors

The standalone server was installed into a separate torch-free web environment
`/tmp/hypergan-server-final-proof-verify` (Python 3.12.13), from the source containing
`0e0d8b6d`. The final installed core-file hashes were independently compared against
the standalone working tree after its integration merge `a6e1f10b`: all match.
Browser-only commits arriving during this proof do not change the measured files.

```sh
python scripts/metrics_server_proof.py --events 1000000 \
  --output /home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/million-event-portable-server-proof.json
```

| Measurement | Result |
| --- | ---: |
| Source/projection background indexing | 66.379 seconds |
| First historical envelope bootstrap | 67.557 seconds |
| Warm compatible query p95, including JSON serialization | 2.288 milliseconds |
| Bootstrap payload / grouped states | 326,427 bytes / 513 |
| One frame delivered to five subscriber queues | 19.556 milliseconds |
| Live server reducer calls | 0, asserted |
| Peak process RSS | 217.53 MiB |
| Source / projection bytes | 263,668,368 / 841,142,342 |
| Fixture generation | 32.765 seconds |

The first full page incurs indexing plus reduction, about 134 seconds here; warm
compatible requests reuse bounded historical state. The one-frame queue result
is **not browser-render latency**, a latency percentile, or network transit time.
Neither this proof nor the CUDA experiment measures end-to-end browser rendering.
The 180-second historical job budget completed this fixture. It does not promise
cheap cold queries or arbitrary history sizes. The earlier report's source-hashed
Python 3.14 measurements remain historical evidence; runtime and host differences
prevent attributing this rerun's timing difference solely to code changes.

Installed core SHA256 values:

| Module | SHA256 |
| --- | --- |
| `hypergan.web_service` | `63961474a3d3222a1d987699795949994c3f0f1a0f16e137f2cb9926021c0b63` |
| `hypergan.event_views` | `4aaa42f287271f59088bb974715c0e35a85dc8e2e2dc8548fa245468c18c5de6` |
| `hypergan.run_events` | `7254b121871febaf7241ff5b9f3b2829a17e1563fc620ef7a0eb48e56d4ddda4` |
| Shared reducer WASM | `727faa849990730404dd6ca1f898526d380b56bd4ad9384f9186a2bc4d7b275d` |

## Durable evidence and packaging

Receipts and adjacent subprocess log directories are retained outside the checkout
under `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation`:

| Receipt | SHA256 |
| --- | --- |
| `cuda-observation-smoke.json` (eight short trials) | `0fa51337ff305b76c78743f0d972bf1208d8142beecf9b68fdc292892ab35f1c` |
| `cuda-observation-throughput.json` (sixteen full trials) | `0c4c0a521c8c8b11bb3d9aa9e589862760be8ff4a6ca32680586cb7241f26d6e` |
| `million-event-portable-server-proof.json` | `28a9a7f9b48062421b6937e7ca7255dcaf0927ba1c4825681048be7f6299ba31` |

`python -m build --sdist --outdir /tmp/hypergan-observation-proof-dist` succeeded.
The archive was inspected and contains `scripts/metrics_training_proof.py`,
`scripts/metrics_server_proof.py`, `scripts/build_reducer.py` and
`scripts/reducer_proof.py`. The new harness is explicitly included in `MANIFEST.in`.
All owned worker/server/trainer processes were reaped and both GPUs released before
handoff. No core runtime edits were made as part of this measurement task.
