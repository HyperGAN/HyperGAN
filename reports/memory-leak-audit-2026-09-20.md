# Memory leak and startup audit — 2026-09-20

Audited `/home/martyn/dev/hypergan/training-runs/start.sh`, its CIFAR configuration, native training, snapshots, evaluation workers, and the `--server --dev` viewer. Three subagents independently reviewed training, data/evaluation, and viewer lifetimes. Repository started at `4d143c87`; concurrent work advanced develop during the audit.

**Result:** reproduced and fixed the startup blocker. No unbounded memory retention was demonstrated in the inspected paths or bounded CUDA probe. This is not a long-duration leak clearance: full FID50k evaluation and a multi-hour browser/training soak were not run.

## Startup findings and repairs

### High: automatic viewer rejects a valid log after filesystem device renumbering

The existing viewer failed in `Projector.__enter__` with:

```text
ValueError: Stale event cursor: the log was replaced; restart without a cursor
```

Evidence from the actual `train-develop` run:

- Saved source identity: device **54**, inode **9846554**.
- Current identity: device **55**, inode **9846554**.
- Saved cursor offset and current log length both **133,698,621 bytes**.
- Saved boundary SHA256 matches the current bytes.
- The machine had rebooted approximately five minutes before inspection. These observations are consistent with device renumbering on remount, rather than changed log contents.
- `hypergan server-status train-develop` reported `Viewer process exited: projector`. The traceback is in `/tmp/hypergan-viewers-1000/bbe2a92f054e0f092b9a1de3fe2a8d594ce403689f61753f1bd9a62bd4d830d5/viewer.log`.

Because the launcher uses `--server`, viewer failure prevents training startup. Read-only `prepare_train` successfully selected the complete checkpoint at step **106000**, so checkpoint selection was not the immediate blocker.

**Fix:** [event_views.py](../src/hypergan/event_views.py) adds opt-in device-only cursor rebinding; [web_autostart.py](../src/hypergan/web_autostart.py) enables it for the automatic viewer. Inode changes still fail. Directory identity, offset, boundary checksum and ordinary cursor validation remain enforced. Generic `Projector` and `read_event_page` retain their strict default. This is bounded validation, not full-log integrity verification. Existing projection bytes and generation are preserved; no history is rebuilt or deleted.

Regression tests cover device-only recovery, subsequent ordinary restart, and rejection of changed inode, changed boundary bytes, and truncation.

### Medium: launcher depends on the caller's directory and cannot execute directly

The original two-line script had mode `0644`, no shebang, and relative configuration/run paths. Direct execution fails on permissions, and running it through a shell outside `training-runs` selects incorrect paths.

**Fix applied to the actual external script:** added a Bash shebang, executable permissions, strict shell handling, paths relative to the script directory, `exec`, and forwarded arguments. GPU 0 remains the default; an existing nonempty `CUDA_VISIBLE_DEVICES` override is honored. The extra arguments permit bounded startup checks.

The external directory is not a Git repository. A [copy of the repaired launcher](memory-leak-audit-2026-09-20/start.sh) is committed with this report; its original backup is `/tmp/hypergan-start-before-audit-2026-09-20.sh`. The report copy documents the external script and is not intended to run from the report directory.

## Memory findings

| Area | Evidence | Assessment |
| --- | --- | --- |
| Training loop | 128 exact-config CUDA updates: live allocated memory **259,460,608 bytes** at updates 32, 64, 96 and 128. Live host RSS changed from 2,322,332 to 2,322,344 KiB. | No retained growth demonstrated after warmup. |
| CUDA allocator | Reserved memory stayed **2,004,877,312 bytes**; maximum live training allocation was **1,943,059,968 bytes**. | Stable reserved memory is distinct from live tensor retention. |
| Preview capture | Three captures each returned to the same live CUDA baseline; transient increase **8,117,248 bytes**. | Bounded copies, not a demonstrated leak. |
| Evaluation snapshot capture | Three captures each returned to baseline; transient increase **24,452,096 bytes**. | Capture only; does not measure the FID worker itself. |
| CIFAR dataset | One retained uint8 image tensor plus labels: **154,000,000 bytes** (~146.87 MiB). A 1,000-batch CPU probe retained the same image storage. | Fixed resident dataset. This config moves training data onto CUDA. |
| Streaming FID | Two 2048-dimensional float64 moment accumulators retain about **64.03 MiB** on CPU. Samples are processed in batches. | Bounded moments plus temporary covariance/workspace allocations. |
| Viewer | Subscriber count/queues, caches, stream/group/lineage indices are capped; projection catalog cache evicts beyond 16 entries. | No obvious unbounded retention in inspected paths. |

Raw measurements and reproducible probe: [training-memory-results.json](memory-leak-audit-2026-09-20/training-memory-results.json), [training-memory-probe.py](memory-leak-audit-2026-09-20/training-memory-probe.py). The probe exited **0** in approximately 23 seconds. It calls garbage collection at observation points and executes training in memory without modifying the saved run. It does not exercise checkpoint serialization, worker rendering, browser sessions, or FID computation.

Relevant implementation references:

- `src/hypergan/preview_snapshot.py:80–91`: deep-copies selected EMA models/prior before freezing CPU state. Temporary CUDA copies are measurable and released.
- `src/hypergan/single_execution.py:148–152`: final inference deep-copies the complete EMA graph, including components unused for generation. This is a bounded avoidable peak; the measured graph state was 19,820,460 bytes versus 3,723,532 generator bytes, plus 4,194,312 prior bytes. No optimization was made in this audit.
- `src/hypergan/image_data.py:37–59`: one-time CIFAR concatenation and device movement; initialization temporarily duplicates image storage on CPU.
- `src/hypergan/image_metrics.py:31–59,101–111`: streaming moments and inference-mode feature extraction.
- `src/hypergan/metric_evaluation_worker.py:77–113`: batchwise generation/reference transfer under inference mode.
- `src/hypergan/metric_evaluation.py:131–140`, `cpu_worker_service.py:526–563`: fresh supervised evaluator with shutdown/reaping.
- `src/hypergan/web_service.py:22–32,75–99,482–490`: explicit cache/index/subscriber bounds and overflow handling.
- `src/hypergan/web_dev.py:26–37`: one sequential browser polling timer; `--dev` does not reload training.

### Medium: periodic evaluation shares training GPU memory

Both `fid_smoke` and `fid50k_train` omit `trigger`/`every_steps`. The resolved configuration defaults each to **interval evaluation every 10,000 steps**, with batch size 128 on CUDA. `CUDA_VISIBLE_DEVICES=0` places evaluation and training on the same physical GPU. A single worker slot and rotation mean one of the simultaneous requests can be skipped.

This can cause transient contention or OOM despite a stable training baseline. Both metrics specify `on_error="fail"`, making an evaluation error capable of failing the run. No evaluation OOM was demonstrated here. If pressure occurs, reduce evaluation batch size or explicitly configure a second visible device; naming `cuda:1` alone will not work while only GPU 0 is visible. Metric configuration was left intact.

### Low: persistent processes and disk growth can resemble leaks

The supervisor, HTTP server, and projector intentionally survive training completion. Stop them with `hypergan stop-server /home/martyn/dev/hypergan/training-runs/train-develop` when finished viewing. Their persistence alone is not a leak.

At inspection, checkpoints occupied **7.5 GiB**, previews **119 MiB**, and the derived projection **499,518,513 bytes**. Append-only history/checkpoint storage grows on disk. This is separate from RAM or VRAM retention. No files were pruned.

The original run's final event recorded a preview worker total-deadline error. That is evidence of an observer timeout, not evidence of a memory leak or the later viewer startup failure. Available current/previous-boot kernel journal searches found no OOM-kill entry; they do not establish why the earlier run ended.

## Validation and final state

- Event-view, viewer-dev and web-service focused tests: **53 passed, 2 deselected**. One excluded isolated CLI test had failed because `/usr/bin/python -I` cannot import this user-site editable installation; the other deselection follows the suite's heavy-test filter.
- Automatic viewer lifecycle suite: **14 passed**.
- CIFAR data and image metric suites: **11 passed**.
- `bash -n` and a fake-command invocation from `/tmp` verified launcher syntax, absolute paths, GPU default and forwarded arguments.
- Real command from `/tmp`: `/home/martyn/dev/hypergan/training-runs/start.sh --stop-after-steps 8` exited **0**. It resumed at **106000**, saved a new complete checkpoint at **106008**, and stopped with `stop_after_steps`. See [startup-result.json](memory-leak-audit-2026-09-20/startup-result.json).

Training is **stopped at step 106008** after the bounded verification. The viewer was left available at `http://127.0.0.1:8766`; another run was using the default port. Running the repaired launcher resumes from the new checkpoint. The existing configuration and history were preserved. No multi-hour endurance claim is made.
