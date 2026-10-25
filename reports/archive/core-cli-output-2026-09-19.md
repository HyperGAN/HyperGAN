# Bounded training CLI output

The public execution-profile cutpoint needs a parent-side progress sink that can
be used by both native and supervised replicated execution. The existing
arbitrary callback API still requires isolated importable functions. A concrete
internal `CLIProgress` sink performs bounded delivery directly in the coordinator;
it does not spawn a callback worker for each update.

`training_output` redirects training CLI Python streams and native stdout/stderr
descriptors through separate disposable output processes. Each process drains its
input independently of writes to the destination. Native library and inherited
worker diagnostics therefore remain available with a healthy consumer, while an
unread or disconnected destination does not hold numerical work or final process
cleanup. Warnings, error handlers and optional viewer announcements must execute
inside the same context; public command integration is the following PR slice.

The stream is explicitly best effort. Each destination has 16 queued lines in the
parent and 16 in its drain, with a 64 KiB UTF-8 limit per line. Each stage also has
one partial/in-flight line and a finite OS pipe buffer. Full queues discard their
oldest queued lines; oversized diagnostic/progress lines are omitted. A slow or
closed consumer can miss progress and the final result. The durable run
`events.jsonl`, `manifest.json`, checkpoints and artifacts remain authoritative;
use `hypergan events` and `hypergan inspect` to reconnect. Native diagnostics are
best effort too; structured worker errors still follow the execution service's
durable failure path.

Normal results now use compact JSON on one line, avoiding partial eviction of a
pretty-printed manifest. `--progress-json` retains its event JSONL and
`{"event":"result","manifest":...}` envelope. A result exceeding 64 KiB is
replaced by a small `output_omitted` event naming `result_exceeds_output_limit`
and the durable manifest location, accompanied by a stderr warning. The output
stream never claims to be an exact event archive.

Shutdown shares a one-second drain grace across stdout/stderr, then kills and
reaps blocked drains. Raw feeder writes use no Python text-stream locks and are
released by closing the killed receiver. An independent watcher exits each drain
when the coordinator dies, including while its destination writer is blocked.
Before restoration, the context flushes cached Python standard streams and
only the owning C runtime's stdout/stderr (libc on Linux/macOS, UCRT on Windows)
through the still-active drains. This prevents buffered native output from
hanging interpreter exit after restoring a full destination. Arbitrary custom
FILEs, separately cached OS handles, separate CRT streams and externally held
FILE locks are outside this descriptor transport. No blanket `fflush(NULL)` or
old-runtime compatibility shim is used. Cleanup still restores/reaps resources
if flush raises. The context restores original descriptors and streams. Native worker lifetime
continues to be governed by the existing worker broker; output drains do not
replace numerical supervision.

Validation:

```sh
PYTHONPATH=src /tmp/hypergan-public-coordinator-cpu/bin/python -m pytest \
  tests/foundation/test_bounded_cli_output.py -q
```

The nine focused tests passed in 6.11 seconds. A fresh wheel rebuilt through
its source distribution at `e55677ec6463686c906bf3e25cff827532cdd437` passed the
complete installed foundation suite: **355 passed in 17.49 seconds**, with no
skips and no numerical extras installed:

```sh
/tmp/hypergan-public-output-verify/bin/python -I -m pytest \
  /home/martyn/dev/hypergan/public-bounded-output/tests/foundation \
  --import-mode=importlib -q
```

[PR #328](https://github.com/HyperGAN/HyperGAN/pull/328) targets `develop`.
Build/test logs are `/tmp/hypergan-public-output-build.log` and
`/tmp/hypergan-public-output-installed-tests.log`; the coordinator retains the
integrated durable receipt and exact-head required CI results.

Twelve lightweight tests cover actual unread, closed and slowly read stdout/stderr
pipes; native and inherited subprocess diagnostic forwarding; complete normal
JSON/JSONL results; oversized result fallback; parent-death cleanup; memory
capture bounds; and exception restoration. New regressions cover buffered cached Python/native
stdio at process exit, healthy buffered diagnostic forwarding, and forced flush
error cleanup. The initially installed 355-test receipt above precedes these
three additional cases; final installed/platform results are recorded in the PR. Installed-wheel and platform CI
results are recorded in the PR and coordinator's integration receipt. The next
slice routes public `train`/`resume` through this context and qualifies actual
CPU and two-GPU stop/resume with disconnected output. No paid compute or release.

## Viewer process cleanup failure found during integration

On PR #328 head `146460014b1b777f1622c96aadc11b6a2a6017bd`, every output/platform
foundation case passed, but the macOS viewer job reached **20 passed** and then
hung until its ten-minute deadline. The failed job log is preserved in the
coordinator's `2026-09-19-public-execution/bounded-output/` evidence directory.
The existing pytest timer had already been removed before interpreter teardown.

A deterministic whole-Viewer fault fixture confirmed a product defect: a server
killed while owning the shared `multiprocessing.Event` condition lock poisoned
that stop signal. The broker blocked while setting it, and the projector's
parent-death watchdog blocked on the same lock. `Viewer.close()` killed the
broker but the projector survived, holding the resource-tracker pipe open. The
parent then hung during interpreter cleanup. The installed pre-fix fixture
failed after **18.09 seconds**; its independent traceback identified
`multiprocessing.resource_tracker._stop_locked` from `__del__`, with five leaked
semaphores. Exact logs and the completed-close receipt are retained as
`stop-owner-before*.log` and `stop-owner-before-closed.json`.

The viewer now uses a one-way, lock-free shared byte: one broker writes 0→1 and
children only read it. Cooperative waits poll at bounded intervals. Independent
parent-death watchdogs remain active during shutdown; broker death forcibly exits
a child even if its main thread cannot handle a Python signal. Cooperative server
shutdown still uses SIGTERM. No Event lock can be poisoned by an abrupt child
exit. The same whole-Viewer fixture passed in **0.24 seconds**, checking every
owned process and credential cleanup; the full 21-case web suite passed in
**8.51 seconds**. This proves the injected failure is fixed; final macOS CI remains
the gate for the original platform symptom.

Viewer CI now runs pytest through a guarded diagnostic wrapper. After pytest
returns it prints child/thread state and arms repeating traceback diagnostics
for normal interpreter cleanup. It preserves pytest's exit code, the ten-minute
job deadline, every test and ordinary process shutdown; there is no forced-success
exit or suppressed failure. Final combined/local and installed receipts accompany
the PR. This cleanup fix is independent of separate WASM resource-lifetime work.

Final stop-fix validation rebuilt the wheel through the source distribution and
installed it in separate base-only and optional-web environments. Outside the
checkout, **358 foundation tests passed in 17.46 seconds** and **21 web tests
passed in 8.30 seconds** through the diagnostic runner. The runner reported no
active children and only the main thread, then exited normally. The earlier
combined checkout probe had 378 passing assertions plus the expected
installed-distribution guard failure; it is preserved as a source probe, not an
installed proof. An initial diagnostic smoke used a stale viewer environment;
its two compatibility failures are retained separately and were superseded by
the fresh installed wheel. Final logs are `stopfix-installed-base.log`,
`stopfix-installed-web.log` and `stopfix-build.log` in the durable evidence
subdirectory. Required exact-head platform CI and integrated GPU/live-viewer
qualification remain coordinator gates.
