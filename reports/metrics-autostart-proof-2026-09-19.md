# Automatic CLI viewer proof — 2026-09-19

M5 adds automatic CLI-only loopback startup when the optional web dependencies
are installed. A spawn supervisor owns separate HTTP and built-in projection
processes. The trainer never maps or reduces for a browser. Explicit `--server`,
`--server-port` and `--open` require successful authenticated startup before
numerical imports; automatic mode does not await readiness. Only `--open` opens a
browser. `--no-server` has no web imports/socket setup, and Python training APIs
remain headless. Missing optional dependencies silently retain headless defaults.

Credentials live outside the run in an exclusive private file. A pending server
does not create the trainer's exclusive run directory. Once the manifest exists,
nonsecret viewer mode/health receipts live under `observations/`. Diagnostics use
stderr, preserving final JSON and streaming JSONL stdout. Viewer/projection
failures remain independent from numerical failure. The producer processes pages
of 100 documents and has a bounded final drain; incomplete history can be
continued with the existing `project` command.

Validation used the built wheel installed in `/tmp/hypergan-autostart-verify`
with Python 3.12.13, torch 2.14.0+cpu, and the qualified web/reducer dependencies.
Dependency directories from the existing verification environments were exposed
read-only; no shared environment installations were changed.

```sh
python -m build --wheel --outdir /tmp/hypergan-autostart-wheel
python -m pytest tests/reference/test_core_cli.py \
  tests/web/test_web_autostart.py tests/foundation/test_cli_viewer_options.py -q
# 14 passed in 23.72 seconds
```

The installed `python -I -m hypergan` test compares an explicit CPU headless
five-step run with a viewer-enabled two-step stop plus resume. Every checkpoint
state field compares exactly, including models, priors, EMA, optimizer state,
buffers, training modes, data, complete RNG state and retained batch. Samples
match after excluding independently assigned run/artifact identities, and all
numerical metric values match (wall-clock timing is intentionally excluded).
Both viewer lifecycle receipts end stopped; stdout parses as the expected JSON
or JSONL throughout.

Separate tests prove an initially absent run directory remains absent until its
writer creates it, a separate producer catches up 251 source frames, no default
readiness wait occurs, occupied required ports fail before numerical imports/run
creation, optional binding failure preserves training, server failure does not
kill training, numerical errors clean up, and forcibly killing the trainer still
removes the private credential and closes the server socket. Foundation checks
reject unexpected imports on the no-server path and qualify missing extras.

This is Linux CPU execution evidence. Cross-platform lifecycle CI and the
coordinator's GPU acceptance remain separate gates. Cleanup is bounded: very
large projection backlogs can remain unfinished when training ends. The automatic
viewer ends with its CLI command; standalone `serve` remains the persistent
inspection command. No distributed rank startup or remote binding is added.
