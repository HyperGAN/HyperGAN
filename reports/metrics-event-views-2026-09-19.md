# File-backed event views — 2026-09-19

M1a implements the accepted [metrics plan](metrics-first-research-2026-09-19.md): authoritative event documents flow through versioned Python maps into bounded, append-only contribution frames. Views select grouping and shared reducers independently. There is no database. [The guide](../docs/event-views.md) documents the Python and CLI contracts.

`hypergan metrics RUN` reads immutable catalogs; `hypergan project RUN [--follow]` explicitly owns the mapper; `hypergan contributions RUN` provides bounded replay pages with acknowledged cursors. The builtin map imports no numerical runtime. Custom maps execute in supervised group-free workers with source identity, deadlines and bounded JSON results. This supervision is not a sandbox for untrusted Python.

Frames atomically represent one source document, including zero emissions, and retain metric definition, attempt and source identity. A single writer repairs unfinished tails; complete corruption fails. Restart reads a bounded tail and validates the source cursor against the last committed source identity. Review reproduced and fixed both append continuity and a cursor that could otherwise skip documents. Copied contribution streams retain portable read cursors; rebuilding from a relocated source requires a new projection because local source cursors deliberately bind their source.

Independent coordinator validation built a wheel through the source distribution, installed it in a clean base-only environment and ran:

```sh
/tmp/hypergan-metrics-base/bin/python -I -m pytest tests/foundation --import-mode=importlib -q
# absolute checkout test path, execution outside checkout: 309 passed in 11.01s
```

The environment has no torch, NumPy or Wasmtime. Tests cover zero/multiple emissions, deterministic replay, strict catalog partitions, bounded pages, copied cursors, custom worker failure, corruption, partial-tail repair and CLI exports. Durable evidence: `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/views-installed-base.log`. GitHub required platform checks remain a merge gate. This slice does not start a server; standalone serving follows separately.
