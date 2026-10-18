# Foldlite DB integration

HyperGAN is the first consumer of Foldlite DB, an embedded map/reduce database for
realtime views over immutable documents. The `foldlitedb` base dependency uses
Python's SQLite and standard library; `hypergan[reducers]` and `hypergan[web]`
request `foldlitedb[reducers]`, which adds the optional Wasmtime runtime.

Foldlite owns general document persistence, source checkpoints, bounded JSONL
pages, append recovery, process-owned locks, atomic metadata, realtime queues,
snapshot/replay handoff and reducers. HyperGAN owns the training document schema,
metric catalogs and definition hashes, run/attempt partitions, recovery lineage,
custom-map worker supervision and HTTP authorization. Existing `events.jsonl`,
`views/<revision>/contributions.jsonl`, cursors and HTTP/SSE payloads remain readable.
The `hypergan.metrics_reducer` module reexports Foldlite's public reducer API.
Rust source, WASM assets and reproducible build qualification live in Foldlite.

## Regenerable historical views

Explicit historical bootstrap jobs materialize the requested metric groups in
`views/<map_revision>/foldlite/<query_digest>.sqlite`. Each cached document holds
one complete projection frame and its source cursor, including empty emissions.
Foldlite atomically commits document identity, source position, reduced state and
change history. Reopening retrieves the last committed document and continues from
its exact cursor; an unchanged query does not rescan the historical projection.

The view descriptor binds the reducer module SHA256, map generation, selected
metrics, bucket width, step bounds and full recovery lineage. HyperGAN retains
its existing 2048-group and 1 MiB response limits. Cache progress may not exceed
the explicitly captured bootstrap watermark. Every reuse validates the requested
and saved projection boundaries, even when no documents need ingesting. This is
bounded cursor validation under the immutable-source contract, not a full history
checksum on every query. Complete corruption encountered by readers fails visibly.

Source files remain authoritative. To regenerate these views, stop observers using
the cache and remove `views/<map_revision>/foldlite/`; the next explicit historical
query reconstructs it. Replacing a projection with a new generation invalidates
its cache. Removing only the derived cache does not alter source documents,
contributions or training checkpoints.

This first integration uses one SQLite database per distinct query and lineage.
It duplicates projection documents across queries and retains cache files until
explicitly removed. Active jobs remain bounded, while disk retention is user
controlled. A new query performs an initial historical pass; reopening the same
query consumes only its suffix. Live fanout continues to forward unchanged frames;
connected viewers reduce their live suffix through the same Foldlite WASM module.

## Installing the extraction branch

Before Foldlite's first package release, install its local source or a built wheel
before installing HyperGAN. For the adjacent checkouts used during extraction:

```sh
python -m pip install -e /nvme/dev/255bits/foldlitedb
python -m pip install -e '.[web,test]'
python -m pytest tests/foundation/test_run_state.py tests/foundation/test_event_views.py tests/web
```

For wheel qualification, build Foldlite separately, install its wheel, then build
and install HyperGAN. HyperGAN declares a normal version range and does not embed
a workstation path or assume an unpublished repository URL. CI's optional
`FOLDLITEDB_SOURCE` repository variable supplies an accessible wheel URL or pinned
pip source reference. Without it, CI requires the compatible release on its package
index. Keep that source reference pinned while reviewing this branch.
