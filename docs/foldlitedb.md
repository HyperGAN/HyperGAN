# Foldlite DB integration

HyperGAN is the first consumer of Foldlite DB, an embedded map/reduce database for
realtime views over immutable documents. The `foldlitedb` base dependency provides
embedded document storage and projection materialization; `hypergan[reducers]` and `hypergan[web]`
request `foldlitedb[reducers]`, which includes packaged WebAssembly reducer assets.

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
`views/<map_revision>/foldlite/<query_digest>.fold` (with companion `.fold.cache` and `.fold.lock` files). Each cached document holds
one complete projection frame and its source cursor, including empty emissions.
Foldlite atomically commits document identity, source position, reduced state and
change history. Reopening retrieves the last committed document and continues from
its exact cursor; an unchanged query does not rescan the historical projection.

The view descriptor binds the reducer module SHA256, map generation, selected
metrics, bucket width, step bounds and full recovery lineage. HyperGAN retains
its existing 2048-group and 1 MiB response limits. A bootstrap includes only its
explicitly captured watermark. If a delayed request predates the durable cache,
Foldlite reconstructs that older snapshot in memory and preserves the newer cache.
An advisory lock serializes each complete cache job across threads and processes.
Every reuse validates the requested and saved projection boundaries, even when no
documents need ingesting. This is
bounded cursor validation under the immutable-source contract, not a full history
checksum on every query. Complete corruption encountered by readers fails visibly.

Source files remain authoritative. To regenerate these views, stop observers using
the cache and remove `views/<map_revision>/foldlite/`; the next explicit historical
query reconstructs it. Replacing a projection with a new generation invalidates
its cache. Removing only the derived cache does not alter source documents,
contributions or training checkpoints.

This integration uses one Foldlite database (`.fold`, `.fold.cache`, `.fold.lock`) per distinct query and lineage.
It duplicates projection documents across queries and retains cache files until
explicitly removed. Active jobs remain bounded, while disk retention is user
controlled. A new query performs an initial historical pass; reopening the same
query consumes only its suffix. Live fanout continues to forward unchanged frames;
connected viewers reduce their live suffix through the same Foldlite WASM module.

## Installing Foldlite DB

Install `foldlitedb` before or alongside HyperGAN:

```sh
python -m pip install "foldlitedb>=0.1.2,<0.2"
python -m pip install -e '.[web,test]'
python -m pytest tests/foundation/test_run_state.py tests/foundation/test_event_views.py tests/web
```

For wheel qualification, build Foldlite separately, install its wheel, then build
and install HyperGAN. HyperGAN declares a normal version range and does not embed
a workstation path or assume an unpublished repository URL. CI's optional
`FOLDLITEDB_SOURCE` repository variable supplies an accessible wheel URL or pinned
pip source reference. Without it, CI requires the compatible release on its package
index. Keep that source reference pinned while reviewing this branch.

## Verifying persistent history

The deterministic service proof creates retained documents, opens a historical
view, restarts the service, and appends one measurement:

```sh
python scripts/foldlite_integration_proof.py --documents 10000 --output /tmp/foldlite-history.json
```

It requires zero reducer additions when reopening unchanged history and exactly
one contribution when processing the new suffix. It reports discovery indexing
separately because that index still rebuilds at service startup. This proof needs
the reducer runtime and uses no training, GPU allocation, or seed variants.
