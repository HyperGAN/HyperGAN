# Event maps and file-backed views

Metrics are facts recorded in the run's authoritative `events.jsonl`. A map consumes
those documents and produces contributions; a view chooses how to group and reduce
the contributions. A new reducer or renderer reuses the same map output. This
module implements the file/worker boundary; it does not start a web server or a
training observer automatically.

```python
from hypergan.event_views import MapSpec, Projector, ViewSpec, read_projection_page

mapping = MapSpec()  # builtin:metrics selects already published scalar metrics
view = ViewSpec(mapping.revision, reducer="envelope/v1", bucket_steps=100)
with Projector("runs/example", mapping) as projector:
    progress = projector.project(limit=100)  # bounded backfill/live-tail work
page = read_projection_page("runs/example", view.map_revision, limit=100)
```

The headless orchestration process explicitly owns `Projector`. HTTP handlers only
call readers: a missing projection raises `FileNotFoundError` and never executes a
mapper. The optional custom-map worker lives for the projector context and is
reused across documents. Start multiprocessing programs under a Python main guard.
The initial worker budget is five seconds per command and 300 seconds total;
long-lived orchestration must close/reopen before its configured lifetime expires.
Only one sequential caller owns a projector. Failure requires closing and reopening
before further projection, so a partial write cannot be followed by duplicate work.

A custom function remains ordinary Python:

```python
# experiment_views.py
# Keep maps deterministic: no mutable global counters, I/O, RNG or live tensors.
def loss_points(event):
    for metric_id, value in event.get("metrics", {}).items():
        yield [metric_id, event["attempt_id"], event["step"]], value
```

Create a `MapSpec("experiment_views:loss_points", version="1",
source_digest=sha256_of_module_file, config={})`. Source digest, version, arguments
and key schema determine its revision. Custom code imports and executes in a
supervised, group-free worker. The worker verifies the pinned module file before
calling the map; dependent code versions belong in the explicit version/config.
`config` is snapshotted at construction. The worker is isolated from trainer state
and reaped on failures and coordinator death using the existing broker. Trusted
Python is not a security sandbox: deadlines bound lifetime, JSON limits bound
protocol payloads, and neither limits arbitrary allocations or unmanaged descendants.

V1 keys are exactly `[metric_id, attempt_id, step]`, with string IDs and a
nonnegative JavaScript-safe integer step. Values are finite scalars. Maps cannot
change the source attempt. Every emitted metric must already exist in that event's
immutable catalog. The framework attaches its verified `definition_hash`; custom
maps cannot remove this partition or invent an unregistered derived metric. Custom
metric publication is a separate producer interface. A view must preserve metric,
definition and attempt partitions. `mean/v1` and `envelope/v1` identify the initially
supported reducer contracts; the executing host supplies the pinned module/state
identity when it builds a bootstrap. View descriptors alone do not execute reduction.

Each map has a directory `views/<map_revision>/`, containing immutable
`projection.json` identity and append-only `contributions.jsonl`. A frame holds all
emissions for exactly one source document, its source identity, the consumed local
source cursor and a contiguous projection sequence. Empty documents commit empty
emission lists. Emission IDs derive from map revision, source event identity and
emission index. There is no independent progress file that could advance before
values were written. New map revisions create separate projections; multiple views
share an identical revision.

The OS lock has one writer. On restart, the projector reads at most the last
128 KiB, validates the final complete frame, repairs only an unfinished tail and
validates its source cursor before mapping more. Complete corrupt final frames
fail. Older history is validated by paginated reads, not rescanned on every restart.
This is a rebuildable observed stream: newline/flush provides process-crash replay;
frames do not call `fsync` or claim durable training progress. A replaced/truncated
source invalidates its local cursor and requires an explicit rebuilt projection.
An unchanged copied projection remains readable, but restarting its projector in a
different source directory requires a fresh projection because source cursors are local.

A read page returns `frames`, a `frame_cursors` entry after each complete frame,
and the final `cursor`, `has_more` and `partial_tail`. The transport forwards frames
unchanged. A consumer commits its cursor only after applying the entire frame.
Reconnect after an earlier cursor intentionally replays later frames; the delivery
engine deduplicates before invoking a reducer. The reducer never stores an unbounded
set of event IDs. Projection cursors bind map revision, generation, offset, sequence,
source identity and boundary digest without embedding local paths or inode numbers;
faithful copies can retain them. Replacement must get a new generation. A cursor is
a continuation marker, not a credential or tamper-proof signature.

Source batches read at most 1 MiB and at most the requested document count. A
projection frame is at most 64 KiB, with at most 128 emissions and 1 KiB per key.
Custom worker emission payloads are capped at 32 KiB to leave room for its envelope.
Catalogs contain at most 4096 metric definitions; the projector caches at most 16
verified catalogs. Reader pages cap requests at 10,000 frames/16 MiB. Raw logs may
continue growing; a reduced view engine separately caps active grouped keys and
window state. It must not keep a state per historical step forever.

`ArtifactDescriptor` keeps artifacts independent of image assumptions. It identifies
an indexed output, `sample`, `measurement` or `diagnostic` role, open modality/media
type, SHA256, byte size and bounded provenance/metadata. Required provenance names
the run, attempt and evaluated step; snapshot/protocol/evaluation IDs can be added.
A sampler produces examples; a measurement quantifies something. Sharing artifact
identity does not make them the same operation. Artifact descriptors do not authorize
arbitrary file access or implement new audio/video/tensor rendering.
