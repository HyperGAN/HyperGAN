# Shared reducer proof (M0)

This is a bounded proof of the proposed view reducer boundary, not the complete
metrics/server implementation. Python event maps stay Python-native. Python and
a dedicated browser worker execute the exact same bundled `reducer.wasm`, with
its SHA-256 checked against `reducer.json`. No WASI, filesystem, clocks or host
imports are available. Installation needs no Rust compiler; rebuilding does.

Install `hypergan[reducers]` for Python reduction. Plain event readers do not
need Wasmtime. Missing Wasmtime raises an actionable error with no Python
mathematics fallback.

```python
from hypergan.metrics_reducer import Reducer
r = Reducer()
state = r.add(r.identity("mean/v1"), [
    {"value": 4.0, "position": [1, "event-1/emission-0"]},
    {"value": 8.0, "position": [2, "event-2/emission-0"]},
])
assert r.finalize(state) == {"count": 2, "value": 6.0}
```

The provisional ABI exports `abi_version`, `input_ptr`, `input_capacity`,
`execute(length)`, `output_ptr` and `memory`. Hosts copy one UTF-8 JSON request
into the fixed 256 KiB input buffer and read the returned length (at most 64 KiB)
from the output buffer. Requests have `op` = `identity`, `add`, `merge`, or
`finalize`; responses contain `ok` or `error`. The module never retains a state
between calls. One instance must not be called concurrently; the Python wrapper
serializes calls. The worker client allows one outstanding request.

`mean/v1` stores count and sum; `envelope/v1` stores count and first/min/max/last
points. Null or absent values do not contribute. Empty means finalize to null,
not zero. Nonfinite values, overflow, unknown fields/reducers, malformed states,
unsafe counts/steps and oversized batches fail. A source position is a
nonnegative JavaScript-safe integer step plus 1–128 ASCII source identity bytes.
First/last order is lexicographic by this declared position, not arrival order.
Min/max ties choose the smallest position. A source emission must have one
immutable value; duplicate delivery is handled outside the mathematical kernel.

The Python `bootstrap`, `append_frame` and `merge_bootstraps` helpers and JS
coverage wrappers demonstrate cursor handling outside the reducer. Their opaque
ASCII identity **must bind run, source stream generation, map revision, view
revision, source metric definition, grouping and bucket alignment**. Offsets are
half-open committed projection positions, not sample counts. A complete frame
advances coverage even with zero emissions. Covered replay is ignored; a gap,
partial overlap, changed identity or module mismatch requires a new bootstrap.
Only adjacent disjoint states merge. This is a small proof helper, not a durable
projection store or portable network cursor. It trusts immutable same-generation
source frames and does not authenticate arbitrary client-generated history.

Reducer state has constant size (bounded source IDs), and each call accepts at
most 1024 contributions. Linear memory cannot exceed 16 MiB. Python sets a
50-million-instruction fuel budget per call; a trapped instance is discarded,
while a new instance can resume from the last validated state. The browser
owns a dedicated worker with an external deadline and terminates it on timeout;
it does not claim instruction-level browser fuel metering. Custom remote WASM
is not supported. Core numerical state has no unbounded deduplication set.
Key cardinality and subscription queues belong to the upcoming view engine.

Floating-point addition is not associative. Fixed ordered partitions have
consistent Python/browser execution; arbitrary merge trees are not promised
bitwise equality. Do not average finalized means when merging unequal counts.

## Rebuild and verification

From repository root, with `cargo`/`rustup` on PATH:

```sh
python scripts/build_reducer.py
python scripts/build_reducer.py --check
python -m pytest tests/reducers tests/browser -q
python scripts/reducer_proof.py --output /tmp/proof.json --benchmark
```

The Rust toolchain is pinned to 1.90.0 and dependencies/checksums to `Cargo.lock`.
The build strips symbols and remaps checkout/Cargo paths. `--check` compares a
fresh build to the shipped bytes without modifying bundled resources. The
bundled `THIRD_PARTY_LICENSES.txt` retains runtime and build dependency notices,
plus Rust's standard-library notices. Sources, lockfile and rebuild scripts
ship in the sdist; the wheel includes the module, manifest and static hosts.

Browser tests require Playwright plus Chromium (no skip when unavailable):
`python -m playwright install chromium --with-deps`. They launch a real worker
on a temporary loopback server and prove Python bootstrap plus live browser
suffix, duplicate replay, bad inputs, coverage rejection and deadline isolation.
To inspect manually, copy `hypergan.metrics_reducer.assets()` resources to a
temporary folder, add the generated `proof.json`, serve with
`python -m http.server --bind 127.0.0.1`, and open `proof.html`. This static fixture
server is not HyperGAN's planned application server.
