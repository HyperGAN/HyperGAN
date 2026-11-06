# Shared metrics reducer proof

[PR #320](https://github.com/HyperGAN/HyperGAN/pull/320), M0 of the [metrics plan](metrics-first-research-2026-09-19.md), implemented on
2026-09-19. This slice supplies a portable reducer and its delivery contract;
training publication, Python event maps and the production viewer follow in
separate PRs. No database, training dependency or JavaScript implementation of
the reducer mathematics was added.

The same bundled Rust/core-WASM module runs in Python/Wasmtime and a dedicated
browser worker. Its SHA256 is
`727faa849990730404dd6ca1f898526d380b56bd4ad9384f9186a2bc4d7b275d`
(153,189 bytes). The source, locked dependency graph, pinned Rust 1.90.0
toolchain, rebuild script and applicable dependency licenses are included.
End users install the optional `reducers` extra; they do not compile Rust.
The base package can still import readers and reducer descriptors without
Wasmtime or a numerical runtime.

## Contract proved

- One pure `identity`, batched `add`, `merge`, `finalize` interface implements
  `mean/v1` and `envelope/v1` (first/min/max/last). Mean state stores sum/count;
  finalized means are not merged as if they had equal sample counts.
- Envelope positions use evaluated step and stable ASCII source identity.
  Late/out-of-order contributions use this declared order, not arrival order.
- Empty state is explicit; missing/null values do not count. Invalid state,
  nonfinite numbers, unsafe counters, overflow and incompatible reducers fail.
- The narrow byte-buffer ABI has no host/WASI imports. Requests are capped at
  256 KiB and 1,024 values, responses at 64 KiB, WASM memory at 16 MiB.
  Python calls have fuel limits; browser worker calls have a deadline and
  terminate on timeout. A trapped instance cannot be reused silently.
- Bootstrap carries the actual state, module digest, identity and committed
  half-open coverage. The identity must bind the stream generation and full
  view/metric partition. Live replay is deduplicated before reduction; gaps,
  partial overlaps and incompatible revisions require a new bootstrap.
  Only adjacent, disjoint compatible summaries can merge through the delivery
  wrapper. A zero-emission frame still advances coverage.

The mathematical kernel deliberately does not retain an event-ID set or own
source cursors. Event-view framing supplies those facts. Neither mathematical
associativity nor this test suite promises bitwise-invariant floating-point
results for arbitrary regroupings.

## Evidence

The coordinator and implementation agent ran 23 Python reducer cases and four
actual Chromium cases. The browser tests load Python-generated historical
state, apply a live suffix in the worker, replay frames, and compare with the
full Python result using the same module digest. Other cases cover malformed
state, unequal-count merging, out-of-order/tied positions, module mismatch,
gaps/overlap, nonfinite JavaScript inputs, memory plateau, fuel traps and a
deliberately looping worker terminated by its deadline.

Commands from the implementation worktree:

```sh
python scripts/build_reducer.py --check
python -m pytest tests/reducers tests/browser -q
python scripts/reducer_proof.py --output /path/to/proof.json --benchmark
```

The independent coordinator run passed all 27 tests in 1.50 seconds. The
pinned rebuild matched the bundled bytes. One local Python proof measurement
reported 46.9 ms first construction, 0.88 ms per 1,024-value mean batch and
1.57 ms per envelope batch. Sending the same number of values as individual
calls took 55.3/73.6 ms. Serialized state measured 72/258 bytes for that fixture;
WASM linear memory remained 1,769,472 bytes. These are reducer microbenchmarks,
not browser rendering, complete process RSS or training-overhead guarantees.
Batching is required for efficient historical bootstrap.

Independent review added rejection of envelopes retaining more distinct positions
than their count. Initial macOS/Windows CI exposed a one-unit test fuel budget
being consumed by constructor ABI discovery, before the tested request. The
constructor now has a separate fixed discovery budget; the unchanged one-unit
request budget still traps and poisons the instance on every platform. No test
was skipped or request limit widened.

The first sdist-rebuilt installed wheel passed 263 base-only tests in 8.09 seconds
and all 27 optional reducer/browser cases in 1.53 seconds outside the checkout.
Final rebuild/test receipts include the reviewed validation fixes above.
Installed-package results, source/artifact hashes and GitHub PR/check receipts
are retained at
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/`.
CI now requires the optional reducer tests on Linux, macOS and Windows,
an actual Chromium worker proof, and a reproducible Linux rebuild alongside
the existing lightweight and CPU reference jobs. The ABI remains internal
until the first integrated view contract is reviewed.

No GPU execution, paid allocation or release was needed for this pure reducer
slice. Next: M1a materializes bounded Python-map contributions; M1b/M2 publish
configurable metrics while preserving numerical recovery.
