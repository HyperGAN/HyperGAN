# Local metrics serving and shared historical views, 2026-09-19

This implementation supplies a standalone loopback API and an SSE observation
service. The browser and external agents use the same routes. Python maps remain
in the explicit headless projection service; serving never executes a mapper or
numerical metric producer. Only historical bootstrap jobs invoke the bundled
shared WASM reducer. Live source/projection frames are forwarded unchanged through
one tail per file and bounded subscriber queues.

The [user/API guide](../docs/local-web.md) records routes, authentication,
reconnection, resource limits, cache freshness and artifact behavior. OpenAPI 3.1
includes actual JSON schemas and query parameters. The public run response omits
absolute paths and recipe arguments; independent progress heartbeats avoid
rebooting charts for every completed training step. Recovery lineage truncates
ancestor measurements at the selected checkpoint, including a zero-update child.

The source reader and projection reader accept a private file-opener adapter so
HTTP access can use directory-anchored, no-follow regular-file access. Catalogs use
the same adapter. Preview publication now includes a SHA256; the server adapts the
existing retained preview index. A bounded background task indexes the selected
final JSON sample once, while overlarge/unavailable samples remain explicit.
No model checkpoint is deserialized in the server.

A historical request captures an exact committed projection cursor H and reduces
in a background task. The response carries grouped mathematical states, definition
and attempt partitions, module digest, bucket alignment and lineage identity.
Consumer reduction continues strictly after H. Jobs coalesce by query, not by each
new head. A completion lease and first-result handoff prevent a moving training
head from starving a 202/refetch workflow; later explicit requests can refresh
stale covered history. No update-driven reduction occurs on the server.

The local test suite covers authenticated API/catalog/pages, Host/Origin rejection,
private credentials, actual CLI process startup/SIGTERM cleanup, real uvicorn SSE
continuation, fixed historical H followed by live frames, two identical viewers,
slow-consumer bounds, corruption, old-checkpoint lineage, future projection and
evaluation stream discovery, pending-run activation, safe artifact access and
request-driven cache freshness. Custom map execution is monkeypatched to fail in
HTTP tests; live reduction is likewise forbidden after bootstrap in the fanout
fixture. At this checkpoint **12 web tests pass**, plus the five preview publication
fault fixtures and 70 combined web/projection/source-reader cases at the prior
intermediate checkpoint. Final installed-package and platform CI results belong
in the coordinator integration receipt.

The browser implementation was independently exercised against this actual
backend by the UI subagent: historical step 40 became live step 41/projection 42
without JavaScript or CSP errors. Scripts remain same-origin with
`wasm-unsafe-eval` solely for the bundled kernel; `style-src 'self'` works with the
chart implementation and was not loosened.

An explicit one-million-update proof generated 263,668,368 bytes of source events
and 841,142,342 bytes of mapped frames. Its temporary data was removed afterward.
The first measured server prototype produced:

| Measurement | Result |
| --- | ---: |
| Source/projection background indexing | 87.42 seconds |
| First historical envelope bootstrap | 74.38 seconds |
| Warm compatible query, p95 including JSON serialization | 2.19 milliseconds |
| Bootstrap payload / grouped states | 326,427 bytes / 513 |
| One new frame delivered to five viewers | 19.9 milliseconds |
| Live server reducer calls | 0, asserted |
| Peak process RSS | 193.15 MiB |
| Synthetic fixture generation | 37.38 seconds |

Command:

```sh
python scripts/metrics_server_proof.py --events 1000000 \
  --output /home/martyn/dev/hypergan/resurrection-backups/2026-09-19-metrics-implementation/million-event-server-proof.json
```

This first run explicitly allowed a 180-second historical job budget. It exposes
the high cost of cold full-history JSONL validation/reduction; the initial default
60-second history budget would reject this particular million-row request. The
warm-query, live-latency and RSS research targets passed in this standalone
synthetic experiment. It does **not** measure training overhead, qualify arbitrary
hardware, or establish a p95 live-latency distribution from one frame. A sparse
historical index/direct-tail initialization is a potential follow-up optimization.
The implemented default was therefore raised to 180 seconds, with a validated
explicit override and the actual limit exposed in capabilities. Source-identical
revalidation is recorded at integration.

No database, paid compute, dataset download, public deployment or release was used.
Source metrics and their catalogs remain canonical; projections are rebuildable,
bootstrap caches disposable, and the first artifact interface does not claim
built-in rendering for every possible modality.
