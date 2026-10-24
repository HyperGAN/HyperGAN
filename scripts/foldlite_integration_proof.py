"""Measure persisted historical views through the real HyperGAN service.

Explicit deterministic fixture: no training, GPU work, image inspection or seed variants.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time

from hypergan.event_views import MapSpec, Projector
from hypergan.metrics_reducer import Reducer
from hypergan.web_service import ObservationService
from metrics_server_proof import build


async def ready(root):
    started = time.perf_counter()
    service = await ObservationService(root, poll_seconds=.01, history_timeout=180).start()
    while not service.streams or not all(stream.caught_up for stream in service.streams.values()):
        for stream in service.streams.values():
            if stream.error:
                await service.close()
                raise RuntimeError(stream.error)
        if time.perf_counter() - started > 180:
            await service.close()
            raise RuntimeError('Fixture indexing timed out')
        await asyncio.sleep(.01)
    return service, time.perf_counter() - started


async def query(service, count):
    started = time.perf_counter()
    job = service.bootstrap(MapSpec().revision, series=['loss/g_total'],
                            bucket_steps=max(1, count // 512))
    await job.task
    if job.error:
        raise RuntimeError(job.error)
    return job.result, time.perf_counter() - started


async def proof(root, count, source):
    service, initial_index_seconds = await ready(root)
    try:
        initial, initial_query_seconds = await query(service, count)
    finally:
        await service.close()

    calls = []
    original_add = Reducer.add
    def tracked_add(self, state, values):
        calls.append(len(values))
        return original_add(self, state, values)
    Reducer.add = tracked_add
    try:
        service, reopen_index_seconds = await ready(root)
        try:
            reopened, reopen_query_seconds = await query(service, count)
            assert reopened == initial
            assert calls == [], 'Reopening reduced already committed history'
        finally:
            await service.close()

        row = dict(source, sequence=count + 2, event='train', step=count + 1,
                   metrics={'loss/g_total': .42})
        with (root / 'events.jsonl').open('ab') as stream:
            stream.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
        with Projector(root) as projector:
            assert projector.project()['documents'] == 1

        service, suffix_index_seconds = await ready(root)
        try:
            suffix, suffix_query_seconds = await query(service, count)
            assert calls == [1], f'Expected only one new contribution, got {calls}'
            assert sum(group['state']['count'] for group in suffix['groups']) == count + 1
        finally:
            await service.close()
    finally:
        Reducer.add = original_add

    cache_bytes = sum(path.stat().st_size for path in root.glob('views/*/foldlite/*.fold*'))
    return {
        'initial_source_documents': count + 1,
        'initial_measurements': count,
        'final_source_documents': count + 2,
        'initial_index_seconds': initial_index_seconds,
        'initial_historical_query_seconds': initial_query_seconds,
        'reopen_index_seconds': reopen_index_seconds,
        'reopened_historical_query_seconds': reopen_query_seconds,
        'reopened_reducer_add_calls': 0,
        'suffix_index_seconds': suffix_index_seconds,
        'suffix_historical_query_seconds': suffix_query_seconds,
        'suffix_reducer_add_batches': calls,
        'persisted_query_cache_bytes': cache_bytes,
        'groups': len(suffix['groups']),
        'module_sha256': initial['module_sha256'],
        'limits': 'Source/projection discovery indexes still rebuild on each service startup; query caches duplicate documents per query.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.documents < 1:
        parser.error('--documents must be positive')
    with tempfile.TemporaryDirectory(prefix='hypergan-foldlite-proof-') as temporary:
        root = Path(temporary)
        source = build(root, args.documents)
        result = asyncio.run(proof(root, args.documents, source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
