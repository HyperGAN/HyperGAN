"""Explicit large-log benchmark; never part of routine unit-test execution.

Run with an installed HyperGAN web/reducer environment:
python scripts/metrics_server_proof.py --events 1000000 --output /tmp/proof.json
The temporary synthetic source and projections are removed after measurements.
"""
import argparse
import asyncio
import json
from pathlib import Path
import resource
import statistics
import tempfile
import time

from hypergan.event_views import MapSpec, Projector
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_service import ObservationService


def build(root, count):
    descriptor = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    descriptor['definition_hash'] = digest(descriptor)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': descriptor}}
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    atomic_json(root / 'manifest.json', dict(schema_version=1, run_id='proof', attempt_id='a',
        status='running', steps=count, total_steps=count + 1, metrics_catalog=revision))
    base = dict(schema_version=2, run_id='proof', stream_id='training', stream_generation='proof',
                attempt_id='a', catalog=revision)
    with (root / 'events.jsonl').open('wb') as stream:
        stream.write((json.dumps(dict(base, sequence=1, event='start', step=0, parent_attempt_id=None, restored_step=0)) + '\n').encode())
        for step in range(1, count + 1):
            row = dict(base, sequence=step + 1, event='train', step=step, metrics={'loss/g_total': (step % 1000) / 1000})
            stream.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
    with Projector(root) as projector:
        while projector.project(limit=10000)['has_more']:
            pass
    return base


async def prove(root, count, base):
    service = await ObservationService(root, poll_seconds=.02, history_timeout=180).start()
    try:
        started = time.perf_counter()
        while not all(stream.caught_up for stream in service.streams.values()):
            for stream in service.streams.values():
                if stream.error:
                    raise RuntimeError(stream.error)
            await asyncio.sleep(.02)
        indexing_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bucket_steps = max(1, count // 512)
        job = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=bucket_steps)
        await job.task
        if job.error:
            raise RuntimeError(job.error)
        cold_seconds = time.perf_counter() - started
        result = job.result
        warm = []
        for _ in range(100):
            started = time.perf_counter()
            replay = service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=bucket_steps)
            encoded = json.dumps(replay.result)
            warm.append((time.perf_counter() - started) * 1000)
        subscribers = [service.events('projection:' + MapSpec().revision, result['cursor']) for _ in range(5)]
        for subscriber in subscribers:
            await anext(subscriber)
        # No reduced call is allowed after historical bootstrap, including five
        # subscribers, changed head and another cached page load.
        from hypergan.metrics_reducer import Reducer
        original_add = Reducer.add
        def unexpected(*args, **kwargs):
            raise AssertionError('Live server executed reduction')
        Reducer.add = unexpected
        try:
            started = time.perf_counter()
            with (root / 'events.jsonl').open('ab') as stream:
                stream.write((json.dumps(dict(base, sequence=count + 2, event='train', step=count + 1,
                                             metrics={'loss/g_total': .42})) + '\n').encode())
            with Projector(root) as projector:
                projector.project()
            for subscriber in subscribers:
                while not (await asyncio.wait_for(anext(subscriber), 30)).startswith(b'event: frame'):
                    pass
            live_seconds = time.perf_counter() - started
            assert service.bootstrap(MapSpec().revision, series=['loss/g_total'], bucket_steps=bucket_steps) is job
        finally:
            Reducer.add = original_add
            for subscriber in subscribers:
                await subscriber.aclose()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
        return dict(events=count, indexing_seconds=indexing_seconds, cold_bootstrap_seconds=cold_seconds,
            warm_query_p95_ms=sorted(warm)[94], warm_query_median_ms=statistics.median(warm),
            bootstrap_bytes=len(encoded.encode()), grouped_states=len(result['groups']),
            five_viewer_live_seconds=live_seconds, live_server_reductions=0, peak_rss_bytes=rss_bytes,
            source_bytes=(root / 'events.jsonl').stat().st_size,
            projection_bytes=(root / 'views' / MapSpec().revision / 'contributions.jsonl').stat().st_size,
            module_sha256=result['module_sha256'])
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.events <= 1000000:
        parser.error('--events must be in 1..1000000')
    with tempfile.TemporaryDirectory(prefix='hypergan-metrics-server-proof-') as directory:
        root = Path(directory)
        started = time.perf_counter()
        base = build(root, args.events)
        generation_seconds = time.perf_counter() - started
        result = asyncio.run(prove(root, args.events, base))
        result['fixture_generation_seconds'] = generation_seconds
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
