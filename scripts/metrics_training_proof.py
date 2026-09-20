"""Explicit local CUDA observation benchmark and exact full-state proof.

Install hypergan[train,web] with a CUDA-enabled torch build, then run:
python scripts/metrics_training_proof.py run --device cuda:1 --output /tmp/proof.json

The benchmark starts only localhost processes on this machine. No dataset is
fetched and no cloud allocation is used. Child modes are private harness plumbing.
"""
import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time

CONDITIONS = ('none', 'metrics', 'server_zero', 'server_five')


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state_digest(value):
    """Hash every saved numerical/RNG/data tensor and scalar with type/shape tags."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b'tensor:')
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b'numpy:')
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b':')
            for child in item:
                visit(child)
            digest.update(b'end')
        else:
            digest.update((type(item).__name__ + ':' + repr(item) + '\0').encode())
    visit(value)
    return digest.hexdigest()


def training(args):
    import torch
    from hypergan.training import train
    from hypergan.checkpoints import read_checkpoint
    started = time.perf_counter()
    manifest = train(args.config, args.run_dir, checkpoint_every=10 ** 9, preview_every=0)
    torch.cuda.synchronize(torch.device(args.device))
    wall_seconds = time.perf_counter() - started
    rows = [json.loads(line) for line in (args.run_dir / 'events.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['event'] == 'train']
    assert len(rows) == args.steps and rows[-1]['step'] == args.steps
    first = next(row for row in rows if row['step'] == args.warmup)
    elapsed = rows[-1]['seconds'] - first['seconds']
    checkpoint = read_checkpoint(args.run_dir)[2]
    result = dict(steps=args.steps, warmup=args.warmup, measured_steps=args.steps - args.warmup,
                  measured_seconds=elapsed, updates_per_second=(args.steps - args.warmup) / elapsed,
                  wall_seconds=wall_seconds, numerical_state_sha256=state_digest(checkpoint),
                  metric_ids=sorted(rows[-1]['metrics']), observed_steps=manifest['steps'],
                  durable_steps=manifest['last_durable_step'], checkpoint_step=checkpoint['step'])
    save(args.result, result)


def bare_training(args):
    """The same validated adapter updates, without controller observation I/O."""
    import torch
    from hypergan.checkpoints import trainer_state
    from hypergan.config import load_config
    from hypergan.single_execution import SingleProcessExecution
    execution = SingleProcessExecution(load_config(args.config))
    started = time.perf_counter()
    try:
        execution.start()
        for step in range(1, args.steps + 1):
            execution.update()
            if step == args.warmup:
                torch.cuda.synchronize(torch.device(args.device))
                measured_started = time.perf_counter()
        torch.cuda.synchronize(torch.device(args.device))
        elapsed = time.perf_counter() - measured_started
        wall_seconds = time.perf_counter() - started
        digest = state_digest(trainer_state(execution._trainer, execution._last_batch))
        save(args.result, dict(steps=args.steps, warmup=args.warmup,
            measured_steps=args.steps - args.warmup, measured_seconds=elapsed,
            updates_per_second=(args.steps - args.warmup) / elapsed,
            wall_seconds=wall_seconds, numerical_state_sha256=digest,
            metric_ids=[], observed_steps=None, durable_steps=None, checkpoint_step=None))
    finally:
        execution.shutdown()


def serving(args):
    import socket
    from hypergan.web_server import run_socket
    from hypergan.web_session import LocalSession
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    session = LocalSession(sock.getsockname()[1], auth='token')
    session.write_credentials(args.credentials)
    # Convert Uvicorn's replay of SIGTERM into a Python unwind, as the product CLI
    # does, so the owned file/socket cleanup below executes on graceful POSIX stop.
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, terminate)
    try:
        run_socket(args.run_dir, sock, session)
    finally:
        signal.signal(signal.SIGTERM, previous)
        sock.close()
        args.credentials.unlink(missing_ok=True)


def projecting(args):
    # Use the public command after its required run directory appears. The process
    # is independent of server requests and numerical training callbacks.
    while not args.run_dir.is_dir():
        time.sleep(.01)
    os.execv(sys.executable, [sys.executable, '-I', '-m', 'hypergan', 'project',
                            str(args.run_dir), '--follow', '--limit', '256'])


async def consuming(args):
    import httpx
    from hypergan.event_views import MapSpec
    credentials = json.loads(args.credentials.read_text())
    headers = {'Authorization': 'Bearer ' + credentials['token']}
    connected = 0
    results = []
    map_revision = MapSpec().revision
    async def consumer(index):
        nonlocal connected
        count, last_sequence, last_step = 0, 0, 0
        cursor, gaps, registered = None, 0, False
        async with httpx.AsyncClient(base_url=credentials['origin'], headers=headers, timeout=None) as client:
            while gaps <= 16:
                params = {'stream_id': 'projection:' + map_revision}
                if cursor is not None:
                    params['cursor'] = cursor
                reconnect = False
                async with client.stream('GET', '/api/v1/stream', params=params) as response:
                    response.raise_for_status()
                    if not registered:
                        registered = True
                        connected += 1
                        if connected == 5:
                            save(args.ready, {'connected': 5})
                    event = None
                    async for line in response.aiter_lines():
                        if line.startswith('event: '):
                            event = line[7:]
                        elif line.startswith('data: '):
                            value = json.loads(line[6:])
                            if event == 'gap' and value.get('recover') == 'reconnect_from_last_applied_cursor' and cursor:
                                gaps += 1
                                reconnect = True
                                break
                            if event in ('gap', 'reset_required'):
                                raise RuntimeError(f'Viewer {index} cannot recover stream continuity: {value}')
                            if event == 'frame':
                                frame = value['frame']
                                sequence = frame['projection_sequence']
                                if sequence != last_sequence + 1:
                                    raise RuntimeError(f'Viewer {index} projection sequence gap')
                                last_sequence = sequence
                                cursor = value['cursor']
                                count += 1
                                for emission in frame['emissions']:
                                    last_step = max(last_step, emission['key'][2])
                                if last_step >= args.steps:
                                    results.append({'viewer': index, 'frames': count,
                                                    'last_sequence': last_sequence, 'last_step': last_step,
                                                    'backpressure_reconnects': gaps})
                                    return
                if not reconnect:
                    raise RuntimeError(f'Viewer {index} disconnected before the final update')
                await asyncio.sleep(.05)
        raise RuntimeError(f'Viewer {index} exceeded the bounded reconnect budget')
    await asyncio.wait_for(asyncio.gather(*(consumer(i) for i in range(5))), timeout=300)
    save(args.result, {'viewers': sorted(results, key=lambda r: r['viewer']),
                       'consumer': 'five real HTTP SSE readers; no browser rendering simulated'})


def telemetry():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,power.draw,clocks.sm',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    return result.stdout.strip().splitlines() if result.returncode == 0 else ['nvidia-smi unavailable']


def wait_file(path, process, timeout=30):
    started = time.monotonic()
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f'Benchmark subprocess exited early with {process.returncode}; inspect its log')
        if time.monotonic() - started > timeout:
            raise TimeoutError(f'Benchmark subprocess did not publish {path.name}')
        time.sleep(.01)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if process.poll() is None:
        raise RuntimeError('Benchmark process was not reaped')


def paired_summary(trials, numerator, denominator, target_percent):
    ratios = []
    for block in sorted({trial['block'] for trial in trials}):
        selected = {trial['condition']: trial for trial in trials if trial['block'] == block}
        ratios.append(selected[numerator]['measured_seconds'] / selected[denominator]['measured_seconds'])
    logs = [math.log(x) for x in ratios]
    mean, error = statistics.mean(logs), statistics.stdev(logs) / math.sqrt(len(logs))
    # Two-sided Student t, 95%, fixed small-n paired design (2..8 blocks).
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}[len(logs)]
    low, high = [(math.exp(mean + sign * critical * error) - 1) * 100 for sign in (-1, 1)]
    return {'comparison': numerator + '/' + denominator,
            'paired_overhead_percent': [(x - 1) * 100 for x in ratios],
            'geometric_mean_overhead_percent': (math.exp(mean) - 1) * 100,
            'paired_t_95_percent_interval': [low, high], 'candidate_target_percent': target_percent,
            'target_established_for_this_fixture': high <= target_percent,
            'method': 'paired log elapsed ratios, two-sided Student-t interval; not an equivalence claim across workloads'}


def orchestrate(args):
    import httpx
    import hypergan
    from hypergan.config import DEFAULT_TOML
    if not args.device.startswith('cuda'):
        raise ValueError('This throughput acceptance requires an explicit CUDA device')
    if not 2 <= args.repetitions <= 8 or not 1 <= args.warmup < args.steps:
        raise ValueError('Require 2..8 repetitions and 1 <= warmup < steps')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_root = args.output.parent / (args.output.stem + '-logs')
    log_root.mkdir(exist_ok=False)
    fixture = DEFAULT_TOML.replace('hidden = [64, 64]', 'hidden = [512, 512]')
    fixture = fixture.replace('num_particles = 20000', 'num_particles = 4096')
    fixture = fixture.replace('steps = 5', f'steps = {args.steps}').replace('batch_size = 16', 'batch_size = 256')
    fixture = fixture.replace('device = "cpu"', f'device = "{args.device}"').replace('count = 256', 'count = 16')
    save(log_root / 'recipe.json', {'toml': fixture, 'warmup': args.warmup, 'measured_steps': args.steps - args.warmup})
    package_root = Path(hypergan.__file__).parent
    sources = {str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package_root.rglob('*.py'))}
    import torch, numpy
    report = {'schema_version': 1, 'device': args.device, 'repetitions': args.repetitions,
              'steps': args.steps, 'warmup_steps': args.warmup,
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__,
                          'hypergan': str(package_root), 'cuda': torch.version.cuda},
              'package_source_sha256': sources, 'initial_gpu_telemetry': telemetry(), 'trials': [],
              'limitations': ['Existing GPU/desktop workloads are not stopped; telemetry records interference.',
                  'Consumers are actual SSE clients, not graphical browsers; browser-render latency is not measured.',
                  'Steady-state elapsed uses completed-update event seconds after warmup, excluding initialization/final export.',
                  'No training callback, per-step profiler or benchmark timestamp write is added to the measured loop.']}
    order = list(CONDITIONS) + (['bare'] if args.include_bare else [])
    random.Random(20260919).shuffle(order)
    state_hash = None
    with tempfile.TemporaryDirectory(prefix='hypergan-observation-training-') as temporary:
        root = Path(temporary)
        for block in range(args.repetitions):
            block_order = order[block % len(order):] + order[:block % len(order)]
            for position, condition in enumerate(block_order):
                trial_name = f'{block:02d}-{position}-{condition}'
                folder = root / trial_name
                folder.mkdir()
                run_dir, config = folder / 'run', folder / 'recipe.toml'
                config.write_text(fixture + ('\n[metrics]\npreset = "none"\n' if condition == 'none' else ''))
                result_path, credentials = folder / 'result.json', folder / 'credentials.json'
                before = telemetry()
                children = []
                with ExitStack() as stack:
                    def launch(role, *arguments):
                        log = stack.enter_context((log_root / f'{trial_name}-{role}.log').open('w'))
                        process = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), role,
                                                    '--run-dir', str(run_dir), *map(str, arguments)],
                                                   stdout=log, stderr=subprocess.STDOUT)
                        children.append(process)
                        return process
                    try:
                        consumer = None
                        if condition.startswith('server_'):
                            server = launch('server', '--credentials', credentials)
                            wait_file(credentials, server)
                            auth = json.loads(credentials.read_text())
                            for _ in range(300):
                                try:
                                    response = httpx.get(auth['origin'] + '/api/v1/capabilities',
                                        headers={'Authorization': 'Bearer ' + auth['token']}, timeout=1)
                                    response.raise_for_status()
                                    break
                                except httpx.ConnectError:
                                    time.sleep(.01)
                            else:
                                raise RuntimeError('Benchmark server did not become ready')
                            launch('projector')
                            if condition == 'server_five':
                                ready = folder / 'consumers-ready.json'
                                consumer = launch('consumers', '--credentials', credentials, '--ready', ready,
                                                  '--result', folder / 'consumers.json', '--steps', args.steps)
                                wait_file(ready, consumer)
                        trainer = launch('bare' if condition == 'bare' else 'train', '--config', config, '--result', result_path,
                                         '--device', args.device, '--steps', args.steps, '--warmup', args.warmup)
                        if trainer.wait(timeout=300) != 0:
                            raise RuntimeError(f'{trial_name} training failed; inspect log')
                        if consumer is not None and consumer.wait(timeout=30) != 0:
                            raise RuntimeError(f'{trial_name} consumer failed; inspect log')
                        trial = json.loads(result_path.read_text())
                        trial.update(block=block, position=position, condition=condition,
                                     gpu_before=before, gpu_after=telemetry())
                        if consumer is not None:
                            trial['consumers'] = json.loads((folder / 'consumers.json').read_text())
                        if state_hash is None:
                            state_hash = trial['numerical_state_sha256']
                        if trial['numerical_state_sha256'] != state_hash:
                            raise AssertionError(f'{trial_name} complete numerical state differs from baseline')
                        report['trials'].append(trial)
                        save(args.output, report)
                        print(json.dumps({'trial': trial_name, 'updates_per_second': trial['updates_per_second'],
                                          'complete_state_equal': True}), flush=True)
                    finally:
                        for process in reversed(children):
                            stop(process)
                # Keep exact metadata/receipts, remove potentially large checkpoint payloads.
                import shutil
                shutil.rmtree(folder)
    report['comparisons'] = [paired_summary(report['trials'], 'metrics', 'none', 1),
                             paired_summary(report['trials'], 'server_five', 'server_zero', 2),
                             paired_summary(report['trials'], 'server_five', 'metrics', 2)]
    if args.include_bare:
        report['comparisons'].append(paired_summary(report['trials'], 'metrics', 'bare', 1))
    report['all_complete_states_equal'] = True
    report['all_children_reaped'] = True
    report['final_gpu_telemetry'] = telemetry()
    save(args.output, report)
    print(json.dumps(report['comparisons'], indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='role', required=True)
    run = sub.add_parser('run')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--include-bare', action='store_true', help='Also compare validated updates without controller observation')
    run.add_argument('--steps', type=int, default=1088)
    run.add_argument('--warmup', type=int, default=64)
    run.add_argument('--repetitions', type=int, default=4)
    run.add_argument('--output', type=Path, required=True)
    for role in ('train', 'bare', 'server', 'projector', 'consumers'):
        child = sub.add_parser(role)
        child.add_argument('--run-dir', type=Path, required=True)
        if role in ('server', 'consumers'):
            child.add_argument('--credentials', type=Path, required=True)
        if role in ('train', 'bare', 'consumers'):
            child.add_argument('--steps', type=int, required=True)
            child.add_argument('--result', type=Path, required=True)
        if role in ('train', 'bare'):
            child.add_argument('--config', type=Path, required=True)
            child.add_argument('--device', required=True)
            child.add_argument('--warmup', type=int, required=True)
        if role == 'consumers':
            child.add_argument('--ready', type=Path, required=True)
    args = parser.parse_args()
    if args.role == 'run':
        orchestrate(args)
    elif args.role == 'train':
        training(args)
    elif args.role == 'bare':
        bare_training(args)
    elif args.role == 'server':
        serving(args)
    elif args.role == 'projector':
        projecting(args)
    else:
        asyncio.run(consuming(args))


if __name__ == '__main__':
    main()
