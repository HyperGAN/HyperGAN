"""Bounded scaling measurements with fixed initialization and synthetic inputs.

Local CUDA example:
    python scripts/benchmark_scaling.py --topologies 1,2 --output reports/local-scaling
Planning a Modal run without importing its SDK or provisioning resources:
    python scripts/benchmark_scaling.py --plan-modal --topologies 1,2,4,2x4
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


# Modal list prices verified 2026-10-26; estimates are not billing receipts.
A10_SECOND = 0.000306
CPU_SECOND = 0.0000131
GIB_SECOND = 0.00000222
CALL_SECONDS = 180
WORKER_SECONDS = 90
BUILD_RESERVE_USD = 0.50


def topologies(value):
    result = []
    for item in value.split(','):
        parts = item.strip().split('x')
        if len(parts) == 1:
            nodes, gpus = 1, int(parts[0])
        elif len(parts) == 2:
            nodes, gpus = map(int, parts)
        else:
            raise ValueError('Topology must be GPUS or NODESxGPUS_PER_NODE')
        if nodes < 1 or gpus < 1 or nodes * gpus > 64:
            raise ValueError('Topology must have 1–64 total ranks')
        pair = (nodes, gpus)
        if pair in result:
            raise ValueError('Duplicate topology')
        result.append(pair)
    return result


def label(nodes, gpus):
    return str(gpus) if nodes == 1 else f'{nodes}x{gpus}'


def options(mode='strong', workload='images', batch_size=64, warmup=10, steps=50):
    if mode not in ('strong', 'weak') or workload not in ('images', 'mlp'):
        raise ValueError('Unknown scaling mode or workload')
    if any(type(value) is not int or value < 1 for value in (batch_size, warmup, steps)):
        raise ValueError('Batch size, warmup and measured steps must be positive integers')
    if warmup + steps > 1000:
        raise ValueError('Smoke benchmarks are limited to 1000 total updates')
    return dict(mode=mode, workload=workload, batch_size=batch_size, warmup=warmup, steps=steps)


def payload(settings, nodes, gpus, backend='nccl'):
    settings = options(**settings)
    if backend not in ('nccl', 'gloo') or nodes < 1 or gpus < 1 or nodes * gpus > 64:
        raise ValueError('Invalid backend or topology')
    world_size = nodes * gpus
    batch = settings['batch_size'] * (world_size if settings['mode'] == 'weak' else 1)
    if batch > 256 or batch % world_size:
        raise ValueError('Global batch must be <=256 and divisible by world size')
    return dict(settings, nodes=nodes, gpus_per_node=gpus, world_size=world_size,
                global_batch_size=batch, backend=backend)


def modal_plan(settings, selected, budget_usd=2.0, call_wait_seconds=CALL_SECONDS):
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError('Budget must be finite and positive')
    if type(call_wait_seconds) is not int or call_wait_seconds < 1:
        raise ValueError('Call wait must be a positive integer number of seconds')
    allowed = {(1, 1), (1, 2), (1, 4), (2, 4)}
    if not selected or any(pair not in allowed for pair in selected):
        raise ValueError('Modal A10 topologies supported here: 1,2,4,2x4')
    profiles = [payload(settings, *pair) for pair in selected]
    # Sequential calls, one CPU core and 4 GiB requested per GPU. Conservatively
    # charge the entire call wait, including queue time, against the estimate.
    ceiling = BUILD_RESERVE_USD + sum(p['world_size'] * call_wait_seconds *
                                     (A10_SECOND + CPU_SECOND + 4 * GIB_SECOND) for p in profiles)
    if ceiling > budget_usd:
        raise ValueError(f'Conservative estimate ${ceiling:.3f} exceeds budget ${budget_usd:.3f}')
    return {'profiles': profiles, 'estimated_ceiling_usd': ceiling, 'budget_usd': budget_usd,
            'build_reserve_usd': BUILD_RESERVE_USD, 'call_wait_seconds': call_wait_seconds,
            'pricing_date': '2026-10-26',
            'cost_scope': 'list-price estimate including reserved build allowance; not an account spending limit'}


def summarize(ranks, spec):
    expected = spec['world_size']
    if len(ranks) != expected or sorted(row['rank'] for row in ranks) != list(range(expected)):
        raise ValueError('Missing, duplicate or unexpected rank evidence')
    for row in ranks:
        if (row['node_rank'] != row['rank'] // spec['gpus_per_node']
                or row['local_rank'] != row['rank'] % spec['gpus_per_node']
                or row['completed_steps'] != spec['warmup'] + spec['steps']
                or row['global_batch_size'] != spec['global_batch_size']
                or row['spec'] != spec or not row['finite'] or not row['parameters_changed']):
            raise ValueError('Rank evidence differs from the requested benchmark')
        if (len(row['step_seconds']) != spec['steps']
                or not all(math.isfinite(t) and t > 0 for t in row['step_seconds'])
                or not math.isfinite(row['measured_seconds']) or row['measured_seconds'] <= 0
                or row['measured_seconds'] < sum(row['step_seconds']) * 0.999999):
            raise ValueError('Invalid measured timing')
    for key in ('initial_sha256', 'final_sha256', 'config_sha256', 'gpu', 'torch', 'cuda', 'replica_check_every'):
        if len({json.dumps(row[key], sort_keys=True) for row in ranks}) != 1:
            raise ValueError(f'Ranks disagree on {key}')
    seconds = max(row['measured_seconds'] for row in ranks)
    step_times = [max(row['step_seconds'][index] for row in ranks) for index in range(spec['steps'])]
    image_rate = spec['global_batch_size'] * spec['steps'] / seconds
    ordered = sorted(step_times)
    return {'topology': label(spec['nodes'], spec['gpus_per_node']), 'spec': spec,
            'samples_per_second': image_rate, 'measured_seconds': seconds,
            'median_step_ms': statistics.median(step_times) * 1000,
            'p95_step_ms': ordered[math.ceil(.95 * len(ordered)) - 1] * 1000,
            'peak_allocated_mib': max(row['peak_allocated_bytes'] for row in ranks) / 2**20,
            'initial_sha256': ranks[0]['initial_sha256'],
            'config_sha256': ranks[0]['config_sha256'],
            'ranks': ranks, 'status': 'passed'}


def comparison(results, gpu_second=None):
    if not results:
        raise ValueError('No complete benchmarks')
    first = results[0]
    base = first['samples_per_second']
    for result in results:
        for field in ('mode', 'workload', 'batch_size', 'warmup', 'steps', 'backend'):
            if result['spec'][field] != first['spec'][field]:
                raise ValueError('Cannot compare different benchmark cohorts')
        if result['initial_sha256'] != first['initial_sha256']:
            raise ValueError('Initial model/prior/EMA state changed across topologies')
        if result['spec']['mode'] == 'strong' and result['config_sha256'] != first['config_sha256']:
            raise ValueError('Strong scaling requires one unchanged resolved recipe')
        if any(result['ranks'][0][field] != first['ranks'][0][field] for field in ('gpu', 'torch', 'cuda')):
            raise ValueError('Hardware or runtime changed across topologies')
        result['throughput_speedup'] = result['samples_per_second'] / base
        result['scaling_efficiency'] = result['throughput_speedup'] / (
            result['spec']['world_size'] / first['spec']['world_size'])
        if gpu_second is not None:
            rate = result['spec']['world_size'] * (gpu_second + CPU_SECOND + 4 * GIB_SECOND)
            result['estimated_steady_samples_per_dollar'] = result['samples_per_second'] / rate
    return {'schema_version': 1, 'status': 'passed', 'baseline': first['topology'], 'results': results,
            'scope': 'complete trainer updates on resident synthetic inputs; normal checks enabled; '
                     'excludes public supervisor, data loading, checkpoint I/O, recovery and image quality',
            'cost_scope': 'steady requested GPU/CPU/RAM list prices; excludes startup, builds and billing adjustments'}


def markdown(report):
    unit = 'Images/sec' if report['results'][0]['spec']['workload'] == 'images' else 'Samples/sec'
    lines = [f'| Topology | {unit} | Speedup | Efficiency | Median ms | p95 ms | Peak MiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['results']:
        lines.append(f"| {row['topology']} | {row['samples_per_second']:.2f} | "
                     f"{row['throughput_speedup']:.2f}x | {100 * row['scaling_efficiency']:.1f}% | "
                     f"{row['median_step_ms']:.2f} | {row['p95_step_ms']:.2f} | {row['peak_allocated_mib']:.1f} |")
    return '\n'.join(lines) + '\n'


def launch_node(spec, node_rank, rendezvous, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    started = time.monotonic()
    try:
        for local_rank in range(spec['gpus_per_node']):
            log = (root / f'rank-{local_rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen(
                [sys.executable, *(['-I'] if sys.flags.isolated else []), str(Path(__file__).resolve()),
                 '--worker', '--payload', json.dumps(spec), '--node-rank', str(node_rank),
                 '--local-rank', str(local_rank), '--rendezvous', rendezvous,
                 '--output', str(root / f'rank-{local_rank}.json')],
                stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
        deadline = started + WORKER_SECONDS
        for child in children:
            child.wait(timeout=max(.1, deadline - time.monotonic()))
        for log in logs:
            log.flush()
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('\n'.join(path.read_text() for path in sorted(root.glob('*.log')))[-32000:])
        rows = [json.loads((root / f'rank-{rank}.json').read_text()) for rank in range(spec['gpus_per_node'])]
        if any(row != rows[0] for row in rows):
            raise ValueError('Local workers disagree on global rank evidence')
        result = summarize(rows[0], spec)
        result['node_wall_seconds'] = time.monotonic() - started
        return result
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for log in logs:
            log.close()


def worker(args):
    started = time.monotonic()
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from hypergan.config import fingerprint, resolve_config
    from hypergan.distributed_training import ReplicatedTrainer
    from hypergan.training import ReferenceTrainer

    spec = json.loads(args.payload)
    validated = payload({k: spec[k] for k in ('mode', 'workload', 'batch_size', 'warmup', 'steps')},
                        spec['nodes'], spec['gpus_per_node'], spec['backend'])
    if spec != validated or not 0 <= args.node_rank < spec['nodes'] or not 0 <= args.local_rank < spec['gpus_per_node']:
        raise ValueError('Invalid worker request')
    rank = args.node_rank * spec['gpus_per_node'] + args.local_rank
    world = spec['world_size']
    torch.set_num_threads(1)
    device = 'cuda' if spec['backend'] == 'nccl' else 'cpu'
    if device == 'cuda':
        if torch.cuda.device_count() < spec['gpus_per_node']:
            raise ValueError('Requested GPU ranks exceed the visible allocation')
        torch.cuda.set_device(args.local_rank)
    if world > 1:
        dist.init_process_group(spec['backend'], init_method=args.rendezvous, rank=rank,
                                world_size=world, timeout=timedelta(seconds=25))
    try:
        image = spec['workload'] == 'images'
        components = ({
            'generator': {'factory': 'hypergan.image_components:CIFARGenerator',
                          'args': {'z_dim': 16}, 'inputs': {'z': 'latent'}},
            'discriminator': {'factory': 'hypergan.image_components:_PixelDiscriminator',
                              'args': {'width': 16}, 'inputs': {'x': 'candidate', 'xt': 'batch.context'}},
        } if image else {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [64, 64]},
                          'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'mlp', 'args': {'input_dim': 2, 'output_dim': 1, 'hidden': [64, 64]},
                              'inputs': {'x': 'candidate'}},
        })
        config = resolve_config({
            'training': {'device': device, 'batch_size': spec['global_batch_size'], 'steps': 100000,
                         'seed': 42, 'initialization': 'deterministic_orthogonal',
                         'backend': {'cudnn_benchmark': False, 'cudnn_deterministic': True,
                                     'deterministic_algorithms': True, 'matmul_allow_tf32': False,
                                     'cudnn_allow_tf32': False, 'cublas_workspace_config': ':4096:8'}},
            'components': components, 'prior': {'args': {'num_particles': 256, 'z_dim': 16 if image else 4}},
            'gradient_penalty': {'lazy_k': 1, 'kappa': .01}, 'sampling': {'count': 4},
        })
        trainer = (ReplicatedTrainer(config, world_size=world, nodes=spec['nodes'], node_rank=args.node_rank,
                                     ranks_per_node=spec['gpus_per_node'], rendezvous=args.rendezvous)
                   if world > 1 else ReferenceTrainer(config))

        def shared(optimizers=False):
            state = {name: getattr(trainer, name).state_dict()
                     for name in ('graph', 'prior', 'ema_graph', 'ema_prior')}
            if optimizers:
                state['optimizers'] = [trainer.opt_g.state_dict(), trainer.opt_d.state_dict()]
                state['base_lrs'] = trainer.base_lrs
            return ReplicatedTrainer._digest(state)

        initial = shared()
        batch_size = spec['global_batch_size']
        width = 3 * 32 * 32 if image else 2
        real = torch.linspace(-1, 1, batch_size * width, device=trainer.device)
        real = real.reshape((batch_size, 3, 32, 32) if image else (batch_size, 2)).chunk(world)[rank]
        batch = {'real': real, **({'context': torch.zeros_like(real)} if image else {})}
        ids = torch.arange(batch_size, device=trainer.device).chunk(world)[rank]

        def update():
            return trainer.update(batch, (trainer.prior.z[ids], ids))[0]

        def synchronize():
            if device == 'cuda':
                torch.cuda.synchronize(trainer.device)

        for _ in range(spec['warmup']):
            update()
        if world > 1:
            trainer._assert_replicas('scaling warmup')
            dist.barrier()
        synchronize()
        warmup_seconds = time.monotonic() - started
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.device)
        timings, metrics = [], []
        window = time.perf_counter()
        for _ in range(spec['steps']):
            before = time.perf_counter()
            row = update()
            synchronize()
            timings.append(time.perf_counter() - before)
            metrics.append(row)
        measured = time.perf_counter() - window
        peak = torch.cuda.max_memory_allocated(trainer.device) if device == 'cuda' else 0
        if world > 1:
            trainer._assert_replicas('scaling final complete state')
        finite = all(math.isfinite(float(row[key])) for row in metrics
                     for key in ('d_loss', 'g_loss', 'prior_loss', 'gradient_penalty'))
        row = {'rank': rank, 'node_rank': args.node_rank, 'local_rank': args.local_rank, 'spec': spec,
               'completed_steps': trainer.step, 'global_batch_size': batch_size,
               'initial_sha256': initial, 'final_sha256': shared(optimizers=True),
               'parameters_changed': shared() != initial, 'finite': finite,
               'config_sha256': fingerprint(config), 'step_seconds': timings, 'measured_seconds': measured,
               'peak_allocated_bytes': peak, 'startup_and_warmup_seconds': warmup_seconds,
               'worker_wall_seconds': time.monotonic() - started,
               'gpu': torch.cuda.get_device_name(args.local_rank) if device == 'cuda' else None,
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'replica_check_every': (trainer.aggregation_strategy.replica_check_every if world > 1 else None),
               'last_losses': {key: float(metrics[-1][key]) for key in ('d_loss', 'g_loss', 'prior_loss', 'gradient_penalty')}}
        rows = [row]
        if world > 1:
            rows = [None] * world
            dist.all_gather_object(rows, row)
        summarize(rows, spec)
        Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topologies', default='1,2')
    parser.add_argument('--mode', choices=('strong', 'weak'), default='strong')
    parser.add_argument('--workload', choices=('images', 'mlp'), default='images')
    parser.add_argument('--batch-size', type=int, default=64, help='Global batch for strong; per-GPU batch for weak')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--output', default='reports/local-scaling')
    parser.add_argument('--plan-modal', action='store_true')
    parser.add_argument('--budget-usd', type=float, default=2.0)
    parser.add_argument('--call-wait-seconds', type=int, default=CALL_SECONDS,
                        help='Modal result wait, including scheduling; execution limits stay fixed')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--payload', help=argparse.SUPPRESS)
    parser.add_argument('--node-rank', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--local-rank', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--rendezvous', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    settings = options(args.mode, args.workload, args.batch_size, args.warmup, args.steps)
    selected = topologies(args.topologies)
    if args.plan_modal:
        print(json.dumps(modal_plan(settings, selected, args.budget_usd, args.call_wait_seconds), indent=2))
        return
    if any(nodes != 1 for nodes, _ in selected):
        raise ValueError('Local driver supports one physical host; use Modal for multiple nodes')
    specs = [payload(settings, *pair, backend=args.backend) for pair in selected]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for spec in specs:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            results.append(launch_node(spec, 0, f'tcp://127.0.0.1:{port}', root / label(spec['nodes'], spec['gpus_per_node'])))
            (root / 'partial.json').write_text(json.dumps(results, indent=2) + '\n')
        report = comparison(results)
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (root / 'comparison.md').write_text(markdown(report))
        print(markdown(report))
    except BaseException as error:
        (root / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
