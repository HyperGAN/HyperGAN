"""Sequential 1/2/4/8-A10 scaling smoke, bounded by an estimated resource budget.

Build one current wheel into dist/ and authenticate Modal locally, then:
    modal run scripts/modal_scaling.py --output reports/modal-scaling
Use benchmark_scaling.py --plan-modal for a provisioning-free estimate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import modal


def benchmark_module(path):
    spec = importlib.util.spec_from_file_location('hypergan_scaling_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[1]
    benchmark = benchmark_module(ROOT / 'scripts' / 'benchmark_scaling.py')
    wheels = sorted((ROOT / 'dist').glob('hypergan-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Build exactly one current HyperGAN wheel into dist/')
    wheel = wheels[0]
    WHEEL_SHA256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install('torch==2.14.1', 'numpy==2.5.4', 'hndl==0.10.0',
                          'particlegan==0.9.3', 'foldlitedb==0.1.2')
             .add_local_file(wheel, remote_path=f'/opt/wheels/{wheel.name}', copy=True)
             .run_commands(f'python -m pip install --no-deps /opt/wheels/{wheel.name}')
             .add_local_file(ROOT / 'scripts' / 'benchmark_scaling.py',
                             remote_path='/opt/benchmark_scaling.py', copy=True)
             .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1'}))
else:
    image = None
app = modal.App('hypergan-scaling-smoke')


def run_node(spec):
    benchmark = benchmark_module('/opt/benchmark_scaling.py')
    node_rank, rendezvous = 0, 'tcp://127.0.0.1:29500'
    if spec['nodes'] > 1:
        cluster = modal.Cluster.from_context()
        addresses = cluster.container_ips()
        if len(addresses) != spec['nodes'] or len(set(addresses)) != spec['nodes']:
            raise ValueError('Managed allocation differs from the requested nodes')
        node_rank = cluster.container_rank()
        rendezvous = f'tcp://[{addresses[0]}]:29500'
    with tempfile.TemporaryDirectory(prefix='hypergan-scaling-') as root:
        return benchmark.launch_node(spec, node_rank, rendezvous, root)


def limits(gpus):
    return dict(image=image, gpu=f'A10:{gpus}', cpu=(gpus, gpus),
                memory=(gpus * 4096, gpus * 4096),
                timeout=105, startup_timeout=45, retries=0, min_containers=0,
                buffer_containers=0, single_use_containers=True)


@app.function(**limits(1), max_containers=1)
def one(spec):
    return run_node(spec)


@app.function(**limits(2), max_containers=1)
def two(spec):
    return run_node(spec)


@app.function(**limits(4), max_containers=1)
def four(spec):
    return run_node(spec)


@app.function(**limits(4), max_containers=2)
@modal.clustered(size=2, rdma=False)
def two_nodes(spec):
    return run_node(spec)


@app.local_entrypoint()
def main(output: str = 'reports/modal-scaling', topologies: str = '1,2,4,2x4',
         mode: str = 'strong', workload: str = 'images', batch_size: int = 64,
         warmup: int = 10, steps: int = 50, budget_usd: float = 2.0,
         call_wait_seconds: int = 180):
    settings = benchmark.options(mode, workload, batch_size, warmup, steps)
    plan = benchmark.modal_plan(settings, benchmark.topologies(topologies), budget_usd, call_wait_seconds)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    functions = {(1, 1): one, (1, 2): two, (1, 4): four, (2, 4): two_nodes}
    results, calls, submitted = [], [], []
    try:
        for spec in plan['profiles']:
            started = time.monotonic()
            call = functions[(spec['nodes'], spec['gpus_per_node'])].spawn(spec)
            calls.append(call)
            submitted.append({'topology': benchmark.label(spec['nodes'], spec['gpus_per_node']),
                              'function_call_id': call.object_id})
            (root / 'calls.json').write_text(json.dumps(submitted, indent=2) + '\n')
            print(f"Submitted {submitted[-1]['topology']}: {call.object_id}; "
                  f"waiting up to {plan['call_wait_seconds']} seconds", flush=True)
            result = call.get(timeout=plan['call_wait_seconds'])
            # Validate the returned rank evidence again on the client.
            checked = benchmark.summarize(result['ranks'], spec)
            checked['node_wall_seconds'] = result['node_wall_seconds']
            checked['call_wall_seconds'] = time.monotonic() - started
            results.append(checked)
            (root / 'partial.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f"{checked['topology']}: {checked['samples_per_second']:.2f} samples/sec", flush=True)
        report = benchmark.comparison(results, gpu_second=benchmark.A10_SECOND)
        report.update(plan=plan, wheel_sha256=WHEEL_SHA256, rdma=False)
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (root / 'comparison.md').write_text(benchmark.markdown(report))
        print(benchmark.markdown(report))
    except BaseException as error:
        for call in calls:
            try:
                call.cancel(terminate_containers=True)
            except Exception as cleanup_error:
                print(f'Call cancellation failed: {cleanup_error}', flush=True)
        (root / 'failure.json').write_text(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                                                     'error': str(error),
                                                     'complete_profiles': len(results)}, indent=2) + '\n')
        raise
