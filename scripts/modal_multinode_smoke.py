"""Tiny managed two-node A10 smoke test; no deployment, retries or data uploads.

Build one current wheel into dist/ first, then run:
    modal run scripts/modal_multinode_smoke.py --output reports/modal-multinode.json

The allocation is two complete A10 nodes (four GPUs each), with a 120-second
function timeout, 90-second worker budget and 240-second call wait limit.
Images are built before GPU work.
Only three fixed-initialization small GAN updates run. This is not recovery,
public launcher, image quality, throughput or RDMA qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import modal


if modal.is_local():
    ROOT = Path(__file__).resolve().parents[1]
    WHEELS = sorted((ROOT / 'dist').glob('hypergan-*.whl'))
    if len(WHEELS) != 1:
        raise RuntimeError('Build exactly one current HyperGAN wheel into dist/ before running this smoke test')
    WHEEL = WHEELS[0]
    WHEEL_SHA256 = hashlib.sha256(WHEEL.read_bytes()).hexdigest()
    REMOTE_WHEEL = f'/opt/wheels/{WHEEL.name}'
    image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install('torch==2.14.1', 'numpy==2.5.4', 'hndl==0.10.0',
                          'particlegan==0.9.3', 'foldlitedb==0.1.2')
             .add_local_file(WHEEL, remote_path=REMOTE_WHEEL, copy=True)
             .run_commands(f'python -m pip install --no-deps {REMOTE_WHEEL}')
             .add_local_file(ROOT / 'scripts' / 'multinode_smoke_worker.py',
                             remote_path='/opt/multinode_smoke_worker.py', copy=True)
             .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1'}))
else:
    image = None
app = modal.App('hypergan-multinode-smoke')


@app.function(image=image, gpu='A10:4', cpu=4, memory=8192,
              timeout=120, startup_timeout=120, retries=0,
              min_containers=0, max_containers=2, buffer_containers=0,
              single_use_containers=True)
@modal.clustered(size=2, rdma=False)
def smoke(wheel_sha256: str):
    import torch

    cluster = modal.Cluster.from_context()
    node_rank = cluster.container_rank()
    addresses = cluster.container_ips()
    if len(addresses) != 2 or len(set(addresses)) != 2 or torch.cuda.device_count() != 4:
        raise RuntimeError('Expected two distinct managed nodes with four A10 GPUs each')
    rendezvous = f'tcp://[{addresses[0]}]:29500'
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='hypergan-smoke-') as directory:
        root = Path(directory)
        children, logs = [], []
        try:
            for local_rank in range(4):
                log = (root / f'rank-{local_rank}.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen(
                    [sys.executable, '/opt/multinode_smoke_worker.py',
                     '--node-rank', str(node_rank), '--ranks-per-node', '4',
                     '--local-rank', str(local_rank), '--rendezvous', rendezvous,
                     '--output', str(root / f'rank-{local_rank}.json')],
                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
            deadline = started + 90
            for child in children:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            for log in logs:
                log.flush()
            if any(child.returncode != 0 for child in children):
                raise RuntimeError('\n'.join(path.read_text() for path in sorted(root.glob('*.log'))))
            results = [json.loads((root / f'rank-{rank}.json').read_text()) for rank in range(4)]
            if any(result != results[0] for result in results):
                raise RuntimeError('Local workers returned inconsistent global rank evidence')
            all_ranks = results[0]
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
            for log in logs:
                log.close()
    if (sorted(rank['rank'] for rank in all_ranks) != list(range(8))
            or len({rank['replica_sha256'] for rank in all_ranks}) != 1
            or any(rank['steps'] != 3 or not rank['finite'] for rank in all_ranks)):
        raise RuntimeError('Incomplete rank evidence or replicas disagree')
    return {'status': 'passed', 'nodes': 2, 'gpus_per_node': 4, 'world_size': 8,
            'steps': 3, 'rdma': False, 'wheel_sha256': wheel_sha256,
            'ranks': all_ranks, 'seconds': time.monotonic() - started,
            'scope': 'managed two-node NCCL complete GAN updates and replica agreement; '
                     'excludes public supervisor, recovery, image quality and throughput'}


@app.local_entrypoint()
def main(output: str = 'reports/modal-multinode-smoke.json'):
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    call = smoke.spawn(WHEEL_SHA256)
    try:
        result = call.get(timeout=240)
    except BaseException as error:
        call.cancel(terminate_containers=True)
        destination.write_text(json.dumps({'status': 'failed', 'wheel_sha256': WHEEL_SHA256,
                                          'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')
        raise
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
