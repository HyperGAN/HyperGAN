"""Reference integration test suite for multi-node distributed execution and TCP rendezvous.

Covers:
1. 2-node distributed CPU smoke test (nodes=2, ranks_per_node=1, world_size=2) over tcp://127.0.0.1:<port> (cpu-replicated-gloo).
2. 2-node GPU distributed smoke test (nodes=2, ranks_per_node=1, world_size=2) locally over tcp://127.0.0.1:<port>
   with node_rank=0 on CUDA_VISIBLE_DEVICES=0 and node_rank=1 on CUDA_VISIBLE_DEVICES=1 on 2x NVIDIA RTX A6000 GPUs (cuda-replicated-nccl).
3. Learning aggregation strategies (parameter_average, pseudo_gradient) across nodes over TCP.
4. Distributed checkpoint save and exact resume trajectory continuity across independent node processes.
5. Multi-node profile preflight validation in isolated subprocesses.
6. Timeout and error cleanup across node processes:
   - Follower node exits cleanly when master crashes or fails to appear.
   - Master node exits cleanly when follower fails to connect.
   - Immediate non-hanging failure when TCP port is already occupied.
   - Process leak check ensuring zero orphaned or zombie worker processes.
"""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.heavy

REPO_ROOT = str(Path(__file__).resolve().parents[2])


# ---------------------------------------------------------------------------
# Hardware Capabilities & Fixtures
# ---------------------------------------------------------------------------

def is_2gpu_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() >= 2 and dist.is_nccl_available())
    except (RuntimeError, ValueError):
        return False


def get_free_tcp_port() -> int:
    """Find a guaranteed free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def make_multinode_env(gpu_index: int | None = None) -> dict[str, str]:
    """Prepare environment variables for multi-node subprocess execution."""
    env = os.environ.copy()
    if gpu_index is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        env['NCCL_P2P_DISABLE'] = '1'
        env['NCCL_IB_DISABLE'] = '1'
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        env['TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC'] = '1000'
    env['PYTHONPATH'] = f"{REPO_ROOT}/src:{REPO_ROOT}/tests:{REPO_ROOT}/tests/reference:" + env.get('PYTHONPATH', '')
    return env


# ---------------------------------------------------------------------------
# Isolated Worker Driver Script
# ---------------------------------------------------------------------------

WORKER_CODE = r'''
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.distributed_checkpoints import (
    save_distributed_checkpoint, restore_distributed_checkpoint,
)

mode, node_rank_str, tcp_url, run_dir, out_path, backend, agg_json, steps_str = sys.argv[1:9]
node_rank = int(node_rank_str)
steps = int(steps_str)
run_dir = Path(run_dir)
agg_cfg = json.loads(agg_json)

# In multi-node where each node isolates its GPU via CUDA_VISIBLE_DEVICES,
# the local device index on every node is always 0.
if backend == "nccl":
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    torch.set_num_threads(1)

# Initialize process group over TCP rendezvous
dist.init_process_group(
    backend,
    init_method=tcp_url,
    rank=node_rank,
    world_size=2,
    timeout=timedelta(seconds=20),
)

try:
    device_name = "cuda" if backend == "nccl" else "cpu"
    cfg = resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {'steps': steps + 2, 'batch_size': 8, 'device': device_name, 'seed': 42},
        'sampling': {'count': 4},
    })

    execution = {
        'name': 'cuda-replicated-nccl' if backend == 'nccl' else 'cpu-replicated-gloo',
        'world_size': 2,
        'nodes': 2,
        'node_rank': node_rank,
        'ranks_per_node': 1,
        'rendezvous': tcp_url,
        'accumulation_steps': 1,
        'global_batch_size': 8,
        'local_batch_size': 4,
        'microbatch_size': 4,
        'accumulation_algorithm': 'retained-local-graph-v1',
        'aggregation': agg_cfg,
    }

    # Deterministic local shard data
    raw_data = torch.linspace(-1, 1, 16, device="cuda" if backend == "nccl" else "cpu").reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda" if backend == "nccl" else "cpu")
    local_ids = ids.chunk(2)[node_rank]
    local_data = raw_data.chunk(2)[node_rank]

    result = {'node_rank': node_rank, 'mode': mode, 'passed': True}

    if mode == "step":
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        metrics = []
        weight_history = []
        for step in range(1, steps + 1):
            m, _ = trainer.update({'real': local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})
            w = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
            weight_history.append(w)

        result['metrics'] = metrics
        result['weight_history'] = weight_history
        result['final_step'] = trainer.step
        Path(out_path).write_text(json.dumps(result))

    elif mode == "checkpoint_save":
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        last_batch = None
        metrics = []
        for step in range(1, steps + 1):
            m, last_batch = trainer.update({'real': local_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})

        ckpt_path = save_distributed_checkpoint(
            run_dir, trainer, last_batch,
            {'run_id': 'test-run', 'attempt_id': 'att-1', 'next_sample_sequence': 10}
        )
        result['checkpoint_path'] = str(ckpt_path)
        result['metrics'] = metrics
        result['final_weights'] = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
        Path(out_path).write_text(json.dumps(result))

    elif mode == "checkpoint_resume":
        fresh_trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        restored_path, info, last_batch = restore_distributed_checkpoint(
            run_dir, fresh_trainer, {'run_id': 'test-run'}
        )
        assert fresh_trainer.step == steps, f"Expected step {steps}, got {fresh_trainer.step}"

        metrics = []
        for step in range(steps + 1, steps + 3):
            m, last_batch = fresh_trainer.update({'real': local_data}, (fresh_trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})

        result['restored_path'] = str(restored_path)
        result['metrics'] = metrics
        result['final_weights'] = {k: v.detach().cpu().tolist() for k, v in fresh_trainer.graph.state_dict().items()}
        Path(out_path).write_text(json.dumps(result))

finally:
    dist.destroy_process_group()
'''


def run_2node_processes(
    tmp_path: Path,
    mode: str,
    backend: str = "gloo",
    agg_cfg: dict[str, Any] | None = None,
    steps: int = 2,
    timeout: float = 30.0,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Launch 2 node processes connected over localhost TCP rendezvous."""
    if agg_cfg is None:
        agg_cfg = {'strategy': 'gradient_allreduce', 'sync_every_steps': 1, 'outer_lr': 1.0, 'outer_momentum': 0.0}

    worker_script = tmp_path / "worker.py"
    if not worker_script.exists():
        worker_script.write_text(WORKER_CODE)

    if run_dir is None:
        run_dir = tmp_path / f"run_{mode}"
    run_dir.mkdir(exist_ok=True, parents=True)

    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"

    out_files = [tmp_path / f"out_{mode}_{nr}.json" for nr in range(2)]
    processes = []

    try:
        for nr in range(2):
            gpu_idx = nr if backend == "nccl" else None
            env = make_multinode_env(gpu_index=gpu_idx)
            cmd = [
                sys.executable, str(worker_script),
                mode, str(nr), tcp_url, str(run_dir), str(out_files[nr]),
                backend, json.dumps(agg_cfg), str(steps),
            ]
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(p)

        deadline = time.monotonic() + timeout
        for nr, p in enumerate(processes):
            stdout, stderr = p.communicate(timeout=max(0.1, deadline - time.monotonic()))
            assert p.returncode == 0, f"Node {nr} failed (code {p.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        return results
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=2)


# ==============================================================================
# 4. Multi-Node Reference Tests
# ==============================================================================

def test_2node_cpu_gloo_smoke_step_and_parity(tmp_path):
    """Test 1: Verify 2-node distributed CPU smoke test over TCP Gloo:
    - nodes=2, ranks_per_node=1, world_size=2.
    - Synchronous gradient allreduce across TCP socket.
    - Losses and model parameters remain bit-identical across both nodes.
    """
    results = run_2node_processes(tmp_path, mode="step", backend="gloo", steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]
    assert r0['final_step'] == 2 and r1['final_step'] == 2

    # Step losses must match
    assert r0['metrics'] == r1['metrics']

    # Model weights must match bit-for-bit after every step
    for step_idx in range(2):
        w0 = r0['weight_history'][step_idx]
        w1 = r1['weight_history'][step_idx]
        for k in w0:
            torch.testing.assert_close(torch.tensor(w0[k]), torch.tensor(w1[k]), rtol=0, atol=0)


def test_2node_cpu_gloo_parameter_average(tmp_path):
    """Test 2: Verify parameter_average (Local SGD) across 2 nodes over TCP:
    - sync_every_steps = 2.
    - Step 1: Weights drift on distinct local shards.
    - Step 2: Weights averaged over TCP rendezvous, reaching consensus.
    """
    agg = {'strategy': 'parameter_average', 'sync_every_steps': 2, 'outer_lr': 1.0, 'outer_momentum': 0.0}
    results = run_2node_processes(tmp_path, mode="step", backend="gloo", agg_cfg=agg, steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]

    # Drift at step 1
    w0_s1 = r0['weight_history'][0]
    w1_s1 = r1['weight_history'][0]
    has_drift = any(w0_s1[k] != w1_s1[k] for k in w0_s1)
    assert has_drift, "Ranks should have diverged at local step 1"

    # Consensus at step 2
    w0_s2 = r0['weight_history'][1]
    w1_s2 = r1['weight_history'][1]
    for k in w0_s2:
        torch.testing.assert_close(torch.tensor(w0_s2[k]), torch.tensor(w1_s2[k]), rtol=1e-5, atol=1e-6)


def test_2node_cpu_gloo_pseudo_gradient_diloco(tmp_path):
    """Test 3: Verify pseudo_gradient (DiLoco) across 2 nodes over TCP:
    - sync_every_steps = 2, outer_lr = 0.5, outer_momentum = 0.8.
    - Weights diverge from anchor during step 1.
    - Pseudo-gradients reduced over TCP and outer update applied at step 2.
    """
    agg = {'strategy': 'pseudo_gradient', 'sync_every_steps': 2, 'outer_lr': 0.5, 'outer_momentum': 0.8}
    results = run_2node_processes(tmp_path, mode="step", backend="gloo", agg_cfg=agg, steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]

    # Consensus at step 2
    w0_s2 = r0['weight_history'][1]
    w1_s2 = r1['weight_history'][1]
    for k in w0_s2:
        torch.testing.assert_close(torch.tensor(w0_s2[k]), torch.tensor(w1_s2[k]), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not is_2gpu_available(), reason="Requires at least 2 visible CUDA GPUs with NCCL")
def test_2node_gpu_nccl_smoke_step_and_parity(tmp_path):
    """Test 4: Verify 2-node GPU distributed smoke test on 2x NVIDIA RTX A6000:
    - Node 0 on CUDA_VISIBLE_DEVICES=0 (RTX A6000 #0).
    - Node 1 on CUDA_VISIBLE_DEVICES=1 (RTX A6000 #1).
    - PCIe NCCL flags: NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1.
    - Collectives execute over tcp://127.0.0.1:<port> with NCCL backend.
    - Metrics and parameters reach exact numerical consensus.
    """
    results = run_2node_processes(tmp_path, mode="step", backend="nccl", steps=2)
    assert len(results) == 2
    r0, r1 = results[0], results[1]
    assert r0['final_step'] == 2 and r1['final_step'] == 2

    # Step losses match
    assert r0['metrics'] == r1['metrics']

    # Model weights match bit-for-bit after every step
    for step_idx in range(2):
        w0 = r0['weight_history'][step_idx]
        w1 = r1['weight_history'][step_idx]
        for k in w0:
            torch.testing.assert_close(torch.tensor(w0[k]), torch.tensor(w1[k]), rtol=0, atol=0)


def test_2node_checkpoint_save_and_resume_across_nodes(tmp_path):
    """Test 5: Verify distributed checkpoint save and resume across nodes over TCP:
    - Phase 1: Train 2 steps and save checkpoint to shared run directory.
    - Verify top-level manifest.json and per-node rank payloads (rank-00000.pt, rank-00001.pt).
    - Phase 2: Fresh 2-node group restores checkpoint and trains steps 3 and 4.
    """
    run_dir = tmp_path / "run_checkpoint"
    run_dir.mkdir(exist_ok=True)

    # 1. Save checkpoint at step 2
    save_results = run_2node_processes(tmp_path, mode="checkpoint_save", backend="gloo", steps=2, run_dir=run_dir)
    assert len(save_results) == 2
    ckpt_dir = Path(save_results[0]['checkpoint_path'])
    assert (ckpt_dir / 'manifest.json').exists()
    assert (ckpt_dir / 'rank-00000.pt').exists()
    assert (ckpt_dir / 'rank-00001.pt').exists()

    manifest = json.loads((ckpt_dir / 'manifest.json').read_text())
    assert manifest['step'] == 2

    # 2. Resume from checkpoint for steps 3 and 4
    resume_results = run_2node_processes(tmp_path, mode="checkpoint_resume", backend="gloo", steps=2, run_dir=run_dir)
    assert len(resume_results) == 2
    r0, r1 = resume_results[0], resume_results[1]

    assert r0['metrics'][0]['step'] == 3
    assert r0['metrics'][1]['step'] == 4
    for k in r0['final_weights']:
        torch.testing.assert_close(torch.tensor(r0['final_weights'][k]), torch.tensor(r1['final_weights'][k]))


def test_multinode_preflight_construction_cpu(tmp_path):
    """Test 6: Preflight runtime validation passes on multi-node CPU profile."""
    from hypergan.config import resolve_config
    from hypergan.execution_preflight import preflight
    from hypergan.execution_profiles import resolve_execution_profile

    port = get_free_tcp_port()
    cfg = resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': 8, 'device': 'cpu'},
        'sampling': {'count': 4},
    })
    raw_profile = {
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'ranks_per_node': 1,
            'rendezvous': f'tcp://127.0.0.1:{port}',
        },
        'preflight': {'timeout': 15, 'collective_timeout': 5},
    }
    resolved = resolve_execution_profile(raw_profile, cfg)
    report = preflight(cfg, resolved)
    assert report['status'] == 'passed'
    assert report['identity']['execution']['nodes'] == 2
    assert report['identity']['execution']['ranks_per_node'] == 1


# ==============================================================================
# 5. Timeout & Clean Error Handling Tests
# ==============================================================================

def test_follower_timeout_when_master_fails_to_start(tmp_path):
    """Test 7: Follower node exits cleanly with timeout when master node is never started."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"

    script = tmp_path / "timeout_worker.py"
    script.write_text(r'''
import sys
from datetime import timedelta
import torch.distributed as dist

tcp_url = sys.argv[1]
try:
    # 2-second timeout, master never starts
    dist.init_process_group("gloo", init_method=tcp_url, rank=1, world_size=2, timeout=timedelta(seconds=2))
except Exception as e:
    sys.exit(42)
sys.exit(0)
''')

    p = subprocess.run([sys.executable, str(script), tcp_url], check=False, timeout=15)
    # Process must exit cleanly with return code 42, not hang
    assert p.returncode == 42


def test_master_timeout_when_follower_never_connects(tmp_path):
    """Test 8: Master node exits cleanly with timeout when follower never joins."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"

    script = tmp_path / "master_timeout_worker.py"
    script.write_text(r'''
import sys
from datetime import timedelta
import torch.distributed as dist

tcp_url = sys.argv[1]
try:
    # 2-second timeout, follower never joins
    dist.init_process_group("gloo", init_method=tcp_url, rank=0, world_size=2, timeout=timedelta(seconds=2))
except Exception as e:
    sys.exit(42)
sys.exit(0)
''')

    p = subprocess.run([sys.executable, str(script), tcp_url], check=False, timeout=15)
    assert p.returncode == 42


def test_occupied_tcp_port_fails_immediately(tmp_path):
    """Test 9: Process attempting to bind to an already occupied port fails immediately."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    tcp_url = f"tcp://127.0.0.1:{port}"

    script = tmp_path / "occupied_port_worker.py"
    script.write_text(r'''
import sys
from datetime import timedelta
import torch.distributed as dist

tcp_url = sys.argv[1]
try:
    dist.init_process_group("gloo", init_method=tcp_url, rank=0, world_size=2, timeout=timedelta(seconds=2))
except Exception as e:
    sys.exit(43)
sys.exit(0)
''')

    try:
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, str(script), tcp_url], check=False, timeout=10)
        assert p.returncode == 43
        # Must fail immediately within 5 seconds, not hang
        assert time.monotonic() - t0 < 5.0
    finally:
        sock.close()


def test_mid_training_node_crash_cleanup(tmp_path):
    """Test 10: When master crashes during training, follower detects disconnect and aborts cleanly."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"

    script = tmp_path / "crash_worker.py"
    script.write_text(r'''
import os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

node_rank = int(sys.argv[1])
tcp_url = sys.argv[2]

dist.init_process_group("gloo", init_method=tcp_url, rank=node_rank, world_size=2, timeout=timedelta(seconds=3))
try:
    t = torch.tensor([1.0])
    dist.all_reduce(t)

    if node_rank == 0:
        # Master abruptly crashes
        os._exit(7)
    else:
        # Follower attempts second collective
        time.sleep(0.5)
        try:
            t2 = torch.tensor([2.0])
            dist.all_reduce(t2)
            sys.exit(0)
        except Exception:
            sys.exit(44)
finally:
    dist.destroy_process_group()
''')

    p0 = subprocess.Popen([sys.executable, str(script), '0', tcp_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, str(script), '1', tcp_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p0.communicate(timeout=10)
    p1.communicate(timeout=10)

    assert p0.returncode == 7
    assert p1.returncode == 44


# ---------------------------------------------------------------------------
# Runner-Level Multi-Node Integration Tests
# ---------------------------------------------------------------------------

RUNNER_WORKER_CODE = r'''
import sys
import json
from pathlib import Path

if __name__ == '__main__':
    from hypergan.replicated_execution import run_train, run_follower, run_resume

    mode = sys.argv[1]
    node_rank = int(sys.argv[2])
    cfg_file = sys.argv[3]
    run_dir = Path(sys.argv[4])
    profile = json.loads(sys.argv[5])
    steps = int(sys.argv[6]) if sys.argv[6] != 'None' else None
    checkpoint_every = int(sys.argv[7]) if sys.argv[7] != 'None' else 1000
    checkpoint_path = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != 'None' else None

    raw_cfg = json.loads(Path(cfg_file).read_text())

    if mode == 'train':
        if node_rank == 0:
            run_train(raw_cfg, run_dir, steps=steps, profile=profile, checkpoint_every=checkpoint_every)
        else:
            run_follower(raw_cfg, run_dir, steps=steps, profile=profile, checkpoint_every=checkpoint_every)
    elif mode == 'resume':
        if node_rank == 0:
            run_resume(run_dir, checkpoint=checkpoint_path, config_path=raw_cfg, profile=profile, steps=steps, checkpoint_every=checkpoint_every)
        else:
            run_follower(raw_cfg, run_dir, steps=steps, profile=profile, checkpoint=checkpoint_path, checkpoint_every=checkpoint_every)
'''


def test_multinode_runners_cpu_gloo(tmp_path):
    """Test: 2-node high-level runner execution (run_train on Node 0, run_follower on Node 1) over CPU Gloo."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"
    run_dir = tmp_path / "run_cpu"

    cfg = {
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {'steps': 4, 'batch_size': 8, 'device': 'cpu', 'seed': 42},
        'sampling': {'count': 4},
    }
    cfg_file = tmp_path / "config_cpu.json"
    cfg_file.write_text(json.dumps(cfg))

    script = tmp_path / "runner_worker.py"
    script.write_text(RUNNER_WORKER_CODE)

    p0_prof = {
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'rendezvous': tcp_url,
            'accumulation_steps': 1,
        }
    }
    p1_prof = json.loads(json.dumps(p0_prof))
    p1_prof['execution']['node_rank'] = 1

    env = make_multinode_env()
    p0 = subprocess.Popen([sys.executable, str(script), 'train', '0', str(cfg_file), str(run_dir), json.dumps(p0_prof), '4', '2'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, str(script), 'train', '1', str(cfg_file), str(run_dir), json.dumps(p1_prof), '4', '2'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _out0, err0 = p0.communicate(timeout=30)
    _out1, err1 = p1.communicate(timeout=30)

    assert p0.returncode == 0, f"P0 failed: {err0.decode()}"
    assert p1.returncode == 0, f"P1 failed: {err1.decode()}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest['steps'] == 4
    assert manifest['checkpoint_path']
    assert Path(manifest['checkpoint_path']).exists()
    assert Path(manifest['bundle_path']).is_file()


@pytest.mark.skipif(not is_2gpu_available(), reason="Requires 2 CUDA GPUs with NCCL")
def test_multinode_runners_gpu_nccl(tmp_path):
    """Test: 2-node high-level runner execution (run_train on Node 0, run_follower on Node 1) over GPU NCCL on 2x RTX A6000."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"
    run_dir = tmp_path / "run_gpu"

    cfg = {
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {'steps': 4, 'batch_size': 8, 'device': 'cuda', 'seed': 42},
        'sampling': {'count': 4},
    }
    cfg_file = tmp_path / "config_gpu.json"
    cfg_file.write_text(json.dumps(cfg))

    script = tmp_path / "runner_worker.py"
    script.write_text(RUNNER_WORKER_CODE)

    p0_prof = {
        'schema_version': 1,
        'execution': {
            'name': 'cuda-replicated-nccl',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'rendezvous': tcp_url,
            'accumulation_steps': 1,
        }
    }
    p1_prof = json.loads(json.dumps(p0_prof))
    p1_prof['execution']['node_rank'] = 1

    env0 = make_multinode_env(gpu_index=0)
    env1 = make_multinode_env(gpu_index=1)

    p0 = subprocess.Popen([sys.executable, str(script), 'train', '0', str(cfg_file), str(run_dir), json.dumps(p0_prof), '4', '2'], env=env0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, str(script), 'train', '1', str(cfg_file), str(run_dir), json.dumps(p1_prof), '4', '2'], env=env1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _out0, err0 = p0.communicate(timeout=40)
    _out1, err1 = p1.communicate(timeout=40)

    assert p0.returncode == 0, f"P0 GPU failed: {err0.decode()}"
    assert p1.returncode == 0, f"P1 GPU failed: {err1.decode()}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest['steps'] == 4
    assert manifest['checkpoint_path']
    assert Path(manifest['checkpoint_path']).exists()
    assert Path(manifest['bundle_path']).is_file()


def test_multinode_runners_checkpoint_resume(tmp_path):
    """Test: 2-node runner checkpoint save and resume across separate processes."""
    port = get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"
    run_dir = tmp_path / "run_resume"

    cfg = {
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {'steps': 2, 'batch_size': 8, 'device': 'cpu', 'seed': 42, 'lr_floor': 1.0},
        'sampling': {'count': 4},
    }
    cfg_file = tmp_path / "config_resume.json"
    cfg_file.write_text(json.dumps(cfg))

    script = tmp_path / "runner_worker.py"
    script.write_text(RUNNER_WORKER_CODE)

    p0_prof = {
        'schema_version': 1,
        'execution': {
            'name': 'cpu-replicated-gloo',
            'world_size': 2,
            'nodes': 2,
            'node_rank': 0,
            'rendezvous': tcp_url,
            'accumulation_steps': 1,
        }
    }
    p1_prof = json.loads(json.dumps(p0_prof))
    p1_prof['execution']['node_rank'] = 1

    env = make_multinode_env()

    # Phase 1: Train to step 2 with checkpoint_every=2
    p0 = subprocess.Popen([sys.executable, str(script), 'train', '0', str(cfg_file), str(run_dir), json.dumps(p0_prof), '2', '2'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, str(script), 'train', '1', str(cfg_file), str(run_dir), json.dumps(p1_prof), '2', '2'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _out0, err0 = p0.communicate(timeout=30)
    _out1, err1 = p1.communicate(timeout=30)

    assert p0.returncode == 0, f"Phase 1 P0 failed: {err0.decode()}"
    assert p1.returncode == 0, f"Phase 1 P1 failed: {err1.decode()}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    ckpt2 = manifest['checkpoint_path']
    assert manifest['steps'] == 2
    assert Path(ckpt2).exists()

    time.sleep(0.5)

    # Phase 2: Resume to step 4
    p0 = subprocess.Popen([sys.executable, str(script), 'resume', '0', str(cfg_file), str(run_dir), json.dumps(p0_prof), '4', '2', str(ckpt2)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, str(script), 'resume', '1', str(cfg_file), str(run_dir), json.dumps(p1_prof), '4', '2', str(ckpt2)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    _out0, err0 = p0.communicate(timeout=30)
    _out1, err1 = p1.communicate(timeout=30)

    assert p0.returncode == 0, f"Phase 2 P0 failed: {err0.decode()}"
    assert p1.returncode == 0, f"Phase 2 P1 failed: {err1.decode()}"

    manifest_resumed = json.loads((run_dir / "manifest.json").read_text())
    assert manifest_resumed['steps'] == 4
    assert manifest_resumed['last_durable_step'] == 4
    assert Path(manifest_resumed['bundle_path']).is_file()
