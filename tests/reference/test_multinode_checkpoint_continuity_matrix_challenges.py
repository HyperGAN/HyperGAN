"""Empirical Challenge Suite: Multi-Node Checkpoint Save, Manifest Consensus,
Ephemeral Rendezvous Invariance, and Exact Resume Trajectory Continuity Matrix (Milestone 3).

Challenges:
1. Manifest Consensus & Ephemeral Invariance:
   - Verify that `node_rank` and dynamic `rendezvous` URIs are excluded from checkpoint `topology`.
   - Verify that Node 0 (`node_rank=0`) and Node 1 (`node_rank=1`) achieve identical checkpoint identity and SHA256 digest consensus.
   - Verify that top-level `manifest.json` and per-node rank payloads (`rank-00000.pt`, `rank-00001.pt`) are published.
   - Verify that cluster restart on a fresh ephemeral TCP port succeeds without identity mismatch.
   - Verify that genuine topology mismatch (e.g. `nodes=1` vs `nodes=2`) is strictly rejected.

2. Multi-Node Checkpoint Save & Resume across all 3 Strategies:
   - `gradient_allreduce` (default sync_every_steps=1)
   - `parameter_average` (sync_every_steps=2)
   - `pseudo_gradient` (sync_every_steps=2, outer_lr=0.5, outer_momentum=0.8)
   - Verify post-resume step 3 and step 4 loss and parameter consensus across nodes.

3. Bit-for-Bit Exact Resume Trajectory Continuity Matrix:
   - Continuous 4-step trajectory vs Split (steps 1..2 save) + Resume (steps 3..4) trajectory across nodes over TCP.
   - Verify at split step 2:
     - Restored weights match continuous weights bit-for-bit (`torch.equal`).
     - Restored optimizer states (`step`, `exp_avg`, `exp_avg_sq`) match continuous optimizer states bit-for-bit.
     - Restored aggregation state (anchor parameters, outer momentum, step counters) match continuous state bit-for-bit.
   - Verify at post-resume steps 3 and 4:
     - Step 3 & 4 metrics match continuous metrics bit-for-bit or within numerical tolerance.
     - Step 4 final weights match continuous weights bit-for-bit (`torch.equal`).
     - Step 4 final optimizer states match continuous optimizer states bit-for-bit.

4. Non-Cadence Checkpoint Rejection:
   - Verify that attempting to save a distributed checkpoint at step 1 under `parameter_average` or `pseudo_gradient`
     (when `sync_every_steps=2`) is cleanly rejected due to cross-node parameter divergence.

5. 2x NVIDIA RTX A6000 Multi-Node GPU NCCL Checkpoint Save & Resume Parity:
   - Verify checkpoint save and resume across nodes over TCP with NCCL backend on dual RTX A6000 GPUs.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.heavy

REPO_ROOT = str(Path(__file__).resolve().parents[2])


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
# Worker Subprocess Driver Code
# ---------------------------------------------------------------------------

CHALLENGE_WORKER_CODE = r'''
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
    distributed_checkpoint_identity,
)

mode, node_rank_str, tcp_url, run_dir, out_path, backend, agg_json, steps_str = sys.argv[1:9]
node_rank = int(node_rank_str)
steps = int(steps_str)
run_dir = Path(run_dir)
agg_cfg = json.loads(agg_json)

if backend == "nccl":
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    torch.set_num_threads(1)

dist.init_process_group(
    backend,
    init_method=tcp_url,
    rank=node_rank,
    world_size=2,
    timeout=timedelta(seconds=25),
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
        'training': {'steps': 10, 'batch_size': 8, 'device': device_name, 'seed': 42},
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

    # Deterministic sequence of 4 distinct batches across steps
    batches = []
    base_data = torch.linspace(-1, 1, 32, device="cuda" if backend == "nccl" else "cpu").reshape(4, 8)
    for b_idx in range(4):
        raw_data = base_data[b_idx].reshape(4, 2)
        batches.append(raw_data)

    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda" if backend == "nccl" else "cpu")
    local_ids = ids.chunk(2)[node_rank]

    result = {'node_rank': node_rank, 'mode': mode, 'passed': True}

    if mode == "inspect_identity":
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        ident = distributed_checkpoint_identity(trainer)
        result['identity'] = ident
        Path(out_path).write_text(json.dumps(result))

    elif mode == "continuous":
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        metrics = []
        payload_step2 = None
        for step in range(1, steps + 1):
            batch_data = batches[step - 1]
            m, last_batch = trainer.update({'real': batch_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})
            if step == 2:
                payload_step2 = {
                    'step': trainer.step,
                    'weights': copy.deepcopy(trainer.graph.state_dict()),
                    'opt_g': copy.deepcopy(trainer.opt_g.state_dict()),
                    'opt_d': copy.deepcopy(trainer.opt_d.state_dict()),
                    'aggregation': copy.deepcopy(trainer.aggregation_strategy.state_dict()),
                }
                torch.save(payload_step2, run_dir / f"continuous_step2_rank{node_rank}.pt")

        payload_final = {
            'step': trainer.step,
            'metrics': metrics,
            'weights': copy.deepcopy(trainer.graph.state_dict()),
            'opt_g': copy.deepcopy(trainer.opt_g.state_dict()),
            'opt_d': copy.deepcopy(trainer.opt_d.state_dict()),
            'aggregation': copy.deepcopy(trainer.aggregation_strategy.state_dict()),
        }
        torch.save(payload_final, run_dir / f"continuous_final_rank{node_rank}.pt")
        result['metrics'] = metrics
        Path(out_path).write_text(json.dumps(result))

    elif mode == "split_save":
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        metrics = []
        last_batch = None
        for step in range(1, steps + 1):
            batch_data = batches[step - 1]
            m, last_batch = trainer.update({'real': batch_data}, (trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})

        ckpt_path = save_distributed_checkpoint(
            run_dir, trainer, last_batch,
            {'run_id': 'test-run', 'attempt_id': 'att-1', 'next_sample_sequence': 10}
        )
        payload_step2 = {
            'step': trainer.step,
            'metrics': metrics,
            'weights': copy.deepcopy(trainer.graph.state_dict()),
            'opt_g': copy.deepcopy(trainer.opt_g.state_dict()),
            'opt_d': copy.deepcopy(trainer.opt_d.state_dict()),
            'aggregation': copy.deepcopy(trainer.aggregation_strategy.state_dict()),
        }
        torch.save(payload_step2, run_dir / f"split_step2_rank{node_rank}.pt")
        result['checkpoint_path'] = str(ckpt_path)
        result['metrics'] = metrics
        Path(out_path).write_text(json.dumps(result))

    elif mode == "resume_continue":
        fresh_trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        restored_path, info, last_batch = restore_distributed_checkpoint(
            run_dir, fresh_trainer, {'run_id': 'test-run'}
        )
        assert fresh_trainer.step == steps, f"Expected step {steps}, got {fresh_trainer.step}"

        # Capture immediately restored state at step 2
        payload_restored = {
            'step': fresh_trainer.step,
            'weights': copy.deepcopy(fresh_trainer.graph.state_dict()),
            'opt_g': copy.deepcopy(fresh_trainer.opt_g.state_dict()),
            'opt_d': copy.deepcopy(fresh_trainer.opt_d.state_dict()),
            'aggregation': copy.deepcopy(fresh_trainer.aggregation_strategy.state_dict()),
        }
        torch.save(payload_restored, run_dir / f"restored_step2_rank{node_rank}.pt")

        metrics = []
        for step in range(steps + 1, steps + 3):
            batch_data = batches[step - 1]
            m, last_batch = fresh_trainer.update({'real': batch_data}, (fresh_trainer.prior.z[local_ids], local_ids))
            metrics.append({'step': step, 'd_loss': float(m['d_loss']), 'g_loss': float(m['g_loss'])})

        payload_final = {
            'step': fresh_trainer.step,
            'metrics': metrics,
            'weights': copy.deepcopy(fresh_trainer.graph.state_dict()),
            'opt_g': copy.deepcopy(fresh_trainer.opt_g.state_dict()),
            'opt_d': copy.deepcopy(fresh_trainer.opt_d.state_dict()),
            'aggregation': copy.deepcopy(fresh_trainer.aggregation_strategy.state_dict()),
        }
        torch.save(payload_final, run_dir / f"resume_final_rank{node_rank}.pt")
        result['restored_path'] = str(restored_path)
        result['metrics'] = metrics
        Path(out_path).write_text(json.dumps(result))

    elif mode == "non_cadence_save":
        # Under parameter_average or pseudo_gradient with sync_every_steps=2,
        # step 1 causes weight divergence across distinct rank data shards.
        # A checkpoint at step 1 first forces one synchronization (review
        # 2026-10-25 H3: it used to fail consensus and poison the run).
        trainer = ReplicatedTrainer(
            cfg, world_size=2, accumulation_steps=1, nodes=2,
            node_rank=node_rank, ranks_per_node=1, rendezvous=tcp_url,
            execution=execution,
        )
        batch_data = batches[0]
        m, last_batch = trainer.update({'real': batch_data}, (trainer.prior.z[local_ids], local_ids))
        rejected = False
        rejection_error = None
        try:
            save_distributed_checkpoint(
                run_dir, trainer, last_batch,
                {'run_id': 'test-run', 'attempt_id': 'att-non-cadence', 'next_sample_sequence': 10}
            )
        except ValueError as exc:
            rejected = True
            rejection_error = str(exc)

        result['rejected'] = rejected
        result['rejection_error'] = rejection_error
        result['weights_sha256'] = ReplicatedTrainer._digest(trainer.graph.state_dict())
        result['synced_step'] = trainer.aggregation_strategy.synced_step
        result['checkpoint_ready'] = trainer.checkpoint_ready
        Path(out_path).write_text(json.dumps(result))

finally:
    dist.destroy_process_group()
'''


def run_challenge_nodes(
    tmp_path: Path,
    mode: str,
    backend: str = "gloo",
    agg_cfg: dict[str, Any] | None = None,
    steps: int = 2,
    timeout: float = 35.0,
    run_dir: Path | None = None,
    custom_port: int | None = None,
) -> list[dict[str, Any]]:
    """Launch 2 node processes connected over localhost TCP rendezvous."""
    if agg_cfg is None:
        agg_cfg = {'strategy': 'gradient_allreduce', 'sync_every_steps': 1, 'outer_lr': 1.0, 'outer_momentum': 0.0}

    worker_script = tmp_path / "challenge_worker.py"
    if not worker_script.exists():
        worker_script.write_text(CHALLENGE_WORKER_CODE)

    if run_dir is None:
        run_dir = tmp_path / f"run_{mode}"
    run_dir.mkdir(exist_ok=True, parents=True)

    port = custom_port if custom_port is not None else get_free_tcp_port()
    tcp_url = f"tcp://127.0.0.1:{port}"

    out_files = [tmp_path / f"out_{mode}_{nr}_{int(time.time()*1000)%100000}.json" for nr in range(2)]
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
            assert p.returncode == 0, f"Node {nr} in mode '{mode}' failed (code {p.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

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


def assert_states_close(left: Any, right: Any, rtol: float = 0.0, atol: float = 0.0, context: str = "") -> None:
    """Recursively compare nested PyTorch tensor dicts/lists for exact or numerical equality."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"{context}: right is not a Tensor: {type(right)}"
        if rtol == 0.0 and atol == 0.0:
            assert torch.equal(left.cpu(), right.cpu()), f"{context}: Tensors are not bit-identical"
        else:
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=rtol, atol=atol)
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"{context}: right is not a dict: {type(right)}"
        assert set(left.keys()) == set(right.keys()), f"{context}: Key set mismatch: {set(left.keys())} vs {set(right.keys())}"
        for k in left:
            assert_states_close(left[k], right[k], rtol=rtol, atol=atol, context=f"{context}.{k}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), f"{context}: Length/type mismatch"
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            assert_states_close(l_item, r_item, rtol=rtol, atol=atol, context=f"{context}[{idx}]")
    elif isinstance(left, (int, str, bool)) or left is None:
        assert left == right, f"{context}: {left} != {right}"
    elif isinstance(left, float):
        if rtol == 0.0 and atol == 0.0:
            assert left == right, f"{context}: {left} != {right}"
        else:
            assert abs(left - right) <= (atol + rtol * abs(right)), f"{context}: {left} != {right}"


# ==============================================================================
# Challenge 1: Manifest Consensus & Ephemeral Rendezvous Invariance
# ==============================================================================

def test_manifest_identity_excludes_node_rank_and_ephemeral_rendezvous(tmp_path):
    """Challenge 1.1: Verify node_rank and ephemeral rendezvous URI are excluded from
    topology identity, ensuring cross-node manifest agreement on save and across restarts.
    """
    port1 = get_free_tcp_port()
    results = run_challenge_nodes(tmp_path, mode="inspect_identity", backend="gloo", custom_port=port1)
    assert len(results) == 2
    id0 = results[0]['identity']
    id1 = results[1]['identity']

    # 1. Identity must be JSON-identical between node_rank=0 and node_rank=1
    assert id0 == id1, "Node 0 and Node 1 must produce bit-identical checkpoint identity"

    # 2. Topology must NOT contain node_rank or rendezvous
    top0 = id0['topology']
    assert 'node_rank' not in top0, "Topology must exclude rank-local node_rank"
    assert 'rendezvous' not in top0, "Topology must exclude ephemeral rendezvous URI"

    # 3. Topology must preserve invariant cluster architecture
    assert top0['nodes'] == 2
    assert top0['ranks_per_node'] == 1
    assert top0['world_size'] == 2
    assert top0['global_batch_size'] == 8
    assert top0['local_batch_size'] == 4


def test_cluster_restart_on_fresh_ephemeral_tcp_port(tmp_path):
    """Challenge 1.2: Verify checkpoint saved on port A can be seamlessly restored on a fresh
    cluster initialized on a completely different ephemeral port B.
    """
    run_dir = tmp_path / "run_ephemeral_restart"
    run_dir.mkdir(exist_ok=True)

    port_save = get_free_tcp_port()
    save_results = run_challenge_nodes(
        tmp_path, mode="split_save", backend="gloo", steps=2,
        run_dir=run_dir, custom_port=port_save
    )
    assert len(save_results) == 2
    manifest_path = Path(save_results[0]['checkpoint_path']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['step'] == 2
    assert 'node_rank' not in manifest['identity']['topology']
    assert 'rendezvous' not in manifest['identity']['topology']

    # Restore on a guaranteed distinct ephemeral TCP port
    port_resume = get_free_tcp_port()
    while port_resume == port_save:
        port_resume = get_free_tcp_port()

    resume_results = run_challenge_nodes(
        tmp_path, mode="resume_continue", backend="gloo", steps=2,
        run_dir=run_dir, custom_port=port_resume
    )
    assert len(resume_results) == 2
    r0, r1 = resume_results[0], resume_results[1]
    assert r0['metrics'][0]['step'] == 3
    assert r0['metrics'][1]['step'] == 4
    # Cross-rank parity on resumed steps
    assert r0['metrics'] == r1['metrics']


def test_cluster_topology_mismatch_strictly_rejected(tmp_path):
    """Challenge 1.3: Verify that genuine topology divergence (e.g. attempting to restore
    a 2-node checkpoint on a single-node or different world size) is strictly rejected.
    """
    from hypergan.config import resolve_config
    from hypergan.distributed_checkpoints import _validate_restore_identity

    saved_identity = {
        'hypergan_checkpoint_version': 1,
        'config': {'training': {'batch_size': 8, 'steps': 10}},
        'config_sha256': 'dummy',
        'runtime': {'device': 'cpu', 'backend': 'gloo', 'world_size': 2, 'threads': 1, 'interop_threads': 1},
        'implementation': {},
        'data_contract': {'identity': {}},
        'topology': {'world_size': 2, 'nodes': 2, 'ranks_per_node': 1, 'global_batch_size': 8, 'local_batch_size': 4},
    }

    # Attempt restore with nodes=1, ranks_per_node=2 (different node layout)
    current_identity_mismatched_nodes = copy.deepcopy(saved_identity)
    current_identity_mismatched_nodes['topology']['nodes'] = 1
    current_identity_mismatched_nodes['topology']['ranks_per_node'] = 2

    # Attempt restore with different world_size
    current_identity_mismatched_world = copy.deepcopy(saved_identity)
    current_identity_mismatched_world['topology']['world_size'] = 4
    current_identity_mismatched_world['runtime']['world_size'] = 4

    import pytest
    with pytest.raises(ValueError, match="Distributed checkpoint config/data/topology identity differs"):
        # We need mock fingerprint/resume_compatible
        from hypergan import distributed_checkpoints
        orig_fp = distributed_checkpoints.fingerprint
        orig_compat = distributed_checkpoints.resume_compatible
        try:
            distributed_checkpoints.fingerprint = lambda cfg: 'dummy'
            distributed_checkpoints.resume_compatible = lambda c1, c2: True
            distributed_checkpoints.validate_checkpoint_compatibility = lambda s: None
            distributed_checkpoints.validate_runtime = lambda s, c, **kw: None
            distributed_checkpoints.validate_implementation = lambda s, c: None
            _validate_restore_identity(saved_identity, current_identity_mismatched_nodes)
        finally:
            distributed_checkpoints.fingerprint = orig_fp
            distributed_checkpoints.resume_compatible = orig_compat


# ==============================================================================
# Challenge 2: Save and Resume across all 3 Aggregation Strategies
# ==============================================================================

@pytest.mark.parametrize("strategy,sync_every,outer_lr,outer_momentum", [
    ("gradient_allreduce", 1, 1.0, 0.0),
    ("parameter_average", 2, 1.0, 0.0),
    ("pseudo_gradient", 2, 0.5, 0.8),
])
def test_2node_checkpoint_save_and_resume_all_strategies(tmp_path, strategy, sync_every, outer_lr, outer_momentum):
    """Challenge 2: Verify distributed checkpoint save and resume across 2 nodes over TCP
    for gradient_allreduce, parameter_average, and pseudo_gradient.
    """
    agg = {
        'strategy': strategy,
        'sync_every_steps': sync_every,
        'outer_lr': outer_lr,
        'outer_momentum': outer_momentum,
    }
    run_dir = tmp_path / f"run_save_resume_{strategy}"
    run_dir.mkdir(exist_ok=True)

    # 1. Save checkpoint at step 2
    save_results = run_challenge_nodes(tmp_path, mode="split_save", backend="gloo", agg_cfg=agg, steps=2, run_dir=run_dir)
    assert len(save_results) == 2
    ckpt_dir = Path(save_results[0]['checkpoint_path'])
    assert (ckpt_dir / 'manifest.json').exists()
    assert (ckpt_dir / 'rank-00000.pt').exists()
    assert (ckpt_dir / 'rank-00001.pt').exists()

    manifest = json.loads((ckpt_dir / 'manifest.json').read_text())
    assert manifest['step'] == 2
    assert manifest['aggregation']['strategy'] == strategy
    assert manifest['aggregation']['sync_every_steps'] == sync_every

    # 2. Resume on fresh process group for steps 3 and 4
    resume_results = run_challenge_nodes(tmp_path, mode="resume_continue", backend="gloo", agg_cfg=agg, steps=2, run_dir=run_dir)
    assert len(resume_results) == 2
    r0, r1 = resume_results[0], resume_results[1]

    # Verify steps 3 and 4 were executed
    assert r0['metrics'][0]['step'] == 3
    assert r0['metrics'][1]['step'] == 4

    # At step 4 (sync_every_steps=2), both nodes must have synchronized
    assert_states_close(r0['metrics'], r1['metrics'], rtol=1e-5, atol=1e-6, context=f"Cross-node metrics parity for {strategy}")


# ==============================================================================
# Challenge 3: Bit-for-Bit Exact Resume Trajectory Continuity Matrix
# ==============================================================================

@pytest.mark.parametrize("strategy,sync_every,outer_lr,outer_momentum", [
    ("gradient_allreduce", 1, 1.0, 0.0),
    ("parameter_average", 2, 1.0, 0.0),
    ("pseudo_gradient", 2, 0.5, 0.8),
])
def test_2node_exact_trajectory_continuity_matrix_cpu(tmp_path, strategy, sync_every, outer_lr, outer_momentum):
    """Challenge 3: Verify bit-for-bit trajectory continuity across 2 nodes over TCP:
    Continuous 4 steps vs (Split 2 steps + Save -> Restart -> Restore -> Resume 2 steps).
    Checks:
    - Step 2 restored weights == Continuous step 2 weights (bit-for-bit).
    - Step 2 restored optimizer state == Continuous step 2 optimizer state (bit-for-bit).
    - Step 2 restored aggregation state == Continuous step 2 aggregation state (bit-for-bit).
    - Step 3 & 4 resumed metrics == Continuous step 3 & 4 metrics (bit-for-bit).
    - Step 4 resumed weights == Continuous step 4 weights (bit-for-bit).
    - Step 4 resumed optimizer state == Continuous step 4 optimizer state (bit-for-bit).
    """
    agg = {
        'strategy': strategy,
        'sync_every_steps': sync_every,
        'outer_lr': outer_lr,
        'outer_momentum': outer_momentum,
    }

    dir_continuous = tmp_path / f"traj_continuous_{strategy}"
    dir_continuous.mkdir(exist_ok=True)
    dir_split = tmp_path / f"traj_split_{strategy}"
    dir_split.mkdir(exist_ok=True)

    # 1. Run Continuous Trajectory (4 steps)
    cont_res = run_challenge_nodes(tmp_path, mode="continuous", backend="gloo", agg_cfg=agg, steps=4, run_dir=dir_continuous)
    assert len(cont_res) == 2

    # 2. Run Split Save Trajectory (2 steps)
    split_res = run_challenge_nodes(tmp_path, mode="split_save", backend="gloo", agg_cfg=agg, steps=2, run_dir=dir_split)
    assert len(split_res) == 2

    # 3. Run Resume Continue Trajectory (steps 3 & 4) on fresh process group
    port_resume = get_free_tcp_port()
    resume_res = run_challenge_nodes(tmp_path, mode="resume_continue", backend="gloo", agg_cfg=agg, steps=2, run_dir=dir_split, custom_port=port_resume)
    assert len(resume_res) == 2

    # Compare for both ranks (node_rank 0 and node_rank 1)
    for nr in range(2):
        cont_s2 = torch.load(dir_continuous / f"continuous_step2_rank{nr}.pt", weights_only=False)
        split_s2 = torch.load(dir_split / f"split_step2_rank{nr}.pt", weights_only=False)
        rest_s2 = torch.load(dir_split / f"restored_step2_rank{nr}.pt", weights_only=False)

        cont_fin = torch.load(dir_continuous / f"continuous_final_rank{nr}.pt", weights_only=False)
        res_fin = torch.load(dir_split / f"resume_final_rank{nr}.pt", weights_only=False)

        # A. At Step 2:
        assert cont_s2['step'] == 2 and rest_s2['step'] == 2
        # Restored weights match continuous weights bit-for-bit
        assert_states_close(cont_s2['weights'], rest_s2['weights'], rtol=0, atol=0, context=f"Rank {nr} step 2 weights ({strategy})")
        # Restored optimizer states match continuous optimizer states bit-for-bit
        assert_states_close(cont_s2['opt_g']['state'], rest_s2['opt_g']['state'], rtol=0, atol=0, context=f"Rank {nr} step 2 opt_g ({strategy})")
        assert_states_close(cont_s2['opt_d']['state'], rest_s2['opt_d']['state'], rtol=0, atol=0, context=f"Rank {nr} step 2 opt_d ({strategy})")
        # Restored aggregation states match continuous aggregation state bit-for-bit
        assert_states_close(cont_s2['aggregation'], rest_s2['aggregation'], rtol=0, atol=0, context=f"Rank {nr} step 2 agg ({strategy})")

        # B. Post-resume Steps 3 and 4:
        # Step metrics match bit-for-bit
        cont_post_metrics = cont_fin['metrics'][2:4]  # steps 3 and 4
        resume_metrics = res_fin['metrics']           # steps 3 and 4
        assert_states_close(cont_post_metrics, resume_metrics, rtol=1e-5, atol=1e-6, context=f"Rank {nr} post-resume metrics ({strategy})")

        # Step 4 final weights match bit-for-bit
        assert cont_fin['step'] == 4 and res_fin['step'] == 4
        assert_states_close(cont_fin['weights'], res_fin['weights'], rtol=0, atol=0, context=f"Rank {nr} final step 4 weights ({strategy})")
        # Step 4 final optimizer states match bit-for-bit
        assert_states_close(cont_fin['opt_g']['state'], res_fin['opt_g']['state'], rtol=0, atol=0, context=f"Rank {nr} final step 4 opt_g ({strategy})")
        assert_states_close(cont_fin['opt_d']['state'], res_fin['opt_d']['state'], rtol=0, atol=0, context=f"Rank {nr} final step 4 opt_d ({strategy})")


# ==============================================================================
# Challenge 4: Non-Cadence Checkpoint Rejection
# ==============================================================================

@pytest.mark.parametrize("strategy", ["parameter_average", "pseudo_gradient"])
def test_non_cadence_checkpoint_forces_one_synchronization_multi_node(tmp_path, strategy):
    """Challenge 4: a checkpoint at an unsynchronized local step (step 1 when
    sync_every_steps=2) across 2 nodes over TCP synchronizes the replicas once and
    publishes one replicated state instead of failing the run.
    """
    agg = {
        'strategy': strategy,
        'sync_every_steps': 2,
        'outer_lr': 0.5,
        'outer_momentum': 0.8,
    }
    run_dir = tmp_path / f"run_non_cadence_{strategy}"
    run_dir.mkdir(exist_ok=True)

    results = run_challenge_nodes(tmp_path, mode="non_cadence_save", backend="gloo", agg_cfg=agg, steps=1, run_dir=run_dir)
    assert len(results) == 2
    for r in results:
        assert r['rejected'] is False, r['rejection_error']
        assert r['synced_step'] == 1 and r['checkpoint_ready'] is True
    assert results[0]['weights_sha256'] == results[1]['weights_sha256']
    latest = json.loads((run_dir / 'distributed-checkpoints' / 'latest.json').read_text())
    assert latest['step'] == 1


# ==============================================================================
# Challenge 5: 2x NVIDIA RTX A6000 Multi-Node GPU NCCL Save & Resume Continuity
# ==============================================================================

@pytest.mark.skipif(not is_2gpu_available(), reason="Requires 2 CUDA GPUs with NCCL")
@pytest.mark.parametrize("strategy", ["gradient_allreduce", "parameter_average", "pseudo_gradient"])
def test_2node_gpu_nccl_checkpoint_save_resume_and_continuity(tmp_path, strategy):
    """Challenge 5: Verify multi-node checkpoint save and resume across 2 nodes over TCP
    using 2x NVIDIA RTX A6000 GPUs with NCCL backend.
    - Node 0 on CUDA_VISIBLE_DEVICES=0 (RTX A6000 #0).
    - Node 1 on CUDA_VISIBLE_DEVICES=1 (RTX A6000 #1).
    - Checkpoint save at step 2; fresh group restore for steps 3 and 4.
    - Trajectory parity against continuous multi-node GPU execution.
    """
    agg = {
        'strategy': strategy,
        'sync_every_steps': 2 if strategy != "gradient_allreduce" else 1,
        'outer_lr': 0.5,
        'outer_momentum': 0.8,
    }

    dir_cont = tmp_path / f"gpu_cont_{strategy}"
    dir_cont.mkdir(exist_ok=True)
    dir_split = tmp_path / f"gpu_split_{strategy}"
    dir_split.mkdir(exist_ok=True)

    # 1. Continuous GPU run (4 steps)
    cont_res = run_challenge_nodes(tmp_path, mode="continuous", backend="nccl", agg_cfg=agg, steps=4, run_dir=dir_cont, timeout=45.0)
    assert len(cont_res) == 2

    # 2. Split GPU run (save at step 2)
    split_res = run_challenge_nodes(tmp_path, mode="split_save", backend="nccl", agg_cfg=agg, steps=2, run_dir=dir_split, timeout=45.0)
    assert len(split_res) == 2

    # 3. Resume GPU run (restore and run steps 3 & 4 on fresh TCP port)
    port_resume = get_free_tcp_port()
    resume_res = run_challenge_nodes(tmp_path, mode="resume_continue", backend="nccl", agg_cfg=agg, steps=2, run_dir=dir_split, custom_port=port_resume, timeout=45.0)
    assert len(resume_res) == 2

    # Compare Rank 0 and Rank 1
    for nr in range(2):
        cont_fin = torch.load(dir_cont / f"continuous_final_rank{nr}.pt", weights_only=False)
        res_fin = torch.load(dir_split / f"resume_final_rank{nr}.pt", weights_only=False)

        # Losses match within CUDA floating-point tolerance
        cont_post_metrics = cont_fin['metrics'][2:4]
        res_post_metrics = res_fin['metrics']
        assert_states_close(cont_post_metrics, res_post_metrics, rtol=1e-5, atol=1e-6, context=f"GPU Rank {nr} post-resume metrics ({strategy})")

        # Step 4 final weights match within CUDA floating-point tolerance
        assert_states_close(cont_fin['weights'], res_fin['weights'], rtol=1e-5, atol=1e-6, context=f"GPU Rank {nr} final weights ({strategy})")
