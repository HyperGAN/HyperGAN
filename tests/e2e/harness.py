"""E2E Test Harness & Verification Utilities for Multi-Device Execution (Issue #186).

Provides:
- Milestone capability detectors (M1 recipe parity/schema, M2 aggregation, M3 multi-node, M4 qualify/RunPod)
- Conditional skip markers adhering to Progressive Testability
- Canonical recipe and execution profile configuration factories
- Exact mathematical reference oracles for parameter averaging (FedAvg) and pseudo-gradient (DiLoCo)
- Qualification receipt schema validator and SHA-256 verifier
- Ephemeral lifecycle and resource cleanup guardrail assertions
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from hypergan.config import resolve_config

# ---------------------------------------------------------------------------
# Hardware & Backend Capabilities
# ---------------------------------------------------------------------------

HAVE_CUDA: bool = torch.cuda.is_available()
CUDA_DEVICE_COUNT: int = torch.cuda.device_count() if HAVE_CUDA else 0
HAVE_2_GPUS: bool = HAVE_CUDA and CUDA_DEVICE_COUNT >= 2
HAVE_GLOO: bool = dist.is_gloo_available()
HAVE_NCCL: bool = dist.is_nccl_available()

# ---------------------------------------------------------------------------
# Milestone Capability Detectors (Progressive Testability)
# ---------------------------------------------------------------------------

def check_m1_recipe_parity() -> bool:
    """Check if validate_replicated_recipe permits prior inputs, reuse, and adversarial terms."""
    try:
        from hypergan.execution_profiles import validate_replicated_recipe
        # Test config with prior inputs
        prior_cfg = resolve_config({
            'prior': {'kind': 'mog', 'fixed_sigma': 0.05, 'args': {'num_particles': 8, 'z_dim': 4}},
            'components': {
                'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [4]}, 'inputs': {'input': 'prior.means'}},
                'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
            },
            'training': {'device': 'cpu', 'steps': 1, 'batch_size': 4},
        })
        validate_replicated_recipe(prior_cfg)
        return True
    except Exception:
        return False


def check_m1_schema() -> bool:
    """Check if resolve_execution_profile validates canonical multi-node and aggregation keys."""
    try:
        from hypergan.execution_profiles import resolve_execution_profile
        cfg = resolve_config({'training': {'device': 'cpu', 'batch_size': 4, 'steps': 1}})
        raw_profile = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'nodes': 2,
                'node_rank': 0,
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
            'aggregation': {
                'strategy': 'gradient_allreduce',
                'sync_every_steps': 1,
            },
        }
        res = resolve_execution_profile(raw_profile, cfg)
        exec_table = res.get('execution', {})
        return 'nodes' in exec_table and 'ranks_per_node' in exec_table
    except Exception:
        return False


def check_aggregation_module() -> bool:
    """Check if hypergan.aggregation is available with standard strategies."""
    try:
        import hypergan.aggregation as agg
        return hasattr(agg, 'AggregationStrategy') and hasattr(agg, 'register_aggregation_strategy')
    except ImportError:
        return False


def check_multinode_runtime() -> bool:
    """Check if multi-node distributed broker and worker support TCP rendezvous."""
    try:
        from hypergan.execution_profiles import resolve_execution_profile
        cfg = resolve_config({'training': {'device': 'cpu', 'batch_size': 4, 'steps': 1}})
        res = resolve_execution_profile({
            'schema_version': 1,
            'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, 'nodes': 2, 'node_rank': 0, 'rendezvous': 'tcp://127.0.0.1:29500'}
        }, cfg)
        return res.get('execution', {}).get('rendezvous') == 'tcp://127.0.0.1:29500'
    except Exception:
        return False


def check_qualify_module() -> bool:
    """Check if hypergan.qualify or hypergan qualify CLI command is implemented."""
    try:
        import hypergan.qualify as qual
        return hasattr(qual, 'run_qualification') or hasattr(qual, 'QualificationRunner')
    except ImportError:
        pass
    try:
        from hypergan.cli import _parser
        parser = _parser()
        # Check if qualify subparser exists
        subparsers = [action for action in parser._actions if action.dest == 'command']
        if subparsers and 'qualify' in subparsers[0].choices:
            return True
    except Exception:
        pass
    return False


def check_runpod_module() -> bool:
    """Check if hypergan.runpod_runner exists with guardrails."""
    try:
        import hypergan.runpod_runner as rpr
        return hasattr(rpr, 'RunPodRunner') or hasattr(rpr, 'provision_ephemeral_pod')
    except ImportError:
        return False


def check_replicated_worker_parity() -> bool:
    """Check if ReplicatedTrainer / replicated_worker supports resolved execution profile schema."""
    try:
        import inspect
        from hypergan.distributed_training import ReplicatedTrainer
        src_trainer = inspect.getsource(ReplicatedTrainer.__init__)
        return 'nodes' in src_trainer and 'ranks_per_node' in src_trainer
    except Exception:
        return False


HAVE_M1_RECIPE_PARITY: bool = check_m1_recipe_parity()
HAVE_M1_SCHEMA: bool = check_m1_schema()
HAVE_REPLICATED_WORKER_PARITY: bool = check_replicated_worker_parity()
HAVE_AGGREGATION: bool = check_aggregation_module()
HAVE_MULTINODE: bool = check_multinode_runtime()
HAVE_QUALIFY: bool = check_qualify_module()
HAVE_RUNPOD: bool = check_runpod_module()

# ---------------------------------------------------------------------------
# Progressive Testability Skip Markers
# ---------------------------------------------------------------------------

requires_m1_recipe_parity = pytest.mark.skipif(
    not HAVE_M1_RECIPE_PARITY,
    reason="M1 recipe parity pending: validate_replicated_recipe does not yet permit prior/reuse/adversarial/presets",
)

requires_m1_schema = pytest.mark.skipif(
    not HAVE_M1_SCHEMA,
    reason="M1 schema pending: resolve_execution_profile does not yet validate nodes/rendezvous/aggregation",
)

requires_replicated_worker_parity = pytest.mark.skipif(
    not HAVE_REPLICATED_WORKER_PARITY,
    reason="Replicated worker parity pending: replicated_worker requires strategy_info to match all execution profile keys (M2/M3)",
)

requires_aggregation = pytest.mark.skipif(
    not HAVE_AGGREGATION,
    reason="M2 aggregation pending: hypergan.aggregation module is not yet implemented",
)

requires_multinode = pytest.mark.skipif(
    not HAVE_MULTINODE,
    reason="M3 multi-node execution pending: TCP rendezvous and multi-node brokering not yet implemented",
)

requires_qualify = pytest.mark.skipif(
    not HAVE_QUALIFY,
    reason="M4 qualification pending: hypergan qualify / hypergan.qualify module not yet implemented",
)

requires_runpod = pytest.mark.skipif(
    not HAVE_RUNPOD,
    reason="M4 RunPod runner pending: hypergan.runpod_runner module not yet implemented",
)

requires_2_gpus = pytest.mark.skipif(
    not HAVE_2_GPUS,
    reason="Hardware requirement: at least 2 CUDA GPUs required for local multi-GPU qualification",
)

# ---------------------------------------------------------------------------
# Configuration Builders
# ---------------------------------------------------------------------------

def make_toy_config(
    device: str = "cpu",
    batch_size: int = 8,
    steps: int = 3,
    extra_training: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal fast deterministic training configuration."""
    base: Dict[str, Any] = {
        'components': {
            'generator': {
                'factory': 'mlp',
                'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
                'inputs': {'x': 'latent'},
            },
            'discriminator': {
                'factory': 'linear',
                'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                'inputs': {'input': 'candidate'},
            },
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {
            'steps': steps,
            'batch_size': batch_size,
            'device': device,
            'seed': 42,
        },
        'sampling': {'count': 4},
    }
    if extra_training:
        base['training'].update(extra_training)
    return resolve_config(base)


def make_prior_bound_config(device: str = "cpu", batch_size: int = 8) -> Dict[str, Any]:
    """Build a config with prior-bound component inputs ('prior.means')."""
    cfg = {
        'prior': {'kind': 'mog', 'fixed_sigma': 0.05, 'args': {'num_particles': 8, 'z_dim': 4}},
        'components': {
            'generator': {
                'factory': 'linear',
                'args': {'in_features': 4, 'out_features': 2, 'bias': False},
                'inputs': {'input': 'latent'},
            },
            'encoder': {
                'factory': 'linear',
                'args': {'in_features': 4, 'out_features': 4, 'bias': False},
                'inputs': {'input': 'prior.means'},
                'trainable': True,
            },
            'discriminator': {
                'factory': 'linear',
                'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                'inputs': {'input': 'candidate'},
            },
        },
        'objectives': [
            {
                'factory': 'mse',
                'inputs': {'input': 'components.encoder', 'target': 'prior.means'},
                'weight': 0.5,
            }
        ],
        'training': {'steps': 2, 'batch_size': batch_size, 'device': device, 'seed': 123},
    }
    return resolve_config(cfg)


def make_reuse_config(device: str = "cpu", batch_size: int = 8) -> Dict[str, Any]:
    """Build a config with component reuse (shared weights across graph positions)."""
    cfg = {
        'components': {
            'generator': {
                'factory': 'mlp',
                'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
                'inputs': {'x': 'latent'},
            },
            'encoder_a': {
                'factory': 'mlp',
                'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [4]},
                'inputs': {'x': 'latent'},
            },
            'encoder_b': {
                'reuse': 'encoder_a',
                'inputs': {'x': 'latent'},
            },
            'discriminator': {
                'factory': 'linear',
                'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                'inputs': {'input': 'candidate'},
            },
        },
        'objectives': [
            {
                'factory': 'mse',
                'inputs': {'input': 'components.encoder_b', 'target': 'batch.real'},
                'weight': 1.0,
            }
        ],
        'prior': {'kind': 'particles', 'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': batch_size, 'device': device, 'seed': 123},
    }
    return resolve_config(cfg)


def make_adversarial_terms_config(device: str = "cpu", batch_size: int = 8) -> Dict[str, Any]:
    """Build a config with extra adversarial terms."""
    cfg = {
        'components': {
            'generator': {
                'factory': 'mlp',
                'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
                'inputs': {'x': 'latent'},
            },
            'discriminator': {
                'factory': 'linear',
                'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                'inputs': {'input': 'candidate'},
            },
        },
        'adversarial_terms': [
            {
                'id': 'extra_term',
                'component': 'discriminator',
                'real': 'batch.real',
                'fake': 'generated',
            }
        ],
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'training': {'steps': 2, 'batch_size': batch_size, 'device': device, 'seed': 123},
    }
    return resolve_config(cfg)


def make_profile_raw(
    name: str = 'cpu-single',
    world_size: Optional[int] = None,
    accumulation_steps: int = 1,
    nodes: Optional[int] = None,
    node_rank: Optional[int] = None,
    rendezvous: Optional[str] = None,
    aggregation: Optional[Dict[str, Any]] = None,
    preflight: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a raw execution profile dictionary conforming to GLOSSARY.md."""
    exec_table: Dict[str, Any] = {'name': name}
    if world_size is not None:
        exec_table['world_size'] = world_size
    if accumulation_steps != 1:
        exec_table['accumulation_steps'] = accumulation_steps
    if nodes is not None:
        exec_table['nodes'] = nodes
    if node_rank is not None:
        exec_table['node_rank'] = node_rank
    if rendezvous is not None:
        exec_table['rendezvous'] = rendezvous

    raw: Dict[str, Any] = {
        'schema_version': 1,
        'execution': exec_table,
    }
    if aggregation is not None:
        raw['aggregation'] = aggregation
    if preflight is not None:
        raw['preflight'] = preflight
    return raw


# ---------------------------------------------------------------------------
# Authoritative Mathematical Oracles
# ---------------------------------------------------------------------------

def oracle_parameter_average(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Authoritative FedAvg mathematical oracle:
    Theta_mean = (1 / W) * sum_{i=1}^W Theta_i
    """
    assert len(weights_list) > 0, "weights_list cannot be empty"
    world_size = len(weights_list)
    avg_state: Dict[str, torch.Tensor] = {}
    for key in weights_list[0].keys():
        tensors = [w[key] for w in weights_list]
        stacked = torch.stack(tensors, dim=0)
        avg_state[key] = torch.mean(stacked, dim=0)
    return avg_state


def oracle_pseudo_gradient_step(
    anchor_state: Dict[str, torch.Tensor],
    replica_states: List[Dict[str, torch.Tensor]],
    outer_lr: float = 1.0,
    outer_momentum: float = 0.0,
    momentum_buffer: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Authoritative DiLoCo / FedOpt mathematical oracle:
    DeltaTheta_i = Theta_anchor - Theta_i
    DeltaTheta = (1 / W) * sum_{i=1}^W DeltaTheta_i
    v_{t} = outer_momentum * v_{t-1} + DeltaTheta
    Theta_anchor^{new} = Theta_anchor - outer_lr * v_{t}
    """
    world_size = len(replica_states)
    new_anchor: Dict[str, torch.Tensor] = {}
    new_momentum: Dict[str, torch.Tensor] = {}

    if momentum_buffer is None:
        momentum_buffer = {k: torch.zeros_like(v) for k, v in anchor_state.items()}

    for key, anchor_val in anchor_state.items():
        # Compute weight deltas: DeltaTheta_i = Theta_anchor - Theta_replica
        deltas = [anchor_val - rep[key] for rep in replica_states]
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)

        # Update outer momentum buffer: v = mu * v + DeltaTheta
        v_prev = momentum_buffer.get(key, torch.zeros_like(anchor_val))
        v_new = outer_momentum * v_prev + mean_delta
        new_momentum[key] = v_new

        # Update anchor: Theta_anchor = Theta_anchor - eta * v
        new_anchor[key] = anchor_val - outer_lr * v_new

    return new_anchor, new_momentum


def assert_states_close(
    left: Any,
    right: Any,
    rtol: float = 3e-5,
    atol: float = 3e-6,
    path: str = "root",
) -> None:
    """Recursively compare state dictionaries, tensors, floats, or collections."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"{path}: right is not a torch.Tensor ({type(right)})"
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=lambda m: f"{path}: {m}")
    elif isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, rel=rtol, abs=atol), f"{path}: {left} != {right}"
    elif isinstance(left, dict):
        assert isinstance(right, dict), f"{path}: right is not a dict ({type(right)})"
        assert set(left.keys()) == set(right.keys()), f"{path}: keys differ ({set(left.keys()) ^ set(right.keys())})"
        for k in left:
            assert_states_close(left[k], right[k], rtol=rtol, atol=atol, path=f"{path}.{k}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)), f"{path}: right is not a sequence ({type(right)})"
        assert len(left) == len(right), f"{path}: sequence length mismatch ({len(left)} vs {len(right)})"
        for idx, (a, b) in enumerate(zip(left, right)):
            assert_states_close(a, b, rtol=rtol, atol=atol, path=f"{path}[{idx}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


# ---------------------------------------------------------------------------
# Receipt Schema Validator & SHA-256 Verifier
# ---------------------------------------------------------------------------

def compute_receipt_sha256(receipt_dict: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 over receipt dict excluding receipt_sha256 key."""
    canonical = {k: v for k, v in receipt_dict.items() if k != "receipt_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_qualification_receipt(receipt: Dict[str, Any]) -> None:
    """Verify that a qualification receipt satisfies PROJECT.md § Interface Contracts."""
    assert isinstance(receipt, dict), "Receipt must be a JSON object"
    required_keys = {
        "schema_version",
        "qualification_id",
        "timestamp",
        "status",
        "duration_seconds",
        "environment",
        "suites",
        "receipt_sha256",
    }
    missing = required_keys - set(receipt.keys())
    assert not missing, f"Qualification receipt missing required keys: {missing}"

    assert receipt["schema_version"] == 1, f"Expected schema_version 1, got {receipt['schema_version']}"
    assert isinstance(receipt["qualification_id"], str) and len(receipt["qualification_id"]) > 0
    assert isinstance(receipt["timestamp"], str) and len(receipt["timestamp"]) > 0
    assert receipt["status"] in ("passed", "failed"), f"Invalid status: {receipt['status']}"
    assert isinstance(receipt["duration_seconds"], (int, float)) and receipt["duration_seconds"] >= 0.0
    assert receipt["duration_seconds"] <= 180.0, f"Qualification exceeded 3-minute hard limit: {receipt['duration_seconds']}s"

    env = receipt["environment"]
    assert isinstance(env, dict), "Receipt environment must be a dictionary"
    for env_key in ("platform", "python", "torch", "device_count"):
        assert env_key in env, f"Receipt environment missing '{env_key}'"

    assert isinstance(receipt["suites"], list), "Receipt suites must be a list"
    assert len(receipt["suites"]) > 0, "Receipt suites list cannot be empty"

    # Verify SHA-256 integrity
    expected_sha = compute_receipt_sha256(receipt)
    assert receipt["receipt_sha256"] == expected_sha, (
        f"Receipt SHA-256 mismatch: recorded {receipt['receipt_sha256']} != calculated {expected_sha}"
    )


# ---------------------------------------------------------------------------
# Ephemeral RunPod Runner Guardrail Mock / Simulator
# ---------------------------------------------------------------------------

class MockRunPodClient:
    """Mock RunPod API client recording lifecycle calls for guardrail verification."""

    def __init__(self, fail_at: Optional[str] = None):
        self.created_pods: List[str] = []
        self.terminated_pods: List[str] = []
        self.fail_at = fail_at
        self._next_id = 1

    def create_pod(self, gpu_type: str, image: str, timeout_seconds: int) -> str:
        if self.fail_at == "create_pod":
            raise RuntimeError("Simulated cloud creation failure")
        pod_id = f"mock-pod-{self._next_id}"
        self._next_id += 1
        self.created_pods.append(pod_id)
        return pod_id

    def terminate_pod(self, pod_id: str) -> bool:
        if pod_id in self.created_pods:
            self.terminated_pods.append(pod_id)
            return True
        return False

    @property
    def all_pods_cleaned_up(self) -> bool:
        return set(self.created_pods) == set(self.terminated_pods)


# ---------------------------------------------------------------------------
# Multi-Rank Subprocess Smoke Test Runner
# ---------------------------------------------------------------------------

def run_replicated_cpu_smoke_test(
    config: Dict[str, Any],
    steps: int = 2,
    world_size: int = 2,
    accumulation_steps: int = 1,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Execute real multi-rank Gloo CPU training in worker subprocesses and return rank outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(config))
        store_uri = (tmp_path / "rendezvous_store").as_uri()

        worker_script = tmp_path / "worker.py"
        worker_script.write_text(r'''
import json, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedCPUTrainer

rank, world_size, accumulation_steps, steps, store_uri, cfg_path, out_path = sys.argv[1:8]
rank = int(rank)
world_size = int(world_size)
accumulation_steps = int(accumulation_steps)
steps = int(steps)

cfg = json.loads(Path(cfg_path).read_text())
dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
try:
    trainer = ReplicatedCPUTrainer(cfg, world_size=world_size, accumulation_steps=accumulation_steps)
    history = []
    data = torch.linspace(-1, 1, 16).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    for step in range(steps):
        local_ids = ids.chunk(world_size)[rank]
        m, _ = trainer.update({'real': data.chunk(world_size)[rank]}, (trainer.prior.z[local_ids], local_ids))
        history.append({
            'step': m.get('step', step + 1),
            'd_loss': float(m.get('d_loss', 0.0)),
            'g_loss': float(m.get('g_loss', 0.0)),
            'gradient_penalty': float(m.get('gradient_penalty', 0.0)),
        })
    weights = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
    Path(out_path).write_text(json.dumps({'rank': rank, 'history': history, 'weights': weights, 'passed': True}))
finally:
    dist.destroy_process_group()
''')

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = f"{repo_root}/src:{repo_root}/tests:{repo_root}/tests/reference:" + env.get('PYTHONPATH', '')

        procs = []
        out_files = []
        for r in range(world_size):
            out_f = tmp_path / f"out_{r}.json"
            out_files.append(out_f)
            p = subprocess.Popen(
                [sys.executable, str(worker_script), str(r), str(world_size), str(accumulation_steps), str(steps), store_uri, str(cfg_file), str(out_f)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=timeout)
            assert p.returncode == 0, f"Worker process failed (exit code {p.returncode}):\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        return results
