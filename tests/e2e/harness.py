"""E2E test harness for multi-device execution (Issue #186).

Provides:
- hardware capability flags and the two-GPU skip marker (the only skips: missing hardware)
- recipe and execution-profile builders for replicated tests
- a multi-rank Gloo CPU smoke runner and a RunPod lifecycle mock for guardrail tests

The toy recipe and aggregation oracles live in ``tests/distributed_fixtures.py``
so unit tests can share them without importing this harness. Every feature these tests cover has shipped, so nothing here
probes for features: a broken import fails collection instead of turning tests
into skips.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.distributed as dist

from hypergan.config import resolve_config

try:
    from tests.distributed_fixtures import (  # noqa: F401 - re-exported for the e2e tests
        assert_states_close,
        make_toy_config,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
    )
except ModuleNotFoundError:
    # pytest's importlib mode imports this directory as the package ``e2e``
    # (it has an __init__.py), and then ``tests`` itself is not importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from distributed_fixtures import (  # noqa: F401
        assert_states_close,
        make_toy_config,
        oracle_parameter_average,
        oracle_pseudo_gradient_step,
    )

# ---------------------------------------------------------------------------
# Hardware & Backend Capabilities
# ---------------------------------------------------------------------------

HAVE_CUDA: bool = torch.cuda.is_available()
CUDA_DEVICE_COUNT: int = torch.cuda.device_count() if HAVE_CUDA else 0
HAVE_2_GPUS: bool = HAVE_CUDA and CUDA_DEVICE_COUNT >= 2
HAVE_GLOO: bool = dist.is_gloo_available()
HAVE_NCCL: bool = dist.is_nccl_available()

# ---------------------------------------------------------------------------
# Hardware skip marker
# ---------------------------------------------------------------------------

requires_2_gpus = pytest.mark.skipif(
    not HAVE_2_GPUS,
    reason="Hardware requirement: at least 2 CUDA GPUs required for local multi-GPU qualification",
)

# ---------------------------------------------------------------------------
# Configuration Builders
# ---------------------------------------------------------------------------

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
    """Build a raw execution profile dictionary (see GLOSSARY.md, execution profiles)."""
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
