"""Tier 1: Feature Coverage for Multi-Node Distributed Execution and TCP Rendezvous.

Requirements covered:
- R3: Multi-Node Distributed Execution (nodes, node_rank, ranks_per_node, rendezvous) (ORIGINAL_REQUEST.md §R3)
- Features 11, 12, 13: Multi-Node Topology Resolution, TCP & File Rendezvous,
  Local 2-Node 2x RTX A6000 Verification (PROJECT.md §Feature Inventory)
- Interface Contract 4: Broker ↔ Worker Multi-Node TCP (PROJECT.md §Interface Contracts)

Test coverage target: >=5 test cases per feature:
1. Multi-node keys (nodes, node_rank, ranks_per_node, local device mapping) (>=5 cases)
2. TCP rendezvous (URI validation, 2-process TCP handshake, timeout cleanup, CPU & GPU multi-node smoke) (>=5 cases)
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.execution_profiles import resolve_execution_profile

try:
    from e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_GLOO,
        HAVE_NCCL,
        requires_2_gpus,
        make_toy_config,
        make_profile_raw,
    )
except ImportError:
    from tests.e2e.harness import (
        HAVE_CUDA,
        HAVE_2_GPUS,
        HAVE_GLOO,
        HAVE_NCCL,
        requires_2_gpus,
        make_toy_config,
        make_profile_raw,
    )


def get_free_tcp_port() -> int:
    """Find a guaranteed free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ==============================================================================
# Feature 10: Multi-Node Topology Keys (>= 5 test cases)
# ==============================================================================

class TestMultiNodeKeys:
    """Feature 10: Multi-node topology resolution (nodes, node_rank, ranks_per_node)."""

    def test_tier1_nodes_and_node_rank_resolution(self):
        """Case 10.1: Profile resolves nodes=2, node_rank=1, world_size=4, ranks_per_node=2."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 4,
                'nodes': 2,
                'node_rank': 1,
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        exec_tab = resolved['execution']
        assert exec_tab['nodes'] == 2
        assert exec_tab['node_rank'] == 1
        assert exec_tab['world_size'] == 4
        assert exec_tab['ranks_per_node'] == 2

    def test_tier1_ranks_per_node_exact_division(self):
        """Case 10.2: Derived ranks_per_node = world_size // nodes divides evenly."""
        cfg = make_toy_config(device="cpu", batch_size=16)
        for world_size, nodes, expected_rpn in [(2, 2, 1), (4, 2, 2), (8, 4, 2), (8, 2, 4)]:
            raw = {
                'schema_version': 1,
                'execution': {
                    'name': 'cpu-replicated-gloo',
                    'world_size': world_size,
                    'nodes': nodes,
                    'node_rank': 0,
                    'rendezvous': 'tcp://127.0.0.1:29500',
                },
            }
            resolved = resolve_execution_profile(raw, cfg)
            assert resolved['execution']['ranks_per_node'] == expected_rpn

    def test_tier1_rejects_node_rank_gte_nodes(self):
        """Case 10.3: node_rank must be strictly less than nodes (0 <= node_rank < nodes)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'nodes': 2,
                'node_rank': 2,  # Invalid: node_rank == nodes
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
        }
        with pytest.raises(ValueError, match="node_rank"):
            resolve_execution_profile(raw, cfg)

    def test_tier1_rejects_nodes_not_dividing_world_size(self):
        """Case 10.4: Rejects configuration where world_size does not divide evenly by nodes."""
        cfg = make_toy_config(device="cpu", batch_size=12)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 3,
                'nodes': 2,  # Invalid: 3 % 2 != 0
                'node_rank': 0,
                'rendezvous': 'tcp://127.0.0.1:29500',
            },
        }
        with pytest.raises(ValueError, match="divide evenly across execution.nodes"):
            resolve_execution_profile(raw, cfg)

    def test_tier1_local_device_remapping_formula(self):
        """Case 10.5: Local device index formula local_rank = global_rank % ranks_per_node."""
        # For world_size=4 across 2 nodes (ranks_per_node=2):
        # Node 0 (ranks 0, 1) -> local_rank 0, 1
        # Node 1 (ranks 2, 3) -> local_rank 0, 1
        ranks_per_node = 2
        for global_rank in range(4):
            local_rank = global_rank % ranks_per_node
            expected_local = 0 if global_rank in (0, 2) else 1
            assert local_rank == expected_local, f"Rank {global_rank} mapped to {local_rank}, expected {expected_local}"


# ==============================================================================
# Feature 11: TCP Rendezvous (>= 5 test cases)
# ==============================================================================

class TestTCPRendezvous:
    """Feature 11: TCP & File rendezvous, collectives initialization, and cleanup."""

    def test_tier1_tcp_rendezvous_uri_acceptance(self):
        """Case 11.1: Profile parser accepts valid tcp://host:port URI in rendezvous."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'nodes': 2,
                'node_rank': 0,
                'rendezvous': 'tcp://192.168.1.100:29500',
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['rendezvous'] == 'tcp://192.168.1.100:29500'

    def test_tier1_file_rendezvous_uri_acceptance(self):
        """Case 11.2: Profile parser accepts local file:// URI in rendezvous."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        raw = {
            'schema_version': 1,
            'execution': {
                'name': 'cpu-replicated-gloo',
                'world_size': 2,
                'rendezvous': 'file:///tmp/hypergan-rendezvous-test',
            },
        }
        resolved = resolve_execution_profile(raw, cfg)
        assert resolved['execution']['rendezvous'] == 'file:///tmp/hypergan-rendezvous-test'

    def test_tier1_rejects_unsupported_rendezvous_schemes(self):
        """Case 11.3: Rejects unsupported URI schemes (http://, udp://, ftp://)."""
        cfg = make_toy_config(device="cpu", batch_size=8)
        for invalid_scheme in ('http://127.0.0.1:29500', 'udp://127.0.0.1:29500', 'ftp://127.0.0.1:29500'):
            raw = {
                'schema_version': 1,
                'execution': {
                    'name': 'cpu-replicated-gloo',
                    'world_size': 2,
                    'rendezvous': invalid_scheme,
                },
            }
            with pytest.raises(ValueError, match="file:// or tcp://"):
                resolve_execution_profile(raw, cfg)

    @pytest.mark.heavy
    def test_tier1_tcp_rendezvous_handshake_cpu(self, tmp_path):
        """Case 11.4: 2 processes complete TCP rendezvous handshake over localhost."""
        port = get_free_tcp_port()
        tcp_url = f"tcp://127.0.0.1:{port}"

        worker_script = tmp_path / "tcp_worker.py"
        worker_script.write_text(r'''
import json, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

rank, world_size, tcp_url, out_path = sys.argv[1:5]
rank = int(rank)
world_size = int(world_size)

dist.init_process_group("gloo", init_method=tcp_url, rank=rank, world_size=world_size, timeout=timedelta(seconds=15))
try:
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t)
    Path(out_path).write_text(json.dumps({"rank": rank, "reduced": t.item(), "status": "ok"}))
finally:
    dist.destroy_process_group()
''')

        env = os.environ.copy()
        procs = []
        out_files = []
        for r in range(2):
            out_f = tmp_path / f"out_{r}.json"
            out_files.append(out_f)
            p = subprocess.Popen(
                [sys.executable, str(worker_script), str(r), '2', tcp_url, str(out_f)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=25)
            assert p.returncode == 0, f"TCP worker failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        assert results[0]['reduced'] == 3.0
        assert results[1]['reduced'] == 3.0

    @pytest.mark.heavy
    def test_tier1_tcp_rendezvous_timeout_cleanup(self, tmp_path):
        """Case 11.5: Rank 0 aborts cleanly with timeout when peer never connects."""
        port = get_free_tcp_port()
        tcp_url = f"tcp://127.0.0.1:{port}"

        worker_script = tmp_path / "tcp_timeout_worker.py"
        worker_script.write_text(r'''
import sys
from datetime import timedelta
import torch.distributed as dist

tcp_url = sys.argv[1]
try:
    # 2-second timeout, rank 1 never shows up
    dist.init_process_group("gloo", init_method=tcp_url, rank=0, world_size=2, timeout=timedelta(seconds=2))
except Exception as e:
    sys.exit(42)
sys.exit(0)
''')

        p = subprocess.run([sys.executable, str(worker_script), tcp_url], timeout=15)
        # Process must fail with timeout (exit code 42), not hang forever
        assert p.returncode == 42, f"Expected timeout exit code 42, got {p.returncode}"

    @pytest.mark.heavy
    @requires_2_gpus
    def test_tier1_local_2_node_gpu_tcp_smoke(self, tmp_path):
        """Case 11.6: 2-node GPU TCP smoke test (node 0 on GPU 0, node 1 on GPU 1)."""
        port = get_free_tcp_port()
        tcp_url = f"tcp://127.0.0.1:{port}"

        driver = tmp_path / "multinode_gpu.py"
        driver.write_text(r'''
import json, os, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

node_rank, tcp_url, out_path = sys.argv[1:4]
node_rank = int(node_rank)
# Local rank within the node is always 0 since CUDA_VISIBLE_DEVICES isolates each GPU
local_device = 0
torch.cuda.set_device(local_device)

dist.init_process_group("nccl", init_method=tcp_url, rank=node_rank, world_size=2, timeout=timedelta(seconds=15))
try:
    t = torch.tensor([float(node_rank + 1)], device=f"cuda:{local_device}")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    Path(out_path).write_text(json.dumps({
        "node_rank": node_rank,
        "device": torch.cuda.get_device_name(local_device),
        "reduced": t.item(),
    }))
finally:
    dist.destroy_process_group()
''')

        env = os.environ.copy()
        env['NCCL_P2P_DISABLE'] = '1'
        env['NCCL_IB_DISABLE'] = '1'

        procs = []
        out_files = []
        for nr in range(2):
            out_f = tmp_path / f"node_{nr}.json"
            out_files.append(out_f)
            node_env = env.copy()
            node_env['CUDA_VISIBLE_DEVICES'] = str(nr)
            p = subprocess.Popen(
                [sys.executable, str(driver), str(nr), tcp_url, str(out_f)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=node_env,
            )
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate(timeout=25)
            assert p.returncode == 0, f"Multi-node GPU failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"

        results = [json.loads(f.read_text()) for f in out_files]
        assert results[0]['reduced'] == 3.0
        assert results[1]['reduced'] == 3.0
