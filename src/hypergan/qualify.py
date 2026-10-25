"""Hardware & Distributed Execution Qualification Harness (Issue #186, Milestone 4).

Provides:
- Fast (<3 minute) qualification smoke suite testing:
  1. Environment & platform diagnostics (Python, PyTorch, CUDA, device count & models)
  2. Single-device execution smoke test
  3. Multi-device CPU Gloo replicated execution
  4. Multi-device CUDA NCCL replicated execution (when >= 2 CUDA GPUs available)
  5. Pluggable learning aggregation strategies (gradient_allreduce, parameter_average, pseudo_gradient)
  6. Multi-node topology over a TCP rendezvous, simulated on one host (both node ranks
     run as local processes; separate hosts and the multi-node launcher are not exercised)
  7. Checkpoint compatibility smoke test
- Standardized machine-readable JSON receipt emitter following PROJECT.md § Interface
  Contracts (schema_version 1). ``receipt_sha256`` is an unkeyed SHA-256 checksum of the
  receipt's own content: it detects accidental corruption or truncation, but anyone who
  edits a receipt can recompute it, so it is not a signature or tamper proof.
- Receipt and suite statuses: ``passed``, ``failed``, ``skipped`` (nothing could run,
  e.g. no GPUs) and ``dry_run`` (``--dry-run``: nothing ran). Only suites that actually
  ran and checked their result report ``passed``.
- Worker subprocess logic for isolated distributed execution
- CLI driver `run_qualification_cli` for `hypergan qualify`
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple
import uuid

MAX_QUALIFICATION_SECONDS: float = 180.0
# Ceiling for one multi-rank worker job; the remaining qualification budget lowers it.
RANK_TIMEOUT_SECONDS: float = 35.0
RECEIPT_STATUSES: Tuple[str, ...] = ("passed", "failed", "skipped", "dry_run")


class ReceiptValidationError(ValueError):
    """A qualification receipt does not satisfy the receipt contract."""


class QualificationCheckFailed(RuntimeError):
    """A qualification suite ran and its result check failed."""


def _require(condition: Any, message: str) -> None:
    # Explicit raise rather than ``assert``: checks must survive ``python -O``.
    if not condition:
        raise QualificationCheckFailed(message)


# ---------------------------------------------------------------------------
# Receipt Schema & SHA-256 Checksum
# ---------------------------------------------------------------------------

def compute_receipt_sha256(receipt_dict: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 over receipt dict excluding receipt_sha256 key.

    This is an unkeyed checksum: it catches accidental edits and corruption, not a
    deliberate edit, because whoever edits the receipt can recompute it.
    """
    canonical = {k: v for k, v in receipt_dict.items() if k != "receipt_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_suite_status(suites: List[Dict[str, Any]]) -> str:
    """Overall status implied by suite results: any failure fails; a pass needs a suite that ran."""
    statuses = {s.get("status") for s in suites}
    if "failed" in statuses:
        return "failed"
    if "passed" in statuses:
        return "passed"
    if "dry_run" in statuses:
        return "dry_run"
    return "skipped"


def validate_qualification_receipt(receipt: Dict[str, Any]) -> None:
    """Verify that a qualification receipt satisfies PROJECT.md § Interface Contracts.

    Raises ReceiptValidationError on the first violation.
    """
    def check(condition: Any, message: str) -> None:
        if not condition:
            raise ReceiptValidationError(message)

    check(isinstance(receipt, dict), "Receipt must be a JSON object")
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
    check(not missing, f"Qualification receipt missing required keys: {missing}")
    check(receipt["schema_version"] == 1, f"Expected schema_version 1, got {receipt['schema_version']}")
    check(isinstance(receipt["qualification_id"], str) and len(receipt["qualification_id"]) > 0,
          "Receipt qualification_id must be a non-empty string")
    check(isinstance(receipt["timestamp"], str) and len(receipt["timestamp"]) > 0,
          "Receipt timestamp must be a non-empty string")
    check(receipt["status"] in RECEIPT_STATUSES, f"Invalid status: {receipt['status']}")
    duration = receipt["duration_seconds"]
    check(isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0.0,
          f"Receipt duration_seconds must be a non-negative number, got {duration!r}")

    env = receipt["environment"]
    check(isinstance(env, dict), "Receipt environment must be a dictionary")
    for env_key in ("platform", "python", "torch", "device_count"):
        check(env_key in env, f"Receipt environment missing '{env_key}'")

    suites = receipt["suites"]
    check(isinstance(suites, list), "Receipt suites must be a list")
    check(len(suites) > 0, "Receipt suites list cannot be empty")
    for suite in suites:
        check(isinstance(suite, dict) and isinstance(suite.get("name"), str),
              f"Receipt suite entries need a string name: {suite!r}")
        check(suite.get("status") in RECEIPT_STATUSES,
              f"Invalid status for suite {suite.get('name')!r}: {suite.get('status')!r}")

    expected_sha = compute_receipt_sha256(receipt)
    check(receipt["receipt_sha256"] == expected_sha,
          f"Receipt SHA-256 mismatch: recorded {receipt['receipt_sha256']} != calculated {expected_sha}")

    # A receipt over the limit is only valid as a failure; the duration is never clamped.
    check(receipt["status"] == "failed" or duration <= MAX_QUALIFICATION_SECONDS,
          f"Qualification exceeded 3-minute hard limit: {duration}s")
    if receipt["status"] != "failed":
        implied = summarize_suite_status(suites)
        check(receipt["status"] == implied,
              f"Receipt status {receipt['status']!r} contradicts its suites, which imply {implied!r}")


def default_receipt_path(qualification_id: str) -> Path:
    """Per-run receipt path under the HyperGAN state directory, never the working directory."""
    from .registry import get_registry_dir
    return get_registry_dir() / "qualification" / f"qualification-receipt-{qualification_id}.json"


def write_receipt(receipt: Dict[str, Any], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Environment Diagnostics Probe
# ---------------------------------------------------------------------------

def inspect_environment() -> Dict[str, Any]:
    """Collect platform, Python, PyTorch, and CUDA runtime metadata."""
    py_ver = platform.python_version()
    plat = sys.platform

    torch_ver = "none"
    device_count = 0
    device_names: List[str] = []
    cuda_available = False
    cuda_ver = None
    nccl_available = False
    gloo_available = False
    device_details: List[Dict[str, Any]] = []

    try:
        import torch
        torch_ver = getattr(torch, "__version__", "unknown")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
            device_count = int(torch.cuda.device_count())
            for idx in range(device_count):
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = f"cuda:{idx}"
                device_names.append(name)
                try:
                    prop = torch.cuda.get_device_properties(idx)
                    device_details.append({
                        "index": idx,
                        "name": name,
                        "total_memory_bytes": getattr(prop, "total_memory", 0),
                        "major": getattr(prop, "major", 0),
                        "minor": getattr(prop, "minor", 0),
                    })
                except Exception:
                    device_details.append({"index": idx, "name": name})
        try:
            import torch.distributed as dist
            nccl_available = bool(dist.is_nccl_available())
            gloo_available = bool(dist.is_gloo_available())
        except Exception:
            pass
    except Exception:
        pass

    return {
        "platform": plat,
        "python": py_ver,
        "torch": torch_ver,
        "device_count": device_count,
        "devices": device_names,
        "cuda": {
            "available": cuda_available,
            "version": cuda_ver,
            "nccl": nccl_available,
            "gloo": gloo_available,
            "devices": device_details,
        },
    }


inspect_platform_environment = inspect_environment


# ---------------------------------------------------------------------------
# Helper: Allocate Guaranteed Free TCP Port
# ---------------------------------------------------------------------------

def get_free_tcp_port() -> int:
    """Find a guaranteed free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Worker Subprocess Driver Script
# ---------------------------------------------------------------------------

WORKER_DRIVER_CODE = r'''
from datetime import timedelta
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.distributed_checkpoints import save_distributed_checkpoint, restore_distributed_checkpoint

mode = sys.argv[1]
rank = int(sys.argv[2])
world_size = int(sys.argv[3])
init_method = sys.argv[4]
backend = sys.argv[5]
out_file = Path(sys.argv[6])
extra_json = sys.argv[7] if len(sys.argv) > 7 else "{}"
extra = json.loads(extra_json)

steps = extra.get("steps", 2)
agg_cfg = extra.get("aggregation", {"strategy": "gradient_allreduce", "sync_every_steps": 1})
nodes = extra.get("nodes", 1)
ranks_per_node = extra.get("ranks_per_node", world_size // nodes)
node_rank = rank // ranks_per_node
run_dir_str = extra.get("run_dir")

if backend == "nccl":
    avail_gpus = torch.cuda.device_count()
    local_dev = rank % avail_gpus if avail_gpus > 0 else 0
    torch.cuda.set_device(local_dev)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    torch.set_num_threads(1)

dist.init_process_group(
    backend,
    init_method=init_method,
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=25),
)

try:
    dev_name = "cuda" if backend == "nccl" else "cpu"
    cfg = resolve_config({
        'components': {
            'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
        },
        'prior': {'args': {'num_particles': 16, 'z_dim': 4}},
        'gradient_penalty': {'lazy_k': 2, 'kappa': 0.01},
        'training': {'steps': steps + 2, 'batch_size': 8, 'device': dev_name, 'seed': 42},
        'sampling': {'count': 4},
    })

    execution = {
        'name': 'cuda-replicated-nccl' if backend == 'nccl' else 'cpu-replicated-gloo',
        'world_size': world_size,
        'nodes': nodes,
        'node_rank': node_rank,
        'ranks_per_node': ranks_per_node,
        'rendezvous': init_method,
        'accumulation_steps': 1,
        'global_batch_size': 8,
        'local_batch_size': 8 // world_size,
        'microbatch_size': 8 // world_size,
        'accumulation_algorithm': 'retained-local-graph-v1',
        'aggregation': agg_cfg,
    }

    raw_data = torch.linspace(-1, 1, 16, device=dev_name).reshape(8, 2)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=dev_name)
    local_ids = ids.chunk(world_size)[rank]
    local_data = raw_data.chunk(world_size)[rank]

    trainer = ReplicatedTrainer(
        cfg, world_size=world_size, accumulation_steps=1, nodes=nodes,
        node_rank=node_rank, ranks_per_node=ranks_per_node, rendezvous=init_method,
        execution=execution,
    )

    losses = []
    weight_history = []
    last_batch = None
    for step in range(1, steps + 1):
        m, last_batch = trainer.update({'real': local_data}, (trainer.prior.z[local_ids], local_ids))
        losses.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})
        weights = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
        weight_history.append(weights)

    out_data = {
        "rank": rank,
        "node_rank": node_rank,
        "passed": True,
        "losses": losses,
        "final_weights": weight_history[-1] if weight_history else {},
        "weight_history": weight_history,
    }

    if mode == "checkpoint" and run_dir_str:
        run_dir = Path(run_dir_str)
        ckpt_path = save_distributed_checkpoint(
            run_dir, trainer, last_batch,
            {"run_id": "qual-ckpt", "attempt_id": "att-1", "next_sample_sequence": 2}
        )
        fresh_trainer = ReplicatedTrainer(
            cfg, world_size=world_size, accumulation_steps=1, nodes=nodes,
            node_rank=node_rank, ranks_per_node=ranks_per_node, rendezvous=init_method,
            execution=execution,
        )
        restored_path, info, _ = restore_distributed_checkpoint(
            run_dir, fresh_trainer, {"run_id": "qual-ckpt"}
        )
        if fresh_trainer.step != steps:
            raise RuntimeError(f"Restored trainer is at step {fresh_trainer.step}, expected {steps}")
        out_data["checkpoint_verified"] = True

    out_file.write_text(json.dumps(out_data))
finally:
    dist.destroy_process_group()
'''


# Known qualification test suites (filter names; matching is by substring)
VALID_SUITES = {
    "single_device",
    "cpu_replicated_gloo",
    "cuda_replicated_nccl",
    "aggregation",
    "aggregation_strategies",
    "multinode",
    "multinode_tcp",
    "multinode_tcp_single_host",
    "checkpoint_compatibility",
}


def _stop_processes(procs: List[subprocess.Popen]) -> None:
    """Kill and reap every rank process that is still running."""
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# Qualification Harness Runner
# ---------------------------------------------------------------------------

class QualificationRunner:
    """Orchestrates fast smoke qualification suites and emits standardized receipt JSON."""

    def __init__(
        self,
        output_path: Optional[Path | str] = None,
        dry_run: bool = False,
        suite_filter: Optional[str] = None,
        timeout: float = MAX_QUALIFICATION_SECONDS,
        verbose: bool = False,
        progress_stream: Optional[TextIO] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.output_path = Path(output_path) if output_path is not None else None
        self.last_output_path: Optional[Path] = None
        self.dry_run = dry_run
        if timeout <= 0 or timeout > MAX_QUALIFICATION_SECONDS:
            raise ValueError(f"Timeout must be between 1 and 180 seconds, got {timeout}")
        self.timeout = timeout
        self.verbose = verbose
        self.progress_stream = progress_stream
        self._clock = clock
        self._deadline: Optional[float] = None

        if suite_filter and suite_filter.lower() not in ("all", "*"):
            norm = suite_filter.strip().lower()
            matches = [s for s in VALID_SUITES if norm in s or s in norm]
            if not matches:
                raise ValueError(
                    f"Unknown qualification suite '{suite_filter}'. Available suites: {sorted(VALID_SUITES)}"
                )
            self.suite_filter = norm
        else:
            self.suite_filter = None

    def should_run_suite(self, name: str) -> bool:
        if not self.suite_filter:
            return True
        return self.suite_filter in name.lower() or name.lower() in self.suite_filter

    def _progress(self, message: str) -> None:
        if self.verbose:
            print(message, file=self.progress_stream or sys.stdout)

    def _remaining(self) -> float:
        if self._deadline is None:
            return self.timeout
        return self._deadline - self._clock()

    @staticmethod
    def _dry_run_result(name: str) -> Dict[str, Any]:
        return {"name": name, "status": "dry_run", "duration_seconds": 0.0,
                "details": {"dry_run": True, "reason": "dry run: suite not executed"}}

    def _run_subprocess_ranks(
        self,
        backend: str,
        world_size: int,
        nodes: int = 1,
        agg_cfg: Optional[Dict[str, Any]] = None,
        steps: int = 2,
        cuda_devices: Optional[List[int]] = None,
        mode: str = "step",
        run_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run one worker process per rank; every rank is killed on timeout or failure."""
        rank_timeout = min(RANK_TIMEOUT_SECONDS, self._remaining())
        if rank_timeout <= 0:
            raise TimeoutError(f"Qualification timeout of {self.timeout}s exhausted before ranks could start")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script = tmp / "worker.py"
            script.write_text(WORKER_DRIVER_CODE)

            port = get_free_tcp_port()
            init_method = f"tcp://127.0.0.1:{port}"

            extra_data = {
                "steps": steps,
                "aggregation": agg_cfg or {"strategy": "gradient_allreduce", "sync_every_steps": 1},
                "nodes": nodes,
                "ranks_per_node": world_size // nodes,
                "run_dir": run_dir,
            }
            extra_str = json.dumps(extra_data)

            out_files = [tmp / f"out_{r}.json" for r in range(world_size)]
            env_base = os.environ.copy()
            # The directory that holds the hypergan package: src/ in a checkout, site-packages when installed.
            package_root = str(Path(__file__).resolve().parents[1])
            env_base["PYTHONPATH"] = package_root + os.pathsep + env_base.get("PYTHONPATH", "")
            if backend == "nccl":
                env_base["NCCL_P2P_DISABLE"] = "1"
                env_base["NCCL_IB_DISABLE"] = "1"
                env_base["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            procs: List[subprocess.Popen] = []
            logs = []
            try:
                for r in range(world_size):
                    env = env_base.copy()
                    if backend == "nccl" and cuda_devices:
                        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[r % len(cuda_devices)])
                    cmd = [
                        sys.executable, str(script),
                        mode, str(r), str(world_size), init_method, backend, str(out_files[r]), extra_str
                    ]
                    # Files, not pipes: a rank blocked on a full pipe would stall its peers.
                    log = (tmp / f"rank_{r}.log").open("w+")
                    logs.append(log)
                    procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True))

                deadline = self._clock() + rank_timeout
                results = []
                for r, p in enumerate(procs):
                    try:
                        p.wait(timeout=max(deadline - self._clock(), 0.001))
                    except subprocess.TimeoutExpired:
                        raise TimeoutError(f"Rank {r} did not finish within {rank_timeout:.1f}s") from None
                    if p.returncode != 0:
                        logs[r].seek(0)
                        raise QualificationCheckFailed(f"Rank {r} failed with code {p.returncode}:\n{logs[r].read()}")
                    _require(out_files[r].exists(), f"Rank {r} did not write output file")
                    results.append(json.loads(out_files[r].read_text()))
                return results
            finally:
                _stop_processes(procs)
                for log in logs:
                    log.close()

    def _run_suite_single_device(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 1: Single device execution smoke test."""
        t0 = time.monotonic()
        if self.dry_run:
            return self._dry_run_result("single_device")
        from hypergan.config import resolve_config
        from hypergan.training import ReferenceTrainer
        cuda_avail = bool(env_info and env_info.get("cuda", {}).get("available"))
        dev = "cuda" if cuda_avail else "cpu"
        cfg = resolve_config({
            'components': {
                'generator': {'factory': 'mlp', 'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]}, 'inputs': {'x': 'latent'}},
                'discriminator': {'factory': 'linear', 'args': {'in_features': 2, 'out_features': 1, 'bias': False}, 'inputs': {'input': 'candidate'}},
            },
            'prior': {'args': {'num_particles': 4, 'z_dim': 4}},
            'training': {'steps': 2, 'batch_size': 4, 'device': dev, 'seed': 42},
        })
        trainer = ReferenceTrainer(cfg)
        m, _ = trainer.update()
        _require(math.isfinite(float(m['d_loss'])) and math.isfinite(float(m['g_loss'])),
                 f"Non-finite single-device losses: d_loss={m['d_loss']}, g_loss={m['g_loss']}")
        return {
            "name": "single_device",
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": {"device": dev, "d_loss": float(m['d_loss']), "g_loss": float(m['g_loss'])},
        }

    def _run_suite_cpu_replicated_gloo(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 2: Multi-device CPU Gloo replicated execution."""
        t0 = time.monotonic()
        if self.dry_run:
            return self._dry_run_result("cpu_replicated_gloo")
        results = self._run_subprocess_ranks(backend="gloo", world_size=2, steps=2)
        _require(len(results) == 2 and all(r.get("passed") for r in results), "CPU Gloo ranks did not all report success")
        w0, w1 = results[0]["final_weights"], results[1]["final_weights"]
        _require(w0 == w1, "CPU Gloo replica weights diverged")
        return {
            "name": "cpu_replicated_gloo",
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": {"world_size": 2, "steps": 2, "consensus": True},
        }

    def _run_suite_cuda_replicated_nccl(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 3: Multi-device CUDA NCCL replicated execution."""
        t0 = time.monotonic()
        if self.dry_run:
            return self._dry_run_result("cuda_replicated_nccl")
        cuda = (env_info or {}).get("cuda", {})
        dev_count = env_info.get("device_count", 0) if env_info else 0
        if not (cuda.get("available") and dev_count >= 2 and cuda.get("nccl")):
            return {
                "name": "cuda_replicated_nccl",
                "status": "skipped",
                "duration_seconds": round(time.monotonic() - t0, 4),
                "details": {"reason": f"Requires >= 2 CUDA GPUs with NCCL (found {dev_count})"},
            }
        results = self._run_subprocess_ranks(
            backend="nccl", world_size=2, steps=2, cuda_devices=[0, 1]
        )
        _require(len(results) == 2 and all(r.get("passed") for r in results), "CUDA NCCL ranks did not all report success")
        w0, w1 = results[0]["final_weights"], results[1]["final_weights"]
        _require(w0 == w1, "CUDA NCCL replica weights diverged")
        return {
            "name": "cuda_replicated_nccl",
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": {"world_size": 2, "steps": 2, "consensus": True},
        }

    def _run_suite_aggregation(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 4: Learning aggregation strategies (gradient_allreduce, parameter_average, pseudo_gradient)."""
        t0 = time.monotonic()
        if self.dry_run:
            return self._dry_run_result("aggregation_strategies")
        backend = "gloo"
        strat_details = {}

        # 1. gradient_allreduce
        res_allreduce = self._run_subprocess_ranks(
            backend=backend, world_size=2, steps=2,
            agg_cfg={"strategy": "gradient_allreduce", "sync_every_steps": 1}
        )
        _require(res_allreduce[0]["final_weights"] == res_allreduce[1]["final_weights"],
                 "gradient_allreduce replica weights diverged")
        strat_details["gradient_allreduce"] = "passed"

        # 2. parameter_average
        res_avg = self._run_subprocess_ranks(
            backend=backend, world_size=2, steps=2,
            agg_cfg={"strategy": "parameter_average", "sync_every_steps": 2}
        )
        w_h0 = res_avg[0]["weight_history"]
        w_h1 = res_avg[1]["weight_history"]
        _require(w_h0[0] != w_h1[0], "Independent local steps did not diverge at step 1")
        _require(w_h0[1] == w_h1[1], "FedAvg parameter averaging did not achieve consensus at step 2")
        strat_details["parameter_average"] = "passed"

        # 3. pseudo_gradient: four steps so the second outer step carries momentum.
        pseudo_cfg = {"strategy": "pseudo_gradient", "sync_every_steps": 2, "outer_lr": 1.0}
        res_pseudo = self._run_subprocess_ranks(
            backend=backend, world_size=2, steps=4, agg_cfg={**pseudo_cfg, "outer_momentum": 0.9}
        )
        w_p0 = res_pseudo[0]["weight_history"]
        w_p1 = res_pseudo[1]["weight_history"]
        _require(w_p0[0] != w_p1[0], "DiLoCo inner steps did not diverge at step 1")
        _require(w_p0[1] == w_p1[1], "DiLoCo outer update did not achieve consensus at step 2")
        _require(w_p0[3] == w_p1[3], "DiLoCo outer update did not achieve consensus at step 4")
        res_plain = self._run_subprocess_ranks(
            backend=backend, world_size=2, steps=4, agg_cfg={**pseudo_cfg, "outer_momentum": 0.0}
        )
        plain = res_plain[0]["weight_history"]
        # The first outer step is v = delta either way; momentum can only show at step 4.
        _require(plain[1] == w_p0[1], "DiLoCo first outer step depended on outer_momentum")
        _require(plain[3] != w_p0[3], "DiLoCo outer_momentum had no effect on the second outer step")
        strat_details["pseudo_gradient"] = "passed"

        return {
            "name": "aggregation_strategies",
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": strat_details,
        }

    def _run_suite_multinode_tcp(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 5: two-node topology over TCP rendezvous, simulated on this host.

        Both node ranks are local processes here; this does not exercise separate hosts,
        CPUWorkerService, run_follower or the multi-node launcher.
        """
        t0 = time.monotonic()
        name = "multinode_tcp_single_host"
        if self.dry_run:
            return self._dry_run_result(name)
        cuda = (env_info or {}).get("cuda", {})
        dev_count = env_info.get("device_count", 0) if env_info else 0
        backend = "nccl" if (cuda.get("available") and dev_count >= 2 and cuda.get("nccl")) else "gloo"
        cuda_devs = [0, 1] if backend == "nccl" else None
        results = self._run_subprocess_ranks(
            backend=backend, world_size=2, nodes=2, steps=2, cuda_devices=cuda_devs
        )
        _require(len(results) == 2 and all(r.get("passed") for r in results), "Multi-node ranks did not all report success")
        node_ranks = [r.get("node_rank") for r in results]
        _require(node_ranks == [0, 1], f"Ranks did not run as distinct node ranks: {node_ranks}")
        _require(results[0]["final_weights"] == results[1]["final_weights"], "Multi-node replica weights diverged")
        return {
            "name": name,
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": {
                "simulated": True,
                "hosts": 1,
                "nodes": 2,
                "ranks_per_node": 1,
                "node_ranks": node_ranks,
                "backend": backend,
                "consensus": True,
                "scope": "both node ranks run as local processes over a loopback TCP rendezvous; "
                         "separate hosts, CPUWorkerService, run_follower and the multi-node launcher are not exercised",
            },
        }

    def _run_suite_checkpoint_compatibility(self, env_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suite 6: Distributed checkpoint manifest persistence & compatibility."""
        t0 = time.monotonic()
        if self.dry_run:
            return self._dry_run_result("checkpoint_compatibility")
        with tempfile.TemporaryDirectory() as td:
            results = self._run_subprocess_ranks(
                backend="gloo", world_size=2, steps=2, mode="checkpoint", run_dir=td
            )
            _require(len(results) == 2 and all(r.get("passed") for r in results), "Checkpoint ranks did not all report success")
            _require(all(r.get("checkpoint_verified") for r in results), "Checkpoint restore was not verified on every rank")
            manifests = list(Path(td).rglob("manifest.json"))
            _require(len(manifests) > 0, "Manifest file missing")
            manifest_data = json.loads(manifests[0].read_text())
            _require(manifest_data.get("run_id") == "qual-ckpt",
                     f"Checkpoint manifest has run_id {manifest_data.get('run_id')!r}, expected 'qual-ckpt'")
        return {
            "name": "checkpoint_compatibility",
            "status": "passed",
            "duration_seconds": round(time.monotonic() - t0, 4),
            "details": {"manifest_verified": True},
        }

    # Public aliases
    run_single_device = _run_suite_single_device
    run_cpu_replicated_gloo = _run_suite_cpu_replicated_gloo
    run_cuda_replicated_nccl = _run_suite_cuda_replicated_nccl
    run_aggregation_strategies = _run_suite_aggregation
    run_multinode_tcp = _run_suite_multinode_tcp
    run_checkpoint_compatibility = _run_suite_checkpoint_compatibility

    def execute(self) -> Dict[str, Any]:
        """Run the selected suites within the timeout and return the receipt without writing it."""
        start_time = self._clock()
        self._deadline = start_time + self.timeout
        env_info = inspect_environment()

        # (receipt name, runner); the filter matches receipt names.
        suite_definitions = [
            ("single_device", self._run_suite_single_device),
            ("cpu_replicated_gloo", self._run_suite_cpu_replicated_gloo),
            ("cuda_replicated_nccl", self._run_suite_cuda_replicated_nccl),
            ("aggregation_strategies", self._run_suite_aggregation),
            ("multinode_tcp_single_host", self._run_suite_multinode_tcp),
            ("checkpoint_compatibility", self._run_suite_checkpoint_compatibility),
        ]

        suite_results: List[Dict[str, Any]] = []
        timed_out = False

        for name, runner_fn in suite_definitions:
            if not self.should_run_suite(name):
                continue
            if self._remaining() <= 0:
                timed_out = True
                suite_results.append({
                    "name": name,
                    "status": "skipped",
                    "duration_seconds": 0.0,
                    "details": {"reason": f"not started: qualification timeout of {self.timeout}s exhausted"},
                })
                continue
            self._progress(f"  Running qualification suite: {name}...")
            t0 = self._clock()
            try:
                res = runner_fn(env_info)
                suite_results.append(res)
                self._progress(f"  [{res['status']}] {res['name']} ({res['duration_seconds']}s)")
            except Exception as e:
                suite_results.append({
                    "name": name,
                    "status": "failed",
                    "duration_seconds": round(self._clock() - t0, 4),
                    "error": str(e),
                })
                self._progress(f"  [failed] {name}: {e}")

        total_duration = round(self._clock() - start_time, 4)
        overall_status = summarize_suite_status(suite_results)
        error = None
        if timed_out or total_duration > self.timeout:
            overall_status = "failed"
            error = f"qualification exceeded its {self.timeout}s timeout ({total_duration}s)"

        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "qualification_id": f"qual-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": overall_status,
            "duration_seconds": total_duration,
            "environment": env_info,
            "suites": suite_results,
        }
        if error:
            receipt["error"] = error
        receipt["receipt_sha256"] = compute_receipt_sha256(receipt)
        return receipt

    def run(self, output: Optional[Path | str] = None) -> Dict[str, Any]:
        """Run qualification suites and write the machine-readable JSON receipt.

        The receipt goes to ``output``, else the runner's output_path, else a per-run file
        under the HyperGAN state directory (see default_receipt_path).
        """
        receipt = self.execute()
        target = output or self.output_path or default_receipt_path(receipt["qualification_id"])
        self.last_output_path = write_receipt(receipt, target)
        return receipt


def run_qualification(
    output: Optional[Path | str] = None,
    dry_run: bool = False,
    suite: Optional[str] = None,
    timeout: float = MAX_QUALIFICATION_SECONDS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Functional entrypoint for qualification harness."""
    runner = QualificationRunner(output_path=output, dry_run=dry_run, suite_filter=suite, timeout=timeout, verbose=verbose)
    return runner.run()


def run_qualification_cli(args: argparse.Namespace) -> int:
    """CLI handler for `hypergan qualify`."""
    output_path = getattr(args, "output", None)
    dry_run = getattr(args, "dry_run", False)
    suite = getattr(args, "suite", None)
    timeout = getattr(args, "timeout", MAX_QUALIFICATION_SECONDS)
    verbose = getattr(args, "verbose", False)
    is_json = getattr(args, "json", False)

    try:
        runner = QualificationRunner(
            output_path=output_path,
            dry_run=dry_run,
            suite_filter=suite,
            timeout=timeout,
            verbose=verbose,
            # Keep stdout pure JSON under --json.
            progress_stream=sys.stderr if is_json else None,
        )
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    if not is_json:
        print(f"HyperGAN Hardware Qualification Harness")
        print(f"  Target receipt: {output_path or 'HyperGAN state directory (qualification/)'}")
        print(f"  Mode:           {'dry-run (no suites execute)' if dry_run else 'live hardware'}")
        print(f"  Suite filter:   {suite or 'all'}")
        print(f"  Timeout ceiling: {timeout}s")
        print("Starting qualification checks...")

    try:
        receipt = runner.run()
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    if is_json:
        print(json.dumps(receipt, indent=2))
    else:
        print(f"\nQualification Finished: {receipt['status'].upper()} in {receipt['duration_seconds']}s")
        if receipt.get("error"):
            print(f"  error: {receipt['error']}")
        symbols = {"passed": "✓", "skipped": "⊘", "dry_run": "-"}
        for s in receipt["suites"]:
            print(f"  [{symbols.get(s['status'], '✗')}] {s['name']}: {s['status']} ({s['duration_seconds']}s)")
        print(f"Receipt written to {runner.last_output_path} (unkeyed SHA-256 checksum: {receipt['receipt_sha256'][:16]}...)")

    # A dry run succeeds when it completes; "skipped" means nothing was qualified.
    return 0 if receipt["status"] in ("passed", "dry_run") else 1
