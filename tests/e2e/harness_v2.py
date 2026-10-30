"""HyperGAN 2.0.0 GA E2E Test Harness & Oracles.

Provides capability detection probes, synthetic dataset generators, CLI execution
helpers, and authoritative mathematical oracles for opaque-box E2E testing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

import hypergan
import hypergan.api as hg
import hypergan.cli as cli
from hypergan.data import ImageFolder


# ---------------------------------------------------------------------------
# Dynamic Capability Probes (Progressive Testability across Milestones M1-M6)
# ---------------------------------------------------------------------------

def _get_cli_subparsers() -> Dict[str, argparse.ArgumentParser]:
    """Inspect CLI parser subcommands without executing."""
    try:
        parser = cli._parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices
        return {}
    except Exception:
        return {}


def _has_cli_subcommand(subcmd: str) -> bool:
    return subcmd in _get_cli_subparsers()


def _subcommand_has_flag(subcmd: str, flag: str) -> bool:
    subparsers = _get_cli_subparsers()
    if subcmd not in subparsers:
        return False
    parser = subparsers[subcmd]
    for action in parser._actions:
        if flag in action.option_strings:
            return True
    return False


# Milestone 1 Probes
HAS_CACHE_MODULE: bool = hasattr(hypergan, "cache") or hasattr(hg, "ensure_cached_url")
try:
    import torch_fidelity
    HAS_TORCH_FIDELITY: bool = True
except ImportError:
    HAS_TORCH_FIDELITY: bool = False

# Milestone 2 Probes
HAS_MULTINODE_CLI: bool = _subcommand_has_flag("train", "--nodes") and _subcommand_has_flag("train", "--strategy")

# Milestone 3 Probes
HAS_SCHEMA_CLI: bool = _has_cli_subcommand("schema")
HAS_OPS_CLI: bool = _has_cli_subcommand("ops")
HAS_GLOBAL_JSON: bool = _subcommand_has_flag("train", "--json")
HAS_QUIET_FLAG: bool = _subcommand_has_flag("train", "--quiet") or _subcommand_has_flag("train", "-q")
HAS_TYPED_RUNINFO: bool = hasattr(hg, "read_run_info") or hasattr(hypergan.api, "read_run_info")

# Milestone 4 Probes
HAS_WEB_BIND_FLAG: bool = _subcommand_has_flag("train", "-b") or _subcommand_has_flag("train", "--bind")
HAS_EXPORT_FORMAT: bool = _subcommand_has_flag("export", "--format")
HAS_SAMPLE_OUTPUT_DIR: bool = _subcommand_has_flag("sample", "--output-dir")
HAS_LOADED_GENERATOR: bool = hasattr(hg, "LoadedGenerator") or hasattr(hypergan.api, "LoadedGenerator")
HAS_LATENT_INTERPOLATE: bool = hasattr(hg, "interpolate") or hasattr(hypergan.api, "interpolate")

# Check ImageFolder parameter support
try:
    import inspect
    _img_params = inspect.signature(ImageFolder.__init__).parameters
    HAS_IMAGEFOLDER_ALPHA: bool = "alpha" in _img_params
    HAS_IMAGEFOLDER_SYMLINKS: bool = "follow_symlinks" in _img_params
except Exception:
    HAS_IMAGEFOLDER_ALPHA = False
    HAS_IMAGEFOLDER_SYMLINKS = False

# Check Static HNDL resolution in validate_network_args
def _check_static_hndl_validation() -> bool:
    try:
        from hypergan.network_config import validate_network_args
        bad_spec = {
            "source": "nonexistent_op_syntax_error(999)",
            "input_shape": ["B", 4],
            "output_shape": ["B", 2],
        }
        errors = validate_network_args(bad_spec)
        return len(errors) > 0 and any("nonexistent_op" in e or "Undefined" in e or "Unknown" in e for e in errors)
    except Exception:
        return False

HAS_STATIC_HNDL_CHECK: bool = _check_static_hndl_validation()


# ---------------------------------------------------------------------------
# CLI In-Process Invocation & Output Capture
# ---------------------------------------------------------------------------

def run_cli(*args: str) -> Tuple[int, str, str]:
    """Execute hypergan CLI in-process, capturing return code, stdout, and stderr."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            rc = cli.main([str(a) for a in args])
            if rc is None:
                rc = 0
        except SystemExit as exc:
            rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except Exception as exc:
            stderr_buf.write(f"\nUnhandled Exception: {exc}\n")
            rc = 1
    return rc, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Synthetic Dataset & Project Builders
# ---------------------------------------------------------------------------

def create_synthetic_image_folder(
    directory: Path,
    count: int = 8,
    resolution: int = 32,
    include_rgba: bool = False,
    include_palette: bool = False,
    create_symlink: bool = False,
) -> Path:
    """Create a synthetic dataset folder with RGB, RGBA, and palette PNGs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        r = int(50 + (i * 20) % 200)
        g = int(80 + (i * 15) % 170)
        b = int(120 + (i * 25) % 130)
        
        if include_rgba and i % 3 == 1:
            # Create RGBA image with translucent alpha channel
            img = Image.new("RGBA", (resolution, resolution), color=(r, g, b, 180))
            img.save(directory / f"img_rgba_{i:03d}.png")
        elif include_palette and i % 3 == 2:
            # Create 8-bit palette (P mode) image
            base = Image.new("RGB", (resolution, resolution), color=(r, g, b))
            img = base.convert("P", palette=Image.Palette.ADAPTIVE)
            img.save(directory / f"img_palette_{i:03d}.png")
        else:
            # Standard RGB image
            img = Image.new("RGB", (resolution, resolution), color=(r, g, b))
            img.save(directory / f"img_{i:03d}.png")

    if create_symlink:
        # Create a subdirectory and symlink it
        sub_dir = directory.parent / f"{directory.name}_linked_source"
        sub_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (resolution, resolution), color=(255, 0, 0))
        img.save(sub_dir / "symlinked_sample.png")
        symlink_target = directory / "symlink_dir"
        if not symlink_target.exists():
            symlink_target.symlink_to(sub_dir, target_is_directory=True)

    return directory


# ---------------------------------------------------------------------------
# Authoritative Mathematical Oracles
# ---------------------------------------------------------------------------

def oracle_critic_margin(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> float:
    """Authoritative formulation for critic margin: E[D(x)] - E[D(G(z))]."""
    return float((real_scores.mean() - fake_scores.mean()).item())


def oracle_critic_accuracy(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> float:
    """Authoritative formulation for critic ranking accuracy: P(D(x) > D(G(z)))."""
    return float((real_scores > fake_scores).float().mean().item())


def oracle_particle_rms(particles: torch.Tensor) -> float:
    """Authoritative formulation for particle table RMS: sqrt(mean(z^2))."""
    return float(torch.sqrt(torch.mean(particles ** 2)).item())


def oracle_sha256(data: bytes) -> str:
    """Authoritative SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def oracle_lerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Authoritative linear interpolation: (1 - t) * v0 + t * v1."""
    return (1.0 - t) * v0 + t * v1


def oracle_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Authoritative spherical linear interpolation (slerp)."""
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True).clamp_min(eps)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True).clamp_min(eps)
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    scale0 = torch.sin((1.0 - t) * omega) / sin_omega
    scale1 = torch.sin(t * omega) / sin_omega
    return scale0 * v0 + scale1 * v1
