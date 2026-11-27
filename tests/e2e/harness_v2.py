"""HyperGAN 2.0.0 GA E2E Test Harness & Oracles.

Provides synthetic dataset generators, CLI execution helpers, and authoritative
mathematical oracles for opaque-box E2E testing.

Every feature these tests cover has shipped, so nothing here probes for
features or catches import/capability errors to convert failures into skips:
a broken import fails collection instead of silently skipping tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch

# Eager imports of shipped public surfaces: broken imports fail collection
# rather than being masked by capability-probing skip guards.
import hypergan
import hypergan.api as hg
import hypergan.artifacts as hg_artifacts
import hypergan.cache as hg_cache
import hypergan.cli as cli
from hypergan.data import ImageDataError, ImageFolder
import hypergan.execution as hg_execution
import hypergan.image_metrics as hg_image_metrics
import hypergan.inspection as hg_inspection
from hypergan.network_config import validate_network_args
from hypergan.run_info import RunInfo, read_run_info

__all__ = [
    "ImageDataError",
    "ImageFolder",
    "RunInfo",
    "cli",
    "create_synthetic_image_folder",
    "hg",
    "hg_artifacts",
    "hg_cache",
    "hg_execution",
    "hg_image_metrics",
    "hg_inspection",
    "hypergan",
    "oracle_critic_accuracy",
    "oracle_critic_margin",
    "oracle_lerp",
    "oracle_particle_rms",
    "oracle_sha256",
    "oracle_slerp",
    "read_run_info",
    "run_cli",
    "validate_network_args",
]


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
    collinear = dot.abs() > 0.9995
    if collinear.all():
        return (1.0 - t) * v0 + t * v1
    safe_dot = torch.where(collinear, torch.zeros_like(dot), dot)
    omega = torch.acos(safe_dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    scale0 = torch.sin((1.0 - t) * omega) / sin_omega
    scale1 = torch.sin(t * omega) / sin_omega
    slerp_val = scale0 * v0 + scale1 * v1
    return torch.where(collinear, (1.0 - t) * v0 + t * v1, slerp_val)
