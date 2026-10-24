from contextlib import contextmanager
import os
import re
import torch


def apply_backend_policy(config):
    """Apply explicit process-wide numerical policy before construction/identity.

    An empty policy preserves caller settings. Deterministic failures remain
    errors: unsupported kernels never fall back to warn-only execution.
    """
    policy = config['training']['backend']
    is_rocm = getattr(torch.version, 'hip', None) is not None

    # CUBLAS_WORKSPACE_CONFIG is NVIDIA cuBLAS specific; gate to no-op on ROCm
    if not is_rocm:
        workspace = policy.get('cublas_workspace_config')
        if workspace is not None:
            if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
                raise ValueError('CUBLAS workspace policy must be set before CUDA initialization; start a fresh process')
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
        if policy.get('deterministic_algorithms') and config['training']['device'].startswith('cuda'):
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
                raise ValueError('Deterministic CUDA execution requires training.backend.cublas_workspace_config or a valid CUBLAS_WORKSPACE_CONFIG environment setting before initialization')

    if 'deterministic_algorithms' in policy:
        torch.use_deterministic_algorithms(policy['deterministic_algorithms'], warn_only=False)
    for key, owner, attribute in (
        ('cudnn_deterministic', torch.backends.cudnn, 'deterministic'),
        ('cudnn_benchmark', torch.backends.cudnn, 'benchmark'),
        ('matmul_allow_tf32', torch.backends.cuda.matmul, 'allow_tf32'),
        ('cudnn_allow_tf32', torch.backends.cudnn, 'allow_tf32'),
    ):
        if key in policy:
            setattr(owner, attribute, policy[key])


def enrich_rocm_determinism_error(exc: Exception) -> Exception:
    """Enrich PyTorch determinism error messages on ROCm with failing op and config setting.

    Identifies the failing operator from PyTorch's RuntimeError and surfaces instructions
    to relax the determinism requirement via training.backend.deterministic_algorithms = false.
    """
    if not isinstance(exc, RuntimeError):
        return exc

    msg = str(exc)
    # Check if this is a PyTorch deterministic algorithms error
    if (
        "does not have a deterministic implementation" in msg
        or "use_deterministic_algorithms" in msg
        or "deterministic behavior was requested" in msg.lower()
    ):
        op_match = re.search(r"(?:aten::)?([a-zA-Z0-9_:]+)\s+does not have a deterministic implementation", msg)
        if not op_match:
            op_match = re.search(r"operation\s+(?:aten::)?([a-zA-Z0-9_:]+)\s+does not have a deterministic implementation", msg)
        if not op_match:
            op_match = re.search(r"(?:op|operator|operation)\s+['\"]?([a-zA-Z0-9_:]+)['\"]?", msg)
        op_name = op_match.group(1) if op_match else "unknown_op"

        hip_ver = getattr(torch.version, 'hip', 'unknown')
        enriched_msg = (
            f"{msg}\n\n"
            f"[ROCm Determinism Error] Operation '{op_name}' does not have a deterministic "
            f"implementation on ROCm (torch.version.hip={hip_ver}). "
            f"To relax this requirement, set `training.backend.deterministic_algorithms = false` in your configuration."
        )
        return RuntimeError(enriched_msg)

    return exc


@contextmanager
def rocm_determinism_guard():
    """Context manager intercepting determinism errors on ROCm and enriching them."""
    try:
        yield
    except RuntimeError as exc:
        if getattr(torch.version, 'hip', None) is not None:
            raise enrich_rocm_determinism_error(exc) from exc
        raise


def backend_info():
    """Read effective flags without probing or initializing a CUDA device."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
