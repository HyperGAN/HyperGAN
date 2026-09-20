"""Explicit backend policy shared by training and disposable evaluation workers."""
import os
import torch


def apply_backend_policy(config):
    """Apply explicit process-wide numerical policy before construction/identity.

    An empty policy preserves caller settings. Deterministic failures remain
    errors: unsupported kernels never fall back to warn-only execution.
    """
    policy = config['training']['backend']
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
