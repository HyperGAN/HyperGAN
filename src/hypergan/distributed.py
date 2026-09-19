"""Internal CPU collective primitives; not a distributed training strategy.

All ranks must call the same operations and differentiated backward passes in the
same order. The caller owns process-group initialization, finite timeouts and
worker cleanup. Only the default Gloo group is supported in this first fixture.
"""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


class GlooCollectives:
    """Fixed-membership, CPU numerical groundwork with collective validation."""

    def __init__(self, world_size):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Initialize the default Gloo process group with a finite timeout first")
        if dist.get_backend() != "gloo":
            raise ValueError("This numerical fixture supports only the default Gloo process group")
        if dist.get_world_size() != world_size:
            raise ValueError("Initialized process group differs from fixed world_size")
        self.world_size = world_size

    def _agree(self, operation, tensor, *, indices=False, num_rows=None):
        """Exchange metadata before tensor collectives, so rank errors agree."""
        error = None
        if not isinstance(tensor, torch.Tensor):
            error = "input must be a tensor"
        elif tensor.device.type != "cpu" or tensor.layout != torch.strided:
            error = "input must be a dense CPU tensor"
        elif indices:
            if tensor.dtype != torch.int64 or tensor.ndim != 1:
                error = "indices must be a one-dimensional int64 tensor"
            elif type(num_rows) is not int or num_rows < 1:
                error = "num_rows must be a positive integer"
            elif tensor.numel() and (tensor.min().item() < 0 or tensor.max().item() >= num_rows):
                error = "indices are outside the replicated prior table"
        elif tensor.dtype not in (torch.float32, torch.float64) or tensor.ndim < 1 or not tensor.numel():
            error = "input must be a nonempty float32/float64 batch tensor"
        metadata = {"operation": operation, "error": error, "num_rows": num_rows,
                    "shape": list(tensor.shape) if error is None else None,
                    "dtype": str(tensor.dtype) if error is None else None,
                    "requires_grad": tensor.requires_grad if error is None else None,
                    "grad_enabled": torch.is_grad_enabled()}
        peers = [None] * self.world_size
        # These small objects come only from trusted members of this CPU group.
        dist.all_gather_object(peers, metadata)
        if any(peer["operation"] != operation for peer in peers):
            raise ValueError("Ranks called different collective operations")
        errors = [f"rank {rank}: {peer['error']}" for rank, peer in enumerate(peers) if peer["error"]]
        if errors:
            raise ValueError("Invalid collective input: " + "; ".join(errors))
        keys = ("num_rows",) if indices else ("shape", "dtype", "requires_grad", "grad_enabled")
        if any(any(peer[key] != peers[0][key] for key in keys) for peer in peers):
            raise ValueError("Ranks must agree on tensor shape/dtype/autograd participation and prior size")
        return peers

    def gather(self, tensor):
        """Concatenate equal local batches in rank order, retaining autograd.

        Backward SUMS contributions from every rank consuming the gathered value.
        It does not divide by world_size. Higher derivatives also communicate.
        """
        self._agree("gather", tensor)
        return torch.cat(all_gather(tensor.contiguous()), dim=0)

    def mean(self, tensor):
        """Differentiable global batch mean, retaining a leading dimension.

        Equal nonempty batch shapes are required. No detach or rank-local mean
        substitution is used; this is suitable for relativistic-average logits.
        """
        self._agree("mean", tensor)
        return torch.cat(all_gather(tensor.contiguous()), dim=0).mean(dim=0, keepdim=True)

    def unique_indices(self, indices, *, num_rows):
        """Sorted union of rank-local prior IDs; empty/unequal lists are allowed.

        Index a REPLICATED raw prior table with the result and evaluate the same
        regularizer on every rank. With DDP's averaged parameter gradients this
        term needs no extra world-size multiplier or divisor. IDs have no gradient.
        """
        peers = self._agree("unique_indices", indices, indices=True, num_rows=num_rows)
        lengths = [peer["shape"][0] for peer in peers]
        maximum = max(lengths)
        if maximum == 0:
            return indices.new_empty(0)
        padded = indices.new_zeros(maximum)
        padded[:len(indices)] = indices
        gathered = [torch.empty_like(padded) for _ in peers]
        dist.all_gather(gathered, padded)
        return torch.cat([value[:length] for value, length in zip(gathered, lengths)]).unique(sorted=True)
