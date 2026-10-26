"""One rank of a `hypergan qualify` multi-rank suite: ``python -m hypergan.qualify_worker``.

``QualificationRunner._run_subprocess_ranks`` starts one process per rank with::

    python -m hypergan.qualify_worker MODE RANK WORLD_SIZE INIT_METHOD BACKEND OUT_FILE [EXTRA_JSON]

Each rank joins the process group, drives ``ReplicatedTrainer`` on a fixed toy recipe and
writes its losses and per-step weights to ``OUT_FILE`` as JSON for the parent to compare.
``MODE`` is ``step``, or ``checkpoint`` to also save and restore a distributed checkpoint
under ``EXTRA_JSON["run_dir"]``. ``EXTRA_JSON`` carries ``steps``, ``aggregation``,
``nodes``, ``ranks_per_node`` and ``run_dir``.
"""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from hypergan.config import resolve_config
from hypergan.distributed_checkpoints import restore_distributed_checkpoint, save_distributed_checkpoint
from hypergan.distributed_training import ReplicatedTrainer
from hypergan.qualify import qualification_recipe

GLOBAL_BATCH_SIZE = 8


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    mode, rank, world_size, init_method, backend, out_file = (
        args[0], int(args[1]), int(args[2]), args[3], args[4], Path(args[5]))
    extra: Dict[str, Any] = json.loads(args[6]) if len(args) > 6 else {}

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
        # Two spare steps: the trainer must not reach the end of its schedule.
        cfg = resolve_config(qualification_recipe(dev_name, steps=steps + 2))

        execution = {
            'name': 'cuda-replicated-nccl' if backend == 'nccl' else 'cpu-replicated-gloo',
            'world_size': world_size,
            'nodes': nodes,
            'node_rank': node_rank,
            'ranks_per_node': ranks_per_node,
            'rendezvous': init_method,
            'accumulation_steps': 1,
            'global_batch_size': GLOBAL_BATCH_SIZE,
            'local_batch_size': GLOBAL_BATCH_SIZE // world_size,
            'microbatch_size': GLOBAL_BATCH_SIZE // world_size,
            'accumulation_algorithm': 'retained-local-graph-v1',
            'aggregation': agg_cfg,
        }

        raw_data = torch.linspace(-1, 1, 2 * GLOBAL_BATCH_SIZE, device=dev_name).reshape(GLOBAL_BATCH_SIZE, 2)
        ids = torch.arange(GLOBAL_BATCH_SIZE, device=dev_name)
        local_ids = ids.chunk(world_size)[rank]
        local_data = raw_data.chunk(world_size)[rank]

        def new_trainer() -> ReplicatedTrainer:
            return ReplicatedTrainer(
                cfg, world_size=world_size, accumulation_steps=1, nodes=nodes,
                node_rank=node_rank, ranks_per_node=ranks_per_node, rendezvous=init_method,
                execution=execution,
            )

        trainer = new_trainer()
        losses = []
        weight_history = []
        last_batch = None
        for step in range(1, steps + 1):
            m, last_batch = trainer.update({'real': local_data}, (trainer.prior.z[local_ids], local_ids))
            losses.append({"step": step, "d_loss": float(m["d_loss"]), "g_loss": float(m["g_loss"])})
            weights = {k: v.detach().cpu().tolist() for k, v in trainer.graph.state_dict().items()}
            weight_history.append(weights)

        out_data: Dict[str, Any] = {
            "rank": rank,
            "node_rank": node_rank,
            "passed": True,
            "losses": losses,
            "final_weights": weight_history[-1] if weight_history else {},
            "weight_history": weight_history,
        }

        if mode == "checkpoint" and run_dir_str:
            run_dir = Path(run_dir_str)
            save_distributed_checkpoint(
                run_dir, trainer, last_batch,
                {"run_id": "qual-ckpt", "attempt_id": "att-1", "next_sample_sequence": 2}
            )
            fresh_trainer = new_trainer()
            restore_distributed_checkpoint(run_dir, fresh_trainer, {"run_id": "qual-ckpt"})
            if fresh_trainer.step != steps:
                raise RuntimeError(f"Restored trainer is at step {fresh_trainer.step}, expected {steps}")
            out_data["checkpoint_verified"] = True

        out_file.write_text(json.dumps(out_data))
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
