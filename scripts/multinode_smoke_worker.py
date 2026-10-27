"""Three fixed-initialization GAN updates for a bounded cross-node smoke test.

This worker needs an external launcher and a reachable TCP rendezvous. It does
not qualify the public supervisor, checkpoint storage, recovery or throughput.
"""
import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import time


def run(args):
    import torch
    import torch.distributed as dist

    from hypergan.config import resolve_config
    from hypergan.distributed_training import ReplicatedTrainer

    world_size = args.nodes * args.ranks_per_node
    rank = args.node_rank * args.ranks_per_node + args.local_rank
    if (args.nodes < 2 or args.ranks_per_node < 1
            or not 0 <= args.node_rank < args.nodes
            or not 0 <= args.local_rank < args.ranks_per_node):
        raise ValueError('Invalid fixed multi-node topology')
    device = 'cuda' if args.backend == 'nccl' else 'cpu'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        if torch.cuda.device_count() != args.ranks_per_node:
            raise ValueError('Allocation must expose exactly one GPU per local rank')
        torch.cuda.set_device(args.local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    dist.init_process_group(args.backend, init_method=args.rendezvous,
                            rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=25))
    try:
        batch_size = 4 * world_size
        config = resolve_config({
            'components': {
                'generator': {'factory': 'mlp',
                              'args': {'input_dim': 4, 'output_dim': 2, 'hidden': [8]},
                              'inputs': {'x': 'latent'}},
                'discriminator': {'factory': 'linear',
                                  'args': {'in_features': 2, 'out_features': 1, 'bias': False},
                                  'inputs': {'input': 'candidate'}},
            },
            'prior': {'args': {'num_particles': max(16, batch_size), 'z_dim': 4}},
            'gradient_penalty': {'lazy_k': 1, 'kappa': 0.01},
            'training': {'steps': 5, 'batch_size': batch_size, 'device': device, 'seed': 42},
            'sampling': {'count': 4},
        })
        trainer = ReplicatedTrainer(config, world_size=world_size, nodes=args.nodes,
                                    node_rank=args.node_rank, ranks_per_node=args.ranks_per_node,
                                    rendezvous=args.rendezvous)
        initial = {name: trainer._digest(module.state_dict()) for name, module in (
            ('generator', trainer.graph.models['generator']),
            ('discriminator', trainer.graph.models['discriminator']),
            ('prior', trainer.prior), ('ema_graph', trainer.ema_graph),
            ('ema_prior', trainer.ema_prior))}
        real = torch.linspace(-1, 1, batch_size * 2, device=trainer.device).reshape(batch_size, 2)
        local_real = real.chunk(world_size)[rank]
        local_ids = torch.arange(batch_size, device=trainer.device).chunk(world_size)[rank]
        losses = []
        for _ in range(3):
            metrics, _ = trainer.update({'real': local_real}, (trainer.prior.z[local_ids], local_ids))
            row = {key: float(metrics[key]) for key in ('d_loss', 'g_loss', 'gradient_penalty')}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f'Nonfinite complete-update metrics: {row}')
            trainer._assert_replicas('cross-node smoke complete state')
            losses.append(row)
        changed = {name: trainer._digest(module.state_dict()) != initial[name]
                   for name, module in (
                       ('generator', trainer.graph.models['generator']),
                       ('discriminator', trainer.graph.models['discriminator']),
                       ('prior', trainer.prior), ('ema_graph', trainer.ema_graph),
                       ('ema_prior', trainer.ema_prior))}
        if not all(changed.values()) or trainer.step != 3 or not trainer.checkpoint_ready:
            raise ValueError(f'Incomplete training updates: {changed}, step={trainer.step}')
        result = {'rank': rank, 'node_rank': args.node_rank, 'local_rank': args.local_rank,
                  'world_size': world_size, 'steps': trainer.step, 'finite': True,
                  'changed': changed, 'losses': losses, 'replica_sha256': trainer._replica_digest(),
                  'seconds': time.monotonic() - started,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'backend': args.backend,
                  'gpu': torch.cuda.get_device_name(args.local_rank) if device == 'cuda' else None}
        peers = [None] * world_size
        dist.all_gather_object(peers, result)
        if (sorted(peer['rank'] for peer in peers) != list(range(world_size))
                or len({peer['replica_sha256'] for peer in peers}) != 1):
            raise ValueError('Final cross-node rank evidence disagrees')
        args.output.write_text(json.dumps(peers, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, default=2)
    parser.add_argument('--ranks-per-node', type=int, required=True)
    parser.add_argument('--node-rank', type=int, required=True)
    parser.add_argument('--local-rank', type=int, required=True)
    parser.add_argument('--rendezvous', required=True)
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
