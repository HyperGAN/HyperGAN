"""Parser/dispatch helpers for the optional manual evaluation command."""
from pathlib import Path


def add_evaluate_parser(commands):
    parser = commands.add_parser('evaluate', aliases=['eval'], help='Evaluate a configured snapshot metric on an immutable EMA snapshot')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--metric', default=None, help='enabled metrics.custom ID with mode=snapshot (default: sole enabled snapshot metric)')
    parser.add_argument('--config', type=Path, help='recipe with evaluation settings; numerical recipe must match')
    parser.add_argument('--bundle', type=Path, help='older model.pt within this run attempts directory')
    parser.add_argument('--checkpoint', default=None, help='target checkpoint: best, latest, step number, or path')
    parser.add_argument('--devices', default=None, help='device assignment (e.g. 1, gpu1, cuda:1, cpu)')
    return parser


def run_evaluate(args):
    from .metric_evaluation import evaluate
    return evaluate(args.run_dir, getattr(args, 'metric', None), config_path=args.config,
                    bundle=args.bundle, checkpoint=getattr(args, 'checkpoint', None),
                    devices=getattr(args, 'devices', None))
