"""Parser/dispatch helpers for the optional manual evaluation command."""
from pathlib import Path


def add_evaluate_parser(commands):
    parser = commands.add_parser('evaluate', help='Evaluate a configured manual metric on an immutable EMA snapshot')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--metric', required=True, help='enabled metrics.custom ID with mode=snapshot and trigger=manual')
    parser.add_argument('--config', type=Path, help='recipe with evaluation settings; numerical recipe must match')
    parser.add_argument('--bundle', type=Path, help='older model.pt within this run attempts directory')
    return parser


def run_evaluate(args):
    from .metric_evaluation import evaluate
    return evaluate(args.run_dir, args.metric, config_path=args.config, bundle=args.bundle)
