"""Torch-free command definition for an explicit generator gradient audit."""
import json
from pathlib import Path


def add_signal_parser(commands, positive_int):
    parser = commands.add_parser('diagnose-signal', help='Measure generator gradients from a config or complete training checkpoint without updates')
    parser.add_argument('config', metavar='SOURCE', type=Path, help='config path for initialization, or existing run directory for saved online G/D')
    parser.add_argument('--checkpoint', type=Path, help='completed checkpoint inside the source run (default: latest complete checkpoint)')
    parser.add_argument('--output', type=Path, required=True, help='new JSON report file; never overwritten')
    parser.add_argument('--objective', choices=('adversarial', 'total'), default='adversarial')
    parser.add_argument('--batch-size', type=positive_int, help='explicit probe batch override (default: configured batch)')
    parser.add_argument('--device', help='evaluation device override, e.g. cuda:1 or cpu; checkpoint source stays unchanged')


def run_signal(args):
    if args.output.exists():
        raise FileExistsError(f'Signal report already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f'Signal report parent directory does not exist: {args.output.parent}')
    from .signal_diagnostic import diagnose
    result = diagnose(args.config, objective=args.objective, batch_size=args.batch_size,
                      device=args.device, checkpoint=args.checkpoint)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return {'report': str(args.output.resolve()), 'phase': result['phase'], 'step': result['step'],
            'objective': result['objective'],
            'probe_batch_size': result['probe_batch_size'], 'summary': result['summary'],
            'state_verification': result['state_verification'], 'seconds': result['seconds']}
