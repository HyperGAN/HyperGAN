#!/usr/bin/env python3
"""Finite extra-G warmup, followed by ordinary native D/G updates.

A round is one native D/G/prior update plus optional G-only updates. The
additional updates use fresh draws, the native generator objective and Adam,
but do not advance D, the learned prior, their optimizer states, or penalties.
This research-only wrapper does not change production --tune behavior.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
import torch
import particlegan.grad_regularizers
from hypergan.config import load_config
from hypergan.objective_program import _bound_scores, _generator_tail, score_candidate
from hypergan.training import update_ema, DeviceAdam
from particlegan import GradientPenalty
from joint_rate_probe import run_probe, _atomic_report
from healthy_control_screen import CONFIGS, stages


def penalty_only_update(trainer, optimizer, penalty, batch, context):
    """Separate Adam history: no carried adversarial momentum on GP-only steps."""
    term = trainer.program.adversarial_terms[0]
    optimizer.zero_grad(set_to_none=True)
    loss = penalty(lambda x: score_candidate(term, x, context, trainer.graph, 'critic'),
                   batch['real'], context['generated'].detach(), step=1,
                   generator=trainer.streams['penalty'])
    trainer._refuse_nonfinite(loss, 'Nonfinite extra penalty', [])
    value = float(loss.detach())
    if value > 0:
        loss.backward()
        trainer._refuse_nonfinite(loss, 'Nonfinite extra penalty',
                                  [('Nonfinite penalty gradient', trainer.program.critic_parameters)])
        optimizer.step()
    return {'loss': value, 'optimizer_step': value > 0}


def extra_generator_update(trainer, *, penalty_optimizer=None, penalty=None):
    """Native single-term G objective, with D and prior held fixed."""
    program = trainer.program
    if len(program.adversarial_terms) != 1 or program.objectives:
        raise ValueError('This diagnostic requires one adversarial term and no auxiliary objectives')
    term = program.adversarial_terms[0]
    batch, latent_draw, penalty_result = None, None, None
    if penalty is not None:
        with torch.no_grad():
            batch, ids, context = trainer._draw(None, None)
        latent_draw = (context['latent'].detach(), ids)
        penalty_result = penalty_only_update(trainer, penalty_optimizer, penalty, batch, context)
    fixed = list(term.module.parameters()) + list(trainer.prior.parameters())
    flags = [p.requires_grad for p in fixed]
    versions = [p._version for p in fixed]
    buffers = [(b, b.detach().clone()) for b in term.module.buffers()]
    try:
        for parameter in fixed:
            parameter.requires_grad_(False)
        trainer.opt_g.zero_grad(set_to_none=True)
        batch, ids, context = trainer._draw(batch, latent_draw)
        _, fake, real_score, fake_score = _bound_scores(
            term, context, trainer.graph, 'generator', term.generator_phase, first='fake')
        adversarial = term.gan.g_loss(fake_score, real_score)
        prior_loss, objectives = _generator_tail(trainer, program, context, ids, fake)
        loss = term.weight * adversarial + prior_loss + sum(objectives)
        loss.backward()
        trainer._refuse_nonfinite(loss, 'Nonfinite extra-G loss',
                                  [('Nonfinite extra-G gradient', program.generator_parameters)])
        # Prior grads are None, so Adam skips both parameter and moment updates.
        assert all(p.grad is None for p in program.prior_parameters)
        trainer.opt_g.step()
        update_ema(trainer.ema_graph.models['generator'], trainer.graph.models['generator'],
                   trainer.config['training']['ema'])
        assert versions == [p._version for p in fixed], 'Extra-G changed D/prior parameters'
        assert all(torch.equal(b, old) for b, old in buffers), 'Extra-G changed critic buffers'
        values = trainer._metric_transfer([loss, adversarial, prior_loss])
        result = dict(zip(('g_loss', 'g_adversarial', 'prior_loss'), values))
        if penalty_result is not None:
            result['extra_penalty'] = penalty_result
        return result
    finally:
        for parameter, flag in zip(fixed, flags):
            parameter.requires_grad_(flag)


def prepare(*, ratio=4, warmup_rounds=32, warmup_g_factor=1., extra_penalty='none'):
    if type(ratio) is not int or ratio < 1 or type(warmup_rounds) is not int or warmup_rounds < 0:
        raise ValueError('Invalid finite warmup schedule')
    if not 0 < warmup_g_factor <= 1:
        raise ValueError('Warmup G factor must be in (0, 1]')
    if extra_penalty not in ('none', 'b_cap', 'e_interp'):
        raise ValueError('Unsupported extra penalty')

    @contextmanager
    def context(trainer):
        original_update = trainer.update
        original_base = trainer.base_lrs[0][0]
        penalty, penalty_optimizer = None, None
        if extra_penalty != 'none':
            options = dict(trainer.config['gradient_penalty'])
            options.update(arm=extra_penalty, lazy_k=1)
            penalty = GradientPenalty(**options)
            penalty_optimizer = DeviceAdam(trainer.program.critic_parameters,
                                            **trainer.opt_d.defaults)
            for group, rate in zip(penalty_optimizer.param_groups, trainer.base_lrs[1]):
                group['lr'] = rate
        proposal = {
            'kind': 'finite-extra-generator-warmup', 'warmup_rounds': warmup_rounds,
            'warmup_g_per_d': ratio, 'warmup_g_lr_factor': warmup_g_factor,
            'after_warmup': 'native 1D:1G at original rates',
            'step_unit': 'one D/G/prior round, including any extra G-only updates',
            'extra_g_semantics': 'fresh real/latent draws; D fixed during G backward/update, optionally penalty-updated beforehand; prior parameters/moments fixed; G EMA per G update',
            'clocks': 'D, penalty, prior, annealing use round count; G Adam and G EMA use G update count',
            'rounds': [], 'g_updates': 0, 'd_updates': 0, 'prior_updates': 0,
            'extra_penalty': extra_penalty, 'penalty_only_updates': 0,
            'penalty_optimizer': 'separate Adam at D rate/betas; no adversarial momentum; skip zero penalty',
            'penalty_schedule': 'before each extra G, coefficient as configured without lazy multiplier; native lazy penalty unchanged',
        }

        def update(*args, **kwargs):
            number = trainer.step + 1
            warming = number <= warmup_rounds
            trainer.base_lrs[0][0] = original_base * (warmup_g_factor if warming else 1.)
            row, batch = original_update(*args, **kwargs)
            proposal['g_updates'] += 1
            proposal['d_updates'] += 1
            proposal['prior_updates'] += 1
            detail = {'round': number, 'warming': warming,
                      'native_g_loss': row['g_loss'], 'extra_g': []}
            for _ in range(ratio - 1 if warming else 0):
                result = extra_generator_update(trainer, penalty_optimizer=penalty_optimizer, penalty=penalty)
                detail['extra_g'].append(result)
                proposal['g_updates'] += 1
                proposal['penalty_only_updates'] += int(result.get('extra_penalty', {}).get('optimizer_step', False))
            detail.update(g_updates=proposal['g_updates'], d_updates=proposal['d_updates'],
                          prior_updates=proposal['prior_updates'], g_lr=trainer.opt_g.param_groups[0]['lr'])
            proposal['rounds'].append(detail)
            # Standard per_step losses retain native paired-step semantics;
            # every extra G loss is separately recorded in proposal.rounds.
            return row, batch

        trainer.update = update
        try:
            yield proposal
        finally:
            del trainer.update
            trainer.base_lrs[0][0] = original_base
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CONFIGS), default='logos')
    parser.add_argument('--ratio', type=int, default=4)
    parser.add_argument('--warmup-rounds', type=int, default=32)
    parser.add_argument('--warmup-g-factor', type=float, default=1.)
    parser.add_argument('--extra-penalty', choices=('none', 'b_cap', 'e_interp'), default='none')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.warmup_rounds < args.steps:
        parser.error('Include at least one ordinary round after warmup')
    source = CONFIGS[args.case]
    config = load_config(source)
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (source, source.parent / 'generator.hndl', source.parent / 'discriminator.hndl'):
        (args.output / path.name).write_bytes(path.read_bytes())
    for name, path in [('runner.py', Path(__file__)), ('joint_rate_probe.py', ROOT / 'reports/joint_rate_probe.py'),
                       ('objective_program.py', ROOT / 'src/hypergan/objective_program.py'),
                       ('grad_regularizers.py', Path(particlegan.grad_regularizers.__file__))]:
        (args.output / name).write_bytes(path.read_bytes())
    (args.output / 'resolved-training-config.json').write_text(json.dumps(config, indent=2) + '\n')
    opt = config['optimizer']
    report = run_probe(source, g_lr=opt['lr'], d_lr=opt['lr'] * opt['d_lr_mult'],
                       steps=args.steps, device=args.device,
                       observe_steps=[0, 1, 4, 8, 16, 32, 33, 48, 64, 96, 128],
                       observe_modules=stages(args.case),
                       prepare=prepare(ratio=args.ratio, warmup_rounds=args.warmup_rounds,
                                       warmup_g_factor=args.warmup_g_factor, extra_penalty=args.extra_penalty),
                       progress_path=args.output / 'report.json')
    report['schedule'] = 'Native D/G/prior rounds with research-only extra G warmup; see proposal for exact counts and clocks'
    report['budget']['extra_generator_updates'] = report['proposal']['g_updates'] - report['completed_updates']
    _atomic_report(args.output / 'report.json', report)
    if report['status'] != 'complete':
        raise RuntimeError(report.get('failure', report.get('audit_failure')))


if __name__ == '__main__':
    main()
