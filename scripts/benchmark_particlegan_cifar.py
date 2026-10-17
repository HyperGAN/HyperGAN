"""Bounded actual-architecture integration benchmark; bulk outputs stay local.

Run with the same Python/environment and one visible GPU for each rule. The
input model provides architecture/data/prior/sampling; the current ParticleGAN
preset supplies all trainer settings. This is a seed-0 performance cohort,
not a CIFAR image-quality qualification or a historical-run continuation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import (PARTICLEGAN_DEFAULT_FIELDS, config_values, load_config,
                             particlegan_recipe_fields, resolve_config)
from hypergan.training import ReferenceTrainer, runtime_info, source_info


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def equal(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            difference = (left.cpu().double() - right.cpu().double()).abs()
            raise AssertionError(f'Resume tensor differs: {path}, max_abs={difference.max().item()}, count={torch.count_nonzero(difference).item()}')
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal(left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, path + '.' + str(index))
    else:
        assert left == right, path


def frozen_state(module):
    frozen = {name: cpu_copy(parameter) for name, parameter in module.named_parameters()
              if not parameter.requires_grad}
    # All backbone buffers must remain fixed. Other trainable-head buffers are
    # also checked for this exact architecture, which uses GroupNorm/LayerNorm.
    frozen.update({'buffer.' + name: cpu_copy(value) for name, value in module.named_buffers()})
    return frozen


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    original = load_config(args.config)
    raw = config_values(original)
    fields = particlegan_recipe_fields()
    fields['optimizer_convolution_rule'] = args.rule
    raw['recipe'] = fields
    for (section, key), field in PARTICLEGAN_DEFAULT_FIELDS.items():
        value = fields[field]
        if field == 'prior_betas' and value is None:
            value = fields['betas']
        raw[section][key] = list(value) if isinstance(value, tuple) else value
    raw['name'] = 'performance/cifar-bispf-' + args.rule
    raw['training'].update(seed=0, initialization='deterministic_orthogonal')
    raw['prior']['initialization_seed'] = 2
    raw['sampling']['seed'] = 10000
    # Bounded execution does not change the recipe's declared 200k horizon.
    raw['training']['steps'] = 200000
    raw['training']['network_lr_horizon_cap'] = None
    # Exact resume is audited under deterministic kernels, as a separate
    # numerical contract from the historical throughput report.
    raw['training']['backend'].update(cudnn_benchmark=False, cudnn_deterministic=True,
                                     deterministic_algorithms=True,
                                     matmul_allow_tf32=False, cudnn_allow_tf32=False)
    config = resolve_config(raw)
    (out / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True))
    started = time.monotonic()
    print('constructing', flush=True)
    trainer = ReferenceTrainer(config)
    before = frozen_state(trainer.graph)
    timing = []
    last_batch = None
    torch.cuda.reset_peak_memory_stats(trainer.device)
    with (out / 'events.jsonl').open('w', buffering=1) as log:
        for index in range(args.updates):
            torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()
            row, last_batch = trainer.update()
            seconds = time.perf_counter() - start
            finite = all(torch.isfinite(torch.tensor(value)) for value in row.values()
                         if isinstance(value, (float, int)))
            assert finite
            record = dict(row, seconds=seconds, measured=index >= args.warmup)
            log.write(json.dumps(record) + '\n')
            print(f'step {trainer.step}: {seconds * 1000:.3f} ms D={row["d_loss"]:.6g} G={row["g_loss"]:.6g}', flush=True)
            if index >= args.warmup:
                timing.append(seconds)
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError('Declared wall-clock benchmark budget exhausted')
    peak = torch.cuda.max_memory_allocated(trainer.device)
    equal(before, frozen_state(trainer.graph), 'frozen')
    (out / 'timing.json').write_text(json.dumps({'median_update_ms': statistics.median(timing) * 1000,
                                                'peak_allocated_bytes': peak, 'seconds': timing}))
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profile:
            for _ in range(3):
                trainer.update()
        profile.export_chrome_trace(str(out / 'profile.json'))
        (out / 'profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=45))
    # Keep checkpoint on disk; restore both optimizers, reference buffers,
    # trainability/modes and every consumed constructor/training/eval RNG.
    saved = cpu_copy(trainer_state(trainer, last_batch))
    torch.save(saved, out / 'state.pt')
    resume = None
    if args.resume:
        for _ in range(3):
            _, last_batch = trainer.update()
        expected = cpu_copy(trainer_state(trainer, last_batch))
        restored = ReferenceTrainer(config)
        restore_trainer(restored, saved)
        for _ in range(3):
            _, restored_batch = restored.update()
        actual = cpu_copy(trainer_state(restored, restored_batch))
        try:
            equal(expected, actual)
        except AssertionError:
            torch.save(expected, out / 'resume-expected.pt')
            torch.save(actual, out / 'resume-actual.pt')
            raise
        resume = {'updates': 3, 'exact_state_and_rng': True}
        del restored, expected, actual
    milliseconds = statistics.median(timing) * 1000
    result = {'rule': args.rule, 'updates': args.updates, 'warmup': args.warmup,
              'median_update_ms': milliseconds, 'images_per_second': 64 / statistics.median(timing),
              'peak_allocated_bytes': peak, 'finite': True, 'frozen_values_unchanged': True,
              'resume': resume, 'performance_gate_ms': 300, 'performance_pass': milliseconds <= 300,
              'scope': 'complete ReferenceTrainer.update including data draw/transfer, D/G/prior/EMA; excludes preview/FID/checkpoint IO',
              'runtime': runtime_info(trainer.device), 'source': source_info(config),
              'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
              'seconds': time.monotonic() - started}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rule', choices=('polar', 'quintic'), required=True)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=450)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
