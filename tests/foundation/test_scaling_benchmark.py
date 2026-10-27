"""Scaling receipts must represent every rank and enforce the declared cohort."""
import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'scaling_benchmark', Path(__file__).resolve().parents[2] / 'scripts' / 'benchmark_scaling.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def receipt(world=2, **overrides):
    settings = benchmark.options(batch_size=8, warmup=1, steps=2)
    request = benchmark.payload(settings, 1, world)
    rows = []
    for rank in range(world):
        rows.append(dict(rank=rank, node_rank=0, local_rank=rank, spec=request,
                         completed_steps=3, global_batch_size=8, finite=True, parameters_changed=True,
                         step_seconds=[1.0, 2.0], measured_seconds=3.0, peak_allocated_bytes=2**20,
                         initial_sha256='initial', final_sha256='final', config_sha256='config',
                         gpu='test GPU', torch='test runtime', cuda='test CUDA', replica_check_every=100))
    rows[-1].update(overrides)
    return rows, request


def test_slowest_rank_window_controls_throughput_and_peak_memory():
    rows, request = receipt(step_seconds=[2.0, 4.0], measured_seconds=6.0,
                            peak_allocated_bytes=5 * 2**20)
    result = benchmark.summarize(rows, request)
    assert result['samples_per_second'] == pytest.approx(16 / 6)
    assert result['median_step_ms'] == 3000
    assert result['p95_step_ms'] == 4000
    assert result['peak_allocated_mib'] == 5


@pytest.mark.parametrize('change', [
    {'rank': 0}, {'node_rank': 1}, {'local_rank': 0}, {'completed_steps': 2},
    {'finite': False}, {'parameters_changed': False}, {'global_batch_size': 4},
    {'step_seconds': [1.0]}, {'step_seconds': [float('nan'), 1.0]},
    {'measured_seconds': float('inf')}, {'measured_seconds': .1},
    {'initial_sha256': 'different'}, {'final_sha256': 'different'}, {'gpu': 'different GPU'},
])
def test_invalid_or_disagreeing_rank_evidence_cannot_pass(change):
    rows, request = receipt(**change)
    with pytest.raises(ValueError):
        benchmark.summarize(rows, request)


def test_missing_rank_is_not_a_successful_small_allocation():
    rows, request = receipt()
    with pytest.raises(ValueError, match='rank evidence'):
        benchmark.summarize(rows[:1], request)


def test_strong_and_weak_batches_have_distinct_meanings():
    strong = benchmark.payload(benchmark.options(batch_size=64), 2, 4)
    weak = benchmark.payload(benchmark.options(mode='weak', batch_size=8), 2, 4)
    assert strong['global_batch_size'] == weak['global_batch_size'] == 64
    assert benchmark.payload(benchmark.options(mode='weak', batch_size=8), 1, 2)['global_batch_size'] == 16
    with pytest.raises(ValueError, match='divisible'):
        benchmark.payload(benchmark.options(batch_size=7), 1, 2)


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf'), .6])
def test_invalid_or_insufficient_modal_budget_refuses_the_matrix(budget):
    with pytest.raises(ValueError):
        benchmark.modal_plan(benchmark.options(), benchmark.topologies('1,2,4,2x4'), budget)


def test_default_modal_matrix_fits_two_dollars_and_includes_all_nodes():
    plan = benchmark.modal_plan(benchmark.options(), benchmark.topologies('1,2,4,2x4'))
    assert [p['world_size'] for p in plan['profiles']] == [1, 2, 4, 8]
    assert plan['estimated_ceiling_usd'] == pytest.approx(.5 + 15 * 180 * (.000306 + .0000131 + 4 * .00000222))
    assert plan['estimated_ceiling_usd'] < 2
    with pytest.raises(ValueError, match='supported'):
        benchmark.modal_plan(benchmark.options(), [(3, 4)])


def test_duplicate_topology_is_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        benchmark.topologies('2,1x2')


def test_comparison_uses_baseline_and_refuses_changed_initialization():
    rows, request = receipt(world=1)
    first = benchmark.summarize(rows, request)
    rows, request = receipt()
    second = benchmark.summarize(rows, request)
    second['samples_per_second'] *= 1.5
    report = benchmark.comparison([first, second], gpu_second=benchmark.A10_SECOND)
    assert report['baseline'] == '1'
    assert second['throughput_speedup'] == 1.5
    assert second['scaling_efficiency'] == .75
    assert second['estimated_steady_samples_per_dollar'] > 0
    changed = copy.deepcopy(second)
    changed['initial_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Initial'):
        benchmark.comparison([first, changed])
    changed = copy.deepcopy(second)
    changed['config_sha256'] = 'changed'
    with pytest.raises(ValueError, match='unchanged resolved recipe'):
        benchmark.comparison([first, changed])
