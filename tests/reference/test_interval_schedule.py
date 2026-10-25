"""Interval evaluation fairness: a short cadence must not starve longer ones."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypergan.config import load_config
from hypergan.interval_evaluation import IntervalEvaluations

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples' / 'logos-tiny-transformer-dinov3-projected-128.toml'


class StepWorker:
    """A fake evaluator that finishes a fixed number of training steps after submit."""
    def __init__(self, manifest, duration):
        self.manifest, self.duration, self.job = manifest, duration, None

    @property
    def busy(self):
        return self.job is not None

    def submit(self, root, selected, metric_id, identity, snapshot):
        assert not self.busy
        self.source = dict(identity, metric_id=metric_id)
        self.job = self.manifest['steps'] + self.duration

    def poll(self):
        if self.job is None or self.manifest['steps'] < self.job:
            return None
        self.job = None
        return {'status': 'complete', 'evaluation_id': self.source['evaluation_id']}

    def abort(self, **kwargs):
        self.job = None


DUE = {'random_spread': 100, 'chroma': 50, 'structure': 50, 'fid10k_heldout': 10}


def simulate(tmp_path, monkeypatch, duration, steps=50000, drain=0):
    from hypergan import interval_evaluation, summary
    monkeypatch.setattr(interval_evaluation, '_recover_abandoned', lambda *args: None)
    monkeypatch.setattr(summary, 'write_summary_json', lambda *args, **kwargs: None)
    config = load_config(EXAMPLE)
    manifest = {'run_id': 'run', 'attempt_id': 'attempt', 'attempt_index': 1, 'steps': 0}
    events = []
    scheduler = IntervalEvaluations(config, tmp_path, manifest,
        SimpleNamespace(evaluation_snapshot=lambda *args: {}),
        lambda event, **values: events.append((event, values)))
    scheduler.start()
    scheduler.worker = StepWorker(manifest, duration)
    # Drain steps fall before the next due point, so they only flush deferred work.
    for step in range(1, steps + drain + 1):
        manifest['steps'] = step
        scheduler.schedule()
    ran = Counter(values['metric_id'] for event, values in events if event == 'evaluation_scheduled')
    return ran, manifest['evaluation_schedule'], scheduler


def test_idle_evaluator_runs_every_configured_metric_on_every_due_point(tmp_path, monkeypatch):
    # The shipped logos schedule: chroma and structure collide with random_spread
    # every 1000 steps and fid10k_heldout every 5000. Collisions are deferred to
    # the next free boundary rather than dropped.
    ran, schedule, scheduler = simulate(tmp_path, monkeypatch, 1, drain=10)
    assert ran == DUE
    assert not scheduler.pending
    assert all(record['skipped_busy'] == 0 for record in schedule.values())


@pytest.mark.parametrize('duration', [300, 700])
def test_slow_evaluator_still_rotates_through_every_metric(tmp_path, monkeypatch, duration):
    # Evaluations outlive the shortest cadence, so repeated due points coalesce,
    # but every metric keeps running and the longest cadence is never dropped.
    ran, schedule, scheduler = simulate(tmp_path, monkeypatch, duration)
    assert set(ran) == set(DUE)
    assert ran['fid10k_heldout'] >= 9
    assert min(ran.values()) >= 9
    for key, due in DUE.items():
        assert ran[key] + schedule[key]['skipped_busy'] + (key in scheduler.pending) == due
