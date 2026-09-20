import io
import json

import pytest

from hypergan.bounded_cli_output import TrainingOutput
from hypergan.console_settings import read_settings, write_settings


def test_default_cadence_filters_only_console_progress():
    stdout, stderr = io.StringIO(), io.StringIO()
    output = TrainingOutput(stdout, stderr, True)
    events = [{'event': 'train', 'step': step, 'metrics': {'loss/g_total': float(step)}} for step in range(1, 202)]
    for event in events:
        output.progress(event)
    for name in ('start', 'checkpoint', 'observer_error', 'stop'):
        output.progress({'event': name, 'step': 201})
    output.result({'steps': 201})
    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [row['step'] for row in rows if row['event'] == 'train'] == [100, 200]
    assert [row['event'] for row in rows[2:]] == ['start', 'checkpoint', 'observer_error', 'stop', 'result']
    assert len(events) == 201 and events[-1]['metrics']['loss/g_total'] == 201.


def test_ui_policy_reload_and_resume_preserve_settings(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr('hypergan.console_settings.time.monotonic', lambda: clock[0])
    stderr = io.StringIO()
    output = TrainingOutput(io.StringIO(), stderr, False)
    output.configure(tmp_path)
    output.progress({'event': 'train', 'step': 1})
    assert stderr.getvalue() == ''
    write_settings(tmp_path, {'progress_every': 2})
    clock[0] = .3
    output.progress({'event': 'train', 'step': 2})
    assert stderr.getvalue() == 'step 2\n'
    resumed = TrainingOutput(io.StringIO(), io.StringIO(), False)
    resumed.configure(tmp_path)
    resumed.progress({'event': 'train', 'step': 4})
    assert resumed.stderr.getvalue() == 'step 4\n'
    resumed.configure(tmp_path, progress_every=3)
    resumed.progress({'event': 'train', 'step': 6})
    assert read_settings(tmp_path) == {'progress_every': 3}
    # Invalid external edits warn, retain last policy and recover on correction.
    (tmp_path / 'console.json').write_text('{')
    clock[0] = 1.
    resumed.progress({'event': 'train', 'step': 9})
    assert 'retaining interval 3' in resumed.stderr.getvalue()
    write_settings(tmp_path, {'progress_every': 5})
    clock[0] = 2.
    resumed.progress({'event': 'train', 'step': 10})
    assert resumed.stderr.getvalue().endswith('step 10\n')


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '100', 1000000001, None])
def test_invalid_interval_cannot_replace_last_setting(tmp_path, value):
    write_settings(tmp_path, {'progress_every': 7})
    with pytest.raises(ValueError):
        write_settings(tmp_path, {'progress_every': value})
    assert read_settings(tmp_path) == {'progress_every': 7}
