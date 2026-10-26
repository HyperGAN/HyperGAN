"""Spawned preview/evaluation workers must never re-run an unguarded training script.

Lightweight: hypergan.main_guard and run preparation never import torch.
"""
import sys
import textwrap
import types

import pytest

from hypergan.execution import prepare_train
from hypergan.main_guard import require_main_guard, unguarded_main_call
from hypergan.run_controller import FatalExecutionError
from tests.config_fixtures import write_default


def _run_as_main(monkeypatch, tmp_path, source, *, spec=None):
    """Execute source as the ``__main__`` script of this process and return what it recorded."""
    script = tmp_path / 'train_script.py'
    script.write_text(textwrap.dedent(source), encoding='utf-8')
    main = types.ModuleType('__main__')
    main.__file__ = str(script)
    main.__spec__ = spec
    main.__dict__['check'] = unguarded_main_call
    monkeypatch.setitem(sys.modules, '__main__', main)
    exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), main.__dict__)
    return script, main.__dict__.get('found')


def test_module_level_training_call_is_detected(monkeypatch, tmp_path):
    script, found = _run_as_main(monkeypatch, tmp_path, '''
        import os
        found = check()
    ''')
    assert found == (str(script), 3)


def test_call_through_a_function_from_module_level_is_detected(monkeypatch, tmp_path):
    script, found = _run_as_main(monkeypatch, tmp_path, '''
        def main():
            return check()
        found = main()
    ''')
    assert found == (str(script), 4)


def test_guarded_call_is_allowed(monkeypatch, tmp_path):
    _, found = _run_as_main(monkeypatch, tmp_path, '''
        def main():
            return check()
        found = 'unset'
        if __name__ == "__main__":
            found = main()
    ''')
    assert found is None


def test_guard_combined_with_and_is_allowed(monkeypatch, tmp_path):
    _, found = _run_as_main(monkeypatch, tmp_path, '''
        enabled = True
        found = 'unset'
        if __name__ == "__main__" and enabled:
            found = check()
    ''')
    assert found is None


def test_guard_combined_with_or_is_not_a_guard(monkeypatch, tmp_path):
    script, found = _run_as_main(monkeypatch, tmp_path, '''
        enabled = True
        if __name__ == "__main__" or enabled:
            found = check()
    ''')
    assert found == (str(script), 4)


def test_package_main_is_not_re_run_by_spawn(monkeypatch, tmp_path):
    spec = types.SimpleNamespace(name='hypergan.__main__')
    _, found = _run_as_main(monkeypatch, tmp_path, 'found = check()\n', spec=spec)
    assert found is None


def test_require_main_guard_raises_a_fatal_error_naming_the_line(monkeypatch, tmp_path):
    def check():
        try:
            require_main_guard('Previews')
        except FatalExecutionError as error:
            return str(error)
    script = tmp_path / 'train_script.py'
    script.write_text('found = check()\n', encoding='utf-8')
    main = types.ModuleType('__main__')
    main.__file__, main.__spec__ = str(script), None
    main.__dict__['check'] = check
    monkeypatch.setitem(sys.modules, '__main__', main)
    exec(compile(script.read_text(), str(script), 'exec'), main.__dict__)
    message = main.__dict__['found']
    assert f'{script}:1' in message and 'if __name__ == "__main__":' in message


def _run_prepared_as_main(monkeypatch, tmp_path, prepared):
    """Call ``prepared.run()`` from an unguarded script's top level; return the error text."""
    def start():
        try:
            prepared.run()
        except FatalExecutionError as error:
            return str(error)
    script = tmp_path / 'train_script.py'
    script.write_text('found = start()\n', encoding='utf-8')
    main = types.ModuleType('__main__')
    main.__file__, main.__spec__ = str(script), None
    main.__dict__['start'] = start
    monkeypatch.setitem(sys.modules, '__main__', main)
    exec(compile(script.read_text(), str(script), 'exec'), main.__dict__)
    return script, main.__dict__['found']


@pytest.mark.parametrize('workers', ['previews', 'custom metrics'])
def test_unguarded_run_with_spawned_workers_fails_before_step_zero(monkeypatch, tmp_path, workers):
    # A large preview interval must not let an unguarded script train until the first preview.
    path, run = write_default(tmp_path / 'project', device='cpu', defaults='hypergan'), tmp_path / 'run'
    if workers == 'previews':
        prepared = prepare_train(path, run, preview_every=5000, stop_after_steps=1)
    else:
        monkeypatch.setattr('hypergan.execution.enabled_custom', lambda config: {'metric': {}})
        prepared = prepare_train(path, run, stop_after_steps=1)
    script, message = _run_prepared_as_main(monkeypatch, tmp_path, prepared)
    assert message is not None and f'{script}:1' in message and 'without a main guard' in message
    assert not run.exists()  # refused before the run directory or any worker existed
