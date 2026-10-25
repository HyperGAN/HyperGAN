"""Spawned preview/evaluation workers must never re-run an unguarded training script."""
import subprocess
import sys
import textwrap
import types

import pytest

from hypergan.run_controller import FatalExecutionError
from hypergan.single_execution import require_main_guard, unguarded_main_call


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


@pytest.mark.heavy
def test_unguarded_script_with_previews_fails_before_running_twice(tmp_path):
    marker = tmp_path / 'top-level-runs.txt'
    script = tmp_path / 'unguarded.py'
    script.write_text(textwrap.dedent(f'''
        import hypergan.api as hg
        with open({str(marker)!r}, 'a') as stream:
            stream.write('ran\\n')
        model = hg.override(hg.packaged('gaussian-grid'), {{'train.device': 'cpu'}})
        hg.train(model, {str(tmp_path / 'run')!r}, steps=2, previews=1, devices='cpu')
    '''), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode != 0
    assert 'without a main guard' in result.stderr
    assert marker.read_text() == 'ran\n'
