"""The explicit headless path has no serving or multiprocessing dependency."""
import builtins

import pytest

from hypergan.cli import _parser, _training_viewer


def test_no_server_never_imports_serving_or_opens_sockets(monkeypatch, tmp_path):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not any(part in name for part in ('web_', 'starlette', 'uvicorn', 'wasmtime', 'socket', 'multiprocessing'))
        return original(name, *args, **kwargs)
    args = _parser().parse_args(['train', 'config.toml', '--run-dir', str(tmp_path / 'run'), '--no-server'])
    monkeypatch.setattr(builtins, '__import__', guarded)
    with _training_viewer(args) as viewer:
        assert viewer is None


@pytest.mark.parametrize('extra', [['--open'], ['--server-port', '8000']])
def test_headless_conflicts_are_clear(extra, tmp_path):
    args = _parser().parse_args(['resume', str(tmp_path), '--no-server', *extra])
    with pytest.raises(ValueError, match='cannot be combined'):
        _training_viewer(args)


def test_missing_web_extra_auto_continues_explicit_fails(monkeypatch, tmp_path, capsys):
    from hypergan import web_launch
    from hypergan.web_autostart import training_viewer
    monkeypatch.setattr(web_launch.importlib.util, 'find_spec', lambda name: None)
    with training_viewer(tmp_path) as viewer:
        assert viewer is None
    assert capsys.readouterr().err == ''
    with pytest.raises(RuntimeError, match=r'hypergan\[web\]'):
        with training_viewer(tmp_path, required=True):
            pytest.fail('training started')


def test_bind_and_auth_cli_options():
    parser = _parser()
    serve = parser.parse_args(['serve', 'run'])
    assert (serve.host, serve.auth) == ('0.0.0.0', 'none')
    serve = parser.parse_args(['serve', 'run', '--host', '127.0.0.1', '--auth', 'token'])
    assert (serve.host, serve.auth) == ('127.0.0.1', 'token')
    for command in [['train', 'config', '--run-dir', 'run'], ['resume', 'run']]:
        args = parser.parse_args([*command, '--server-host', '192.0.2.1', '--auth', 'token'])
        assert (args.server_host, args.auth) == ('192.0.2.1', 'token')
