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


@pytest.mark.parametrize('extra', [['--open'], ['--port', '8000'], ['--server-port', '8000']])
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


def _free_base(count):
    """A base with `count` consecutive ports proven free a moment ago."""
    from hypergan.web_launch import bind_loopback, bind_server
    for _ in range(50):
        with bind_loopback() as probe:
            base = probe.getsockname()[1]
        held = []
        try:
            for offset in range(count):
                held.append(bind_server(base + offset, '127.0.0.1'))
        except OSError:
            continue
        finally:
            for listener in held:
                listener.close()
        return base
    raise AssertionError('no run of consecutive free ports available')


def test_default_port_is_stable_and_steps_past_occupied_ports(monkeypatch):
    """One bookmarked URL survives restarts; a busy default moves by one."""
    from hypergan import web_launch
    base = _free_base(3)
    monkeypatch.setattr(web_launch, 'DEFAULT_VIEWER_PORT', base)
    with web_launch.bind_available(host='127.0.0.1') as first:
        assert first.getsockname()[1] == base
        with web_launch.bind_available(host='127.0.0.1') as second:
            assert second.getsockname()[1] == base + 1
            with web_launch.bind_available(host='127.0.0.1') as third:
                assert third.getsockname()[1] == base + 2
    # Releasing the default returns the next launch to the same stable port.
    with web_launch.bind_available(host='127.0.0.1') as repeated:
        assert repeated.getsockname()[1] == base


def test_exhausted_default_range_names_the_conflict(monkeypatch):
    from hypergan import web_launch
    base = _free_base(1)
    monkeypatch.setattr(web_launch, 'DEFAULT_VIEWER_PORT', base)
    with web_launch.bind_server(base, '127.0.0.1'):
        with pytest.raises(OSError, match=f'No free viewer port in {base}..{base}'):
            web_launch.bind_available(host='127.0.0.1', attempts=1)


def test_requested_port_is_strict_and_zero_still_means_any_port():
    from hypergan.web_launch import bind_available, bind_loopback
    with bind_loopback() as occupied:
        port = occupied.getsockname()[1]
        # An explicit request never relocates silently to a neighbouring port.
        with pytest.raises(OSError):
            bind_available(port, '127.0.0.1')
    with bind_available(0, '127.0.0.1') as ephemeral:
        assert ephemeral.getsockname()[1] > 0


def test_viewer_port_options_share_one_default():
    from hypergan import web_launch
    from hypergan.ports import DEFAULT_VIEWER_PORT
    parser = _parser()
    # Serving and the automatic viewer read the same single definition.
    assert web_launch.DEFAULT_VIEWER_PORT == DEFAULT_VIEWER_PORT
    # None defers to the stable default; every explicit value stays literal.
    assert parser.parse_args(['serve', 'run']).port is None
    assert parser.parse_args(['serve', 'run', '--port', '0']).port == 0
    assert parser.parse_args(['serve', 'run', '--port', '8123']).port == 8123
    for command in [['train', 'config', '--run-dir', 'run'], ['resume', 'run']]:
        assert parser.parse_args(command).server_port is None
        assert parser.parse_args([*command, '--port', '8123']).server_port == 8123
        assert parser.parse_args([*command, '--server-port', '8123']).server_port == 8123
        assert parser.parse_args([*command, '--port', '0']).server_port == 0
