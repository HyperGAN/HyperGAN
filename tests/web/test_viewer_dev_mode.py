"""Refreshing the browser must show edited assets without restarting training."""
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_dev import ASSETS_VARIABLE, ENABLED_VARIABLE, checkout, dev_assets
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession


def fixture_run(root):
    definition = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': definition}}
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    atomic_json(root / 'manifest.json', dict(schema_version=1, run_id='run', attempt_id='a',
                                             steps=0, status='running', metrics_catalog=revision))


def editable_assets(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'index.html').write_text('<html><body><main>viewer</main></body></html>', encoding='utf-8')
    (directory / 'app.js').write_bytes(b'export const build = 1;\n')
    return directory


def client_for(root, **kwargs):
    session = LocalSession(8123, auth='none')
    app = create_app(root, session, poll_seconds=.01, **kwargs)
    return TestClient(app, base_url=session.origin)


def test_normal_mode_keeps_packaged_assets_and_hides_dev_routes(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    monkeypatch.delenv(ENABLED_VARIABLE, raising=False)
    with client_for(tmp_path) as client:
        index = client.get('/')
        assert index.status_code == 200
        assert '/dev/reload.js' not in index.text
        assert index.headers['cache-control'] == 'no-store'
        assert 'etag' not in index.headers and 'last-modified' not in index.headers
        assert client.get('/dev/version').status_code == 404
        assert client.get('/dev/reload.js').status_code == 404


def test_dev_mode_serves_edits_on_refresh_without_restarting(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    assets = editable_assets(tmp_path / 'checkout' / 'web_assets')
    monkeypatch.setenv(ENABLED_VARIABLE, '1')
    monkeypatch.setenv(ASSETS_VARIABLE, str(assets))
    with client_for(tmp_path) as client:
        first = client.get('/assets/app.js')
        assert first.text == 'export const build = 1;\n'
        assert first.headers['cache-control'] == 'no-store'
        # Exactly one policy: the server-wide header must not be appended twice.
        assert first.headers.get_list('cache-control') == ['no-store']
        assert 'etag' not in first.headers and 'last-modified' not in first.headers
        before = client.get('/dev/version').json()['version']

        # The same long-lived server, as during training; only the file changed.
        (assets / 'app.js').write_bytes(b'export const build = 2;\n')
        assert client.get('/assets/app.js').text == 'export const build = 2;\n'
        assert client.get('/dev/version').json()['version'] != before


def test_dev_mode_injects_a_public_reload_module(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    assets = editable_assets(tmp_path / 'checkout' / 'web_assets')
    monkeypatch.setenv(ENABLED_VARIABLE, 'yes')
    monkeypatch.setenv(ASSETS_VARIABLE, str(assets))
    session = LocalSession(8123, auth='token')
    session.write_credentials(tmp_path / 'session.json')
    with TestClient(create_app(tmp_path, session, poll_seconds=.01), base_url=session.origin) as client:
        index = client.get('/')
        assert index.text.count('<script type="module" src="/dev/reload.js"></script>') == 1
        assert index.text.endswith('</body></html>')
        # Unauthenticated, like the assets the page already loads before sign-in.
        assert client.get('/api/v1/capabilities').status_code == 401
        reload_module = client.get('/dev/reload.js')
        assert reload_module.status_code == 200
        assert reload_module.headers['content-type'].startswith('text/javascript')
        assert '/dev/version' in reload_module.text
        assert json.loads(client.get('/dev/version').text)['version']


def test_dev_mode_prefers_the_working_checkout(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'HyperGAN'
    (root / 'frontend').mkdir(parents=True)
    (root / 'frontend' / 'build.mjs').write_text('// build\n', encoding='utf-8')
    editable_assets(root / 'src' / 'hypergan' / 'web_assets')
    monkeypatch.setenv(ENABLED_VARIABLE, '1')
    monkeypatch.delenv(ASSETS_VARIABLE, raising=False)
    monkeypatch.chdir(root / 'frontend')
    assert checkout() == root.resolve()

    fixture_run(tmp_path / 'run')
    with client_for(tmp_path / 'run') as client:
        assert client.get('/assets/app.js').text == 'export const build = 1;\n'
    # The resolved directory is announced once, so the source is never a guess.
    assert str(root.resolve()) in capsys.readouterr().err


def test_dev_switch_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv(ENABLED_VARIABLE, 'sometimes')
    with pytest.raises(ValueError, match='1/0/true/false'):
        dev_assets()
    monkeypatch.setenv(ENABLED_VARIABLE, 'off')
    assert dev_assets().active is False
    monkeypatch.setenv(ENABLED_VARIABLE, 'on')
    monkeypatch.setenv(ASSETS_VARIABLE, str(tmp_path / 'absent'))
    with pytest.raises(ValueError, match='existing directory'):
        dev_assets()


def test_watch_script_is_wired_into_the_frontend():
    frontend = Path(__file__).resolve().parents[2] / 'frontend'
    package = json.loads((frontend / 'package.json').read_text(encoding='utf-8'))
    assert package['scripts']['watch'] == 'node build.mjs --watch'
    source = (frontend / 'build.mjs').read_text(encoding='utf-8')
    assert 'context(' in source and 'ctx.watch()' in source
