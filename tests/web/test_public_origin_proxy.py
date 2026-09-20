"""A TLS proxy in front of the viewer, named explicitly and trusted no further."""
import json

import pytest
from starlette.testclient import TestClient

from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_server import create_app
from hypergan.web_session import LocalSession, normalize_public_origin

PUBLIC = 'https://mlserver.tail1234.ts.net'
PROXIED = {'origin': PUBLIC}


def fixture_run(root):
    definition = {'kind': 'scalar', 'source': 'g_loss', 'label': 'Generator loss'}
    definition['definition_hash'] = digest(definition)
    catalog = {'schema_version': 1, 'metrics': {'loss/g_total': definition}}
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    atomic_json(root / 'manifest.json', dict(schema_version=1, run_id='run', attempt_id='a',
                                             steps=0, status='running', metrics_catalog=revision))


def client_for(root, session, base_url=None):
    return TestClient(create_app(root, session, poll_seconds=.01), base_url=base_url or session.origin)


def test_proxied_https_request_signs_in_and_writes_the_console(tmp_path):
    """tailscale serve forwards the public Host; that request is a first-class one."""
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth='token', public_origin=PUBLIC)
    console = '/api/v1/runs/run/console'
    with client_for(tmp_path, session, base_url=PUBLIC) as client:
        assert client.get('/api/v1/capabilities', headers=PROXIED).status_code == 401
        login = client.post('/api/v1/session', json={'token': session._token}, headers=PROXIED)
        assert login.status_code == 200
        cookie = login.headers['set-cookie']
        assert 'Secure' in cookie and 'HttpOnly' in cookie and 'samesite=strict' in cookie.lower()
        assert client.get('/api/v1/capabilities', headers=PROXIED).json()['public_origin'] == PUBLIC
        assert client.put(console, json={'progress_every': 9}, headers=PROXIED).json() == {'progress_every': 9}
        # The page loads its own assets and API through relative URLs, so the
        # single 'self' policy already names the proxy's origin.
        policy = client.get('/', headers=PROXIED).headers['content-security-policy']
        assert "connect-src 'self'" in policy and 'http://' not in policy


@pytest.mark.parametrize('headers', [
    {'origin': 'https://evil.test'},
    {'origin': 'https://mlserver.tail1234.ts.net.evil.test'},
    {'origin': 'http://mlserver.tail1234.ts.net'},
    {'host': 'other.tail1234.ts.net', 'origin': PUBLIC},
])
def test_other_https_origins_stay_rejected(tmp_path, headers):
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth='token', public_origin=PUBLIC)
    with client_for(tmp_path, session, base_url=PUBLIC) as client:
        assert client.post('/api/v1/session', json={'token': session._token},
                           headers=headers).status_code == 403
        assert client.put('/api/v1/runs/run/console', json={'progress_every': 4},
                          headers=headers).status_code == 403


def test_direct_plain_http_login_still_works_beside_the_proxy(tmp_path):
    """A Secure cookie would never be stored over loopback HTTP, so it is not sent."""
    fixture_run(tmp_path)
    session = LocalSession(8123, host='0.0.0.0', auth='token', public_origin=PUBLIC)
    with client_for(tmp_path, session) as client:
        login = client.post('/api/v1/session', json={'token': session._token})
        assert login.status_code == 200
        assert 'secure' not in login.headers['set-cookie'].lower()
        assert 'HttpOnly' in login.headers['set-cookie']
        # The browser kept the cookie, so the authenticated routes stay usable.
        assert client.get('/api/v1/capabilities').json()['public_origin'] == PUBLIC
        assert client.put('/api/v1/runs/run/console', json={'progress_every': 6}).status_code == 200
        assert client.get('/api/v1/capabilities', headers={'host': 'evil.example'}).status_code == 403


def test_forwarded_headers_count_only_when_a_public_origin_is_configured(tmp_path):
    """A proxy may rewrite Host; anyone can forge the header, so it needs the opt-in."""
    fixture_run(tmp_path)
    rewritten = {'host': '127.0.0.1:8123', 'origin': PUBLIC,
                 'x-forwarded-host': 'mlserver.tail1234.ts.net', 'x-forwarded-proto': 'https'}
    unset = LocalSession(8123, host='0.0.0.0', auth='token')
    with client_for(tmp_path, unset) as client:
        assert client.post('/api/v1/session', json={'token': unset._token},
                           headers=rewritten).status_code == 403

    session = LocalSession(8123, host='0.0.0.0', auth='token', public_origin=PUBLIC)
    with client_for(tmp_path, session) as client:
        login = client.post('/api/v1/session', json={'token': session._token}, headers=rewritten)
        assert login.status_code == 200
        assert 'Secure' in login.headers['set-cookie']
        # A forwarded authority that is not the configured one buys nothing.
        assert client.post('/api/v1/session', json={'token': session._token},
                           headers=dict(rewritten, **{'x-forwarded-host': 'evil.test'})).status_code == 403


def test_public_origin_values_are_validated_strictly():
    for value in ['https://box.ts.net', 'https://box.ts.net/', 'https://box.ts.net:443',
                  'HTTPS://Box.TS.net']:
        assert normalize_public_origin(value) == 'https://box.ts.net'
    assert normalize_public_origin('http://box.lan:8765') == 'http://box.lan:8765'
    assert normalize_public_origin(None) is None
    for value in ['', '   ', 'box.ts.net', '//box.ts.net', 'ftp://box.ts.net', 'wss://box.ts.net',
                  'https://', 'https://box.ts.net/viewer', 'https://box.ts.net?a=1',
                  'https://box.ts.net#x', 'https://user:pw@box.ts.net', 'https://box.ts.net:0',
                  'https://box.ts.net:70000', 'https://box .ts.net', 'https://bòx.ts.net', 8765]:
        with pytest.raises(ValueError):
            normalize_public_origin(value)
    with pytest.raises(ValueError, match='public origin'):
        LocalSession(8123, public_origin='https://box.ts.net/viewer')
    plain = LocalSession(8123)
    assert (plain.public_origin, plain.cookie_secure, plain.browser_origin) == (None, False, plain.origin)
    proxied = LocalSession(8123, public_origin='http://box.lan:8765')
    # An http proxy is accepted, but nothing marks its cookie Secure.
    assert proxied.cookie_secure is False and proxied.browser_origin == 'http://box.lan:8765'


def test_public_origin_reaches_the_detached_supervisor_and_its_reports(tmp_path):
    """The supervised viewer is a subprocess; the state file is what reaches it."""
    import http.client
    from hypergan.web_autostart import Viewer, stop_viewer, viewer_status

    fixture_run(tmp_path)
    viewer = Viewer(tmp_path, host='127.0.0.1', auth='token', public_origin=PUBLIC + '/')
    try:
        viewer.wait_ready()
        assert viewer.session.public_origin == PUBLIC
        assert viewer.session.browser_origin == PUBLIC
        credentials = json.loads(viewer.credential_path.read_text())
        assert credentials['public_origin'] == PUBLIC
        status = viewer_status(tmp_path)
        assert status['public_origin'] == PUBLIC and status['origin'].startswith('http://127.0.0.1:')
        receipt = json.loads((tmp_path / 'observations' /
                              f'viewer-{viewer.session.instance_id}.json').read_text())
        assert receipt['public_origin'] == PUBLIC

        # A real proxied request against the running server, not a test client.
        connection = http.client.HTTPConnection(viewer.session.host, timeout=5)
        try:
            connection.request('POST', '/api/v1/session',
                               body=json.dumps({'token': credentials['token']}),
                               headers={'content-type': 'application/json',
                                        'host': 'mlserver.tail1234.ts.net', 'origin': PUBLIC})
            response = connection.getresponse()
            body = response.read()
            assert response.status == 200, body
            assert 'Secure' in response.getheader('set-cookie')
        finally:
            connection.close()

        with pytest.raises(ValueError, match='different bind/auth'):
            Viewer(tmp_path, host='127.0.0.1', auth='token', public_origin='https://other.ts.net')
    finally:
        viewer.detach()
        stop_viewer(tmp_path)
