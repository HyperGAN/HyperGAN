import json
import os

import pytest

from hypergan.web_session import LocalSession


def test_credentials_are_private_new_and_rotate(tmp_path):
    session = LocalSession(8123, auth="token")
    path = session.write_credentials(tmp_path / "credentials.json")
    credential = json.loads(path.read_text())
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        session.write_credentials(path)
    cookie = session.exchange(credential["token"])
    assert session.authenticated(cookie=cookie)
    assert session.authenticated(authorization="Bearer " + credential["token"])
    assert not LocalSession(8123, auth="token").authenticated(cookie=cookie)
    assert not LocalSession(8123, auth="token").authenticated(authorization="Bearer " + credential["token"])


def test_host_origin_and_login_validation():
    session = LocalSession(8123, auth="token")
    assert session.cookie_name != LocalSession(8124, auth="token").cookie_name
    assert session.permits_request("127.0.0.1:8123")
    assert session.permits_request("127.0.0.1:8123", "http://127.0.0.1:8123")
    for host, origin in [("localhost:8123", None), ("evil.test:8123", None),
                         (session.host, "null"), (session.host, "http://evil.test"),
                         (session.host, "http://127.0.0.1:8124")]:
        assert not session.permits_request(host, origin)
    for token in [None, {}, "wrong", "é" * 43]:
        assert not session.authenticated(cookie=token, authorization=token)
        with pytest.raises(ValueError, match="Invalid or expired"):
            session.exchange(token)


def test_expired_sessions_reject_both_clients(tmp_path, monkeypatch):
    session = LocalSession(8123, auth="token", lifetime=1)
    token = json.loads(session.write_credentials(tmp_path / "credentials").read_text())["token"]
    cookie = session.exchange(token)
    monkeypatch.setattr("hypergan.web_session.time.monotonic", lambda: session._expires + 1)
    assert not session.active
    assert not session.authenticated(cookie=cookie)
    assert not session.authenticated(authorization="Bearer " + token)
    with pytest.raises(ValueError):
        session.exchange(token)


def test_default_no_auth_and_wildcard_remote_same_origin(tmp_path):
    session = LocalSession(8123, host='0.0.0.0')
    assert session.origin == 'http://127.0.0.1:8123'
    assert session.authenticated()
    assert session.permits_request('training.example:8123', 'http://training.example:8123')
    assert not session.permits_request('training.example:8123', 'http://foreign.example:8123')
    assert not session.permits_request('training.example:8124')
    assert not session.permits_request('training.example:8123@foreign.example')
    record = json.loads(session.write_credentials(tmp_path / 'session.json').read_text())
    assert record['auth_mode'] == 'none'
    assert 'token' not in record
    with pytest.raises(ValueError, match='auth'):
        LocalSession(8123, auth='unknown')
