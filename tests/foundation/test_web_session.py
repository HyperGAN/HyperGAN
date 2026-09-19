import json
import os

import pytest

from hypergan.web_session import LocalSession


def test_credentials_are_private_new_and_rotate(tmp_path):
    session = LocalSession(8123)
    path = session.write_credentials(tmp_path / "credentials.json")
    credential = json.loads(path.read_text())
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        session.write_credentials(path)
    cookie = session.exchange(credential["token"])
    assert session.authenticated(cookie=cookie)
    assert session.authenticated(authorization="Bearer " + credential["token"])
    assert not LocalSession(8123).authenticated(cookie=cookie)
    assert not LocalSession(8123).authenticated(authorization="Bearer " + credential["token"])


def test_host_origin_and_login_validation():
    session = LocalSession(8123)
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
    session = LocalSession(8123, lifetime=1)
    token = json.loads(session.write_credentials(tmp_path / "credentials").read_text())["token"]
    cookie = session.exchange(token)
    monkeypatch.setattr("hypergan.web_session.time.monotonic", lambda: session._expires + 1)
    assert not session.active
    assert not session.authenticated(cookie=cookie)
    assert not session.authenticated(authorization="Bearer " + token)
    with pytest.raises(ValueError):
        session.exchange(token)
