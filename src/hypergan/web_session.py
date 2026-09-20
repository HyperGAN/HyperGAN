"""Local viewer credentials, independent of HTTP and numerical runtimes."""

import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit


class LocalSession:
    """One server incarnation with optional private bearer and browser session.

    Credentials never enter a run artifact or URL. The owner reads the private
    session file for the browser login or an external client's Authorization
    header. Restarting the server rotates both credentials.
    """

    cookie_name = "hypergan_session"

    def __init__(self, port, *, host="127.0.0.1", auth="none", lifetime=None):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("viewer port must be between 1 and 65535")
        if lifetime is not None and (type(lifetime) not in (int, float) or not 1 <= lifetime <= 86400):
            raise ValueError("session lifetime must be between 1 and 86400 seconds")
        if auth not in {"none", "token"}:
            raise ValueError("auth must be 'none' or 'token'")
        self.auth_mode = auth
        self.bind_host = host
        self.port = port
        browser_host = "127.0.0.1" if host == "0.0.0.0" else host
        self.host = f"{browser_host}:{port}"
        self.origin = f"http://{self.host}"
        # Cookies are scoped by host/path, not TCP port. Multiple experiment
        # servers on loopback must not overwrite each other's browser session.
        self.cookie_name = f"hypergan_session_{port}"
        self.instance_id = secrets.token_hex(16)
        self._token = secrets.token_urlsafe(32)
        self._cookie = secrets.token_urlsafe(32)
        self._expires = None if lifetime is None else time.monotonic() + lifetime
        self._lifetime = lifetime

    @property
    def active(self):
        return self._expires is None or time.monotonic() < self._expires

    @staticmethod
    def _equal(candidate, expected):
        return (isinstance(candidate, str) and len(candidate) == len(expected)
                and candidate.isascii() and hmac.compare_digest(candidate, expected))

    def permits_request(self, host, origin=None):
        # Wildcard binding accepts remote hostnames, while browser requests must
        # still be same-origin. Explicit binding restricts the Host authority.
        if not isinstance(host, str) or any(char in host for char in "/\\@?# "):
            return False
        try:
            parsed = urlsplit("http://" + host)
            valid = bool(parsed.hostname) and (parsed.port or 80) == self.port
        except ValueError:
            return False
        allowed = self.bind_host == "0.0.0.0" or host == self.host
        return valid and allowed and (origin is None or origin == "http://" + host)

    def authenticated(self, *, authorization=None, cookie=None):
        if self.auth_mode == "none":
            return True
        if not self.active:
            return False
        return (self._equal(authorization, "Bearer " + self._token)
                or self._equal(cookie, self._cookie))

    def exchange(self, token):
        if self.auth_mode != "token" or not self.active or not self._equal(token, self._token):
            raise ValueError("Invalid or expired viewer credential")
        return self._cookie

    def write_credentials(self, path):
        """Create a private, new file; never replace an existing path/symlink."""
        path = Path(path)
        # An existing pathname is also the readiness signal for CLI clients.
        # Publish only after close, with a hard link's atomic no-clobber contract.
        # The private staging file is on the same filesystem as the destination.
        descriptor, staging = tempfile.mkstemp(prefix=".hypergan-session-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump({"schema_version": 1, "origin": self.origin,
                           "server_instance_id": self.instance_id,
                           "auth_mode": self.auth_mode, "bind_host": self.bind_host,
                           **({"token": self._token, "expires_in_seconds": self._lifetime}
                              if self.auth_mode == "token" else {})}, output)
                output.write("\n")
            os.link(staging, path)
        finally:
            Path(staging).unlink(missing_ok=True)
        return path
