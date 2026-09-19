"""Local viewer credentials, independent of HTTP and numerical runtimes."""

import hmac
import json
import os
from pathlib import Path
import secrets
import time


class LocalSession:
    """One loopback server incarnation, one private bearer and browser session.

    Credentials never enter a run artifact or URL. The owner reads the private
    session file for the browser login or an external client's Authorization
    header. Restarting the server rotates both credentials.
    """

    cookie_name = "hypergan_session"

    def __init__(self, port, *, lifetime=86400):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("viewer port must be between 1 and 65535")
        if type(lifetime) not in (int, float) or not 1 <= lifetime <= 86400:
            raise ValueError("session lifetime must be between 1 and 86400 seconds")
        self.origin = f"http://127.0.0.1:{port}"
        self.host = f"127.0.0.1:{port}"
        # Cookies are scoped by host/path, not TCP port. Multiple experiment
        # servers on loopback must not overwrite each other's browser session.
        self.cookie_name = f"hypergan_session_{port}"
        self.instance_id = secrets.token_hex(16)
        self._token = secrets.token_urlsafe(32)
        self._cookie = secrets.token_urlsafe(32)
        self._expires = time.monotonic() + lifetime
        self._lifetime = lifetime

    @property
    def active(self):
        return time.monotonic() < self._expires

    @staticmethod
    def _equal(candidate, expected):
        return (isinstance(candidate, str) and len(candidate) == len(expected)
                and candidate.isascii() and hmac.compare_digest(candidate, expected))

    def permits_request(self, host, origin=None):
        # No permissive CORS, alternate DNS names or file:// login origins.
        return host == self.host and (origin is None or origin == self.origin)

    def authenticated(self, *, authorization=None, cookie=None):
        if not self.active:
            return False
        return (self._equal(authorization, "Bearer " + self._token)
                or self._equal(cookie, self._cookie))

    def exchange(self, token):
        if not self.active or not self._equal(token, self._token):
            raise ValueError("Invalid or expired viewer credential")
        return self._cookie

    def write_credentials(self, path):
        """Create a private, new file; never replace an existing path/symlink."""
        path = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump({"schema_version": 1, "origin": self.origin,
                           "server_instance_id": self.instance_id,
                           "token": self._token, "expires_in_seconds": self._lifetime}, output)
                output.write("\n")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
