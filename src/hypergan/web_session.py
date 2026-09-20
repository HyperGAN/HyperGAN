"""Local viewer credentials, independent of HTTP and numerical runtimes."""

import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit


def normalize_public_origin(value):
    """Canonical ``scheme://host[:port]`` for a TLS proxy's public origin.

    A browser behind a proxy sends back exactly this string in ``Origin``, so
    only an absolute http/https URL naming a host and an optional port can be
    compared against it. A path, query, fragment or embedded credential would
    make the value something other than an origin, and is refused rather than
    silently trimmed. ``None`` means no proxy is configured.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('public origin must be a URL string such as https://host.example')
    text = value.strip()
    if not text:
        raise ValueError('public origin must not be empty')
    if not text.isascii() or any(character.isspace() for character in text):
        raise ValueError(f'public origin must be a plain ASCII URL without spaces, not {value!r}')
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f'public origin {value!r} is not a valid URL: {error}') from error
    scheme = parsed.scheme.lower()
    if scheme not in {'http', 'https'}:
        raise ValueError(f'public origin must start with http:// or https://, not {value!r}')
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f'public origin must not carry a username or password: {value!r}')
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment or '?' in text or '#' in text:
        raise ValueError(f'public origin must name a host and optional port only, not {value!r}')
    host = parsed.hostname
    if not host:
        raise ValueError(f'public origin must name a host, not {value!r}')
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f'public origin port must be between 1 and 65535, not {value!r}')
    literal = f'[{host}]' if ':' in host else host
    default = 443 if scheme == 'https' else 80
    authority = literal if port in (None, default) else f'{literal}:{port}'
    return f'{scheme}://{authority}'


def _first(value):
    """The nearest hop's value from a possibly comma-joined forwarded header."""
    if not isinstance(value, str):
        return None
    head = value.split(',')[0].strip()
    return head or None


class LocalSession:
    """One server incarnation with optional private bearer and browser session.

    Credentials never enter a run artifact or URL. The owner reads the private
    session file for the browser login or an external client's Authorization
    header. Restarting the server rotates both credentials.
    """

    cookie_name = "hypergan_session"

    def __init__(self, port, *, host="127.0.0.1", auth="none", lifetime=None, public_origin=None):
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
        # An explicitly named TLS proxy is the only other accepted authority.
        self.public_origin = normalize_public_origin(public_origin)
        self.public_scheme = None if self.public_origin is None else self.public_origin.split("://", 1)[0]
        self.public_host = None if self.public_origin is None else self.public_origin.split("://", 1)[1]
        self.cookie_secure = self.public_scheme == "https"
        self.browser_origin = self.public_origin or self.origin
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

    def _direct_host(self, host):
        # Wildcard binding accepts remote hostnames, while browser requests must
        # still be same-origin. Explicit binding restricts the Host authority.
        if not isinstance(host, str) or any(char in host for char in "/\\@?# "):
            return False
        try:
            parsed = urlsplit("http://" + host)
            valid = bool(parsed.hostname) and (parsed.port or 80) == self.port
        except ValueError:
            return False
        return valid and (self.bind_host == "0.0.0.0" or host == self.host)

    def _public_host(self, host):
        """True when an authority names exactly the configured public origin."""
        if self.public_origin is None or not isinstance(host, str):
            return False
        try:
            return normalize_public_origin(self.public_scheme + "://" + host) == self.public_origin
        except ValueError:
            return False

    def match_request(self, host, origin=None, *, forwarded_proto=None, forwarded_host=None):
        """Name the channel a request arrived on: 'public', 'direct' or None.

        Direct access is unchanged and never weakened. A configured public
        origin adds exactly one further accepted authority, and is the only
        thing that makes `X-Forwarded-*` readable at all: without it the
        forwarded headers are ignored, because anyone can send them.
        """
        direct = self._direct_host(host)
        if self.public_origin is not None and (origin is None or origin == self.public_origin):
            # tailscale serve forwards the public Host unchanged, so that alone
            # decides. A proxy that rewrites Host to the local authority must
            # name the public one in X-Forwarded-Host, and its own Host must
            # still be an authority this server would have accepted directly.
            if self._public_host(host):
                return "public"
            forwarded = _first(forwarded_host)
            if direct and forwarded is not None and self._public_host(forwarded):
                proto = _first(forwarded_proto)
                if proto is None or proto.lower() == self.public_scheme:
                    return "public"
        if direct and (origin is None or origin == "http://" + host):
            return "direct"
        return None

    def permits_request(self, host, origin=None, *, forwarded_proto=None, forwarded_host=None):
        return self.match_request(host, origin, forwarded_proto=forwarded_proto,
                                  forwarded_host=forwarded_host) is not None

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
                           **({"public_origin": self.public_origin} if self.public_origin else {}),
                           **({"token": self._token, "expires_in_seconds": self._lifetime}
                              if self.auth_mode == "token" else {})}, output)
                output.write("\n")
            os.link(staging, path)
        finally:
            Path(staging).unlink(missing_ok=True)
        return path
