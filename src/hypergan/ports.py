"""Local viewer port policy, free of serving, socket and numerical imports.

A stable default port 8765 keeps one bookmarked browser URL valid across
restarts and across runs under the same runs root. At most one persistent
dashboard process runs per runs root on fixed port 8765. An occupied port is
an error naming its owner, never a search for another port.
"""

DEFAULT_VIEWER_PORT = 8765
