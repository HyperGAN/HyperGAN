"""Local viewer port policy, free of serving, socket and numerical imports.

A stable default keeps one bookmarked browser URL valid across restarts. The
bounded upward search keeps a second concurrent run from failing to start, and
is small enough that an exhausted range is a real conflict worth reporting.
"""

DEFAULT_VIEWER_PORT = 8765
PORT_SEARCH_LIMIT = 100
