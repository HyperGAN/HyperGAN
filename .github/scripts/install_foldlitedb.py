"""Install Foldlite DB from an explicit source/wheel or the configured index.

Before its first package release, set repository variable FOLDLITEDB_SOURCE to
an accessible wheel URL or a pip source reference pinned to an exact revision.
Local developers can point the same variable at their Foldlite checkout.
"""
import os
import subprocess
import sys

source = os.environ.get('FOLDLITEDB_SOURCE', '').strip()
requirement = source or 'foldlitedb>=0.1.2,<0.2'
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', requirement], check=True)
except subprocess.CalledProcessError:
    raise SystemExit(
        'Foldlite DB is required. Before its first release, configure '
        'FOLDLITEDB_SOURCE with an accessible wheel URL or a pinned pip source '
        'reference; local development may use the Foldlite checkout path.'
    )
