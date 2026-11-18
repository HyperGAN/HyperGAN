"""Install ParticleGAN from an explicit source/wheel, PyPI, or pinned upstream commit.

Uses PARTICLEGAN_SOURCE when configured, otherwise installs particlegan>=0.10.1,<0.11
from the package index, falling back to the pinned HyperGAN/ParticleGAN v0.10.1 commit
if the PyPI index does not yet serve 0.10.1.
"""
import os
import subprocess
import sys

PINNED_FALLBACK = 'git+https://github.com/HyperGAN/ParticleGAN.git@d7bc20166c022fe037cfb9266272d2cdd7624399'


def main():
    source = os.environ.get('PARTICLEGAN_SOURCE', '').strip()
    if source:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', source], check=True)
        return
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'particlegan>=0.10.1,<0.11'])
    if result.returncode == 0:
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', PINNED_FALLBACK], check=True)


if __name__ == '__main__':
    main()
