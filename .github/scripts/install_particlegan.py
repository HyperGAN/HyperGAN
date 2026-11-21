"""Install ParticleGAN from an explicit source/wheel, PyPI, or pinned upstream commit.

Uses PARTICLEGAN_SOURCE when configured, otherwise installs particlegan>=0.11.2,<0.12
from the package index, falling back to the pinned HyperGAN/ParticleGAN v0.11.2 commit
if the PyPI index does not yet serve 0.11.2.
"""
import os
import subprocess
import sys

PINNED_FALLBACK = 'git+https://github.com/HyperGAN/ParticleGAN.git@985d1cde661c0446c8f9d19e8f51773b9ebfe94b'


def main():
    source = os.environ.get('PARTICLEGAN_SOURCE', '').strip()
    if source:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', source], check=True)
        return
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'particlegan>=0.11.2,<0.12'])
    if result.returncode == 0:
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', PINNED_FALLBACK], check=True)


if __name__ == '__main__':
    main()
