"""Install ParticleGAN from an explicit source/wheel, PyPI, or pinned upstream commit.

Uses PARTICLEGAN_SOURCE when configured, otherwise installs particlegan>=0.10.0,<0.11
from the package index, falling back to the pinned HyperGAN/ParticleGAN v0.10.0 commit
if the PyPI index does not yet serve 0.10.0.
"""
import os
import subprocess
import sys

PINNED_FALLBACK = 'git+https://github.com/HyperGAN/ParticleGAN.git@952c3b62bc7d90224778679f9688318c78b30fb4'


def main():
    source = os.environ.get('PARTICLEGAN_SOURCE', '').strip()
    if source:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', source], check=True)
        return
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'particlegan>=0.10.0,<0.11'])
    if result.returncode == 0:
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', PINNED_FALLBACK], check=True)


if __name__ == '__main__':
    main()
