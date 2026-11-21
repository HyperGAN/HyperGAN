"""Install ParticleGAN from an explicit source/wheel or upstream develop.

Uses PARTICLEGAN_SOURCE when configured, otherwise installs HyperGAN/ParticleGAN
from the develop branch so develop tracks unreleased ParticleGAN fixes.
"""
import os
import subprocess
import sys

DEVELOP_SOURCE = 'git+https://github.com/HyperGAN/ParticleGAN.git@develop'


def main():
    source = os.environ.get('PARTICLEGAN_SOURCE', '').strip() or DEVELOP_SOURCE
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', source], check=True)


if __name__ == '__main__':
    main()
