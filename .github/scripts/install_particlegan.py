"""Install ParticleGAN from an explicit source/wheel or upstream develop.

Uses PARTICLEGAN_SOURCE when configured (reinstalling with --force-reinstall so
an explicit override always wins over pyproject.toml). When PARTICLEGAN_SOURCE is
unset and particlegan is not yet installed, falls back to upstream develop.
"""
import importlib.util
import os
import subprocess
import sys

DEVELOP_SOURCE = 'git+https://github.com/HyperGAN/ParticleGAN.git@develop'


def main():
    explicit = os.environ.get('PARTICLEGAN_SOURCE', '').strip()
    if explicit:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', explicit],
            check=True,
        )
        return
    if importlib.util.find_spec('particlegan') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', DEVELOP_SOURCE], check=True)


if __name__ == '__main__':
    main()
