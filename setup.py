"""Stamp source provenance; project metadata lives in pyproject.toml."""

import json
from pathlib import Path
import runpy

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def _source():
    package = Path(__file__).resolve().parent / 'src' / 'hypergan'
    return runpy.run_path(str(package / 'provenance.py'))['hypergan_source'](package)


def _stamp(destination, source):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / '_build_provenance.json').write_text(json.dumps({
        'schema_version': 1, 'commit': source['hypergan_commit'],
        'dirty': source['hypergan_dirty'],
    }, sort_keys=True) + '\n')


class ProvenanceBuild(build_py):
    def run(self):
        source = _source()
        super().run()
        _stamp(Path(self.build_lib) / 'hypergan', source)

    def get_outputs(self, include_bytecode=True):
        outputs = super().get_outputs(include_bytecode)
        stamp = str(Path(self.build_lib) / 'hypergan' / '_build_provenance.json')
        return outputs if stamp in outputs else [*outputs, stamp]


class ProvenanceSdist(sdist):
    def run(self):
        # Capture before setuptools creates its untracked release staging tree.
        self._release_provenance = _source()
        super().run()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        _stamp(Path(base_dir) / 'src' / 'hypergan', self._release_provenance)

setup(cmdclass={'build_py': ProvenanceBuild, 'sdist': ProvenanceSdist})
