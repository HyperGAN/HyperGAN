"""Docs state the contracts the code and CI actually have: tested torch, receipt location, CLI names.

These read files that only a source checkout has, so they skip in an extracted sdist.
"""
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
import pytest

from hypergan.qualify import RECEIPT_STATUSES

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / '.github').is_dir(), reason='needs a source checkout, not an sdist')


def _read(relative):
    return (ROOT / relative).read_text(encoding='utf-8')


def _shell_lines(relative):
    """Lines inside the shell code fences of a markdown file."""
    for block in re.findall(r'^```(?:sh|bash|shell|console|zsh)?\n(.*?)^```', _read(relative), re.M | re.S):
        yield from block.splitlines()


def test_the_ci_tested_torch_is_stated_beside_the_train_extra_and_in_the_readme():
    tested = set(re.findall(r'"torch==([0-9.]+)"', _read('.github/workflows/ci.yml')))
    assert len(tested) == 1, tested
    version = tested.pop()
    train = tomllib.loads(_read('pyproject.toml'))['project']['optional-dependencies']['train']
    torch = next(Requirement(r) for r in train if Requirement(r).name == 'torch')
    assert version in torch.specifier
    pyproject = _read('pyproject.toml')
    comment = pyproject.split('\ntrain = [', 1)[0].rsplit('\n', 1)[-1]
    assert f'torch=={version}' in comment, comment
    assert f'CI tests only `torch=={version}`' in _read('README.md')


def test_glossary_receipt_row_names_the_state_dir_default_and_every_status():
    row = next(line for line in _read('GLOSSARY.md').splitlines() if line.startswith('| **receipt**'))
    assert '<state dir>/qualification/' in row
    assert 'default `qualification-receipt.json`' not in row
    for status in RECEIPT_STATUSES:
        assert f'`{status}`' in row, status


def test_rocm_preflight_doc_derives_the_cuda_partition_from_the_marker():
    rocm = _read('docs/rocm.md')
    assert not re.search(r'\b\d+\s+(single|multi)-GPU tests', rocm, re.I)
    assert '`multi_gpu`' in rocm


def test_docs_teach_gc_and_view_not_the_prune_and_serve_aliases():
    offenders = []
    for relative in ['README.md', *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'docs').glob('*.md'))]:
        for line in _shell_lines(relative):
            if re.match(r'\s*hypergan prune\b', line):
                offenders.append(f'{relative}: {line.strip()}')
    for line in _shell_lines('docs/local-web.md'):
        if re.match(r'\s*hypergan serve\b', line):
            offenders.append(f'docs/local-web.md: {line.strip()}')
    assert not offenders, offenders
