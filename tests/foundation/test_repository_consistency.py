"""The repository's docs, packaging metadata and CI agree with each other and with the code.

These read files that only a source checkout has (README, docs, CI workflow,
Dockerfile), so they skip in an extracted sdist.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from hypergan.cli import _parser
from hypergan import model_file
from hypergan.provenance import hypergan_source

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / '.github').is_dir(), reason='needs a source checkout, not an sdist')

DOC_FILES = [*(p for p in ('README.md', 'CONTRIBUTING.md', 'tests/README.md') if (ROOT / p).is_file()),
             *sorted(str(p.relative_to(ROOT).as_posix()) for p in (ROOT / 'docs').glob('*.md'))]
SHELL_FENCES = {'', 'sh', 'bash', 'shell', 'console', 'zsh'}
PYTHON_FENCES = {'python', 'py'}
COMMAND = re.compile(r'^(?:[A-Z_][A-Z0-9_]*=\S+\s+)*(?:hypergan|python3? -m hypergan)\s+(.*)$')


def _read(relative):
    return (ROOT / relative).read_text(encoding='utf-8')


def _pyproject():
    return tomllib.loads(_read('pyproject.toml'))


def _shell_blocks(text):
    language, lines = None, []
    for line in text.splitlines():
        fence = re.fullmatch(r'\s*```(\w*)\s*', line)
        if fence and language is None:
            language, lines = fence.group(1), []
        elif line.strip() == '```' and language is not None:
            if language in SHELL_FENCES:
                yield lines
            language = None
        elif language is not None:
            lines.append(line)


def _documented_commands():
    for relative in DOC_FILES:
        for lines in _shell_blocks(_read(relative)):
            logical, current = [], ''
            for line in lines:
                if line.rstrip().endswith('\\'):
                    current += line.rstrip()[:-1] + ' '
                    continue
                logical.append(current + line)
                current = ''
            for line in logical:
                line = line.strip().removeprefix('$ ')
                if line.startswith('#'):
                    continue
                for part in re.split(r'\s*(?:&&|\|\||;|\|)\s*', line):
                    match = COMMAND.match(part.strip())
                    if match:
                        yield relative, part.strip(), re.sub(r'\s+#.*$', '', match.group(1))


def test_every_documented_hypergan_command_parses():
    """Each `hypergan ...` line in a README or docs shell block parses with the real CLI parser (nothing runs)."""
    parser = _parser()
    commands = list(_documented_commands())
    # The docs hold about 140 commands; a much smaller count means extraction broke.
    assert len(commands) >= 100, len(commands)
    failures = []
    for relative, line, rest in commands:
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                parser.parse_args(shlex.split(rest))
        except SystemExit as exc:
            if exc.code:
                failures.append(f'{relative}: {line}\n    {stderr.getvalue().strip().splitlines()[-1:]}')
    assert not failures, 'documented commands that do not parse:\n' + '\n'.join(failures)


def _python_blocks():
    for relative in DOC_FILES:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8')
        language = None
        lines = []
        start_line = 0
        for idx, line in enumerate(text.splitlines(), start=1):
            fence = re.fullmatch(r'\s*```(\w*)\s*', line)
            if fence and language is None:
                language = fence.group(1)
                start_line = idx
                lines = []
            elif line.strip() == '```' and language is not None:
                if language in PYTHON_FENCES:
                    yield relative, start_line, idx, '\n'.join(lines)
                language = None
            elif language is not None:
                lines.append(line)


def test_every_documented_python_block_parses_ast():
    """Verify that every ```python code block across all documentation parses as valid Python AST."""
    blocks = list(_python_blocks())
    assert len(blocks) >= 40, f'Expected at least 40 Python blocks in documentation, found {len(blocks)}'

    failures = []
    for relative, start, end, code in blocks:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            failures.append(f'{relative}:{start}-{end}: SyntaxError: {exc}')

    assert not failures, 'Documented Python code blocks failed AST parse:\n' + '\n'.join(failures)


def test_no_hndl_blocks_erroneously_tagged_as_python():
    """Verify that HNDL DSL blocks are tagged with ```hndl and not ```python."""
    failures = []
    for relative, start, end, code in _python_blocks():
        if re.search(r'\b(pretrained\(\$\{|diff_augment\(x,\s*transforms=)', code):
            if 'import ' not in code and 'def ' not in code:
                failures.append(
                    f'{relative}:{start}-{end}: Contains raw HNDL DSL; must be fenced as ```hndl'
                )

    assert not failures, 'HNDL DSL snippets erroneously tagged as ```python:\n' + '\n'.join(failures)


def _ci_matrices():
    return re.findall(r'^\s*python: \[([^\]]*)\]', _read('.github/workflows/ci.yml'), re.M)


def test_python_classifiers_match_the_ci_lightweight_matrix_and_readme():
    matrices = [sorted(re.findall(r'"(3\.\d+)"', row)) for row in _ci_matrices()]
    assert len(matrices) == 2, 'expected the lightweight and heavy-lightweight matrices'
    assert matrices[0] == matrices[1]
    tested = matrices[0]
    project = _pyproject()['project']
    classified = sorted(c.rsplit(' :: ', 1)[1] for c in project['classifiers']
                        if re.fullmatch(r'Programming Language :: Python :: 3\.\d+', c))
    assert classified == tested
    low, high = re.fullmatch(r'>=(3\.\d+),<(3\.\d+)', project['requires-python']).groups()
    minor = lambda version: int(version.split('.')[1])
    assert minor(tested[0]) == minor(low) and minor(tested[-1]) == minor(high) - 1
    assert f'Python {tested[0]}–{tested[-1]}' in _read('README.md')


def test_pre_1_0_dependencies_have_an_upper_bound():
    project = _pyproject()['project']
    requirements = list(project['dependencies'])
    for extra in project['optional-dependencies'].values():
        requirements += extra
    unbounded = []
    for requirement in requirements:
        lower = re.search(r'>=\s*0\.', requirement)
        if lower and '<' not in requirement and '==' not in requirement:
            unbounded.append(requirement)
    assert not unbounded, f'pre-1.0 dependencies need an upper bound: {unbounded}'


def test_every_default_test_tier_ships_in_the_sdist_and_is_self_contained():
    manifest = _read('MANIFEST.in')
    for path in _pyproject()['tool']['pytest']['ini_options']['testpaths']:
        assert f'recursive-include {path} *.py' in manifest, path
    for helper in sorted((ROOT / 'tests').glob('*_fixtures.py')):
        assert f'include tests/{helper.name}' in manifest, helper.name
    # Only the e2e suite may import its harness; shared helpers live in tests/*_fixtures.py.
    offenders = []
    for source in sorted((ROOT / 'tests').rglob('*.py')):
        if 'e2e' in source.relative_to(ROOT / 'tests').parts[:1]:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            if any(re.match(r'(tests\.)?e2e(\.|$)', name) for name in names):
                offenders.append(str(source.relative_to(ROOT)))
    assert not offenders, offenders


def test_one_changelog_with_an_unreleased_section_citing_real_constants():
    assert not (ROOT / 'docs' / 'changelog.md').exists(), 'keep the root changelog.md only'
    changelog = _read('changelog.md')
    headings = re.findall(r'^## (.+)$', changelog, re.M)
    assert headings[0] == 'Unreleased', headings[:2]
    sources = '\n'.join(p.read_text(encoding='utf-8') for p in (ROOT / 'src' / 'hypergan').rglob('*.py'))
    for name, value in re.findall(r'`([A-Z][A-Z0-9_]+) = (\d+)`', changelog):
        assert re.search(rf'^{name} = {value}\b', sources, re.M), f'changelog cites {name} = {value}, which the code does not define'


def test_glossary_names_model_file_tables_and_keys_that_exist():
    glossary = _read('GLOSSARY.md')
    section = glossary.split('## 1. Model File', 1)[1].split('\n## ', 1)[0]
    rows = [line for line in section.splitlines() if line.startswith('| **')]
    assert rows
    tables, keys = set(), set()
    for row in rows:
        model_file_column = row.split(' | ')[1]
        for token in re.findall(r'`([^`]+)`', model_file_column):
            table = re.fullmatch(r'\[\[?([a-z_]+)(?:\.[^\]]*)?\]\]?', token)
            if table:
                tables.add(table.group(1))
            for prefix, key in re.findall(r'\b(train|networks\.<name>)\.([a-z_]+)', token):
                keys.add((prefix, key))
    assert {'data', 'prior', 'networks', 'losses', 'train'} <= tables
    assert tables <= model_file.TOP_LEVEL, tables - model_file.TOP_LEVEL
    train_fields = model_file.TRAINING_FIELDS | model_file.OPTIMIZER_FIELDS | model_file.RUN_CADENCE_FIELDS
    for prefix, key in keys:
        allowed = train_fields if prefix == 'train' else model_file.NETWORK_FIELDS
        assert key in allowed, f'GLOSSARY documents {prefix}.{key}, which model files do not accept'


def _markdown_links(relative):
    text = re.sub(r'```.*?```', '', _read(relative), flags=re.S)
    for target in re.findall(r'\]\(([^)\s]+)\)', text):
        if re.match(r'^[a-z]+:', target) or target.startswith('#'):
            continue
        yield target.split('#', 1)[0]


def test_docs_index_links_every_doc_and_repository_links_resolve():
    index = set(_markdown_links('docs/README.md'))
    missing = [p.name for p in sorted((ROOT / 'docs').glob('*.md')) if p.name != 'README.md' and p.name not in index]
    assert not missing, f'docs/README.md does not link: {missing}'
    broken = []
    for relative in [*DOC_FILES, 'GLOSSARY.md', 'SECURITY.md', 'changelog.md']:
        for target in _markdown_links(relative):
            if not ((ROOT / relative).parent / target).exists():
                broken.append(f'{relative} -> {target}')
    assert not broken, broken


def _dockerfile_runs():
    joined = re.sub(r'\\\n', ' ', _read('Dockerfile'))
    return [line.strip()[4:] for line in joined.splitlines() if line.strip().startswith('RUN ')]


@pytest.mark.skipif(os.name == 'nt' or shutil.which('sh') is None, reason='runs the Dockerfile step with a POSIX shell')
@pytest.mark.parametrize('commit', ['0123456789abcdef0123456789abcdef01234567', ''])
def test_docker_build_records_the_commit_build_argument(tmp_path, commit):
    """.dockerignore drops .git, so the builder stage stamps HYPERGAN_COMMIT for the wheel's provenance."""
    assert '.git' in _read('.dockerignore').split()
    assert re.search(r'^ARG HYPERGAN_COMMIT=', _read('Dockerfile'), re.M)
    (step,) = [run for run in _dockerfile_runs() if 'HYPERGAN_COMMIT' in run]
    package = tmp_path / 'build' / 'src' / 'hypergan'
    package.mkdir(parents=True)
    stub = tmp_path / 'bin' / 'python'
    stub.parent.mkdir()
    stub.write_text('#!/bin/sh\nexit 0\n')  # stands in for `pip wheel`; only the stamp matters here
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {**os.environ, 'PATH': f'{stub.parent}{os.pathsep}{os.environ.get("PATH", "")}', 'HYPERGAN_COMMIT': commit}
    subprocess.run(['sh', '-c', step], cwd=package.parent.parent, env=env, check=True, timeout=30)
    source = hypergan_source(package)
    if commit:
        assert source == {'hypergan_commit': commit, 'hypergan_dirty': None, 'hypergan_provenance': 'build'}
    else:
        assert source['hypergan_commit'] is None and source['hypergan_provenance'] == 'unknown'
