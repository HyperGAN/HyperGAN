"""Signal command parsing stays torch-free and validates report destinations."""
from types import SimpleNamespace

import pytest

from hypergan.cli import _parser
from hypergan.signal_diagnostic_cli import run_signal


def test_signal_options_and_defaults():
    args = _parser().parse_args(['diagnose-signal', 'config.toml', '--output', 'signal.json'])
    assert args.objective == 'adversarial'
    assert args.batch_size is None and args.device is None


def test_signal_wont_overwrite_before_constructing_models(tmp_path):
    output = tmp_path / 'config.toml'
    output.write_text('original config')
    with pytest.raises(FileExistsError):
        run_signal(SimpleNamespace(output=output))
    assert output.read_text() == 'original config'


def test_signal_batch_must_be_positive():
    with pytest.raises(SystemExit):
        _parser().parse_args(['diagnose-signal', 'x', '--output', 'y', '--batch-size', '0'])
