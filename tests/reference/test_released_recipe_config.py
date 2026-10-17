"""Installed ParticleGAN preset configuration resolves without importing Torch."""
from pathlib import Path
import subprocess
import sys

import pytest

from hypergan.config import (
    DEFAULT, PARTICLEGAN_DEFAULT_FIELDS, config_values, load_config, resolve_config, write_default)


def test_default_roundtrip_without_runtime(tmp_path):
    path = write_default(tmp_path / "project", device="cpu")
    config = load_config(path.parent)
    assert config['recipe']['name'] == 'bispf_optimized'
    assert config['training']['initialization'] == 'deterministic_orthogonal'
    subprocess.run([sys.executable, "-c", "from hypergan.config import load_config; import sys; load_config(sys.argv[1]); assert 'torch' not in sys.modules", str(path)], cwd=tmp_path, check=True)
    with pytest.raises(FileExistsError):
        write_default(path, device="cpu")
    legacy = write_default(tmp_path / 'legacy', device='cpu', defaults='hypergan')
    assert config_values(load_config(legacy)) == DEFAULT


def test_new_project_defaults_to_cuda_without_loading_runtime(tmp_path):
    path = write_default(tmp_path / "gpu-project")
    assert load_config(path)["training"]["device"] == "cuda"
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; from hypergan.config import load_config; "
         "assert load_config(sys.argv[1])['training']['device'] == 'cuda'; "
         "assert 'torch' not in sys.modules", str(path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_particlegan_defaults_fill_only_omitted_fields_without_torch():
    check = """
import sys
from hypergan.config import PARTICLEGAN_DEFAULT_FIELDS, resolve_config
config = resolve_config({'defaults': 'particlegan', 'optimizer': {'lr': 0.001}})
assert 'torch' not in sys.modules and 'particlegan' not in sys.modules
from particlegan import get_recipe
recipe = get_recipe()
for (section, key), field in PARTICLEGAN_DEFAULT_FIELDS.items():
    expected = getattr(recipe, field)
    if expected is None and field == 'prior_betas':
        expected = recipe.betas
    expected = list(expected) if isinstance(expected, tuple) else expected
    assert config[section][key] == (0.001 if field == 'lr' else expected), (section, key)
assert 'defaults' not in config
"""
    subprocess.run([sys.executable, "-c", check], check=True)
    assert resolve_config({"defaults": "hypergan"}) == resolve_config({})
    with pytest.raises(ValueError, match="defaults must be"):
        resolve_config({"defaults": "other"})


def test_all_examples_load_with_load_config():
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    for example in sorted(examples_dir.glob("*.toml")):
        resolved = load_config(example)
        assert isinstance(resolved, dict)
        assert "components" in resolved
