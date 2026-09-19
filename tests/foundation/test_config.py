"""Configuration checks run without importing the numerical runtime."""
from copy import deepcopy
import sys
import subprocess

import pytest

from hypergan.config import DEFAULT, config_values, load_config, resolve_config, write_default


def test_default_roundtrip_without_runtime(tmp_path):
    path = write_default(tmp_path / "project")
    config = load_config(path.parent)
    assert config_values(config) == DEFAULT
    assert config["qualification"]["recipe_match"]
    subprocess.run([sys.executable, "-c", "from hypergan.config import load_config; import sys; load_config(sys.argv[1]); assert 'torch' not in sys.modules", str(path)], cwd=tmp_path, check=True)
    with pytest.raises(FileExistsError):
        write_default(path)


def test_custom_constructor_validates_without_importing():
    raw = deepcopy(DEFAULT)
    raw["components"]["generator"]["factory"] = "not_installed.example:CustomGenerator"
    config = resolve_config(raw)
    assert not config["qualification"]["recipe_match"]
    assert any("trusted Python" in warning for warning in config["warnings"])
    assert "not_installed.example" not in sys.modules


@pytest.mark.parametrize("changes", [{"typo": 1}, {"training": {"stepps": 3}}, {"gradient_penalty": {"lazy_k": 0}}, {"training": {"device": "cuda"}}, {"sampling": {"count": 0}}])
def test_factual_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        resolve_config(changes)


def test_disconnected_components_and_cycles_are_errors():
    raw = deepcopy(DEFAULT)
    raw["components"]["encoder"] = {"factory": "linear", "args": {"in_features": 2, "out_features": 2}, "inputs": {"input": "batch.real"}}
    with pytest.raises(ValueError, match="Disconnected"):
        resolve_config(raw)
    raw["components"]["generator"]["inputs"]["x"] = "components.encoder"
    raw["components"]["encoder"]["inputs"]["input"] = "components.generator"
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_config(raw)


def test_gaussian_requires_explicit_no_table_regularization():
    raw = {"prior": {"kind": "gaussian", "args": {"z_dim": 4}}}
    with pytest.raises(ValueError, match="no learned table"):
        resolve_config(raw)
    raw["prior_regularizer"] = {"weight": 0.0}
    cfg = resolve_config(raw)
    assert any("not applicable" in warning for warning in cfg["warnings"])
