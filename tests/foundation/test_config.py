"""Configuration checks run without importing the numerical runtime."""
from copy import deepcopy
import sys
import subprocess

import pytest

from hypergan.config import DEFAULT, config_values, load_config, resolve_config, write_default


def test_default_roundtrip_without_runtime(tmp_path):
    path = write_default(tmp_path / "project", device="cpu")
    config = load_config(path.parent)
    assert config_values(config) == DEFAULT
    assert config["qualification"]["recipe_match"]
    subprocess.run([sys.executable, "-c", "from hypergan.config import load_config; import sys; load_config(sys.argv[1]); assert 'torch' not in sys.modules", str(path)], cwd=tmp_path, check=True)
    with pytest.raises(FileExistsError):
        write_default(path, device="cpu")


def test_custom_constructor_validates_without_importing():
    raw = deepcopy(DEFAULT)
    raw["components"]["generator"]["factory"] = "not_installed.example:CustomGenerator"
    config = resolve_config(raw)
    assert not config["qualification"]["recipe_match"]
    assert any("trusted Python" in warning for warning in config["warnings"])
    assert "not_installed.example" not in sys.modules


@pytest.mark.parametrize("changes", [{"typo": 1}, {"training": {"stepps": 3}}, {"gradient_penalty": {"lazy_k": 0}}, {"training": {"device": "cuda:-1"}}, {"sampling": {"count": 0}}])
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


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:12"])
def test_explicit_device_is_structural_and_prior_cannot_override_it(tmp_path, device):
    path = write_default(tmp_path / "project", device=device)
    config = load_config(path)
    assert config["training"]["device"] == device
    values = config_values(config)
    values["prior"]["args"]["device"] = device
    assert resolve_config(values)["prior"]["args"]["device"] == device
    values["prior"]["args"]["device"] = "cuda:99" if device == "cpu" else "cpu"
    with pytest.raises(ValueError, match="Training owns prior device"):
        resolve_config(values)


@pytest.mark.parametrize("device", [None, True, 1, "", "mps", "cuda:-1", "cuda:01", "cpu:0", "cuda:1.0"])
def test_invalid_device_does_not_create_project(tmp_path, device):
    path = tmp_path / "project"
    with pytest.raises(ValueError, match="training.device"):
        write_default(path, device=device)
    assert not path.exists()
