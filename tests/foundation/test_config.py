"""Configuration checks run without importing the numerical runtime."""
from copy import deepcopy
import sys
import subprocess

import pytest

from hypergan.config import (
    DEFAULT, config_values, fingerprint, load_config, numerical_values, resolve_config, resume_compatible, write_default)

LEGACY_FINGERPRINT = "11dd870ea803a25a38cba344d0131c74f1d5e0575b9560f47bf7a5863a6ecda5"


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


@pytest.mark.parametrize('sampling', [
    {'generated': 'components.discriminator'}, {'generated': 'batch.real'},
    {'views': {'random': 'components.missing'}}, {'views': {'g': 'generated'}},
    {'comparison': [{'label': 'X', 'binding': 'batch.real'}]},
    {'comparison': [{'label': 'X\n', 'binding': 'batch.real'}, {'label': 'G', 'binding': 'generated'}]},
    {'particle_ids': 2},
])
def test_invalid_sampling_output_views_and_columns_are_rejected(sampling):
    with pytest.raises(ValueError):
        resolve_config({'sampling': sampling})


def _extra_term(**overrides):
    term = {"id": "extra", "component": "discriminator", "real": "batch.real", "fake": "generated"}
    term.update(overrides)
    return term


def test_legacy_fingerprint_ignores_a_missing_adversarial_terms_key():
    resolved = resolve_config({})
    assert "adversarial_terms" not in resolved
    assert resolved["qualification"]["recipe_match"]
    assert "adversarial_terms" not in numerical_values(resolved)
    assert "adversarial_terms" not in config_values(resolved)
    assert fingerprint(resolved) == LEGACY_FINGERPRINT
    assert fingerprint(resolved) == fingerprint(dict(resolved))
    assert fingerprint(resolved) == fingerprint(config_values(resolved))
    assert fingerprint(resolved) == fingerprint(numerical_values(resolved))
    checkpoint = dict(numerical_values(resolved))
    assert "adversarial_terms" not in checkpoint
    assert numerical_values(checkpoint)["schema_version"] == 1
    assert fingerprint(checkpoint) == fingerprint(resolved)
    assert resume_compatible(resolved, config_values(resolved))
    extra = resolve_config({"adversarial_terms": [_extra_term()]})
    assert extra["adversarial_terms"][0]["loss_type"] == resolved["adversarial"]["loss_type"]
    assert extra["adversarial_terms"][0]["mode"] == resolved["adversarial"]["mode"]
    assert extra["adversarial_terms"][0]["weight"] == 1.0
    assert extra["adversarial_terms"][0]["penalty"] is False
    assert extra["adversarial_terms"][0]["penalty_coeff"] == resolved["gradient_penalty"]["coeff"]
    assert extra["adversarial_terms"][0]["inputs"] == resolved["components"]["discriminator"]["inputs"]
    assert not extra["qualification"]["recipe_match"]
    assert fingerprint(extra) != fingerprint(resolved)
    assert numerical_values(extra)["adversarial_terms"] == extra["adversarial_terms"]
    assert config_values(extra)["adversarial_terms"] == extra["adversarial_terms"]


def test_adversarial_term_rules_and_reachability():
    with pytest.raises(ValueError, match="non-empty list"):
        resolve_config({"adversarial_terms": []})
    with pytest.raises(ValueError, match="Unknown adversarial_terms"):
        resolve_config({"adversarial_terms": [_extra_term(bonus=1)]})
    with pytest.raises(ValueError, match="unique"):
        resolve_config({"adversarial_terms": [_extra_term(), _extra_term(id="extra")]})
    with pytest.raises(ValueError, match="exactly one input"):
        resolve_config({"adversarial_terms": [_extra_term(inputs={"x": "batch.real", "y": "generated"})]})
    raw = deepcopy(DEFAULT)
    raw["components"]["alias"] = {"reuse": "generator", "inputs": {"x": "latent"}}
    raw["adversarial_terms"] = [_extra_term(component="alias")]
    with pytest.raises(ValueError, match="non-reuse"):
        resolve_config(raw)
    raw = deepcopy(DEFAULT)
    raw["components"]["encoder"] = {"factory": "linear", "args": {"in_features": 2, "out_features": 2}, "inputs": {"input": "batch.real"}}
    raw["adversarial_terms"] = [_extra_term(fake="components.encoder")]
    assert resolve_config(raw)["adversarial_terms"][0]["fake"] == "components.encoder"
    raw["adversarial_terms"] = [_extra_term(inputs={"x": "candidate", "condition": "components.encoder"})]
    with pytest.raises(ValueError, match="only discriminator conditioning, which is detached"):
        resolve_config(raw)
    raw["components"]["encoder"]["trainable"] = False
    assert resolve_config(raw)["components"]["encoder"]["trainable"] is False
    raw = deepcopy(DEFAULT)
    raw["adversarial_terms"] = [_extra_term(weight=0, penalty=True, penalty_coeff=0.25, loss_type="hinge", mode="vanilla")]
    term = resolve_config(raw)["adversarial_terms"][0]
    assert term["weight"] == 0 and term["penalty"] is True and term["penalty_coeff"] == 0.25
    assert term["loss_type"] == "hinge" and term["mode"] == "vanilla"


def test_sampling_view_does_not_make_a_dormant_trainable_component_reachable():
    import copy
    raw = copy.deepcopy(DEFAULT)
    raw['components']['unused'] = copy.deepcopy(raw['components']['generator'])
    raw['sampling']['views'] = {'random': 'components.unused'}
    with pytest.raises(ValueError, match='Disconnected'):
        resolve_config(raw)
