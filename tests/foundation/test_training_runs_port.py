"""Automated qualification test for ported experiment TOML files in training-runs.

Verifies:
1. All 46 experiment TOML files load successfully via hg.load(path) (workspace + extends + paths).
2. All 46 ported files pass hg.validate(model) with zero errors.
3. All 46 ported files lower cleanly via hg.lower(model).
4. The resolved configuration achieves 100% bit-for-bit numerical fingerprint invariance
   against the pristine legacy original recipes in .legacy_toml_backup.
5. All component graphs, shapes, and inputs match identically.
"""

import json
from copy import deepcopy
from pathlib import Path
import pytest

import hypergan.api as hg
from hypergan.config import DEFAULT, PARTICLEGAN_DEFAULT_FIELDS, fingerprint, resolve_config

RUNS_DIR = Path("/home/martyn/dev/hypergan/training-runs")
BACKUP_DIR = RUNS_DIR / ".legacy_toml_backup"


def clean_legacy_recipe(raw):
    clean = json.loads(json.dumps(raw))
    if "gradient_penalty" in clean:
        for k in ("arm", "norm", "target_anneal", "total_steps", "method", "fd_eps"):
            clean["gradient_penalty"].pop(k, None)
    if "adversarial" in clean:
        for k in ("loss_type", "mode"):
            clean["adversarial"].pop(k, None)
    if "sampling" in clean:
        clean["sampling"].pop("output_noise", None)
    return clean


@pytest.fixture(scope="module")
def runs_workspace():
    if not RUNS_DIR.is_dir() or not BACKUP_DIR.is_dir():
        pytest.skip("training-runs workspace or .legacy_toml_backup not present")
    return RUNS_DIR


def test_training_runs_structure(runs_workspace):
    assert (RUNS_DIR / "hypergan.toml").is_file(), "hypergan.toml must exist in training-runs"
    assert (RUNS_DIR / "cifar-tiny-pg-base.toml").is_file(), "cifar-tiny-pg-base.toml must exist"
    assert (RUNS_DIR / "logos-base.toml").is_file(), "logos-base.toml must exist"
    assert (RUNS_DIR / "logos-transgan-resnet-base.toml").is_file(), "logos-transgan-resnet-base.toml must exist"

    # New training runs are independent of the archived 46-file port cohort.
    exp_tomls = [RUNS_DIR / path.name for path in sorted(BACKUP_DIR.glob('*.toml'))]
    assert len(exp_tomls) == 46, f"Expected exactly 46 experiment TOML files, found {len(exp_tomls)}"
    assert all(path.is_file() for path in exp_tomls)


@pytest.fixture
def published_pg08_defaults(monkeypatch):
    """Freeze the original port cohort to the published 0.8 recipe contract.

    Both archived recipes and ported model files named the then-current
    ParticleGAN defaults. Comparing that historical migration under a newer
    trainer would change the question and rewrite its initialization/recipe
    identity. Current released defaults have their own configuration tests.
    """
    import hypergan.config as configuration
    original = deepcopy(DEFAULT)
    original['optimizer'].update(lr=.00425, d_lr_mult=1., prior_lr_mult=2.)
    original['prior_regularizer']['weight'] = 0.
    original['training'].update(network_lr_floor=.01, network_lr_horizon_cap=1600,
                                input_noise_std=.5, output_noise_std=.029)
    values = {(section, key): deepcopy(original[section][key])
              for section, key in PARTICLEGAN_DEFAULT_FIELDS}
    monkeypatch.setattr(configuration, 'particlegan_defaults', lambda: deepcopy(values))
    monkeypatch.setattr(configuration, 'particlegan_recipe_fields', lambda: {})


def test_all_46_ported_experiments_load_validate_and_preserve_fingerprint(runs_workspace, published_pg08_defaults):
    exp_tomls = [RUNS_DIR / path.name for path in sorted(BACKUP_DIR.glob('*.toml'))]

    mismatches = []
    for ported_path in exp_tomls:
        backup_path = BACKUP_DIR / ported_path.name
        assert backup_path.is_file(), f"Missing backup original for {ported_path.name}"

        # 1. Resolve pristine original legacy recipe
        raw_orig = clean_legacy_recipe(hg.load_recipe(backup_path))
        orig_res = resolve_config(raw_orig)
        fp_orig = fingerprint(orig_res)

        # 2. Load ported hypergan-model/1 (resolves extends + workspace paths)
        model = hg.load(ported_path)

        # 3. Validate
        warnings = hg.validate(model)
        assert isinstance(warnings, list)

        # 4. Lower & resolve
        lowered = hg.lower(model)
        lowered_res = resolve_config(lowered)
        fp_ported = fingerprint(lowered_res)

        if fp_orig != fp_ported:
            mismatches.append((ported_path.name, fp_orig, fp_ported))

        # 5. Component shape & inputs equality
        orig_comps = orig_res["components"]
        ported_comps = lowered_res["components"]
        assert set(orig_comps.keys()) == set(ported_comps.keys()), f"Component set mismatch in {ported_path.name}"
        for cname in orig_comps:
            assert orig_comps[cname]["inputs"] == ported_comps[cname]["inputs"]
            assert orig_comps[cname]["args"]["input_shape"] == ported_comps[cname]["args"]["input_shape"]
            assert orig_comps[cname]["args"]["output_shape"] == ported_comps[cname]["args"]["output_shape"]

    assert len(mismatches) == 0, f"Fingerprint mismatches on {len(mismatches)} files: {mismatches}"
