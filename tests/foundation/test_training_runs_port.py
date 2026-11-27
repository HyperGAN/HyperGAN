"""Automated qualification test for model-file loading, validation, and legacy port parity.

Verifies:
1. All shipped examples/*.toml files use hypergan-model/1, validate cleanly, and
   contain no hardcoded workstation paths.
2. Ported hypergan-model/1 experiment TOMLs (workspace + extends + ${paths.*})
   load, validate, lower, and achieve bit-for-bit numerical fingerprint and
   component graph invariance against legacy engine recipes.
"""

from copy import deepcopy
from pathlib import Path
import tomllib
import pytest

import hypergan.api as hg
from hypergan.config import DEFAULT, PARTICLEGAN_DEFAULT_FIELDS, fingerprint, load_config, resolve_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEMO_SCRIPT = REPO_ROOT / "scripts" / "start-cifar-tiny-transformer-resnet-features-pg07-lr2e-4-demo.sh"


def test_all_examples_use_hypergan_model_format_and_load_cleanly(published_pg08_defaults):
    example_tomls = sorted(EXAMPLES_DIR.glob("*.toml")) + sorted((EXAMPLES_DIR / "archive").glob("*.toml"))
    assert len(example_tomls) >= 23, f"Expected at least 23 examples/*.toml files, found {len(example_tomls)}"

    for path in example_tomls:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        assert raw.get("format") == "hypergan-model/1", f"{path.name} must declare format = 'hypergan-model/1'"
        assert "components" not in raw, f"{path.name} must not contain legacy [components]"
        assert "schema_version" not in raw, f"{path.name} must not contain legacy schema_version"
        assert "networks" in raw, f"{path.name} must declare [networks]"
        assert "losses" in raw, f"{path.name} must declare [[losses]]"

        model = hg.load(path)
        warnings = hg.validate(model)
        assert isinstance(warnings, list)

        lowered = hg.lower(model)
        resolved = load_config(path)
        assert fingerprint(resolve_config(lowered)) == fingerprint(resolved)
        assert hg.fingerprint(model) == fingerprint(resolved)


def test_examples_and_demo_script_contain_no_hardcoded_workstation_paths():
    forbidden = ("/mnt/ml7tb", "/home/martyn", "/nvme/dev")
    for path in sorted(EXAMPLES_DIR.glob("*.toml")) + sorted((EXAMPLES_DIR / "archive").glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains hardcoded workstation path {token!r}"

    if DEMO_SCRIPT.is_file():
        script_text = DEMO_SCRIPT.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in script_text, f"{DEMO_SCRIPT.name} contains hardcoded workstation path {token!r}"
        assert "REPO_ROOT=" in script_text
        assert "PYTHON_BIN=" in script_text


def clean_legacy_recipe(raw):
    clean = deepcopy(raw)
    if "gradient_penalty" in clean:
        for k in ("arm", "norm", "target_anneal", "total_steps", "method", "fd_eps"):
            clean["gradient_penalty"].pop(k, None)
    if "adversarial" in clean:
        for k in ("loss_type", "mode"):
            clean["adversarial"].pop(k, None)
    if "sampling" in clean:
        clean["sampling"].pop("output_noise", None)
    return clean


@pytest.fixture
def synthetic_runs_workspace(tmp_path):
    """Self-contained workspace mirroring a multi-file legacy-to-model-file port."""
    ws_dir = tmp_path / "training-runs"
    backup_dir = ws_dir / ".legacy_toml_backup"
    nets_dir = ws_dir / "nets"
    ws_dir.mkdir()
    backup_dir.mkdir()
    nets_dir.mkdir()

    gen_hndl = nets_dir / "gen.hndl"
    disc_hndl = nets_dir / "disc.hndl"
    gen_hndl.write_text("linear(2)\n", encoding="utf-8")
    disc_hndl.write_text("linear(1)\n", encoding="utf-8")

    (ws_dir / "hypergan.toml").write_text(
        '[workspace]\nruns_root = "${workspace.root}/runs"\n\n'
        '[paths]\nnets = "${workspace.root}/nets"\n',
        encoding="utf-8",
    )

    (ws_dir / "base-model.toml").write_text(
        'format = "hypergan-model/1"\n'
        'defaults = "particlegan"\n\n'
        '[data]\nbuiltin = "gaussian_grid"\n\n'
        '[prior]\nkind = "particles"\nz_dim = 4\nnum_particles = 20000\n\n'
        '[networks.generator]\nrole = "generator"\nhndl = "${paths.nets}/gen.hndl"\n'
        'input_shape = ["B", 4]\noutput_shape = ["B", 2]\ninputs = { x = "latent" }\n\n'
        '[networks.discriminator]\nrole = "critic"\nhndl = "${paths.nets}/disc.hndl"\n'
        'input_shape = ["B", 2]\noutput_shape = ["B", 1]\ninputs = { x = "candidate" }\n\n'
        '[[losses]]\ntype = "adversarial"\ncritic = "discriminator"\npenalty = 1.0\n\n'
        '[[losses]]\ntype = "prior"\nweight = 0.0\n',
        encoding="utf-8",
    )

    (ws_dir / "exp_a.toml").write_text(
        'format = "hypergan-model/1"\n'
        'name = "exp-a"\n'
        'extends = "base-model.toml"\n\n'
        '[train]\nsteps = 50\nbatch_size = 32\nseed = 7\ndevice = "cpu"\n',
        encoding="utf-8",
    )

    (backup_dir / "exp_a.toml").write_text(
        'schema_version = 1\n'
        'name = "exp-a"\n'
        'defaults = "particlegan"\n\n'
        '[data]\nfactory = "gaussian_grid"\nargs = {}\n\n'
        '[prior]\nkind = "particles"\n[prior.args]\nz_dim = 4\nnum_particles = 20000\n\n'
        '[components.generator]\nfactory = "hndl"\ninputs = { x = "latent" }\n'
        f'[components.generator.args]\nfile = "{gen_hndl.as_posix()}"\ninput_shape = ["B", 4]\noutput_shape = ["B", 2]\n\n'
        '[components.discriminator]\nfactory = "hndl"\ninputs = { x = "candidate" }\n'
        f'[components.discriminator.args]\nfile = "{disc_hndl.as_posix()}"\ninput_shape = ["B", 2]\noutput_shape = ["B", 1]\n\n'
        '[adversarial]\nweight = 1.0\nmode = "rp"\nloss_type = "logistic"\n\n'
        '[gradient_penalty]\ncoeff = 1.0\narm = "legacy"\n\n'
        '[prior_regularizer]\nweight = 0.0\n\n'
        '[training]\nsteps = 50\nbatch_size = 32\nseed = 7\ndevice = "cpu"\n',
        encoding="utf-8",
    )

    return ws_dir, backup_dir


def test_training_runs_structure(synthetic_runs_workspace):
    ws_dir, backup_dir = synthetic_runs_workspace
    assert (ws_dir / "hypergan.toml").is_file()
    assert (ws_dir / "base-model.toml").is_file()
    exp_tomls = [ws_dir / path.name for path in sorted(backup_dir.glob("*.toml"))]
    assert len(exp_tomls) >= 1
    assert all(path.is_file() for path in exp_tomls)


@pytest.fixture
def published_pg08_defaults(monkeypatch):
    """Freeze the port fixture to a fixed historical ParticleGAN default snapshot."""
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


def test_ported_experiments_load_validate_and_preserve_fingerprint(synthetic_runs_workspace, published_pg08_defaults):
    ws_dir, backup_dir = synthetic_runs_workspace
    exp_tomls = [ws_dir / path.name for path in sorted(backup_dir.glob("*.toml"))]

    mismatches = []
    for ported_path in exp_tomls:
        backup_path = backup_dir / ported_path.name
        assert backup_path.is_file(), f"Missing backup original for {ported_path.name}"

        raw_orig = clean_legacy_recipe(hg.load_recipe(backup_path))
        orig_res = resolve_config(raw_orig)
        fp_orig = fingerprint(orig_res)

        model = hg.load(ported_path)
        warnings = hg.validate(model)
        assert isinstance(warnings, list)

        lowered = hg.lower(model)
        lowered_res = resolve_config(lowered)
        fp_ported = fingerprint(lowered_res)

        if fp_orig != fp_ported:
            mismatches.append((ported_path.name, fp_orig, fp_ported))

        orig_comps = orig_res["components"]
        ported_comps = lowered_res["components"]
        assert set(orig_comps.keys()) == set(ported_comps.keys()), f"Component set mismatch in {ported_path.name}"
        for cname in orig_comps:
            assert orig_comps[cname]["inputs"] == ported_comps[cname]["inputs"]
            assert orig_comps[cname]["args"]["input_shape"] == ported_comps[cname]["args"]["input_shape"]
            assert orig_comps[cname]["args"]["output_shape"] == ported_comps[cname]["args"]["output_shape"]

    assert len(mismatches) == 0, f"Fingerprint mismatches on {len(mismatches)} files: {mismatches}"
