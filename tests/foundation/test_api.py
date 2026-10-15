"""Tests for hypergan.api: public functional interface, builders, explain, and guards."""
import inspect
from pathlib import Path
import pytest

import hypergan.api as hg
from hypergan.config import DEFAULT, resolve_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples' / 'api' / 'config_first'


def dummy_fn(x, hidden=32):
    return x * hidden


def test_import_path_and_callable_rejection():
    # Valid module:qualname
    path = hg.import_path(dummy_fn, "Test")
    assert path == "test_api:dummy_fn" or "dummy_fn" in path

    # Invalid string
    with pytest.raises(ValueError, match="must be an importable object"):
        hg.import_path("invalid_string_no_colon", "Test")

    # Local function rejection
    def local_fn():
        pass

    with pytest.raises(ValueError, match="local to a function; define it at module level"):
        hg.import_path(local_fn, "Test")

    # Lambda rejection
    lambda_fn = lambda x: x
    with pytest.raises(ValueError, match="local to a function; define it at module level"):
        hg.import_path(lambda_fn, "Test")


def test_builder_signature_verification():
    # Calling dummy_fn with correct args
    net = hg.module(dummy_fn, inputs={'x': 'latent'}, role='generator', hidden=64)
    assert net['args']['hidden'] == 64

    # Calling with misspelled argument
    with pytest.raises(TypeError, match="unexpected keyword argument 'hiden'"):
        hg.module(dummy_fn, inputs={'x': 'latent'}, role='generator', hiden=64)


def test_save_body_hash_guard(tmp_path):
    model = hg.packaged('gaussian-grid')
    dest = tmp_path / "model.toml"
    saved = hg.save(model, dest)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "# body-sha256:" in content

    # Saving again without edits should succeed even without overwrite=True
    hg.save(model, dest, overwrite=False)

    # Tampering with the file
    tampered = content + "\n# added comment\n"
    dest.write_text(tampered, encoding="utf-8")

    # Saving over an edited file without overwrite=True must raise FileExistsError
    with pytest.raises(FileExistsError, match="edited after hg.save wrote it"):
        hg.save(model, dest, overwrite=False)

    # With overwrite=True it succeeds
    hg.save(model, dest, overwrite=True)


def test_explain_diagnostics():
    model = hg.load(EXAMPLES / "two_critics.toml")
    text = hg.explain(model)
    assert "decoder [generator]" in text
    assert "pair_critic [critic]" in text
    assert "marginal_critic [critic]" in text
    assert "critic Adam (shared by all critics)" in text
    assert "generator-side Adam" in text
    assert "inputs:" in text
    assert "gradient from:" in text


def test_lift_converts_legacy_recipe():
    legacy = resolve_config({})
    model = hg.lift(legacy)
    assert isinstance(model, hg.Model)
    assert model.spec['format'] == 'hypergan-model/1'
    assert 'generator' in model.spec['networks']
    assert 'discriminator' in model.spec['networks']
    # Lowering the lifted model gives matching numerical values
    lowered = hg.lower(model)
    assert lowered['components']['generator']['factory'] == legacy['components']['generator']['factory']


def test_override_dotted_path():
    model = hg.packaged('gaussian-grid')
    modified = hg.override(model, {'train.steps': 500, 'losses.0.weight': 2.0})
    assert modified.spec['train']['steps'] == 500
    assert modified.spec['losses'][0]['weight'] == 2.0


def test_packaged_list():
    models = hg.packaged_list()
    assert isinstance(models, list)
    names = {m['name'] for m in models}
    assert 'gaussian-grid' in names


def test_model_constructor():
    m = hg.model(
        name='test/custom-gan',
        data={'factory': 'gaussian_grid', 'args': {'dim': 2}},
        prior={'kind': 'particles', 'args': {'z_dim': 16}},
        networks={
            'g': hg.module(dummy_fn, inputs={'x': 'latent'}, role='generator'),
            'd': hg.module(dummy_fn, inputs={'x': 'candidate'}, role='critic'),
        },
        losses=[
            {'type': 'adversarial', 'generator': 'g', 'critic': 'd'},
        ],
        train={'device': 'cpu', 'steps': 100},
    )
    assert isinstance(m, hg.Model)
    assert m.spec['format'] == 'hypergan-model/1'
    assert m.spec['name'] == 'test/custom-gan'
    assert 'g' in m.spec['networks']
    assert 'd' in m.spec['networks']
    assert m.spec['train']['steps'] == 100


def test_lift_role_assignment_and_reuse():
    cifar_path = ROOT / 'examples' / 'cifar-pretrained-sagan.toml'
    if cifar_path.is_file():
        recipe = hg.load_recipe(cifar_path)
        lifted = hg.lift(recipe, base=cifar_path.parent)
        assert lifted.spec['format'] == 'hypergan-model/1'
        valid_roles = {'generator', 'critic', 'encoder', 'auxiliary'}
        for net_name, net in lifted.spec.get('networks', {}).items():
            role = net.get('role')
            assert role in valid_roles, f"Component {net_name} has invalid role {role!r}"
            assert role not in ('shared', 'frozen')
        explained = hg.explain(lifted)
        assert isinstance(explained, str)


def test_lift_snapshot_metrics_partitioning():
    legacy = {
        'schema_version': 1,
        'components': {
            'generator': {'factory': 'hndl', 'args': {'source': 'linear(2)\n'}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'hndl', 'args': {'source': 'linear(1)\n'}, 'inputs': {'x': 'candidate'}},
        },
        'metrics': {
            'custom': {
                'my_scalar': {'weight': 1.0},
                'fid_snapshot': {'trigger': 'every_steps', 'evaluation': 'fid', 'interval': 100},
            }
        }
    }
    lifted = hg.lift(legacy)
    assert 'fid_snapshot' in lifted.spec.get('evaluations', {})
    assert 'my_scalar' in lifted.spec.get('metrics', {}).get('custom', {})
    assert 'fid_snapshot' not in lifted.spec.get('metrics', {}).get('custom', {})


def test_lift_preserves_hyperparameters_and_sampling():
    dcgan_path = ROOT / 'examples' / 'dcgan-128.toml'
    recipe = hg.load_recipe(dcgan_path)
    lifted = hg.lift(recipe, base=dcgan_path.parent)
    assert 'train' in lifted.spec
    assert lifted.spec['train']['lr'] == recipe['optimizer']['lr']
    assert 'sampling' in lifted.spec
    assert lifted.spec['sampling']['count'] == recipe['sampling']['count']
    from hypergan.config import fingerprint, numerical_values, resolve_config
    import json
    orig_res = resolve_config(recipe)
    lowered_res = hg.resolve(lifted)
    assert fingerprint(orig_res) == fingerprint(lowered_res)
    assert json.dumps(numerical_values(lowered_res), sort_keys=True) == json.dumps(numerical_values(orig_res), sort_keys=True)

    # Test recipe where sampling has generated mapped to output
    ae_path = ROOT / 'examples' / 'logos-ae-gan-256.toml'
    ae_recipe = hg.load_recipe(ae_path)
    ae_lifted = hg.lift(ae_recipe, base=ae_path.parent)
    assert 'output' in ae_lifted.spec['sampling']
    assert 'generated' not in ae_lifted.spec['sampling']


def test_lift_all_examples():
    example_files = list((ROOT / 'examples').glob('*.toml'))
    assert len(example_files) >= 20
    for path in example_files:
        recipe = hg.load_recipe(path)
        lifted = hg.lift(recipe, base=path.parent)
        assert lifted.spec['format'] == 'hypergan-model/1'
        warnings = hg.validate(lifted)
        assert isinstance(warnings, list)

