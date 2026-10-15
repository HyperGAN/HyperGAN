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
