"""FID from fake Inception features: real moment merge, real torch-fidelity distance, cache.

No weights are read or downloaded. A fixed random projection stands in for the
Inception extractor, so every assertion compares against a NumPy/SciPy oracle
computed from the exact features the metric saw.
"""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from hypergan.artifacts import bundle_state
from hypergan.config import config_values, resolve_config
from hypergan.data import EvaluationDataExhausted
from hypergan.image_metrics import (INCEPTION_SHA256, TORCH_FIDELITY_VERSION, InceptionFID,
                                    read_reference_statistics, write_reference_statistics)
from hypergan.metric_evaluation_worker import MAX_DATA_IDENTITY_BYTES, evaluate_snapshot
from hypergan.metric_plugins import _factory
from hypergan.training import ReferenceTrainer

FEATURES = 5


class FakeExtractor:
    """Deterministic uint8 image -> feature projection that records what it saw."""
    def __init__(self, log):
        self.log = log

    def to(self, device):
        return self

    def __call__(self, images):
        assert images.dtype == torch.uint8
        flat = images.reshape(len(images), -1).double() / 255
        projection = torch.from_numpy(np.random.default_rng(7).normal(size=(flat.shape[1], FEATURES)))
        features = (flat @ projection + flat.square() @ projection.abs()).float()
        self.log.append((images.clone(), features.numpy().copy()))
        return (features,)


class FakeFID(InceptionFID):
    """InceptionFID with the extractor swapped; everything else is the real metric."""
    log = []

    def __init__(self, reference_cache=None):
        self.weights_path, self.weights_sha256 = None, INCEPTION_SHA256
        self.torch_fidelity_version = TORCH_FIDELITY_VERSION
        self.reference_cache = reference_cache

    def _extractor(self):
        return FakeExtractor(FakeFID.log)


class Half(nn.Module):
    """A generated view that differs measurably from its reference images."""
    def forward(self, x):
        return x * 0.5


def oracle(generated, reference):
    """Textbook FID with SciPy's matrix square root over float64 features."""
    linalg = pytest.importorskip('scipy.linalg', reason="the oracle needs scipy from the 'fid' extra")
    mu1, mu2 = generated.mean(0), reference.mean(0)
    sigma1, sigma2 = np.cov(generated, rowvar=False), np.cov(reference, rowvar=False)
    root = linalg.sqrtm(sigma1 @ sigma2).real
    return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * root))


def needs_torch_fidelity():
    pytest.importorskip('torch_fidelity.metric_fid', reason="FID distance needs the 'fid' extra")


def images(rng, count):
    return torch.from_numpy(rng.uniform(-1, 1, size=(count, 3, 4, 4))).float()


def protocol(**changes):
    return {'data_identity_sha256': 'a' * 64, 'sample_count': 13, 'batch_size': 5, 'seed': 3,
            'device': 'cpu', 'backend': {'cudnn_allow_tf32': False}, **changes}


def run(metric, generated, reference, sizes, reference_identity):
    FakeFID.log.clear()
    batches, start = [], 0
    for size in sizes:
        batches.append({'generated': generated[start:start + size], 'reference': reference[start:start + size]})
        start += size
    return metric.evaluate(batches=iter(batches), context={
        'sample_count': start, 'reference_identity': reference_identity})


def logged_features(log, names):
    # The metric extracts generated then reference for each batch.
    result = {name: [] for name in set(names)}
    for name, (_, features) in zip(names, log):
        result[name].append(features)
    return {name: np.concatenate(values).astype(np.float64) for name, values in result.items()}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_streamed_fid_matches_scipy_oracle_for_uneven_batches(tmp_path):
    needs_torch_fidelity()
    rng = np.random.default_rng(11)
    generated, reference = images(rng, 13) * 0.7 + 0.1, images(rng, 13)
    value = run(FakeFID(reference_cache=False), generated, reference, (5, 5, 3), protocol())
    features = logged_features(FakeFID.log, ['generated', 'reference'] * 3)
    assert len(features['generated']) == len(features['reference']) == 13
    expected = oracle(features['generated'], features['reference'])
    assert expected > 1
    assert value == pytest.approx(expected, rel=1e-9)
    # Identical feature sets give zero distance up to the matrix square root.
    assert run(FakeFID(reference_cache=False), reference, reference, (13,), protocol()) == pytest.approx(0, abs=1e-6)


def test_reference_statistics_are_cached_by_protocol_and_reused_exactly(tmp_path):
    needs_torch_fidelity()
    rng = np.random.default_rng(12)
    generated, reference = images(rng, 13), images(rng, 13) * 0.5
    metric = FakeFID(reference_cache=str(tmp_path))
    first = run(metric, generated, reference, (5, 5, 3), protocol())
    assert len(FakeFID.log) == 6
    [cached] = tmp_path.glob('*.fidref')
    # A hit extracts only the generated images and reproduces the value exactly.
    second = run(metric, generated, reference, (5, 5, 3), protocol())
    assert len(FakeFID.log) == 3
    assert second == first
    # Any protocol change is a separate entry, never a stale hit.
    run(metric, generated, reference, (5, 5, 3), protocol(batch_size=4))
    assert len(FakeFID.log) == 6
    assert len(list(tmp_path.glob('*.fidref'))) == 2
    # A damaged file is a miss that is recomputed and rewritten.
    damaged = bytearray(cached.read_bytes())
    damaged[-1] ^= 1
    cached.write_bytes(bytes(damaged))
    assert run(metric, generated, reference, (5, 5, 3), protocol()) == first
    assert len(FakeFID.log) == 6
    assert run(metric, generated, reference, (5, 5, 3), protocol()) == first
    assert len(FakeFID.log) == 3
    # No protocol identity, or reference_cache=false, never touches the cache.
    for metric, identity in ((metric, None), (FakeFID(reference_cache=False), protocol())):
        assert run(metric, generated, reference, (5, 5, 3), identity) == first
        assert len(FakeFID.log) == 6
    with pytest.raises(ValueError, match='reference_cache'):
        InceptionFID('unused', reference_cache='')


def test_cache_file_is_digest_identity_and_count_checked(tmp_path):
    path = tmp_path / 'entry.fidref'
    rng = np.random.default_rng(13)
    statistics = {'mu': rng.normal(size=3), 'sigma': np.cov(rng.normal(size=(9, 3)), rowvar=False)}
    identity = {'kind': 'test', 'sample_count': 9}
    write_reference_statistics(path, identity, statistics, 9)
    loaded = read_reference_statistics(path, identity, 9)
    assert loaded['mu'].tobytes() == statistics['mu'].tobytes()
    assert loaded['sigma'].tobytes() == statistics['sigma'].tobytes()
    assert read_reference_statistics(path, dict(identity, sample_count=8), 9) is None
    assert read_reference_statistics(path, identity, 8) is None
    assert read_reference_statistics(tmp_path / 'missing.fidref', identity, 9) is None
    content = path.read_bytes()
    header, _, payload = content.partition(b'\n')
    path.write_bytes(header + b'\n' + payload[:-1] + bytes([payload[-1] ^ 1]))
    assert read_reference_statistics(path, identity, 9) is None
    # Even a correctly digested file must hold finite float64 moments.
    write_reference_statistics(path, identity, {'mu': np.full(3, np.nan), 'sigma': statistics['sigma']}, 9)
    assert read_reference_statistics(path, identity, 9) is None
    assert [p.name for p in tmp_path.iterdir()] == ['entry.fidref']


def folder(root, count):
    root.mkdir()
    for index in range(count):
        # Distinct pixels, so a repeated reference image would be visible.
        Image.new('RGB', (2, 2), (index % 256, index // 256, 7)).save(root / f'{index:04d}.png')
    return root


def recipe(root, sample_count, cache, shuffle=False):
    config = resolve_config({})
    config['training']['device'] = 'cpu'
    config['prior']['args']['num_particles'] = 16
    config['components']['dimmed'] = {'factory': f'{__name__}:Half', 'inputs': {'x': 'batch.real'},
                                    'trainable': False}
    config['objectives'] = [{'factory': 'mse', 'inputs': {'input': 'generated', 'target': 'components.dimmed'}}]
    config['metrics']['custom'] = {'fid': {
        'factory': f'{__name__}:FakeFID', 'args': {'reference_cache': str(cache)},
        'mode': 'snapshot', 'trigger': 'manual',
        'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
        'evaluation': {'device': 'cpu', 'sample_count': sample_count, 'batch_size': 64, 'seed': 83,
                       'generated': 'components.dimmed',
                       'data': {'factory': 'image_folder', 'args': {
                           'root': str(root), 'height': 2, 'width': 2, 'shuffle': shuffle}}}}}
    return config_values(config)


def snapshot(tmp_path, config):
    trainer = ReferenceTrainer(resolve_config(config))
    trainer.artifact_identity = {'run_id': 'fixture', 'attempt_id': 'attempt'}
    path = tmp_path / 'snapshot.pt'
    torch.save(bundle_state(trainer, {'real': torch.zeros(2, 2)}), path)
    return trainer, path


def evaluate(path, spec, identity):
    expected = _factory(spec['factory'], spec['args'])[1]
    return evaluate_snapshot(spec, expected, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), identity)


def test_snapshot_fid_over_image_folder_uses_each_reference_once_and_caches(tmp_path):
    needs_torch_fidelity()
    # Enough images that the folder identity alone exceeds the 64 KiB result budget.
    root = folder(tmp_path / 'images', 400)
    trainer, path = snapshot(tmp_path, recipe(root, 400, tmp_path / 'cache'))
    spec = trainer.config['metrics']['custom']['fid']
    FakeFID.log.clear()
    result = evaluate(path, spec, trainer.artifact_identity)
    names = ['generated', 'reference'] * (len(FakeFID.log) // 2)
    assert len(names) == 2 * 7  # ceil(400 / 64) batches
    reference_images = torch.cat([images for name, (images, _) in zip(names, FakeFID.log) if name == 'reference'])
    assert len({tuple(image.flatten().tolist()) for image in reference_images}) == 400
    features = logged_features(FakeFID.log, names)
    assert result['value'] == pytest.approx(oracle(features['generated'], features['reference']), rel=1e-7)
    data_identity = result['protocol']['data_identity']
    assert len(json.dumps(data_identity)) < MAX_DATA_IDENTITY_BYTES
    assert data_identity['entries']['items'] == 400 and data_identity['shuffle'] is False
    FakeFID.log.clear()
    again = evaluate(path, spec, trainer.artifact_identity)
    assert len(FakeFID.log) == len(names) // 2  # generated batches only
    assert again['value'] == result['value']


def test_snapshot_rejects_sample_count_above_folder_size_before_generating(tmp_path):
    root = folder(tmp_path / 'images', 5)
    trainer, path = snapshot(tmp_path, recipe(root, 6, tmp_path / 'cache'))
    FakeFID.log.clear()
    with pytest.raises(EvaluationDataExhausted, match='sample_count 6 exceeds the 5 samples'):
        evaluate(path, trainer.config['metrics']['custom']['fid'], trainer.artifact_identity)
    assert FakeFID.log == []


def test_snapshot_rejects_shuffled_reference_folder_even_without_preflight(tmp_path):
    root = folder(tmp_path / 'images', 5)
    with pytest.raises(ValueError, match='shuffle = false'):
        resolve_config(recipe(root, 5, tmp_path / 'cache', shuffle=True))
    trainer, path = snapshot(tmp_path, recipe(root, 5, tmp_path / 'cache'))
    spec = copy.deepcopy(trainer.config['metrics']['custom']['fid'])
    spec['evaluation']['data']['args']['shuffle'] = True
    with pytest.raises(ValueError, match='Reference evaluation data must not shuffle'):
        evaluate(path, spec, trainer.artifact_identity)
