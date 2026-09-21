import hashlib
import json

import pytest
import torch
from PIL import Image

from hypergan.colorization_data import ColorizationData, prepare_manifest, approve_prepared_manifest
from hypergan.colorization_metrics import ChromaDistributionDistance, GrayscaleStructureDistance, RepeatedConditionChromaDiversity


@pytest.fixture
def logos(tmp_path):
    root = tmp_path / 'logos'
    root.mkdir()
    # Different contents provide both hash-based partitions with fixed fixtures.
    for i in range(100):
        Image.new('RGB', (8, 4), (i * 2, i, 255 - i)).save(root / f'{i:03}.png')
    Image.new('RGBA', (4, 8), (0, 0, 0, 0)).save(root / 'transparent.png')
    Image.new('1', (8, 8), 1).save(root / 'one_bit.png')
    (root / 'archive.tgz').write_bytes(b'Explicitly ignored non-image archive')
    manifest = tmp_path / 'manifest.json'
    info = prepare_manifest(root, manifest, height=8, width=8, workers=2)
    return root, manifest, info


def test_pairs_split_and_full_sampler_recovery(logos):
    root, manifest, info = logos
    args = dict(root=root, manifest=manifest, manifest_sha256=info['sha256'])
    training, heldout = ColorizationData(**args), ColorizationData(**args, split='heldout', shuffle=False)
    assert set(e['sha256'] for e in training.entries).isdisjoint(e['sha256'] for e in heldout.entries)
    assert heldout.entries == sorted(heldout.entries, key=lambda entry: (entry['sha256'], entry['path']))
    assert heldout.resume_identity()['order'] == 'sha256,path'
    assert 'order' not in training.resume_identity()  # Preserve existing training sampler identity.
    assert info['ignored_paths'] == ['archive.tgz']
    rng = torch.Generator().manual_seed(33)
    batch = training(9, generator=rng)
    assert batch['real'].shape == (9, 3, 8, 8)
    assert batch['gray'].shape == (9, 1, 8, 8)
    expected_gray = (batch['real'] * torch.tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    torch.testing.assert_close(batch['gray'], expected_gray)
    state, rng_state = training.state_dict(), rng.get_state()
    expected = training(104, generator=rng)  # Crosses the epoch boundary.
    restored = ColorizationData(**args)
    restored.load_state_dict(state)
    rng.set_state(rng_state)
    actual = restored(104, generator=rng)
    assert all(torch.equal(actual[key], expected[key]) for key in actual)
    assert len(json.dumps(training.resume_identity())) < 4096
    with pytest.raises(ValueError, match='identity mismatch'):
        heldout.load_state_dict(state)


def test_alpha_white_and_pad(logos):
    root, manifest, info = logos
    inventory = json.loads(manifest.read_text())
    entry = next(e for e in inventory['entries'] if e['path'] == 'transparent.png')
    data = ColorizationData(root, manifest, info['sha256'], split=entry['split'], shuffle=False)
    sample = data(len(data.entries), generator=torch.Generator())
    idx = next(i for i, e in enumerate(data.entries) if e['path'] == 'transparent.png')
    assert torch.equal(sample['real'][idx], torch.ones(3, 8, 8))
    assert torch.equal(sample['gray'][idx], torch.ones(1, 8, 8))
    opaque = next(i for i, e in enumerate(data.entries) if e['path'] != 'transparent.png')
    assert torch.equal(sample['real'][opaque, :, :2], torch.ones(3, 2, 8))
    with pytest.raises(StopIteration, match='never wraps'):
        data(1, generator=torch.Generator())
    one_bit = next(e for e in inventory['entries'] if e['path'] == 'one_bit.png')
    assert one_bit['source_mode'] == '1'
    bit_data = ColorizationData(root, manifest, info['sha256'], split=one_bit['split'], shuffle=False)
    bit_sample = bit_data(len(bit_data.entries), generator=torch.Generator())
    bit_index = next(i for i, e in enumerate(bit_data.entries) if e['path'] == 'one_bit.png')
    assert torch.equal(bit_sample['real'][bit_index], torch.ones(3, 8, 8))


def test_pinned_manifest_and_file_changes_fail_without_sampler_rng_advance(logos):
    root, manifest, info = logos
    data = ColorizationData(root, manifest, info['sha256'], shuffle=False)
    rng = torch.Generator().manual_seed(10)
    original_state, original_rng = data.state_dict(), rng.get_state()
    (root / data.entries[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        data(2, generator=rng)
    assert data.state_dict() == original_state
    assert torch.equal(rng.get_state(), original_rng)
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        ColorizationData(root, manifest, info['sha256'])


def test_corrupt_images_never_silently_skipped(tmp_path):
    (tmp_path / 'broken.png').write_bytes(b'not an image')
    with pytest.raises(ValueError, match='no images were skipped'):
        prepare_manifest(tmp_path, tmp_path / 'manifest.json', workers=1)
    assert not (tmp_path / 'manifest.json').exists()


def test_all_failures_reported_and_explicit_exclusions_pinned(tmp_path):
    for name in ('broken.png', 'also_broken.jpg'):
        (tmp_path / name).write_bytes(b'not an image')
    Image.new('RGB', (8, 8)).save(tmp_path / 'valid.png')
    output = tmp_path / 'manifest.json'
    with pytest.raises(ValueError, match='2 invalid images'):
        prepare_manifest(tmp_path, output, workers=2)
    report_path = tmp_path / 'manifest.rejections.json'
    report = json.loads(report_path.read_text())
    assert {e['path'] for e in report['rejections']} == {'broken.png', 'also_broken.jpg'}
    assert all(e['sha256'] == hashlib.sha256(b'not an image').hexdigest() for e in report['rejections'])
    assert not output.exists()
    approved = approve_prepared_manifest(report_path, hashlib.sha256(report_path.read_bytes()).hexdigest(), output)
    manifest = json.loads(output.read_text())
    assert manifest['excluded_images'] == report['rejections']
    assert approved['excluded_count'] == 2
    assert (tmp_path / 'broken.png').read_bytes() == b'not an image'


def test_duplicate_bytes_stay_in_same_split(logos):
    root, manifest, _ = logos
    (root / 'duplicate.png').write_bytes((root / '000.png').read_bytes())
    second = manifest.with_name('second.json')
    prepare_manifest(root, second, height=8, width=8, workers=2)
    entries = {e['path']: e for e in json.loads(second.read_text())['entries']}
    assert entries['duplicate.png']['split'] == entries['000.png']['split']


@pytest.mark.parametrize('metric', [ChromaDistributionDistance(), GrayscaleStructureDistance()])
def test_metrics_identity_and_malformed_input(metric):
    rgb = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
    assert metric.evaluate(batches=[{'generated': rgb, 'reference': rgb}], context={}) == 0
    with pytest.raises(ValueError, match='nonempty'):
        metric.evaluate(batches=[], context={})
    with pytest.raises(ValueError, match='finite'):
        metric.evaluate(batches=[{'generated': rgb * float('nan'), 'reference': rgb}], context={})


def test_chroma_detects_color_shift_with_identical_luminance():
    reference = torch.tensor([.8, .3, .2])[None, :, None, None].expand(2, 3, 8, 8).clone()
    generated = reference.clone()
    generated[:, 0] -= .2
    generated[:, 1] += .2 * .299 / .587
    batch = {'generated': generated * 2 - 1, 'reference': reference * 2 - 1}
    assert ChromaDistributionDistance().evaluate(batches=[batch], context={}) > .01
    assert GrayscaleStructureDistance().evaluate(batches=[batch], context={}) == 0


def test_chroma_ignores_pixel_order_structure_detects_it():
    reference = torch.full((1, 3, 8, 8), -1.)
    reference[:, :, :, :4] = 1
    generated = reference.roll(2, dims=-1)
    batch = {'generated': generated, 'reference': reference}
    assert ChromaDistributionDistance().evaluate(batches=[batch], context={}) == 0
    assert GrayscaleStructureDistance().evaluate(batches=[batch], context={}) > .1


def test_metric_batch_partition_does_not_change_result():
    rng = torch.Generator().manual_seed(17)
    generated, reference = (torch.rand(4, 3, 8, 8, generator=rng) * 2 - 1 for _ in range(2))
    for metric in (ChromaDistributionDistance(), GrayscaleStructureDistance()):
        whole = metric.evaluate(batches=[{'generated': generated, 'reference': reference}], context={})
        split = metric.evaluate(batches=[{'generated': generated[i:i + 1], 'reference': reference[i:i + 1]} for i in range(4)], context={})
        assert split == pytest.approx(whole, abs=1e-14)


def test_repeated_condition_data_recovery_and_metric_cross_batch_groups(logos):
    root, manifest, info = logos
    data = ColorizationData(root, manifest, info['sha256'], split='heldout', shuffle=False, repeats=4)
    rng = torch.Generator()
    first = data(3, generator=rng)
    state = data.state_dict()
    second = data(5, generator=rng)
    data.load_state_dict(state)
    assert torch.equal(data(5, generator=rng)['real'], second['real'])
    assert torch.equal(first['real'][0], second['real'][0])
    reference = torch.cat([first['real'], second['real']])
    generated = reference.clone()
    generated[1::2] = -generated[1::2]
    batches = [{'generated': generated[:3], 'reference': reference[:3]},
               {'generated': generated[3:], 'reference': reference[3:]}]
    metric = RepeatedConditionChromaDiversity(repeats=4)
    assert metric.evaluate(batches=batches, context={}) > 0
    assert metric.evaluate(batches=[{'generated': reference, 'reference': reference}], context={}) == 0
    with pytest.raises(ValueError, match='incomplete group'):
        metric.evaluate(batches=[batches[0]], context={})
    bad_reference = reference.clone()
    bad_reference[1] = -bad_reference[1]
    with pytest.raises(ValueError, match='differs inside a group'):
        metric.evaluate(batches=[{'generated': generated, 'reference': bad_reference}], context={})
