"""Evaluation data must hold the declared sample count without wrapping or repeating."""
import pytest

from hypergan.config import resolve_config
from hypergan.data import EvaluationDataExhausted, ImageFolder
from hypergan.metric_plugins import check_evaluation_data

REFERENCE = {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'}


def snapshot(data_args, inputs=REFERENCE):
    return {'factory': 'uninstalled.metrics:FID', 'mode': 'snapshot', 'trigger': 'manual',
            'inputs': dict(inputs),
            'evaluation': {'data': {'factory': 'image_folder', 'args': dict(data_args)},
                           'sample_count': 8, 'batch_size': 4, 'seed': 5}}


def resolve(spec):
    return resolve_config({'metrics': {'custom': {'fid': spec}}})


@pytest.mark.parametrize('args', [{'root': '/images', 'height': 4, 'width': 4},
                                  {'root': '/images', 'height': 4, 'width': 4, 'shuffle': True}])
def test_preflight_rejects_shuffled_image_folder_reference_data(args):
    # The image_folder default shuffles, and a shuffled folder wraps into a new
    # epoch, so reference statistics would silently count images twice.
    with pytest.raises(ValueError, match='fid.evaluation.data: image_folder reference data must set shuffle = false'):
        resolve(snapshot(args))


def test_preflight_accepts_unshuffled_reference_and_shuffled_conditioning_only_data():
    resolve(snapshot({'root': '/images', 'height': 4, 'width': 4, 'shuffle': False}))
    # Without a reference binding the folder only conditions generation.
    resolve(snapshot({'root': '/images', 'height': 4, 'width': 4}, {'generated': 'evaluation.generated'}))


class Finite:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class Unsized:
    pass


def folder(shuffle, length=3):
    # The check reads only the sampler settings, so no images are needed.
    data = object.__new__(ImageFolder)
    data.shuffle, data.entries = shuffle, [{}] * length
    return data


def test_sample_count_above_dataset_size_fails_before_any_draw():
    with pytest.raises(EvaluationDataExhausted, match='sample_count 8 exceeds the 7 samples'):
        check_evaluation_data(Finite(7), {'sample_count': 8}, REFERENCE)
    with pytest.raises(EvaluationDataExhausted, match='sample_count 4 exceeds the 3 samples'):
        check_evaluation_data(folder(False), {'sample_count': 4}, REFERENCE)
    check_evaluation_data(Finite(8), {'sample_count': 8}, REFERENCE)
    check_evaluation_data(folder(False), {'sample_count': 3}, REFERENCE)
    # Unsized data (random draws, streams) relies on the exhaustion error instead.
    check_evaluation_data(Unsized(), {'sample_count': 10 ** 6}, REFERENCE)


def test_shuffled_image_folder_cannot_supply_reference_images():
    with pytest.raises(ValueError, match='Reference evaluation data must not shuffle'):
        check_evaluation_data(folder(True), {'sample_count': 2}, REFERENCE)
    check_evaluation_data(folder(True), {'sample_count': 2}, {'generated': 'evaluation.generated'})
