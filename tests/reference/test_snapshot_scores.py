"""Public MisGAN evaluation from paired EMA outputs, including actual worker publication."""
import json

import pytest
import torch

import hypergan as hg
from tests.reference.test_misgan import model


class PairedScores:
    def describe(self):
        return {'kind': 'scalars', 'outputs': {
            f'score{i}': {'label': f'Paired score {i}', 'unit': 'value', 'direction': 'minimize'}
            for i in range(12)}}

    def evaluate(self, *, batches, context):
        means = []
        sizes = []
        for tensors in batches:
            x, mask = tensors['x'], tensors['mask']
            assert x.device.type == mask.device.type == 'cpu'
            assert not x.requires_grad and not mask.requires_grad
            assert x.shape == mask.shape and x.shape[1] == 585
            assert torch.equal(tensors['critic_input'], torch.cat((x * mask, mask), dim=1))
            sizes.append(len(x))
            means.append(x.mean(dim=1))
        assert sizes == [127] * 16 + [16]
        assert sum(sizes) == context['sample_count'] == 2048
        rows = torch.cat(means)
        return {f'score{i}': rows[i].item() for i in range(12)}


def score_model(model, tmp_path, device='cpu'):
    path = tmp_path / 'scores.toml'
    spec = hg.override(model, {'train.device': device, 'train.steps': 2,
        'sampling.output': 'pair',
        'evaluations.paired': {
            'factory': __name__ + ':PairedScores', 'every': 1, 'device': 'cpu',
            'samples': 2048, 'batch_size': 127, 'seed': 123, 'prior_seed': 7,
            'inputs': {'x': 'gx', 'mask': 'gm', 'critic_input': 'pair'},
            'data': {'builtin': 'gaussian_grid'}, 'timeout': 60,
        }})
    hg.save(spec, path)
    return path


def check_stream(root, receipt):
    path = root / 'metrics' / 'evaluations' / receipt['evaluation_id']
    event = json.loads((path / 'events.jsonl').read_text())
    assert json.loads((path / 'stream.json').read_text())['stream_id'] == event['stream_id']
    assert event['step'] == receipt['source_step']
    assert event['run_id'] == receipt['run_id']
    assert event['attempt_id'] == receipt['attempt_id'] == event['snapshot_identity']['attempt_id']
    assert event['evaluation_protocol']['evaluation']['prior_seed'] == 7
    assert len(event['protocol_sha256']) == len(event['snapshot_sha256']) == 64
    assert event['metrics'] == {f'paired/{k}': v for k, v in receipt['result']['values'].items()}
    assert len(event['metrics']) == 12


@pytest.mark.heavy
def test_scheduled_scores_resume_manual_retry_and_pruning(model, tmp_path):
    from hypergan.training import train, resume
    root = tmp_path / 'run'
    config = score_model(model, tmp_path)
    stopped = train(config, root, stop_after_steps=1, checkpoint_every=1)
    completed = resume(root)
    assert stopped['attempt_id'] != completed['attempt_id']
    histories = hg.evaluations(root)
    assert len(histories) == 12
    for rows in histories.values():
        assert [row['step'] for row in rows] == [1, 2]
        assert all(row['status'] == 'complete' for row in rows)
    receipt = hg.evaluate(root, 'paired', checkpoint=2, devices='cpu')
    assert receipt['evaluation_id'] == histories['paired/score0'][-1]['evaluation_id']
    check_stream(root, receipt)
    assert hg.evaluations(root) == histories
    assert len(list((root / 'checkpoints').glob('*-step-*'))) == 1
    assert len(list((root / 'metrics/evaluations').iterdir())) == 2
    assert not list(root.rglob('snapshot.pt'))


@pytest.mark.heavy
def test_replicated_training_publishes_one_named_set(model, tmp_path):
    from hypergan.replicated_execution import run_train
    path = score_model(model, tmp_path)
    root = tmp_path / 'replicated'
    profile = {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo',
               'world_size': 2, 'accumulation_steps': 1}}
    result = run_train(path, root, profile=profile, stop_after_steps=1, checkpoint_every=1)
    [path] = (root / 'metrics/evaluations').glob('*/receipt.json')
    receipt = json.loads(path.read_text())
    assert receipt['source_step'] == result['steps'] == 1
    check_stream(root, receipt)
    assert not list(root.rglob('snapshot.pt'))
