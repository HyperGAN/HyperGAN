"""Replicated resume identity ignores where a launch joins; followers never check node 0's run.

The run directory is metadata only: a node 0 manifest and one distributed
checkpoint generation whose topology, like the real writer's
(distributed_checkpoints._distributed_checkpoint_identity), omits node_rank and
rendezvous. No workers start, so a resume that passes routing stops at a stub.
"""
import json

import pytest

from hypergan import replicated_execution, run_controller
from hypergan.checkpoint_compatibility import CURRENT_VERSION
from hypergan.config import config_values, fingerprint, resolve_config
from hypergan.execution import prepare_resume, prepare_train
from hypergan.execution_profiles import resolve_execution_profile

CHECKPOINT = 'attempt-step-00000002-a'


def _config():
    return resolve_config({'training': {'device': 'cpu', 'batch_size': 8, 'steps': 4}})


def _profile(**execution):
    return {'schema_version': 1, 'execution': {'name': 'cpu-replicated-gloo', 'world_size': 2, **execution}}


def _write_run(run_dir, profile, *, topology=None):
    """A node 0 run as the controller leaves it: the full execution in the manifest."""
    config = _config()
    execution = resolve_execution_profile(profile, config)['execution']
    if topology is None:
        topology = {k: v for k, v in execution.items() if k not in ('node_rank', 'rendezvous')}
    manifest = {'schema_version': 1, 'run_id': 'run-a', 'config': config_values(config),
                'config_sha256': fingerprint(config), 'next_sample_sequence': 0,
                'execution': execution, 'events_compacted': True}
    root = run_dir / 'distributed-checkpoints'
    (root / CHECKPOINT).mkdir(parents=True)
    (run_dir / 'manifest.json').write_text(json.dumps(manifest))
    identity = {'hypergan_checkpoint_version': CURRENT_VERSION, 'config_sha256': manifest['config_sha256'],
                'config': manifest['config'], 'topology': topology}
    info = {'schema_version': 1, 'kind': 'hypergan-distributed-training-checkpoint', 'step': 2,
            'run_id': 'run-a', 'attempt_id': 'attempt', 'identity': identity, 'event_boundary': {}}
    (root / CHECKPOINT / 'manifest.json').write_text(json.dumps(info))
    (root / 'latest.json').write_text(json.dumps({'schema_version': 1, 'kind': info['kind'],
                                                  'checkpoint': CHECKPOINT, 'step': 2}))
    return execution


def test_saved_profile_resume_accepts_checkpoint_topology_without_launch_keys(tmp_path):
    """H1: every execution key, node_rank and rendezvous included, was compared with the topology."""
    _write_run(tmp_path, _profile())
    prepared = prepare_resume(tmp_path)
    assert prepared.checkpoint == (tmp_path / 'distributed-checkpoints' / CHECKPOINT).resolve()
    # The repeated-train route reaches the same check.
    assert prepare_train(_config(), tmp_path).operation == 'resume'


def test_resume_with_a_new_rendezvous_port_is_accepted(tmp_path):
    _write_run(tmp_path, _profile(rendezvous='tcp://127.0.0.1:29500'))
    prepared = prepare_resume(tmp_path, profile=_profile(rendezvous='tcp://127.0.0.1:29501'))
    assert prepared.profile['execution']['rendezvous'] == 'tcp://127.0.0.1:29501'


def test_numerical_execution_differences_are_still_refused(tmp_path):
    execution = _write_run(tmp_path, _profile(), topology={})
    with pytest.raises(ValueError, match='Resume checkpoint numerical execution identity differs'):
        prepare_resume(tmp_path)
    topology = {k: v for k, v in execution.items() if k not in ('node_rank', 'rendezvous')}
    with pytest.raises(ValueError, match='^Resume numerical execution identity differs'):
        prepare_resume(tmp_path, profile=_profile(accumulation_steps=2))
    other = tmp_path / 'other'
    _write_run(other, _profile(), topology=dict(topology, accumulation_steps=2))
    with pytest.raises(ValueError, match='Resume checkpoint numerical execution identity differs'):
        prepare_resume(other)


class _Restored(Exception):
    pass


class _Execution:
    def __init__(self, profile):
        self.profile = profile

    def configure_attempt(self, context, *, preview_every, on_event):
        return {'execution': self.profile['execution'], 'service_policy': {}}

    def restore(self, *args):
        raise _Restored

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Factory:
    def __init__(self, profile):
        self.profile = resolve_execution_profile(profile, _config())

    def __call__(self, config):
        return _Execution(self.profile)


def test_controller_resume_with_a_new_rendezvous_port_reaches_restore(tmp_path):
    """The controller repeated the exact manifest comparison after preparation."""
    _write_run(tmp_path, _profile(rendezvous='tcp://127.0.0.1:29500'))
    with pytest.raises(_Restored):
        run_controller.run_resume(tmp_path, execution_factory=_Factory(_profile(rendezvous='tcp://127.0.0.1:29501')))
    with pytest.raises(ValueError, match='Resume numerical execution identity differs'):
        run_controller.run_resume(tmp_path, execution_factory=_Factory(_profile(accumulation_steps=2)))


FOLLOWER = _profile(nodes=2, node_rank=1, rendezvous='tcp://node0:29500')


@pytest.fixture
def followed(monkeypatch):
    calls = []

    def run_follower(config, run_dir, steps=None, *, profile, service_policy=None):
        calls.append(profile['execution']['node_rank'])
        return {'status': 'completed', 'node_rank': profile['execution']['node_rank'], 'commands': 0}
    monkeypatch.setattr(replicated_execution, 'run_follower', run_follower)
    return calls


def test_follower_train_on_node_0_run_routes_to_run_follower(tmp_path, followed):
    """H-cli-follower: a follower compared its node_rank with node 0's manifest and refused."""
    _write_run(tmp_path, dict(FOLLOWER, execution=dict(FOLLOWER['execution'], node_rank=0)))
    prepared = prepare_train(_config(), tmp_path, profile=FOLLOWER)
    assert prepared.operation == 'train'
    assert prepared.run()['node_rank'] == 1
    assert followed == [1]


def test_follower_resume_skips_node_0_checks(tmp_path, followed):
    # Node 0's latest checkpoint is absent, and the follower must not look for it.
    _write_run(tmp_path, dict(FOLLOWER, execution=dict(FOLLOWER['execution'], node_rank=0)))
    (tmp_path / 'distributed-checkpoints' / 'latest.json').unlink()
    prepared = prepare_resume(tmp_path, profile=FOLLOWER)
    assert prepared.run()['node_rank'] == 1
    assert followed == [1]
