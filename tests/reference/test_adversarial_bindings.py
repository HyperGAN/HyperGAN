"""Sample bindings and phase detach policies are executable declarations."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer


class Marker(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.tensor([0.25, -0.5]))

    def forward(self, x):
        return self.value.view(1, -1).expand(len(x), -1).contiguous()


class LatentGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class MarkedGenerator(LatentGenerator):
    """Reads the marker so config reachability accepts it. The values stay the latent map."""

    def forward(self, x, side):
        return self.linear(x) + side * 0


class RecordingCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.linear(x)


class CandidateCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class JointGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 2)

    def forward(self, x, condition):
        return self.linear(torch.cat((x, condition), -1))


class RecordingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.condition_requires_grad = []

    def forward(self, x, condition):
        self.condition_requires_grad.append(bool(condition.requires_grad))
        return self.linear(torch.cat((x, condition), -1))


class _RecordingPenalty:
    def __init__(self, inner):
        self.inner = inner
        self.fake = None

    def __call__(self, critic, real, fake, step=1, generator=None):
        self.fake = fake.detach().clone()
        return self.inner(critic, real, fake, step=step, generator=generator)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _binding_config():
    return resolve_config({
        "components": {
            "generator": {"factory": f"{__name__}:MarkedGenerator",
                          "inputs": {"x": "latent", "side": "components.marker"}},
            "discriminator": {"factory": f"{__name__}:RecordingCandidate", "inputs": {"x": "candidate"}},
            "marker": {"factory": f"{__name__}:Marker", "inputs": {"x": "batch.real"}, "trainable": False}},
        "training": {"steps": 1, "batch_size": 8},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}}})


def _condition_config():
    return resolve_config({
        "components": {
            "generator": {"factory": f"{__name__}:JointGenerator",
                          "inputs": {"x": "latent", "condition": "components.conditioner"}},
            "discriminator": {"factory": f"{__name__}:RecordingCritic",
                              "inputs": {"x": "candidate", "condition": "components.conditioner"}},
            "conditioner": {"factory": f"{__name__}:Conditioner", "inputs": {"x": "batch.real"}}},
        # Relativistic pairing subtracts a shared condition out of both scores.
        "adversarial": {"mode": "vanilla"},
        "gradient_penalty": {"lazy_k": 2},
        "training": {"steps": 2, "batch_size": 8},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}}})


def _encoder_config():
    return resolve_config({
        "components": {
            "generator": {"factory": f"{__name__}:LatentGenerator", "inputs": {"x": "latent"}},
            "discriminator": {"factory": f"{__name__}:CandidateCritic", "inputs": {"x": "candidate"}},
            "encoder": {"factory": f"{__name__}:Encoder", "inputs": {"x": "batch.real"}}},
        # Detached and weight 0: only keeps the encoder reachable. It is not a gradient source.
        "objectives": [{"factory": "mse", "weight": 0,
                        "inputs": {"input": "components.encoder", "target": "batch.real"},
                        "detach": ["input", "target"]}],
        "training": {"steps": 2, "batch_size": 8},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}}})


def _marker(trainer, like):
    return trainer.graph.models["marker"].value.view(1, -1).expand_as(like)


def _term(trainer):
    return trainer.program.adversarial_terms[0]


def _replace_term(trainer, term):
    terms = list(trainer.program.adversarial_terms)
    terms[0] = term
    trainer.program = replace(trainer.program, adversarial_terms=tuple(terms))


def _install_penalty(trainer):
    term = _term(trainer)
    recorder = _RecordingPenalty(term.penalty_fn)
    _replace_term(trainer, replace(term, penalty_fn=recorder))
    return recorder


def _retarget_critic_fake(trainer, path):
    term = _term(trainer)
    phase = replace(term.critic_phase, fake=replace(term.critic_phase.fake, path=path))
    _replace_term(trainer, replace(term, critic_phase=phase))


def _attach_condition(trainer):
    term = _term(trainer)
    routes = tuple(replace(route, detach_critic=False) if route.argument == "condition" else route for route in term.routes)
    _replace_term(trainer, replace(term, routes=routes))


def _generator_real_policy(trainer, *, detach_score, detach_route):
    term = _term(trainer)
    real = replace(term.generator_phase.real, path="components.encoder", detach_sample=False, detach_score=detach_score)
    routes = tuple(
        replace(route, detach_generator=detach_route) if route.path == "candidate" else route
        for route in term.routes)
    _replace_term(trainer, replace(term, generator_phase=replace(term.generator_phase, real=real), routes=routes))


def _snapshot(optimizer, parameters):
    captured = {}
    original = optimizer.step

    def step(*args, **kwargs):
        captured["grads"] = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
        return original(*args, **kwargs)

    optimizer.step = step
    return captured


def _snapshot_encoder_phase(trainer):
    """Capture encoder gradients at the generator step, while the critic is frozen."""
    captured = {}
    original = trainer.opt_g.step

    def step(*args, **kwargs):
        assert all(not parameter.requires_grad for parameter in trainer.graph.models["discriminator"].parameters())
        captured["grads"] = [None if parameter.grad is None else parameter.grad.detach().clone()
                             for parameter in trainer.graph.models["encoder"].parameters()]
        return original(*args, **kwargs)

    trainer.opt_g.step = step
    return captured


def _no_gradient(grads):
    assert grads
    assert all(grad is None for grad in grads)


def _some_gradient(grads):
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


def test_sample_binding_selects_the_tensor_the_critic_and_penalty_score():
    legacy = ReferenceTrainer(_binding_config())
    moved = ReferenceTrainer(_binding_config())
    assert legacy.program.adversarial_terms[0].critic_phase.fake.path == "generated"
    assert legacy.program.adversarial_terms[0].critic_phase.fake.detach_sample
    assert not legacy.program.adversarial_terms[0].critic_phase.fake.detach_score
    legacy_penalty = _install_penalty(legacy)
    moved_penalty = _install_penalty(moved)
    _retarget_critic_fake(moved, "components.marker")
    legacy.update()
    moved.update()
    legacy_seen = legacy.graph.models["discriminator"].seen
    moved_seen = moved.graph.models["discriminator"].seen
    assert len(legacy_seen) >= 2 and len(moved_seen) >= 2
    assert torch.equal(moved_seen[1], _marker(moved, moved_seen[1]))
    assert torch.equal(moved_penalty.fake, _marker(moved, moved_penalty.fake))
    assert not torch.equal(legacy_seen[1], _marker(legacy, legacy_seen[1]))
    assert not torch.equal(legacy_penalty.fake, _marker(legacy, legacy_penalty.fake))


def test_critic_phase_route_detachment_is_executable():
    detached = ReferenceTrainer(_condition_config())
    attached = ReferenceTrainer(_condition_config())
    condition = next(route for route in detached.program.adversarial_terms[0].routes if route.argument == "condition")
    assert condition.path == "components.conditioner"
    assert condition.detach_critic and condition.detach_generator
    _attach_condition(attached)

    def capture(trainer):
        recorded = _snapshot(trainer.opt_d, list(trainer.graph.models["conditioner"].parameters()))
        original = trainer.opt_d.step

        def step(*args, **kwargs):
            recorded["seen"] = list(trainer.graph.models["discriminator"].condition_requires_grad)
            return original(*args, **kwargs)

        trainer.opt_d.step = step
        return recorded

    detached_phase = capture(detached)
    attached_phase = capture(attached)
    detached.update()
    attached.update()
    assert detached_phase["seen"]
    assert not any(detached_phase["seen"])
    _no_gradient(detached_phase["grads"])
    assert attached_phase["seen"]
    assert all(attached_phase["seen"])
    _some_gradient(attached_phase["grads"])


def test_generator_phase_real_score_detachment_is_executable():
    legacy = ReferenceTrainer(_encoder_config())
    route_detached = ReferenceTrainer(_encoder_config())
    attached = ReferenceTrainer(_encoder_config())
    real = legacy.program.adversarial_terms[0].generator_phase.real
    assert real.path == "batch.real" and real.detach_score and not real.detach_sample
    candidate = next(route for route in legacy.program.adversarial_terms[0].routes if route.path == "candidate")
    assert not candidate.detach_generator
    _generator_real_policy(legacy, detach_score=True, detach_route=False)
    _generator_real_policy(route_detached, detach_score=False, detach_route=True)
    _generator_real_policy(attached, detach_score=False, detach_route=False)
    for trainer, expect_gradient in ((legacy, False), (route_detached, False), (attached, True)):
        captured = _snapshot_encoder_phase(trainer)
        trainer.update()
        if expect_gradient:
            _some_gradient(captured["grads"])
        else:
            _no_gradient(captured["grads"])
