"""Extra adversarial terms keep separate losses, penalties, and parameter groups."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer


class JointGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 2)

    def forward(self, x, condition):
        return self.linear(torch.cat((x, condition), -1))


class LatentGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class CountingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


class Words(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x, condition):
        return self.linear(torch.cat((x, condition), -1))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class _PenaltyCalls:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _tiny(**overrides):
    raw = {
        "training": {"steps": 2, "batch_size": 8, "seed": 42},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}},
    }
    raw.update(overrides)
    return resolve_config(raw)


def _shared_config():
    return _tiny(
        components={
            "generator": {"factory": f"{__name__}:JointGenerator",
                          "inputs": {"x": "latent", "condition": "components.condition"}},
            "discriminator": {"factory": f"{__name__}:Critic", "inputs": {"x": "candidate"}},
            "words": {"factory": f"{__name__}:Words",
                      "inputs": {"x": "candidate", "condition": "components.condition"}},
            "condition": {"factory": f"{__name__}:Encoder", "inputs": {"x": "batch.real"}},
            "source": {"factory": f"{__name__}:Encoder", "inputs": {"x": "batch.real"}}},
        adversarial_terms=[
            {"id": "again", "component": "discriminator", "real": "batch.real", "fake": "components.source"},
            {"id": "words", "component": "words", "real": "batch.real", "fake": "generated",
             "inputs": {"x": "candidate", "condition": "components.condition"}},
        ])


def _grads(parameters):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def _no_gradient(grads):
    assert grads
    assert all(grad is None for grad in grads)


def _some_gradient(grads):
    assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


def _clone(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def _moved(parameters, before):
    return any(not torch.equal(parameter, old) for parameter, old in zip(parameters, before))


def test_shared_and_second_critics_own_phase_gradients_and_updates():
    trainer = ReferenceTrainer(_shared_config())
    terms = trainer.program.adversarial_terms
    assert [term.id for term in terms] == ["adversarial", "again", "words"]
    assert terms[0].module is terms[1].module
    assert terms[2].module is not terms[0].module
    assert terms[0].gan is trainer.gan and terms[0].penalty_fn is trainer.penalty and terms[0].penalty is True
    again = terms[1]
    assert again.critic_phase.fake.path == "components.source"
    assert again.critic_phase.fake.detach_sample and not again.critic_phase.fake.detach_score
    assert again.generator_phase.fake.path == "components.source"
    assert not again.generator_phase.fake.detach_sample and not again.generator_phase.fake.detach_score
    assert again.generator_phase.real.detach_score
    condition = next(route for route in terms[2].routes if route.argument == "condition")
    candidate = next(route for route in terms[2].routes if route.path == "candidate")
    assert condition.detach_critic and condition.detach_generator
    assert not candidate.detach_critic and not candidate.detach_generator
    graph = trainer.graph.models
    discriminator = list(graph["discriminator"].parameters())
    words = list(graph["words"].parameters())
    condition_parameters = list(graph["condition"].parameters())
    source = list(graph["source"].parameters())
    generator = list(graph["generator"].parameters())
    critic_ids = {id(parameter) for parameter in trainer.program.critic_parameters}
    generator_ids = {id(parameter) for parameter in trainer.program.generator_parameters}
    assert [id(parameter) for parameter in discriminator + words] == [id(parameter) for parameter in trainer.program.critic_parameters]
    assert critic_ids.isdisjoint(generator_ids)
    assert all(id(parameter) not in generator_ids for parameter in discriminator + words)
    assert all(id(parameter) in generator_ids for parameter in generator + source)
    assert all(id(parameter) not in critic_ids for parameter in condition_parameters + source + generator)
    before = {
        "discriminator": _clone(discriminator), "words": _clone(words), "condition": _clone(condition_parameters),
        "source": _clone(source), "generator": _clone(generator)}
    seen = {}
    original_d, original_g = trainer.opt_d.step, trainer.opt_g.step

    def d_step(*args, **kwargs):
        seen["d"] = {
            "discriminator": _grads(discriminator), "words": _grads(words), "condition": _grads(condition_parameters),
            "source": _grads(source), "generator": _grads(generator)}
        return original_d(*args, **kwargs)

    def g_step(*args, **kwargs):
        assert _moved(discriminator, before["discriminator"])
        assert _moved(words, before["words"])
        assert not _moved(condition_parameters, before["condition"])
        assert not _moved(source, before["source"])
        assert not _moved(generator, before["generator"])
        assert all(not parameter.requires_grad for parameter in discriminator + words)
        seen["g"] = {"source": _grads(source), "generator": _grads(generator)}
        held = {"discriminator": _clone(discriminator), "words": _clone(words)}
        result = original_g(*args, **kwargs)
        assert _moved(generator, before["generator"])
        assert _moved(source, before["source"])
        assert not _moved(discriminator, held["discriminator"])
        assert not _moved(words, held["words"])
        return result

    trainer.opt_d.step = d_step
    trainer.opt_g.step = g_step
    trainer.update()
    _some_gradient(seen["d"]["discriminator"])
    _some_gradient(seen["d"]["words"])
    _no_gradient(seen["d"]["condition"])
    _no_gradient(seen["d"]["source"])
    _no_gradient(seen["d"]["generator"])
    _some_gradient(seen["g"]["source"])
    _some_gradient(seen["g"]["generator"])


def test_zero_weight_extra_term_still_runs_and_its_penalty_steps_the_critic():
    config = _tiny(
        components={
            "generator": {"factory": f"{__name__}:LatentGenerator", "inputs": {"x": "latent"}},
            "discriminator": {"factory": f"{__name__}:Critic", "inputs": {"x": "candidate"}},
            "extra": {"factory": f"{__name__}:CountingCritic", "inputs": {"x": "candidate"}}},
        adversarial={"weight": 0},
        gradient_penalty={"kappa": .001},
        adversarial_terms=[{
            "id": "extra", "component": "extra", "weight": 0, "penalty": True,
            "real": "batch.real", "fake": "generated"}])
    trainer = ReferenceTrainer(config)
    term = trainer.program.adversarial_terms[1]
    assert term.weight == 0 and term.penalty is True and term.penalty_fn is not trainer.penalty
    assert term.penalty_fn.kappa == pytest.approx(.001)
    assert trainer.penalty.kappa == pytest.approx(.001)
    module = trainer.graph.models["extra"]
    before = _clone(module.parameters())
    row, _ = trainer.update()
    assert module.calls >= 2
    assert row["d_adversarial_weighted"] == 0
    assert row["g_adversarial_weighted"] == 0
    assert row["gradient_penalty"] != 0
    assert _moved(list(module.parameters()), before)


def test_penalty_coefficients_stay_per_term_when_a_module_is_reused():
    config = _tiny(
        components={
            "generator": {"factory": f"{__name__}:LatentGenerator", "inputs": {"x": "latent"}},
            "discriminator": {"factory": f"{__name__}:Critic", "inputs": {"x": "candidate"}}},
        gradient_penalty={"kappa": .001},
        adversarial_terms=[
            {"id": "pen-a", "component": "discriminator", "penalty": True, "penalty_coeff": 0.25,
             "real": "batch.real", "fake": "generated"},
            {"id": "pen-b", "component": "discriminator", "penalty": True, "penalty_coeff": 0.5,
             "real": "batch.real", "fake": "generated"},
            {"id": "plain", "component": "discriminator", "penalty": False,
             "real": "batch.real", "fake": "generated"},
        ])
    trainer = ReferenceTrainer(config)
    terms = trainer.program.adversarial_terms
    assert terms[0].penalty_fn is trainer.penalty
    assert trainer.penalty.coeff == 1.0
    assert trainer.config["gradient_penalty"]["coeff"] == 1.0
    assert terms[1].penalty_fn is not terms[2].penalty_fn
    assert terms[1].penalty_fn is not trainer.penalty
    assert terms[1].penalty_fn.coeff == 0.25
    assert terms[2].penalty_fn.coeff == 0.5
    assert terms[3].penalty is False and terms[3].penalty_fn is None
    counters = []
    replaced = []
    for term in terms:
        if term.penalty_fn is None:
            counters.append(None)
            replaced.append(term)
            continue
        counter = _PenaltyCalls(term.penalty_fn)
        counters.append(counter)
        replaced.append(replace(term, penalty_fn=counter))
    trainer.program = replace(trainer.program, adversarial_terms=tuple(replaced))
    trainer.update()
    assert [None if counter is None else counter.calls for counter in counters] == [1, 1, 1, None]
    assert trainer.penalty.coeff == 1.0
    assert terms[1].penalty_fn.coeff == 0.25
    assert terms[2].penalty_fn.coeff == 0.5
