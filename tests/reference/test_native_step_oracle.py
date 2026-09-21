"""The native step matches the frozen pre-program update, including resume."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import load_config, resolve_config
from hypergan.objective_program import compile_legacy_program
from hypergan.training import ReferenceTrainer

from tests.reference.legacy_native_oracle import legacy_native_update
from tests.reference.test_image_phase_policy import config as reconstruction_config


class CountingMSE(nn.Module):
    """Records that a zero-weight objective still runs."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input, target):
        self.calls += 1
        return nn.functional.mse_loss(input, target)


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class JointGenerator(nn.Module):
    """Reads the conditioner, so that module can receive a legitimate generator gradient."""
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


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _close(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for item, other in zip(left, right):
            _close(item, other)
    else:
        assert left == right


def _assert_rows(produced, reference):
    assert produced.keys() == reference.keys()
    for key, value in produced.items():
        assert value == reference[key]


def _assert_trainers(produced, reference):
    assert produced.step == reference.step
    names = sorted(produced.graph.models)
    assert names == sorted(reference.graph.models)
    for name in names:
        _close(produced.graph.models[name].state_dict(), reference.graph.models[name].state_dict())
        for actual, expected in zip(produced.graph.models[name].parameters(), reference.graph.models[name].parameters()):
            assert (actual.grad is None) == (expected.grad is None)
            if actual.grad is not None:
                _close(actual.grad, expected.grad)
    _close(produced.prior.state_dict(), reference.prior.state_dict())
    assert (produced.prior.z.grad is None) == (reference.prior.z.grad is None)
    if produced.prior.z.grad is not None:
        _close(produced.prior.z.grad, reference.prior.z.grad)
    _close(produced.opt_g.state_dict(), reference.opt_g.state_dict())
    _close(produced.opt_d.state_dict(), reference.opt_d.state_dict())
    _close(produced.ema_graph.state_dict(), reference.ema_graph.state_dict())
    _close(produced.ema_prior.state_dict(), reference.ema_prior.state_dict())
    for name in produced.streams:
        assert torch.equal(produced.streams[name].get_state(), reference.streams[name].get_state())


def _matched(config, updates, resumed):
    produced, reference = ReferenceTrainer(config), ReferenceTrainer(config)
    last_produced = last_reference = None
    for _ in range(updates):
        row, last_produced = produced.update()
        other, last_reference = legacy_native_update(reference)
        _assert_rows(row, other)
        _close(last_produced, last_reference)
        _assert_trainers(produced, reference)
    restored_produced, restored_reference = ReferenceTrainer(config), ReferenceTrainer(config)
    restore_trainer(restored_produced, copy.deepcopy(trainer_state(produced, last_produced)))
    restore_trainer(restored_reference, copy.deepcopy(trainer_state(reference, last_reference)))
    for _ in range(resumed):
        row, last_produced = restored_produced.update()
        other, last_reference = legacy_native_update(restored_reference)
        _assert_rows(row, other)
        _close(last_produced, last_reference)
        _assert_trainers(restored_produced, restored_reference)
    return restored_produced


def test_default_particle_recipe_matches_legacy_oracle_across_resume():
    produced = _matched(resolve_config({}), updates=3, resumed=1)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in produced.graph.models["discriminator"].parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in produced.graph.models["generator"].parameters())


def test_lazy_penalty_boundary_matches_on_skip_and_application():
    config = resolve_config({"gradient_penalty": {"lazy_k": 2, "kappa": .001}, "training": {"steps": 4, "batch_size": 8},
                             "prior": {"args": {"num_particles": 64, "z_dim": 4}}})
    produced = ReferenceTrainer(config)
    reference = ReferenceTrainer(config)
    penalties = []
    for _ in range(3):
        row, _ = produced.update()
        other, _ = legacy_native_update(reference)
        _assert_rows(row, other)
        _assert_trainers(produced, reference)
        penalties.append(row["gradient_penalty"])
    assert penalties[0] == 0
    assert penalties[1] > 0


def test_paired_linear_matches_legacy_oracle():
    config = load_config(Path(__file__).parents[2] / "examples" / "paired-linear.toml")
    _matched(config, updates=3, resumed=1)


def test_reconstruction_recipe_matches_legacy_oracle():
    produced = _matched(reconstruction_config(), updates=2, resumed=1)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in produced.graph.models["encoder"].parameters())


def test_zero_adversarial_weight_still_applies_penalty_and_steps_critic():
    config = resolve_config({"adversarial": {"weight": 0}, "gradient_penalty": {"kappa": .001},
                             "training": {"steps": 3, "batch_size": 8},
                             "prior": {"args": {"num_particles": 64, "z_dim": 4}}})
    produced = ReferenceTrainer(config)
    reference = ReferenceTrainer(config)
    before = [parameter.detach().clone() for parameter in produced.graph.models["discriminator"].parameters()]
    row, _ = produced.update()
    other, _ = legacy_native_update(reference)
    _assert_rows(row, other)
    _assert_trainers(produced, reference)
    assert row["d_adversarial_weighted"] == 0
    assert row["gradient_penalty"] != 0
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, produced.graph.models["discriminator"].parameters()))


def test_zero_weight_objective_still_executes():
    config = resolve_config({
        "objectives": [{"factory": f"{__name__}:CountingMSE", "weight": 0,
                        "inputs": {"input": "generated", "target": "batch.real"}}],
        "training": {"steps": 3, "batch_size": 8},
        "prior": {"args": {"num_particles": 64, "z_dim": 4}}})
    produced = ReferenceTrainer(config)
    reference = ReferenceTrainer(config)
    row, _ = produced.update()
    other, _ = legacy_native_update(reference)
    _assert_rows(row, other)
    _assert_trainers(produced, reference)
    assert produced.objectives[0].calls == 1
    assert reference.objectives[0].calls == 1


def test_program_parameter_order_matches_legacy_optimizer_groups():
    trainer = ReferenceTrainer(resolve_config({}))
    assert trainer.program.schedule == "d-then-g-v1"
    assert tuple(trainer.graph.generator_parameters()) == trainer.program.generator_parameters
    expected_critic = tuple(parameter for parameter in trainer.graph.models["discriminator"].parameters() if parameter.requires_grad)
    assert expected_critic == trainer.program.critic_parameters
    assert tuple(trainer.opt_g.param_groups[0]["params"]) == trainer.program.generator_parameters
    assert tuple(trainer.opt_d.param_groups[0]["params"]) == trainer.program.critic_parameters
    assert tuple(trainer.opt_g.param_groups[1]["params"]) == trainer.program.prior_parameters


def test_duplicate_generator_parameters_keep_the_first_occurrence():
    config = resolve_config({"training": {"steps": 1, "batch_size": 4}, "prior": {"args": {"num_particles": 8, "z_dim": 4}}})
    trainer = ReferenceTrainer(config)
    original = trainer.graph.generator_parameters
    parameters = original()
    trainer.graph.generator_parameters = lambda: [parameters[0], parameters[0], *parameters[1:]]
    program = compile_legacy_program(
        trainer.graph, trainer.prior, trainer.config, trainer.objectives, trainer.gan, trainer.penalty, trainer.spread)
    assert program.generator_parameters == tuple(parameters)


def test_overlapping_update_groups_are_rejected():
    config = resolve_config({"training": {"steps": 1, "batch_size": 4}, "prior": {"args": {"num_particles": 8, "z_dim": 4}}})
    trainer = ReferenceTrainer(config)
    shared = next(trainer.graph.models["discriminator"].parameters())
    original = trainer.graph.generator_parameters
    trainer.graph.generator_parameters = lambda: [shared, *original()]
    with pytest.raises(ValueError, match="both the critic and generator"):
        compile_legacy_program(
            trainer.graph, trainer.prior, trainer.config, trainer.objectives, trainer.gan, trainer.penalty, trainer.spread)


def test_component_condition_is_detached_while_generator_path_can_train_it():
    config = resolve_config({
        "components": {
            "generator": {"factory": f"{__name__}:JointGenerator",
                          "inputs": {"x": "latent", "condition": "components.conditioner"}},
            "discriminator": {"factory": f"{__name__}:RecordingCritic",
                              "inputs": {"x": "candidate", "condition": "components.conditioner"}},
            "conditioner": {"factory": f"{__name__}:Conditioner", "inputs": {"x": "batch.real"}}},
        "training": {"steps": 2, "batch_size": 8},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}}})
    trainer = ReferenceTrainer(config)
    direct = trainer.graph.models["conditioner"](torch.zeros(8, 2))
    assert direct.requires_grad
    trainer.update()
    critic = trainer.graph.models["discriminator"]
    assert critic.condition_requires_grad
    assert not any(critic.condition_requires_grad)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in trainer.graph.models["conditioner"].parameters())
