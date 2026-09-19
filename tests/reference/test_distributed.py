"""Two real Gloo processes compared with global-batch autograd references."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from particlegan import GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer

from hypergan.distributed import GlooCollectives


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-5 if expected.dtype == torch.float32 else 1e-9,
                               atol=2e-6 if expected.dtype == torch.float32 else 1e-10)


def _average(value):
    result = value.detach().clone()
    dist.all_reduce(result)
    return result / dist.get_world_size()


def _gather_double_backward(group, rank, dtype):
    full = torch.tensor([[-0.8, 0.3], [0.1, 1.2], [0.6, -0.5], [1.4, 0.9]], dtype=dtype)
    direction = torch.arange(1., 9., dtype=dtype).reshape(4, 2) / 9
    local = full.chunk(2)[rank].clone().requires_grad_()
    gathered = group.gather(local)
    # The same full objective is replicated on both ranks; divide ONLY here to
    # compare gradients of independent input shards with one global objective.
    loss = (gathered.pow(4).mean() + gathered.mean().pow(3)) / 2
    first = torch.autograd.grad(loss, local, create_graph=True)[0]
    second = torch.autograd.grad((first * direction.chunk(2)[rank]).sum(), local)[0]
    reference = full.clone().requires_grad_()
    expected_loss = reference.pow(4).mean() + reference.mean().pow(3)
    expected_first = torch.autograd.grad(expected_loss, reference, create_graph=True)[0]
    expected_second = torch.autograd.grad((expected_first * direction).sum(), reference)[0]
    _close(loss * 2, expected_loss)
    _close(first, expected_first.chunk(2)[rank])
    _close(second, expected_second.chunk(2)[rank])


def _ra_and_double_backward(group, rank):
    real = torch.tensor([[1.2, -.7], [.4, 1.1], [-.9, .6], [.8, .3]], dtype=torch.float64)
    fake = real.flip(0) * .6 + .4
    theta = torch.tensor([.8, -.3], dtype=torch.float64, requires_grad=True)
    dr, df = real.chunk(2)[rank] @ theta, fake.chunk(2)[rank] @ theta
    # Match ParticleGAN RA logistic kernel with differentiable GLOBAL means.
    loss = (F.softplus(-(dr - group.mean(df))).mean()
            + F.softplus(df - group.mean(dr)).mean()) * .5
    gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
    direction = theta.new_tensor([.4, -.7])
    hessian_vector = torch.autograd.grad((gradient * direction).sum(), theta)[0]
    reference = theta.detach().clone().requires_grad_()
    expected_loss = GANLoss(mode="ra").d_loss(real @ reference, fake @ reference)
    expected_gradient = torch.autograd.grad(expected_loss, reference, create_graph=True)[0]
    expected_hvp = torch.autograd.grad((expected_gradient * direction).sum(), reference)[0]
    _close(_average(loss), expected_loss)
    _close(_average(gradient), expected_gradient)
    _close(_average(hessian_vector), expected_hvp)
    # Local means give a measurably different objective for this fixture.
    wrong = GANLoss(mode="ra").d_loss(dr, df)
    assert not torch.isclose(_average(wrong), expected_loss, rtol=1e-5, atol=1e-6)


def _prior_population(group, rank):
    ids = [torch.tensor([0, 0, 2]), torch.tensor([2, 4, 4])][rank]
    selected = group.unique_indices(ids, num_rows=7)
    assert selected.tolist() == [0, 2, 4]
    unequal = group.unique_indices(torch.tensor([], dtype=torch.int64) if rank == 0 else ids, num_rows=7)
    assert unequal.tolist() == [2, 4]
    assert group.unique_indices(torch.tensor([], dtype=torch.int64), num_rows=7).numel() == 0
    prior = MoGParticlePrior(num_particles=7, z_dim=3, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(83))
    rows = prior.z.detach().clone().requires_grad_()
    regularizer = ParticleRegularizer()
    # Standardized MoG means depend on the FULL raw table, even unsampled rows.
    loss = prior.means()[ids].square().mean() + regularizer(prior.z[selected])
    gradient = torch.autograd.grad(loss, prior.z, create_graph=True)[0]
    direction = torch.linspace(.1, .8, rows.numel(), dtype=torch.float64).reshape_as(rows)
    hvp = torch.autograd.grad((gradient * direction).sum(), prior.z)[0]
    all_ids = torch.tensor([0, 0, 2, 2, 4, 4])
    centers = (rows - rows.mean(0)) / (rows.std(0) + 1e-6)
    expected_loss = centers[all_ids].square().mean() + regularizer(rows[all_ids.unique()])
    expected_gradient = torch.autograd.grad(expected_loss, rows, create_graph=True)[0]
    expected_hvp = torch.autograd.grad((expected_gradient * direction).sum(), rows)[0]
    _close(_average(loss), expected_loss)
    _close(_average(gradient), expected_gradient)
    _close(_average(hvp), expected_hvp)
    assert torch.all(expected_gradient[[1, 3, 5, 6]].abs().sum(1) > 0)
    wrong = regularizer(rows[ids.unique()])
    assert not torch.isclose(_average(wrong), regularizer(rows[selected]), rtol=1e-5, atol=1e-6)


def _exact_penalty(rank):
    data = torch.tensor([[.8, 1.2], [.9, .7], [1.4, .5], [.6, 1.3]], dtype=torch.float64)
    parameter = torch.tensor([.9, .8], dtype=torch.float64, requires_grad=True)
    critic = lambda value: (value @ parameter).pow(3)
    # Real==fake removes random interpolation differences while retaining the
    # actual input-autograd/create_graph/parameter-backward b-cap computation.
    local = data.chunk(2)[rank]
    penalty = GradientPenalty(lazy_k=2)
    inactive = penalty(critic, local, local, step=1)
    assert inactive.item() == 0
    active = penalty(critic, local, local, step=2)
    gradient = torch.autograd.grad(active, parameter)[0]
    reference = parameter.detach().clone().requires_grad_()
    expected = penalty(lambda value: (value @ reference).pow(3), data, data, step=2)
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    _close(_average(active), expected)
    _close(_average(gradient), expected_gradient)


def _worker(mode, rank, rendezvous, result):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=3 if mode in ("exit", "stall") else 15))
    group = GlooCollectives(2)
    try:
        if mode == "numerics":
            for dtype in (torch.float32, torch.float64):
                _gather_double_backward(group, rank, dtype)
            _ra_and_double_backward(group, rank)
            _prior_population(group, rank)
            _exact_penalty(rank)
            outcome = {"passed": True}
        elif mode in ("mismatch", "bad-index", "operation", "autograd"):
            try:
                if mode == "mismatch":
                    group.gather(torch.ones(2 + rank, 2, dtype=torch.float64))
                elif mode == "autograd":
                    group.gather(torch.ones(2, 2, requires_grad=rank == 0))
                elif mode == "bad-index":
                    group.unique_indices(torch.tensor([8 if rank else 0]), num_rows=7)
                elif rank:
                    group.mean(torch.ones(2, 2))
                else:
                    group.gather(torch.ones(2, 2))
            except ValueError as error:
                outcome = {"error": str(error)}
            else:
                raise AssertionError("Mismatched ranks were accepted")
        elif mode in ("exit", "stall"):
            dist.barrier()
            if rank:
                if mode == "exit":
                    os._exit(17)
                time.sleep(4)
                outcome = {"stalled": True}
            else:
                started = time.monotonic()
                try:
                    group.gather(torch.ones(2, 2))
                except RuntimeError as error:
                    outcome = {"error": str(error), "elapsed": time.monotonic() - started}
                else:
                    raise AssertionError("Missing rank did not fail collective")
        else:
            raise AssertionError(mode)
        Path(result).write_text(json.dumps(outcome))
    finally:
        dist.destroy_process_group()


def _launch(tmp_path, mode):
    rendezvous = tmp_path / "rendezvous"
    processes = []
    deadline = time.monotonic() + 45
    try:
        for rank in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, *(["-I"] if sys.flags.isolated else []), str(Path(__file__).resolve()), mode, str(rank),
                 str(rendezvous), str(tmp_path / f"rank{rank}.json")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        for rank, process in enumerate(processes):
            stdout, stderr = process.communicate(timeout=max(.1, deadline - time.monotonic()))
            expected = 17 if mode == "exit" and rank == 1 else 0
            assert process.returncode == expected, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    return [json.loads(path.read_text()) for path in sorted(tmp_path.glob("rank*.json"))]


def test_two_process_global_objectives_gradients_and_double_backward(tmp_path):
    assert _launch(tmp_path, "numerics") == [{"passed": True}, {"passed": True}]


@pytest.mark.parametrize("mode", ["mismatch", "bad-index", "operation", "autograd"])
def test_rank_input_errors_are_collective_and_bounded(tmp_path, mode):
    results = _launch(tmp_path, mode)
    assert len(results) == 2 and results[0]["error"] == results[1]["error"]


@pytest.mark.parametrize("mode", ["exit", "stall"])
def test_lost_or_nonparticipating_rank_fails_within_timeout(tmp_path, mode):
    result = _launch(tmp_path, mode)[0]
    assert result["error"] and result["elapsed"] < 8


def test_collectives_require_explicit_initialized_group():
    with pytest.raises(RuntimeError, match="Initialize"):
        GlooCollectives(2)
    with pytest.raises(ValueError, match="positive integer"):
        GlooCollectives(True)


if __name__ == "__main__":
    _worker(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
