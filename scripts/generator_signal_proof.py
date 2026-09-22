"""Deterministic counterexamples for generator-signal diagnostics; no training runs.

Run: python3 scripts/generator_signal_proof.py --output reports/generator-signal-proof-2026-09-21.json
Requires torch. Uses explicit CPU float64 tensors, no datasets, downloads or RNG.
This is a research calculation, not a metric plugin or trainer implementation.
"""
import argparse
import json
import math
import platform
from pathlib import Path

import torch


def tensor(value, *, grad=False):
    return torch.tensor(value, dtype=torch.float64, device="cpu", requires_grad=grad)


def rms(value):
    return float(value.detach().square().mean().sqrt())


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"{actual} != {expected}")


def cosine(a, b):
    a, b = a.detach().flatten(), b.detach().flatten()
    denominator = float(a.norm() * b.norm())
    return None if denominator == 0 else float(a.dot(b)) / denominator


def depth_case(gain, depth=16):
    z = tensor([0.25, 0.5, 0.75, 1.0])
    weights = [tensor(gain, grad=True) for _ in range(depth)]
    biases = [tensor(0.0, grad=True) for _ in range(depth)]
    h, activations = z, []
    for w, b in zip(weights, biases):
        h = h * w + b
        activations.append(h)
    gradients = torch.autograd.grad(-h.mean(), (*activations, *weights, *biases))
    profile = [rms(len(z) * g) for g in gradients[:depth]]
    close(profile[0], gain ** (depth - 1))
    close(profile[-1], 1.0)
    close(abs(float(gradients[2 * depth])), gain ** (depth - 1))
    close(abs(float(gradients[-1])), 1.0)
    return {
        "gain": gain, "depth": depth,
        "activation_gradient_rms_batch_normalized": profile,
        "first_to_output_gradient_ratio": profile[0] / profile[-1],
        "first_weight_gradient_abs": abs(float(gradients[depth])),
        "last_weight_gradient_abs": abs(float(gradients[2 * depth - 1])),
        "first_bias_gradient_abs": abs(float(gradients[2 * depth])),
        "last_bias_gradient_abs": abs(float(gradients[-1])),
    }


def saturation_case():
    pre = tensor([4.0, 6.0, 8.0, 10.0], grad=True)
    x = pre.tanh()
    q, before = torch.autograd.grad(-x.mean(), (x, pre))
    expected = 1 - x.detach().square()
    close(rms(len(x) * before), rms(expected))
    close(rms(len(x) * q), 1.0)
    return {"output_gradient_rms": rms(len(x) * q),
            "pre_tanh_gradient_rms": rms(len(x) * before),
            "mean_tanh_derivative": float(expected.mean())}


def reparameterization_case(scale):
    z = tensor([0.25, 0.5, 0.75, 1.0])
    a, b = tensor(scale, grad=True), tensor(1 / scale, grad=True)
    h = a * z
    x = b * h
    qh, ga, gb = torch.autograd.grad(-x.mean(), (h, a, b))
    close(rms(x - z), 0.0)
    close(rms(len(z) * qh), 1 / scale)
    sensitivity = rms(h) * rms(len(z) * qh)
    close(sensitivity, rms(z))
    return {"hidden_scale": scale, "output": x.detach().tolist(),
            "hidden_gradient_rms": rms(len(z) * qh),
            "activation_rms_times_gradient_rms": sensitivity,
            "parameter_gradient_norm": float(torch.stack((ga, gb)).norm())}


def translation_case(critic_sign, step_size=0.1, loss_scale=1.0):
    # Equal-weight 1-D atoms: sorting gives exact distributional W2 squared.
    z = tensor([-1.0, -0.5, 0.5, 1.0])
    reference = z + 2
    theta = tensor(0.0, grad=True)
    x = theta + z
    loss = -loss_scale * critic_sign * x.mean()
    q, g = torch.autograd.grad(loss, (x, theta))
    delta = -step_size * g
    before = (x.detach().sort().values - reference.sort().values).square().mean()
    after = ((x.detach() + delta).sort().values - reference.sort().values).square().mean()
    close(float(before), 4.0)
    close(float(after), (float(delta) - 2) ** 2)
    # Each sample has the same parameter gradient, including for the bad critic.
    sample_gradients = tensor([-loss_scale * critic_sign] * len(z))
    variance = float(sample_gradients.var(unbiased=True))
    close(variance, 0.0)
    return {"critic_sign": critic_sign, "loss_scale": loss_scale,
            "step_size": step_size, "output_gradient_rms": rms(len(z) * q),
            "parameter_gradient_abs": abs(float(g)),
            "per_sample_gradient_variance": variance,
            "batch_gradient_cosine": 1.0,
            "update": float(delta), "oracle_update_cosine": cosine(tensor(4.0), delta),
            "w2_squared_before": float(before), "w2_squared_after": float(after),
            "oracle_progress": float(before - after)}


def collapse_case():
    z = tensor([-1.0, -0.5, 0.5, 1.0])
    scale = tensor(1.0, grad=True)
    x = scale * z
    # A stale/misspecified critic D(x)=-x^2/2 rewards contraction.
    loss = x.square().mean() / 2
    g, = torch.autograd.grad(loss, (scale,))
    delta = -0.1 * g
    after = (scale.detach() + delta) * z
    quality_after = (after.sort().values - z.sort().values).square().mean()
    close(float(g), 0.625)
    close(float(quality_after), 0.625 * 0.0625 ** 2)
    # Both disjoint symmetric batches have gradients pointing the same way.
    return {"parameter_gradient": float(g), "batch_gradient_cosine": 1.0,
            "critic_loss_before": float(loss.detach()),
            "critic_loss_after": float(after.square().mean() / 2),
            "w2_squared_before": 0.0, "w2_squared_after": float(quality_after),
            "generated_variance_before": float(z.var(unbiased=False)),
            "generated_variance_after": float(after.var(unbiased=False))}


def noise_estimator_case():
    # Enumerate every IID pair from a two-element gradient distribution.
    # E[g]=1, Var[g]=1. Negative per-probe estimates must not be silently clamped.
    values = (0.0, 2.0)
    estimates = []
    for a in values:
        for b in values:
            mean = (a + b) / 2
            sample_variance = ((a - mean) ** 2 + (b - mean) ** 2)
            signal = mean ** 2 - sample_variance / 2
            estimates.append({"draw": [a, b], "signal_squared": signal,
                              "batch_covariance_trace": sample_variance})
    close(sum(e["signal_squared"] for e in estimates) / 4, 1.0)
    close(sum(e["batch_covariance_trace"] for e in estimates) / 4, 1.0)
    # Add an opposing pair to exercise the unresolved-signal status.
    opposing_signal = 0.0 - 2.0 / 2
    close(opposing_signal, -1.0)
    return {"iid_pair_enumeration": estimates, "expected_signal_squared": 1.0,
            "expected_batch_covariance_trace": 1.0,
            "opposing_pair_signal_squared": opposing_signal,
            "opposing_pair_snr_status": "unresolved_signal"}


def blocked_case():
    theta = tensor([1.0, 2.0], grad=True)
    mask = tensor([-1.0, -2.0], grad=True)
    x = theta * mask.relu()
    q, g = torch.autograd.grad(-x.mean(), (x, theta))
    close(rms(len(x) * q), 1.0)
    close(rms(g), 0.0)
    if cosine(-g, g) is not None:
        raise AssertionError("Zero-gradient cosine must be unavailable")
    return {"output_gradient_rms": rms(len(x) * q), "parameter_gradient_rms": rms(g),
            "gradient_cosine": None, "status": "zero_gradient"}


def initialization_drift_case():
    # A possible parameter path, not a simulated training trajectory.
    # A scalar chain has exact gain product(w); no singular-vector ambiguity.
    depth = 64
    rows = []
    for gain in (1.0, 0.99, 0.95, 1.05):
        start = tensor(1.0, grad=True)
        end = start
        for _ in range(depth):
            end = end * gain
        transmitted, = torch.autograd.grad(-end, (start,))
        close(abs(float(transmitted)), gain ** depth)
        rows.append({"per_layer_gain": gain, "depth": depth,
                     "input_to_output_gradient_ratio": abs(float(transmitted))})
    return {"kind": "constructed_parameter_path_not_training", "states": rows}


def critic_direction_drift_case():
    # Identical G Jacobian, different unit output cotangents.
    rows = []
    for direction in ([1.0, 0.0], [0.0, 1.0]):
        theta = tensor([1.0, 1.0], grad=True)
        x = theta * tensor([1.0, 0.001])
        loss = -(x * tensor(direction)).sum()
        q, g = torch.autograd.grad(loss, (x, theta))
        rows.append({"critic_direction": direction, "output_gradient_norm": float(q.norm()),
                     "parameter_gradient_norm": float(g.norm())})
    close(rows[0]["output_gradient_norm"], rows[1]["output_gradient_norm"])
    close(rows[0]["parameter_gradient_norm"] / rows[1]["parameter_gradient_norm"], 1000.0)
    return {"generator_jacobian_diagonal": [1.0, 0.001], "states": rows}


def frozen_module_case():
    # Explicit stand-in for imported state, not a real pretrained model.
    class FrozenTransform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(tensor([2.0, 4.0]), requires_grad=False)
            self.register_buffer("offset", tensor([0.25, -0.25]))

        def forward(self, value):
            return self.weight * value + self.offset

    frozen = FrozenTransform().eval()
    before = {key: value.clone() for key, value in frozen.state_dict().items()}
    owned_input = tensor([0.5, 1.0], grad=True)
    (-frozen(owned_input).mean()).backward()
    close(float(owned_input.grad.norm()), math.sqrt(5.0))
    unchanged = all(torch.equal(before[key], value) for key, value in frozen.state_dict().items())
    if not unchanged or frozen.weight.grad is not None:
        raise AssertionError("Frozen module state or parameter gradients changed")
    return {"input_gradient_norm": float(owned_input.grad.norm()),
            "parameter_gradient_present": frozen.weight.grad is not None,
            "parameters_and_buffers_unchanged": unchanged,
            "calibratable_pretrained_parameters": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = {
        "depth_identity": depth_case(1.0), "depth_attenuation": depth_case(0.5),
        "output_saturation": saturation_case(),
        "coordinates_original": reparameterization_case(1.0),
        "coordinates_rescaled": reparameterization_case(1000.0),
        "critic_helpful": translation_case(1.0),
        "critic_harmful": translation_case(-1.0),
        "helpful_direction_overshoot": translation_case(1.0, step_size=5.0),
        "loss_rescaled_same_update": translation_case(1.0, step_size=0.0001, loss_scale=1000.0),
        "collapse_despite_agreement": collapse_case(),
        "noise_estimator": noise_estimator_case(), "blocked_generator": blocked_case(),
        "initialization_drift": initialization_drift_case(),
        "critic_direction_drift": critic_direction_drift_case(),
        "frozen_module_transmits_signal": frozen_module_case(),
    }
    close(cases["critic_helpful"]["oracle_progress"], 0.39)
    close(cases["critic_harmful"]["oracle_progress"], -0.41)
    close(cases["helpful_direction_overshoot"]["oracle_progress"], -5.0)
    close(cases["critic_helpful"]["update"], cases["loss_rescaled_same_update"]["update"])
    close(cases["coordinates_original"]["activation_rms_times_gradient_rms"],
          cases["coordinates_rescaled"]["activation_rms_times_gradient_rms"])
    report = {"schema_version": 1, "kind": "deterministic-generator-signal-counterexamples",
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "device": "cpu", "dtype": "float64"},
              "uses_rng": False, "training_runs": 0, "checks": "passed", "cases": cases}
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"Analytic checks passed; wrote {len(cases)} cases to {args.output}")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
