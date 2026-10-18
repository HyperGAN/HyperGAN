"""Data-less constraint target distributions, regularizers, and snapshot metrics.

Provides synthetic simplex/logit target distributions for constraint-driven GANs
where a frozen perceptual/VLM head maps generated images into decision logits,
along with per-option probability monitors, Pseudo-Huber total variation
smoothness, and snapshot evaluation plugins.
"""
from collections.abc import Mapping
import copy
import math


def _validate_target_probs(target_probs):
    if not isinstance(target_probs, (list, tuple)) or not target_probs:
        raise ValueError("target_probs must be a nonempty 1D or 2D list/tuple of positive finite numbers")
    is_2d = isinstance(target_probs[0], (list, tuple))
    raw_rows = target_probs if is_2d else [target_probs]
    if any(not isinstance(row, (list, tuple)) for row in raw_rows):
        raise ValueError("target_probs must be either a 1D sequence or a 2D sequence of rows")
    num_classes = len(raw_rows[0])
    if num_classes < 2:
        raise ValueError("target_probs requires at least K >= 2 options per row")
    normalized_rows = []
    for row in raw_rows:
        if len(row) != num_classes:
            raise ValueError("All target_probs rows must have the same length K >= 2")
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("target_probs entries must be positive finite numbers")
        row_sum = math.fsum(float(value) for value in row)
        if not math.isfinite(row_sum) or row_sum <= 0.0:
            raise ValueError("target_probs row sum must be positive and finite")
        normalized_rows.append([float(value) / row_sum for value in row])
    return normalized_rows, is_2d


class ConstraintTargetData:
    """Stateless target distribution sampler over K-option logits or probabilities.

    Supports both single-mode target specifications (1D sequence of length K >= 2)
    and multi-mode specifications (2D sequence of M >= 1 rows, each of length K >= 2).
    In logits mode, targets are zero-mean centered log-probabilities with optional
    isotropic Gaussian jitter re-centered on the zero-sum hyperplane. In probs mode,
    jittered log-probabilities are mapped onto the simplex via softmax.
    """
    resume_stateless = True

    def __init__(self, target_probs, *, mode="logits", jitter_std=0.15):
        import torch
        normalized_rows, is_2d = _validate_target_probs(target_probs)
        if mode not in ("logits", "probs"):
            raise ValueError("ConstraintTargetData mode must be 'logits' or 'probs'")
        if type(jitter_std) not in (int, float) or not math.isfinite(jitter_std) or jitter_std < 0.0:
            raise ValueError("ConstraintTargetData jitter_std must be a finite nonnegative number")
        self.target_probs = normalized_rows if is_2d else normalized_rows[0]
        self.mode = mode
        self.jitter_std = float(jitter_std)
        self.num_modes = len(normalized_rows)
        self.num_classes = len(normalized_rows[0])
        probs = torch.tensor(normalized_rows, dtype=torch.float32)
        log_probs = probs.log()
        self._centered_logits = log_probs - log_probs.mean(dim=-1, keepdim=True)

    def __call__(self, batch_size, *, generator=None):
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        device = generator.device if generator is not None else torch.device("cpu")
        base = self._centered_logits.to(device=device)
        if self.num_modes > 1:
            indices = torch.randint(0, self.num_modes, (batch_size,), generator=generator, device=device)
            out = base[indices].clone()
        else:
            out = base.expand(batch_size, -1).clone()
        if self.jitter_std > 0.0:
            noise = torch.randn(
                (batch_size, self.num_classes),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            out = out + self.jitter_std * noise
        if self.mode == "logits":
            out = out - out.mean(dim=-1, keepdim=True)
        else:
            out = out.softmax(dim=-1)
        return {"real": out}

    def resume_identity(self):
        return {
            "target_probs": copy.deepcopy(self.target_probs),
            "mode": self.mode,
            "jitter_std": self.jitter_std,
        }


class ConstraintHiddenTargetData:
    """Stateless target distribution sampler over [c, h_perp] in R^hidden_dim.

    Samples K-option centered target logits, transforms the first K coordinates into
    target-vs-rest contrastive log-odds around ``target_index``, and samples the
    remaining ``hidden_dim - K`` orthogonal nullspace coordinates from N(0, null_std^2 I).
    """
    resume_stateless = True

    def __init__(self, target_probs, *, hidden_dim=2560, jitter_std=0.05, null_std=0.05, target_index=0):
        import torch
        normalized_rows, is_2d = _validate_target_probs(target_probs)
        num_classes = len(normalized_rows[0])
        if type(hidden_dim) is not int or hidden_dim <= num_classes:
            raise ValueError(f"ConstraintHiddenTargetData hidden_dim must be an integer > K={num_classes}")
        if type(jitter_std) not in (int, float) or not math.isfinite(jitter_std) or jitter_std < 0.0:
            raise ValueError("ConstraintHiddenTargetData jitter_std must be a finite nonnegative number")
        if type(null_std) not in (int, float) or not math.isfinite(null_std) or null_std < 0.0:
            raise ValueError("ConstraintHiddenTargetData null_std must be a finite nonnegative number")
        if type(target_index) is not int or not 0 <= target_index < num_classes:
            raise ValueError(f"ConstraintHiddenTargetData target_index must be an integer in [0, {num_classes})")
        self.target_probs = normalized_rows if is_2d else normalized_rows[0]
        self.hidden_dim = int(hidden_dim)
        self.jitter_std = float(jitter_std)
        self.null_std = float(null_std)
        self.target_index = int(target_index)
        self.num_modes = len(normalized_rows)
        self.num_classes = num_classes
        probs = torch.tensor(normalized_rows, dtype=torch.float32)
        log_probs = probs.log()
        self._centered_logits = log_probs - log_probs.mean(dim=-1, keepdim=True)

    def __call__(self, batch_size, *, generator=None):
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        device = generator.device if generator is not None else torch.device("cpu")
        base = self._centered_logits.to(device=device)
        if self.num_modes > 1:
            indices = torch.randint(0, self.num_modes, (batch_size,), generator=generator, device=device)
            y = base[indices].clone()
        else:
            y = base.expand(batch_size, -1).clone()
        if self.jitter_std > 0.0:
            noise = torch.randn(
                (batch_size, self.num_classes),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            y = y + self.jitter_std * noise
        y = y - y.mean(dim=-1, keepdim=True)
        zt = y[:, self.target_index : self.target_index + 1]
        z_other = torch.cat([y[:, :self.target_index], y[:, self.target_index + 1:]], dim=-1)
        c_target = zt - torch.logsumexp(z_other, dim=-1, keepdim=True)
        c_other = z_other - zt
        c = torch.cat([c_other[:, :self.target_index], c_target, c_other[:, self.target_index:]], dim=-1)
        null_dim = self.hidden_dim - self.num_classes
        if self.null_std > 0.0:
            h_perp = self.null_std * torch.randn(
                (batch_size, null_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            h_perp = torch.zeros((batch_size, null_dim), device=device, dtype=torch.float32)
        return {"real": torch.cat([c, h_perp], dim=-1)}

    def resume_identity(self):
        return {
            "target_probs": copy.deepcopy(self.target_probs),
            "hidden_dim": self.hidden_dim,
            "jitter_std": self.jitter_std,
            "null_std": self.null_std,
            "target_index": self.target_index,
        }


class OptionProbability:
    """Stateless objective / monitor computing mean probability of option ``index``.

    When configured in ``[[objectives]]`` with ``detach = ["logits"]`` and
    ``weight = 1.0``, HyperGAN records ``loss/objectives/<id>`` in ``metrics.jsonl``
    while contributing zero gradient to the generator.
    """
    resume_stateless = True

    def __init__(self, index=0, *, input_mode="logits", num_classes=None, target_index=0):
        if type(index) is not int or index < 0:
            raise ValueError("OptionProbability index must be a nonnegative integer")
        if input_mode not in ("logits", "probs", "hidden_state"):
            raise ValueError("OptionProbability input_mode must be 'logits', 'probs', or 'hidden_state'")
        if input_mode == "hidden_state":
            if type(num_classes) is not int or num_classes < 2:
                raise ValueError("OptionProbability with input_mode='hidden_state' requires integer num_classes >= 2")
            if type(target_index) is not int or not 0 <= target_index < num_classes:
                raise ValueError("OptionProbability target_index must be in [0, num_classes)")
            if index >= num_classes:
                raise ValueError(f"OptionProbability index {index} is out of bounds for num_classes={num_classes}")
        self.index = index
        self.input_mode = input_mode
        self.num_classes = num_classes
        self.target_index = target_index

    def __call__(self, logits):
        import torch
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or not logits.is_floating_point() or logits.shape[0] < 1
                or logits.shape[1] < 2):
            raise ValueError("OptionProbability requires a nonempty 2D floating-point tensor [B, K] with K >= 2")
        if self.input_mode == "hidden_state":
            if logits.shape[1] < self.num_classes:
                raise ValueError(
                    f"OptionProbability hidden_state tensor width {logits.shape[1]} < num_classes={self.num_classes}"
                )
            c = logits[:, :self.num_classes]
            rel = torch.cat(
                [c[:, :self.target_index], torch.zeros_like(c[:, :1]), c[:, self.target_index + 1 : self.num_classes]],
                dim=-1,
            )
            probs = rel.softmax(dim=-1)
            return probs[:, self.index].mean()
        if self.index >= logits.shape[1]:
            raise ValueError(f"OptionProbability index {self.index} is out of bounds for K={logits.shape[1]}")
        probs = logits.softmax(dim=-1) if self.input_mode == "logits" else logits
        return probs[:, self.index].mean()


class TotalVariationSmoothness:
    """Stateless Pseudo-Huber (Charbonnier) total variation spatial smoothness objective."""
    resume_stateless = True

    def __init__(self, delta=0.05):
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("TotalVariationSmoothness delta must be a positive finite number")
        self.delta = float(delta)

    def __call__(self, image):
        import torch
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or not image.is_floating_point() or image.numel() == 0
                or min(image.shape[-2:]) < 2):
            raise ValueError(
                "TotalVariationSmoothness requires a nonempty 4D floating-point NCHW tensor with height/width >= 2"
            )
        dh = image[:, :, 1:, :] - image[:, :, :-1, :]
        dw = image[:, :, :, 1:] - image[:, :, :, :-1]
        delta_sq = self.delta ** 2
        return ((dh.square() + delta_sq).sqrt() - self.delta).mean() + (
            (dw.square() + delta_sq).sqrt() - self.delta
        ).mean()


def _extract_probs(value, input_mode):
    import torch
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or not value.is_floating_point() or value.shape[0] < 1
            or value.shape[1] < 2 or not bool(torch.isfinite(value).all())):
        raise ValueError("ConstraintSatisfactionMetric requires finite nonempty 2D floating-point tensors [B, K] with K >= 2")
    detached = value.detach().to(device="cpu", dtype=torch.float64)
    if input_mode == "logits":
        return detached.softmax(dim=-1)
    if bool((detached < -1e-6).any() or (detached > 1.0 + 1e-6).any()):
        raise ValueError("ConstraintSatisfactionMetric with input_mode='probs' requires values in [0, 1]")
    row_sums = detached.sum(dim=-1, keepdim=True)
    if bool(((row_sums - 1.0).abs() > 1e-3).any()):
        raise ValueError("ConstraintSatisfactionMetric with input_mode='probs' requires rows summing to 1")
    clamped = detached.clamp_min(0.0)
    return clamped / clamped.sum(dim=-1, keepdim=True)


class ConstraintSatisfactionMetric:
    """Snapshot evaluation metric for constraint satisfaction and mode coverage."""

    _STATISTICS = ("target_prob", "unknown_prob", "target_accuracy", "marginal_entropy")

    def __init__(self, statistic="target_prob", target_index=0, input_mode="logits"):
        if statistic not in self._STATISTICS:
            raise ValueError(
                f"ConstraintSatisfactionMetric statistic must be one of {self._STATISTICS}"
            )
        if type(target_index) is not int or target_index < 0:
            raise ValueError("ConstraintSatisfactionMetric target_index must be a nonnegative integer")
        if input_mode not in ("logits", "probs"):
            raise ValueError("ConstraintSatisfactionMetric input_mode must be 'logits' or 'probs'")
        self.statistic = statistic
        self.target_index = target_index
        self.input_mode = input_mode

    def describe(self):
        if self.statistic == "target_prob":
            return {
                "kind": "scalar",
                "label": f"Mean target option probability (index {self.target_index})",
                "unit": "probability",
                "direction": "maximize",
                "description": (
                    f"Mean predicted probability of target option {self.target_index} "
                    "across evaluated samples."
                ),
            }
        if self.statistic == "unknown_prob":
            return {
                "kind": "scalar",
                "label": "Mean unknown/reject option probability",
                "unit": "probability",
                "direction": "minimize",
                "description": (
                    "Mean predicted probability of the final (unknown/reject) option "
                    "across evaluated samples."
                ),
            }
        if self.statistic == "target_accuracy":
            return {
                "kind": "scalar",
                "label": f"Target option top-1 accuracy (index {self.target_index})",
                "unit": "ratio",
                "direction": "maximize",
                "description": (
                    f"Fraction of evaluated samples where option {self.target_index} "
                    "has the highest predicted probability."
                ),
            }
        return {
            "kind": "scalar",
            "label": "Marginal option distribution entropy",
            "unit": "nats",
            "direction": "none",
            "description": (
                "Shannon entropy (in nats) of the mean predicted option distribution "
                "across evaluated samples; monitors multi-mode balance."
            ),
        }

    def evaluate(self, *, batches, context):
        prob_sum = None
        total_count = 0
        correct_count = 0
        num_classes = None
        for batch in batches:
            if not isinstance(batch, Mapping) or not batch:
                raise ValueError("ConstraintSatisfactionMetric batches must be nonempty mappings")
            if "generated" in batch:
                raw = batch["generated"]
            elif "logits" in batch:
                raw = batch["logits"]
            elif len(batch) == 1:
                raw = next(iter(batch.values()))
            else:
                raise ValueError("ConstraintSatisfactionMetric batch must contain 'generated' or 'logits'")
            probs = _extract_probs(raw, self.input_mode)
            if num_classes is None:
                num_classes = probs.shape[1]
                if self.target_index >= num_classes:
                    raise ValueError(
                        f"ConstraintSatisfactionMetric target_index {self.target_index} "
                        f"is out of bounds for K={num_classes}"
                    )
            elif probs.shape[1] != num_classes:
                raise ValueError("ConstraintSatisfactionMetric option count K changed between batches")
            batch_sum = probs.sum(dim=0)
            prob_sum = batch_sum if prob_sum is None else prob_sum + batch_sum
            correct_count += int((probs.argmax(dim=-1) == self.target_index).sum().item())
            total_count += int(probs.shape[0])
        if total_count == 0 or prob_sum is None:
            raise ValueError("ConstraintSatisfactionMetric requires nonempty evaluation samples")
        mean_probs = prob_sum / total_count
        if self.statistic == "target_prob":
            return float(mean_probs[self.target_index].item())
        if self.statistic == "unknown_prob":
            return float(mean_probs[-1].item())
        if self.statistic == "target_accuracy":
            return float(correct_count / total_count)
        return float(-sum(p * math.log(p) for p in mean_probs.tolist() if p > 0.0))
