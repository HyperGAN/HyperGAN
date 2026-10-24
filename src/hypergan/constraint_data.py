"""Data-less constraint target distributions, regularizers, and snapshot metrics.

Provides synthetic simplex/logit target distributions for constraint-driven GANs
where a frozen perceptual/VLM head maps generated images into decision logits,
along with per-option probability monitors, Pseudo-Huber total variation
smoothness, and snapshot evaluation plugins.
"""
from collections.abc import Mapping
import copy
import json
import math
import warnings
from hypergan.pretrained_providers import (
    _imajev_bank_signature,
    _symmetric_log_odds,
    get_cached_or_deterministic_state_embed,
)

try:
    import torch
    from torch import nn
    _ModuleBase = nn.Module
except ImportError:
    _ModuleBase = object


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


class JointMultiQueryConstraintData:
    """Stateless joint multi-query target distribution sampler over R^(Q * K_max).

    Configured with Q semantic queries and their target options/modes (e.g. Mode 0 for
    apple/subject, Modes 0..2 for color, bites, sitting_on, style, stem_leaf).
    On each call, samples independent target mode combinations across all Q queries
    (spanning the full Cartesian product of attributes, e.g. 243 modes for 6 queries),
    and constructs the corresponding joint target soft-capped log-odds vector
    U^target in R^(B x (Q * K_max)) matching the calibrated temperature (T=1.30515697)
    and soft-capping threshold (tau=3.0) of the frozen VLM critic.

    Returns dict {"real": U_target, "modes": mode_indices}.
    """
    resume_stateless = True

    def __init__(
        self,
        queries,
        *,
        tau=3.0,
        temperature=1.30515697,
        jitter_std=0.0,
        target_prob=0.992,
    ):
        import torch
        if not isinstance(queries, (list, tuple)) or not 1 <= len(queries) <= 64:
            raise ValueError("JointMultiQueryConstraintData queries must be a sequence of 1..64 query mappings")
        if isinstance(tau, bool) or type(tau) not in (int, float) or not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("JointMultiQueryConstraintData tau must be a positive finite number")
        if isinstance(temperature, bool) or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("JointMultiQueryConstraintData temperature must be a positive finite number")
        if isinstance(jitter_std, bool) or type(jitter_std) not in (int, float) or not math.isfinite(jitter_std) or jitter_std < 0.0:
            raise ValueError("JointMultiQueryConstraintData jitter_std must be a finite nonnegative number")
        if isinstance(target_prob, bool) or type(target_prob) not in (int, float) or not math.isfinite(target_prob) or not (0.5 < target_prob < 1.0):
            raise ValueError("JointMultiQueryConstraintData target_prob must be a float in (0.5, 1.0)")

        self.tau = float(tau)
        self.temperature = float(temperature)
        self.jitter_std = float(jitter_std)
        self.target_prob = float(target_prob)
        self.num_queries = len(queries)

        normalized_queries = []
        mode_probs_list = []
        num_classes_list = []
        num_modes_list = []

        for idx, q in enumerate(queries):
            if not isinstance(q, Mapping):
                raise ValueError(f"JointMultiQueryConstraintData queries[{idx}] must be a mapping")

            question = q.get("question")
            if question is not None and (not isinstance(question, str) or not question.strip()):
                raise ValueError(f"JointMultiQueryConstraintData queries[{idx}].question must be a non-empty string when set")

            options = q.get("options")
            if options is not None:
                if not isinstance(options, (list, tuple)) or not 1 <= len(options) <= 63:
                    raise ValueError(f"JointMultiQueryConstraintData queries[{idx}].options must contain 1..63 strings")
                clean_options = []
                seen = set()
                for opt in options:
                    if not isinstance(opt, str) or not opt.strip():
                        raise ValueError(f"JointMultiQueryConstraintData queries[{idx}].options must be non-empty strings")
                    if opt in seen:
                        raise ValueError(f"JointMultiQueryConstraintData queries[{idx}].options must be unique")
                    seen.add(opt)
                    clean_options.append(opt)
            else:
                clean_options = None

            if "target_probs" in q:
                normalized_rows, is_2d = _validate_target_probs(q["target_probs"])
                num_classes = len(normalized_rows[0])
                if clean_options is not None:
                    if num_classes not in (len(clean_options), len(clean_options) + 1):
                        raise ValueError(
                            f"JointMultiQueryConstraintData queries[{idx}].target_probs length {num_classes} "
                            f"must match options length {len(clean_options)} or {len(clean_options) + 1}"
                        )
                raw_target_probs = normalized_rows if is_2d else normalized_rows[0]
                target_modes = q.get("target_modes")
                if target_modes is not None:
                    if not isinstance(target_modes, (list, tuple)):
                        target_modes = [target_modes]
                    if len(target_modes) != len(normalized_rows):
                        raise ValueError(
                            f"JointMultiQueryConstraintData queries[{idx}].target_modes length "
                            f"must match target_probs rows ({len(normalized_rows)})"
                        )
                else:
                    target_modes = [int(torch.tensor(row).argmax().item()) for row in normalized_rows]
            elif "target_modes" in q or "target_options" in q or "target_index" in q:
                raw_modes = q.get("target_modes", q.get("target_options", q.get("target_index")))
                if isinstance(raw_modes, int) and not isinstance(raw_modes, bool):
                    target_modes = [raw_modes]
                elif isinstance(raw_modes, (list, tuple)) and len(raw_modes) > 0 and all(type(m) is int and not isinstance(m, bool) for m in raw_modes):
                    target_modes = [int(m) for m in raw_modes]
                else:
                    raise ValueError(
                        f"JointMultiQueryConstraintData queries[{idx}] target_modes must be an int or sequence of ints"
                    )

                if clean_options is not None:
                    num_classes = len(clean_options) + 1
                else:
                    num_classes = max(target_modes) + 2

                if any(m < 0 or m >= num_classes for m in target_modes):
                    raise ValueError(
                        f"JointMultiQueryConstraintData queries[{idx}] target_modes entries must be in [0, {num_classes})"
                    )

                normalized_rows = []
                other_prob = (1.0 - self.target_prob) / float(num_classes - 1)
                for m in target_modes:
                    row = [other_prob] * num_classes
                    row[m] = self.target_prob
                    normalized_rows.append(row)
                raw_target_probs = normalized_rows if len(normalized_rows) > 1 else normalized_rows[0]
            else:
                raise ValueError(
                    f"JointMultiQueryConstraintData queries[{idx}] requires 'target_probs', 'target_modes', or 'target_index'"
                )

            num_modes = len(normalized_rows)
            num_classes_list.append(num_classes)
            num_modes_list.append(num_modes)

            entry = {
                "target_probs": raw_target_probs,
                "target_modes": list(target_modes),
            }
            if question is not None:
                entry["question"] = question
            if clean_options is not None:
                entry["options"] = clean_options
            if "weight" in q:
                entry["weight"] = float(q["weight"])
            if "state" in q:
                entry["state"] = copy.deepcopy(q["state"])
            if "aspect" in q:
                entry["aspect"] = str(q["aspect"])
            normalized_queries.append(entry)

            probs_tensor = torch.tensor(normalized_rows, dtype=torch.float32)
            mode_probs_list.append(probs_tensor)

        self.queries = normalized_queries
        self._num_classes = num_classes_list
        self._num_modes = num_modes_list
        self.k_max = max(num_classes_list)

        self._centered_logits = []
        self._mode_targets = []

        for q_idx in range(self.num_queries):
            probs = mode_probs_list[q_idx]
            log_p = probs.log()
            scaled_logits = log_p / self.temperature
            centered = scaled_logits - scaled_logits.mean(dim=-1, keepdim=True)
            self._centered_logits.append(centered)

            u = _symmetric_log_odds(centered)
            u_capped = self.tau * torch.tanh(u / self.tau)

            K_q = self._num_classes[q_idx]
            if K_q < self.k_max:
                pad = torch.full(
                    (len(probs), self.k_max - K_q),
                    -self.tau,
                    dtype=torch.float32,
                )
                u_capped = torch.cat([u_capped, pad], dim=-1)

            self._mode_targets.append(u_capped)

    def __call__(self, batch_size, *, generator=None):
        import torch
        if type(batch_size) is not int or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        device = generator.device if generator is not None else torch.device("cpu")

        sampled_modes = []
        sampled_u = []

        for q_idx in range(self.num_queries):
            num_modes = self._num_modes[q_idx]
            if num_modes > 1:
                m_indices = torch.randint(0, num_modes, (batch_size,), generator=generator, device=device)
            else:
                m_indices = torch.zeros((batch_size,), dtype=torch.long, device=device)
            sampled_modes.append(m_indices)

            if self.jitter_std > 0.0:
                base = self._centered_logits[q_idx].to(device=device)[m_indices]
                K_q = self._num_classes[q_idx]
                noise = torch.randn(
                    (batch_size, K_q),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                y = base + self.jitter_std * noise
                y = y - y.mean(dim=-1, keepdim=True)
                u = _symmetric_log_odds(y)
                u_capped = self.tau * torch.tanh(u / self.tau)
                if K_q < self.k_max:
                    pad = torch.full(
                        (batch_size, self.k_max - K_q),
                        -self.tau,
                        device=device,
                        dtype=torch.float32,
                    )
                    u_capped = torch.cat([u_capped, pad], dim=-1)
                sampled_u.append(u_capped)
            else:
                targets = self._mode_targets[q_idx].to(device=device)
                sampled_u.append(targets[m_indices])

        U_target = torch.cat(sampled_u, dim=-1)
        mode_indices = torch.stack(sampled_modes, dim=-1)

        return {"real": U_target, "modes": mode_indices}

    def resume_identity(self):
        return {
            "queries": copy.deepcopy(self.queries),
            "tau": self.tau,
            "temperature": self.temperature,
            "jitter_std": self.jitter_std,
            "target_prob": self.target_prob,
        }


class ConstraintHiddenTargetData:
    """[DEPRECATED] Stateless target distribution sampler over [c, h_perp] in R^hidden_dim.

    Deprecated: The nullspace embedding h_perp causes batch mode collapse (null_std=0)
    or unconstrained off-manifold noise excitation (null_std>0).
    Use JointMultiQueryConstraintData instead.
    """
    resume_stateless = True

    def __init__(self, target_probs, *, hidden_dim=2560, jitter_std=0.05, null_std=0.05, target_index=0):
        warnings.warn(
            "ConstraintHiddenTargetData is deprecated and will be removed in a future release. "
            "Use JointMultiQueryConstraintData for joint multi-query semantic constraints.",
            DeprecationWarning,
            stacklevel=2,
        )
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


class MultiQueryConstraintData:
    """Stateless multi-query target distribution sampler over [s_q, u_q] in R^(state_embed_dim + K).

    Samples one active query per training step from a bank of state/question/action queries,
    pushes its index onto the active query synchronization queue for the ImaJev provider,
    and returns concatenated `[s_q, u_q]` real targets where `s_q` is the cached/projected
    prompt state embedding and `u_q` is the symmetric one-vs-rest log-odds vector.
    """
    resume_stateless = True

    def __init__(
        self,
        queries,
        *,
        state_embed_dim=16,
        jitter_std=0.05,
        base_model_path=None,
        adapter_path=None,
    ):
        import torch
        warnings.warn(
            "MultiQueryConstraintData is deprecated. Use JointMultiQueryConstraintData instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(queries, (list, tuple)) or len(queries) < 2 or len(queries) > 64:
            raise ValueError("MultiQueryConstraintData queries must be a sequence of 2..64 query mappings")
        if type(state_embed_dim) is not int or not 4 <= state_embed_dim <= 256:
            raise ValueError("MultiQueryConstraintData state_embed_dim must be an integer in [4, 256]")
        if type(jitter_std) not in (int, float) or not math.isfinite(jitter_std) or jitter_std < 0.0:
            raise ValueError("MultiQueryConstraintData jitter_std must be a finite nonnegative number")
        for attr_name, attr_val in (("base_model_path", base_model_path), ("adapter_path", adapter_path)):
            if attr_val is not None and (not isinstance(attr_val, str) or not attr_val.strip()):
                raise ValueError(f"MultiQueryConstraintData {attr_name} must be a non-empty string when set")

        normalized_queries = []
        centered_logits_bank = []
        num_modes_bank = []
        weights = []
        expected_options_len = None

        for idx, q in enumerate(queries):
            if not isinstance(q, Mapping):
                raise ValueError(f"MultiQueryConstraintData queries[{idx}] must be a mapping")
            question = q.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"MultiQueryConstraintData queries[{idx}].question must be a non-empty string")
            options = q.get("options")
            if not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 63:
                raise ValueError(f"MultiQueryConstraintData queries[{idx}].options must contain 2..63 strings")
            if expected_options_len is None:
                expected_options_len = len(options)
            elif len(options) != expected_options_len:
                raise ValueError(
                    "MultiQueryConstraintData requires all queries to have the same number of options"
                )
            seen = set()
            clean_options = []
            for opt in options:
                if not isinstance(opt, str) or not opt.strip():
                    raise ValueError(f"MultiQueryConstraintData queries[{idx}].options must be non-empty strings")
                if opt == "__unknown__":
                    raise ValueError(
                        f"MultiQueryConstraintData queries[{idx}].options must not include '__unknown__'"
                    )
                if opt in seen:
                    raise ValueError(f"MultiQueryConstraintData queries[{idx}].options must be unique")
                seen.add(opt)
                clean_options.append(opt)
            if "target_probs" not in q:
                raise ValueError(f"MultiQueryConstraintData queries[{idx}] requires 'target_probs'")
            normalized_rows, is_2d = _validate_target_probs(q["target_probs"])
            num_classes = len(normalized_rows[0])
            if num_classes != expected_options_len + 1:
                raise ValueError(
                    f"MultiQueryConstraintData queries[{idx}].target_probs length {num_classes} "
                    f"must equal len(options) + 1 ({expected_options_len + 1})"
                )
            state = q.get("state")
            if state is not None:
                if isinstance(state, str):
                    if not state.strip():
                        raise ValueError(
                            f"MultiQueryConstraintData queries[{idx}].state string must not be empty"
                        )
                    clean_state = state
                elif isinstance(state, Mapping):
                    try:
                        json.dumps(dict(state), sort_keys=True, allow_nan=False, ensure_ascii=False)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"MultiQueryConstraintData queries[{idx}].state mapping must be JSON-serializable"
                        ) from exc
                    clean_state = copy.deepcopy(dict(state))
                else:
                    raise ValueError(
                        f"MultiQueryConstraintData queries[{idx}].state must be a string or mapping"
                    )
            else:
                clean_state = None
            weight = q.get("weight", 1.0)
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"MultiQueryConstraintData queries[{idx}].weight must be a positive finite number")
            weights.append(float(weight))
            probs = torch.tensor(normalized_rows, dtype=torch.float32)
            log_probs = probs.log()
            centered_logits_bank.append(log_probs - log_probs.mean(dim=-1, keepdim=True))
            num_modes_bank.append(len(normalized_rows))
            entry = {
                "question": question,
                "options": clean_options,
                "target_probs": normalized_rows if is_2d else normalized_rows[0],
                "weight": float(weight),
            }
            if clean_state is not None:
                entry["state"] = clean_state
            normalized_queries.append(entry)

        self.queries = normalized_queries
        self._raw_queries = normalized_queries
        self.state_embed_dim = int(state_embed_dim)
        self.jitter_std = float(jitter_std)
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.num_queries = len(normalized_queries)
        self.num_classes = expected_options_len + 1
        self._centered_logits_bank = centered_logits_bank
        self._num_modes_bank = num_modes_bank
        weight_sum = math.fsum(weights)
        self._query_weights = torch.tensor([w / weight_sum for w in weights], dtype=torch.float32)
        self._bank_signature = _imajev_bank_signature(normalized_queries)

    def __call__(self, batch_size, *, generator=None):
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        device = generator.device if generator is not None else torch.device("cpu")
        weights = self._query_weights.to(device=device)
        q_idx = int(torch.multinomial(weights, 1, generator=generator).item())

        base = self._centered_logits_bank[q_idx].to(device=device)
        num_modes = self._num_modes_bank[q_idx]
        if num_modes > 1:
            indices = torch.randint(0, num_modes, (batch_size,), generator=generator, device=device)
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
        u = _symmetric_log_odds(y)
        state_embed = get_cached_or_deterministic_state_embed(
            self._raw_queries[q_idx],
            self.state_embed_dim,
            base_model_path=self.base_model_path,
            adapter_path=self.adapter_path,
            device=device,
        ).to(device=device, dtype=torch.float32)
        s_batch = state_embed.expand(batch_size, -1)
        return {"real": torch.cat([s_batch, u], dim=-1)}

    def resume_identity(self):
        return {
            "queries": copy.deepcopy(self.queries),
            "state_embed_dim": self.state_embed_dim,
            "jitter_std": self.jitter_std,
            "base_model_path": self.base_model_path,
            "adapter_path": self.adapter_path,
        }


class OptionProbability:
    """Stateless objective / monitor computing mean probability of option ``index``.

    When configured in ``[[objectives]]`` with ``detach = ["logits"]`` and
    ``weight = 1.0``, HyperGAN records ``loss/objectives/<id>`` in ``metrics.jsonl``
    while contributing zero gradient to the generator.
    """
    resume_stateless = True

    def __init__(self, index=0, *, input_mode="logits", num_classes=None, target_index=0, query_index=0):
        if type(index) is int:
            if index < 0:
                raise ValueError("OptionProbability index must be a nonnegative integer")
            indices = (index,)
        elif isinstance(index, (list, tuple)) and len(index) > 0:
            if any(type(i) is not int or i < 0 for i in index):
                raise ValueError("OptionProbability index sequence must contain nonnegative integers")
            if len(set(index)) != len(index):
                raise ValueError("OptionProbability index sequence must not contain duplicates")
            indices = tuple(index)
        else:
            raise ValueError(
                "OptionProbability index must be a nonnegative integer or a nonempty sequence of nonnegative integers"
            )
        if input_mode not in ("logits", "probs", "hidden_state", "state_conditioned_log_odds", "joint_log_odds"):
            raise ValueError(
                "OptionProbability input_mode must be 'logits', 'probs', 'hidden_state', 'state_conditioned_log_odds', or 'joint_log_odds'"
            )
        if input_mode in ("hidden_state", "state_conditioned_log_odds", "joint_log_odds"):
            if type(num_classes) is not int or num_classes < 2:
                raise ValueError(
                    f"OptionProbability with input_mode={input_mode!r} requires integer num_classes >= 2"
                )
            for idx in indices:
                if idx >= num_classes:
                    raise ValueError(f"OptionProbability index {idx} is out of bounds for num_classes={num_classes}")
        if input_mode == "hidden_state":
            if type(target_index) is not int or not 0 <= target_index < num_classes:
                raise ValueError("OptionProbability target_index must be in [0, num_classes)")
        if type(query_index) is not int or isinstance(query_index, bool) or query_index < 0:
            raise ValueError("OptionProbability query_index must be a nonnegative integer")
        self.index = index if type(index) is int else list(indices)
        self.indices = indices
        self.input_mode = input_mode
        self.num_classes = num_classes
        self.target_index = target_index
        self.query_index = query_index

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
        elif self.input_mode == "state_conditioned_log_odds":
            if logits.shape[1] < self.num_classes:
                raise ValueError(
                    f"OptionProbability state_conditioned_log_odds tensor width {logits.shape[1]} "
                    f"< num_classes={self.num_classes}"
                )
            u = logits[:, -self.num_classes:]
            probs = torch.sigmoid(u)
        elif self.input_mode == "joint_log_odds":
            q_idx = int(self.query_index)
            K = self.num_classes
            q_start = q_idx * K
            q_end = q_start + K
            if logits.shape[1] < q_end:
                raise ValueError(
                    f"OptionProbability joint_log_odds tensor width {logits.shape[1]} < required {q_end}"
                )
            u_slice = logits[:, q_start:q_end]
            probs = torch.sigmoid(u_slice)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            for idx in self.indices:
                if idx >= logits.shape[1]:
                    raise ValueError(f"OptionProbability index {idx} is out of bounds for K={logits.shape[1]}")
            probs = logits.softmax(dim=-1) if self.input_mode == "logits" else logits
        if len(self.indices) == 1:
            return probs[:, self.indices[0]].mean()
        return probs[:, list(self.indices)].sum(dim=-1).mean()


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


class OpticalWavefrontPrior(_ModuleBase):
    """Optical Wavefront Prior regularizing spatial, boundary, and spectral coherence.

    Penalizes:
    1. Patch-seam gradient penalty along 8px and 16px ViT boundaries:
       R_seam = E[||grad_seam I||_1] / E[||grad_interior I||_1]
       L_seam = max(0, R_seam - seam_threshold)^2 (ensuring R_seam <= 1.15)
    2. Biharmonic curvature regularization:
       L_biharmonic = ||Delta I||_1 = ||nabla^2 I||_1 via discrete 2D 5-point or 9-point Laplacian.
    3. Radial power spectral density (PSD) natural image prior (1/f^alpha, alpha approx 2.0):
       Penalizes excess high-frequency spectral power and spikes above natural 1/f falloff via 2D FFT.
    """
    resume_stateless = True

    def __init__(
        self,
        *,
        weight_seam=1.0,
        weight_biharmonic=0.1,
        weight_psd=0.01,
        patch_size=16,
        secondary_patch_size=8,
        seam_threshold=1.05,
        laplacian_kernel="5point",
        alpha=2.0,
        margin_db=1.0,
        padding_mode="reflect",
        delta=0.01,
    ):
        if _ModuleBase is not object:
            super().__init__()
        if type(weight_seam) not in (int, float) or isinstance(weight_seam, bool) or not math.isfinite(weight_seam) or weight_seam < 0.0:
            raise ValueError("OpticalWavefrontPrior weight_seam must be a finite nonnegative number")
        if type(weight_biharmonic) not in (int, float) or isinstance(weight_biharmonic, bool) or not math.isfinite(weight_biharmonic) or weight_biharmonic < 0.0:
            raise ValueError("OpticalWavefrontPrior weight_biharmonic must be a finite nonnegative number")
        if type(weight_psd) not in (int, float) or isinstance(weight_psd, bool) or not math.isfinite(weight_psd) or weight_psd < 0.0:
            raise ValueError("OpticalWavefrontPrior weight_psd must be a finite nonnegative number")
        if type(patch_size) is not int or isinstance(patch_size, bool) or patch_size < 2:
            raise ValueError("OpticalWavefrontPrior patch_size must be an integer >= 2")
        if type(secondary_patch_size) is not int or isinstance(secondary_patch_size, bool) or secondary_patch_size < 2:
            raise ValueError("OpticalWavefrontPrior secondary_patch_size must be an integer >= 2")
        if type(seam_threshold) not in (int, float) or isinstance(seam_threshold, bool) or not math.isfinite(seam_threshold) or seam_threshold <= 0.0:
            raise ValueError("OpticalWavefrontPrior seam_threshold must be a positive finite number")
        if laplacian_kernel not in ("5point", "9point"):
            raise ValueError("OpticalWavefrontPrior laplacian_kernel must be '5point' or '9point'")
        if type(alpha) not in (int, float) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("OpticalWavefrontPrior alpha must be a positive finite number")
        if type(margin_db) not in (int, float) or isinstance(margin_db, bool) or not math.isfinite(margin_db) or margin_db < 0.0:
            raise ValueError("OpticalWavefrontPrior margin_db must be a finite nonnegative number")
        if padding_mode not in ("reflect", "replicate", "zeros"):
            raise ValueError("OpticalWavefrontPrior padding_mode must be 'reflect', 'replicate', or 'zeros'")
        if type(delta) not in (int, float) or isinstance(delta, bool) or not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("OpticalWavefrontPrior delta must be a positive finite number")

        self.weight_seam = float(weight_seam)
        self.weight_biharmonic = float(weight_biharmonic)
        self.weight_psd = float(weight_psd)
        self.patch_size = int(patch_size)
        self.secondary_patch_size = int(secondary_patch_size)
        self.seam_threshold = float(seam_threshold)
        self.laplacian_kernel = laplacian_kernel
        self.alpha = float(alpha)
        self.margin_db = float(margin_db)
        self.padding_mode = padding_mode
        self.delta = float(delta)

    def forward(self, image=None, x=None, return_dict=False):
        import torch
        import torch.nn.functional as F
        import torch.fft as fft

        img = image if image is not None else x
        if img is None:
            raise ValueError("OpticalWavefrontPrior requires an image tensor ('image' or 'x')")
        if (
            not isinstance(img, torch.Tensor)
            or img.ndim != 4
            or not img.is_floating_point()
            or img.shape[1] != 3
            or img.numel() == 0
        ):
            raise ValueError("OpticalWavefrontPrior requires a nonempty 4D floating-point NCHW tensor with C=3")
        B, C, H, W = img.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"OpticalWavefrontPrior image dimensions ({H}, {W}) must be >= patch_size ({self.patch_size})"
            )

        device = img.device
        dtype = img.dtype
        losses = {}

        # 1. Patch-seam gradient penalty across 8px and 16px ViT boundaries
        if self.weight_seam > 0.0:
            delta_sq = self.delta ** 2
            dh = img[:, :, 1:, :] - img[:, :, :-1, :]
            dw = img[:, :, :, 1:] - img[:, :, :, :-1]
            gy = F.pad((dh.square() + delta_sq).sqrt() - self.delta, (0, 0, 0, 1))
            gx = F.pad((dw.square() + delta_sq).sqrt() - self.delta, (0, 1, 0, 0))
            grad_mag = (gx + gy).mean(dim=1)  # [B, H, W]

            y_coords = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
            x_coords = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
            seam_mask_2d = (
                (y_coords % self.patch_size == 0)
                | (y_coords % self.secondary_patch_size == 0)
                | (x_coords % self.patch_size == 0)
                | (x_coords % self.secondary_patch_size == 0)
            )
            border_mask = (y_coords == 0) | (y_coords == H - 1) | (x_coords == 0) | (x_coords == W - 1)
            seam_mask_2d = seam_mask_2d & (~border_mask)
            interior_mask_2d = (~seam_mask_2d) & (~border_mask)

            mean_seam = grad_mag[:, seam_mask_2d].mean()
            mean_interior = grad_mag[:, interior_mask_2d].mean()
            R_seam = mean_seam / mean_interior.clamp_min(1e-7)
            loss_seam = torch.relu(R_seam - self.seam_threshold).square()
            losses["loss_seam"] = loss_seam
            losses["R_seam"] = R_seam
        else:
            loss_seam = img.new_zeros(())
            losses["loss_seam"] = loss_seam
            losses["R_seam"] = img.new_zeros(())

        # 2. Biharmonic curvature regularization (||Delta I||_1)
        if self.weight_biharmonic > 0.0:
            if self.laplacian_kernel == "9point":
                k = torch.tensor([
                    [1.0, 4.0, 1.0],
                    [4.0, -20.0, 4.0],
                    [1.0, 4.0, 1.0],
                ], device=device, dtype=dtype) / 6.0
            else:
                k = torch.tensor([
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0],
                ], device=device, dtype=dtype)
            weight = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
            pad_mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            img_padded = F.pad(img, (1, 1, 1, 1), mode=pad_mode)
            lap = F.conv2d(img_padded, weight, groups=C)
            delta_sq = self.delta ** 2
            loss_biharmonic = ((lap.square() + delta_sq).sqrt() - self.delta).mean()
            losses["loss_biharmonic"] = loss_biharmonic
        else:
            loss_biharmonic = img.new_zeros(())
            losses["loss_biharmonic"] = loss_biharmonic

        # 3. Radial power spectral density (PSD) natural image prior (1/f^alpha)
        if self.weight_psd > 0.0:
            fy = fft.fftfreq(H, device=device).unsqueeze(1) * H
            fx = fft.rfftfreq(W, device=device).unsqueeze(0) * W
            f_rad = torch.sqrt(fy ** 2 + fx ** 2)

            max_r = min(H, W) // 2
            r_idx = f_rad.round().long().clamp(0, max_r)
            r_flat = r_idx.view(-1)
            num_bins = max_r + 1

            counts = torch.bincount(r_flat, minlength=num_bins).to(dtype=dtype).clamp_min(1.0)
            F_x = fft.rfft2(img, norm="ortho")
            psd = (F_x.real ** 2 + F_x.imag ** 2).mean(dim=1)  # [B, H, W//2 + 1]

            psd_flat = psd.reshape(B, -1)
            zero_init = torch.zeros(B, num_bins, device=device, dtype=dtype)
            radial_sum = zero_init.scatter_add(1, r_flat.unsqueeze(0).expand(B, -1), psd_flat)
            radial_psd = radial_sum / counts.unsqueeze(0)

            eps = 1e-8
            ref_power = radial_psd[:, 1:2].clamp_min(1e-7)
            r_vals = torch.arange(1, num_bins, device=device, dtype=dtype)
            target_power = ref_power.detach() / (r_vals / r_vals[0]).unsqueeze(0).pow(self.alpha)
            log_ratio = torch.log(radial_psd[:, 1:] + eps) - torch.log(target_power + eps)
            ac_power = psd.sum(dim=(-2, -1)) - psd[:, 0, 0]
            gate = torch.tanh(ac_power / 1e-3).view(B, 1)
            loss_psd = (gate * torch.relu(log_ratio - self.margin_db).square()).mean()
            losses["loss_psd"] = loss_psd
        else:
            loss_psd = img.new_zeros(())
            losses["loss_psd"] = loss_psd

        total_loss = (
            self.weight_seam * loss_seam
            + self.weight_biharmonic * loss_biharmonic
            + self.weight_psd * loss_psd
        )
        losses["total_loss"] = total_loss

        if return_dict:
            return losses
        return total_loss

    def __call__(self, *args, **kwargs):
        if _ModuleBase is not object:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def resume_identity(self):
        return {
            "weight_seam": self.weight_seam,
            "weight_biharmonic": self.weight_biharmonic,
            "weight_psd": self.weight_psd,
            "patch_size": self.patch_size,
            "secondary_patch_size": self.secondary_patch_size,
            "seam_threshold": self.seam_threshold,
            "laplacian_kernel": self.laplacian_kernel,
            "alpha": self.alpha,
            "margin_db": self.margin_db,
            "padding_mode": self.padding_mode,
            "delta": self.delta,
        }


class MarginalShannonEntropyRegularizer(_ModuleBase):
    """Marginal Shannon entropy regularization on generated option probabilities.

    Enforces H_q >= min_entropy_ratio * ln(M_q) across multi-mode queries
    by minimizing the squared deficit:
        L_q = max(0, min_entropy_ratio * ln(M_q) - H_q)^2
    """
    resume_stateless = True

    def __init__(
        self,
        query_indices=None,
        query_index=None,
        *,
        num_classes=5,
        num_modes=3,
        min_entropy_ratio=0.85,
        hinge=True,
        input_mode="joint_log_odds",
        eps=1e-8,
        target_attraction_weight=0.0,
        single_mode_queries=None,
    ):
        if _ModuleBase is not object:
            super().__init__()
        import math
        if query_indices is None:
            if query_index is None:
                query_indices = [0]
            else:
                if type(query_index) is not int or query_index < 0:
                    raise ValueError("MarginalShannonEntropyRegularizer query_index must be a nonnegative integer")
                query_indices = [query_index]
        elif isinstance(query_indices, int):
            query_indices = [query_indices]
        elif not isinstance(query_indices, (list, tuple)) or not query_indices:
            raise ValueError("MarginalShannonEntropyRegularizer query_indices must be a nonempty sequence of ints")

        if any(type(q) is not int or q < 0 for q in query_indices):
            raise ValueError("query_indices must contain nonnegative integers")
        if len(set(query_indices)) != len(query_indices):
            raise ValueError("query_indices must not contain duplicates")

        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")

        if isinstance(num_modes, int):
            if num_modes < 2:
                raise ValueError("num_modes must be >= 2 for multi-mode queries")
            modes_dict = {q: num_modes for q in query_indices}
        elif isinstance(num_modes, dict):
            modes_dict = {q: int(num_modes.get(q, 3)) for q in query_indices}
        elif isinstance(num_modes, (list, tuple)):
            if len(num_modes) != len(query_indices):
                raise ValueError("num_modes sequence length must match query_indices length")
            modes_dict = {q: int(m) for q, m in zip(query_indices, num_modes)}
        else:
            raise ValueError("num_modes must be an int, sequence of ints, or dict")

        if type(min_entropy_ratio) not in (int, float) or isinstance(min_entropy_ratio, bool) or not (0.0 < min_entropy_ratio <= 1.0):
            raise ValueError("min_entropy_ratio must be a float in (0.0, 1.0]")
        if input_mode not in ("joint_log_odds", "logits", "probs"):
            raise ValueError("input_mode must be 'joint_log_odds', 'logits', or 'probs'")
        if type(eps) not in (int, float) or eps <= 0.0:
            raise ValueError("eps must be a positive float")

        self.query_indices = list(query_indices)
        self.num_classes = int(num_classes)
        self.num_modes = modes_dict
        self.min_entropy_ratio = float(min_entropy_ratio)
        self.hinge = bool(hinge)
        self.input_mode = input_mode
        self.eps = float(eps)
        self.target_attraction_weight = float(target_attraction_weight)
        self.single_mode_queries = list(single_mode_queries) if single_mode_queries is not None else []

    def forward(self, logits=None, **kwargs):
        import math
        import torch
        import torch.nn.functional as F

        if logits is None:
            if "u" in kwargs:
                logits = kwargs["u"]
            elif len(kwargs) == 1:
                logits = next(iter(kwargs.values()))
            else:
                raise ValueError("MarginalShannonEntropyRegularizer requires logits input")

        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_floating_point():
            raise ValueError("MarginalShannonEntropyRegularizer requires 2D floating-point tensor")

        B = logits.shape[0]
        losses = []

        for q in self.query_indices:
            M_q = self.num_modes[q]
            if self.input_mode == "joint_log_odds":
                q_start = q * self.num_classes
                q_end = q_start + self.num_classes
                if logits.shape[1] < q_end:
                    raise ValueError(
                        f"MarginalShannonEntropyRegularizer joint_log_odds tensor width {logits.shape[1]} < required {q_end}"
                    )
                u_slice = logits[:, q_start:q_end]
                u_modes = u_slice[:, :M_q] if self.target_attraction_weight > 0.0 else u_slice
                probs = torch.sigmoid(u_modes)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            elif self.input_mode == "logits":
                probs = logits.softmax(dim=-1)
            else:
                probs = logits

            # Compute batch-marginal probability distribution
            p_bar = probs.mean(dim=0)
            H = - torch.sum(p_bar * torch.log(p_bar + self.eps))
            target_H = self.min_entropy_ratio * math.log(M_q)

            if self.hinge:
                loss_q = torch.relu(target_H - H).pow(2)
            else:
                loss_q = -H
            losses.append(loss_q)

        entropy_loss = torch.stack(losses).mean()
        if self.target_attraction_weight <= 0.0 or self.input_mode != "joint_log_odds":
            return entropy_loss

        attr_losses = []
        for q in self.single_mode_queries:
            q_start = q * self.num_classes
            q_end = q_start + self.num_classes
            if logits.shape[1] >= q_end:
                u_q = logits[:, q_start:q_end]
                l_pos = F.relu(2.85 - u_q[:, 0]).mean()
                l_neg = F.relu(u_q[:, 1:] + 2.85).mean()
                l_nu = F.relu(u_q[:, -2:] + 2.90).mean()
                attr_losses.append(l_pos + l_neg + 0.5 * l_nu)

        for q in self.query_indices:
            M_q = self.num_modes[q]
            q_start = q * self.num_classes
            q_end = q_start + self.num_classes
            u_q = logits[:, q_start:q_end]
            scores = u_q[:, :M_q].detach()
            scores_norm = (scores - scores.mean(dim=0, keepdim=True)) / (scores.std(dim=0, keepdim=True) + 1e-5)
            assigned = torch.full((B,), -1, dtype=torch.long, device=logits.device)
            cap = (B + M_q - 1) // M_q
            counts = [0] * M_q
            flat_idx = torch.argsort(scores_norm.reshape(-1), descending=True)
            for idx in flat_idx.tolist():
                b_i, m_i = idx // M_q, idx % M_q
                if assigned[b_i] == -1 and counts[m_i] < cap:
                    assigned[b_i] = m_i
                    counts[m_i] += 1
            assigned = assigned.clamp_min(0)
            pos_mask = torch.zeros_like(u_q, dtype=torch.bool)
            pos_mask.scatter_(1, assigned.unsqueeze(-1), True)
            l_pos = F.relu(2.85 - u_q[pos_mask]).mean()
            l_inv = F.relu(u_q[:, M_q:] + 2.90).mean() if M_q < self.num_classes else u_q.new_zeros(())
            attr_losses.append(l_pos + 1.5 * l_inv)

        return entropy_loss + self.target_attraction_weight * torch.stack(attr_losses).mean()

    def __call__(self, *args, **kwargs):
        if _ModuleBase is not object:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def resume_identity(self):
        res = {
            "query_indices": self.query_indices,
            "num_classes": self.num_classes,
            "num_modes": self.num_modes,
            "min_entropy_ratio": self.min_entropy_ratio,
            "hinge": self.hinge,
            "input_mode": self.input_mode,
            "eps": self.eps,
        }
        if self.target_attraction_weight > 0.0:
            res["target_attraction_weight"] = self.target_attraction_weight
            res["single_mode_queries"] = self.single_mode_queries
        return res


class NormalizedVisionCosineRepulsion(_ModuleBase):
    r"""Normalized vision feature cosine repulsion on pooled_vision (v in R^(B x 2560)).

    Penalizes pairwise cosine similarity exceeding margin:
        L_repulse = \frac{1}{B(B-1)} \sum_{i \neq j} \max(0, cos(v_i, v_j) - margin)

    With margin <= 0.88, enforces mean pairwise cosine distance 1 - cos(v_i, v_j) >= 0.12.
    """
    resume_stateless = True

    def __init__(self, margin=0.88, eps=1e-8):
        if _ModuleBase is not object:
            super().__init__()
        import math
        if type(margin) not in (int, float) or isinstance(margin, bool) or not math.isfinite(margin) or not (0.0 <= margin < 1.0):
            raise ValueError("NormalizedVisionCosineRepulsion margin must be in [0.0, 1.0)")
        if type(eps) not in (int, float) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("NormalizedVisionCosineRepulsion eps must be a positive finite number")
        self.margin = float(margin)
        self.eps = float(eps)

    def forward(self, features=None, margin=None, **kwargs):
        import torch
        import torch.nn.functional as F

        if features is None:
            if "v" in kwargs:
                features = kwargs["v"]
            elif "pooled_vision" in kwargs:
                features = kwargs["pooled_vision"]
            elif len(kwargs) == 1:
                features = next(iter(kwargs.values()))
            else:
                raise ValueError("NormalizedVisionCosineRepulsion requires feature tensor input")

        if not isinstance(features, torch.Tensor) or features.ndim != 2 or not features.is_floating_point():
            raise ValueError("NormalizedVisionCosineRepulsion requires a 2D floating-point tensor [B, D]")

        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        eff_margin = self.margin if margin is None else float(margin)
        v_norm = F.normalize(features.float(), p=2, dim=-1, eps=self.eps)
        sim = torch.mm(v_norm, v_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        off_diag = sim[mask]
        penalty = F.relu(off_diag - eff_margin)
        return penalty.mean().to(dtype=features.dtype)

    def __call__(self, *args, **kwargs):
        if _ModuleBase is not object:
            return super().__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def resume_identity(self):
        return {
            "margin": self.margin,
            "eps": self.eps,
        }
