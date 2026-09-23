# Grouped first-G contract

64px revalidation at the source rates G = D = 2e-4. The 128px, G = D = 1e-4 ranking is a candidate to confirm or reject on new banks. It is not a transferred rate rule, not a matched leaderboard row, and not a FLeRM target. One disposable probe, then `propose()` applies the gates below to the stored raw numbers. No second probe, no replacement group, no seed sweep, no grid, no nested training. A finished probe file uses 0 for a stage it skipped. `propose()` copies stored costs, and uses JSON `null` only when a cost key is absent. Never invent a zero.

`propose(context)` returns schema 1. Abstention is a successful result and is exactly:

```python
{'schema_version': 1, 'evidence': {'decision': 'abstain', ...cost fields...}}
```

Those objects contain no `g_lr`, `d_lr`, `init_scales`, or `layer_lr_multipliers`. Do not store a decision computed under any other bar. If a raw field this contract requires is missing, abstain `incomplete_evidence` and do not impute it.

## Assumptions this probe does and does not satisfy

FLeRM defines block response as the RMS of that block's Jacobian times an optimizer displacement, and allows a startup measurement. This probe estimates that quantity for the actual first generator Adam step, with four fixed Rademacher projections, and it keeps the sum of projected responses so cross terms stay visible. It does not satisfy FLeRM matching. Matching needs a recorded base profile, taken at unit rate and normally with the paper's EMA Kronecker estimator, and sets each block rate from the base-to-current response ratio. There is no successful same-task reference profile. Four projections are not that estimator. FLeRM also does not supply a universal ideal response or a reason to equalize layers; this probe does not equalize them and does not map blocks across depth or resolution.

GeN models loss along one optimizer direction, including Adam, as `phi(s) = phi(0) + a s + k s^2 / 2`, and takes `s* = -a/k` only for descent and positive curvature. This probe keeps that sign rule on two declared groups and uses symmetric axis steps. It does not run GeN. Algorithm 1 refits during training, smooths (example γ = 0.9), and adapts on a period (example 8). A three-point interpolant has residual zero, so GeN's fitting-point R² > 0.99 is not a check and is not imported. GeN may raise the previous rate; this contract never emits a multiplier above 1. The paper's GAN evidence is not a startup calibration of this model. GeN-Adam is not invariant to rescaling the raw gradient, which is why the intervention is an optimizer-group learning rate.

## Anchor, groups, banks

Run one native D-then-G update from a fresh step-0 trainer at the source rates. The anchor is the observer's first generator snapshot: parameters and RNG from before that G step, displacement equal to the Adam step just taken, opponent already at its first D step. Prior parameters, pretrained weights, and buffers are never displaced. Every later evaluation restores that snapshot, writes only owned generator parameters, and restores it again before returning. Protected parameter and buffer hashes must match.

Group A is exactly these owned generator tensors, or the probe stops with `group_identity`:

- `graph.models.generator.network.nodes.n_stage8_block1_ffn.down.weight`
- `graph.models.generator.network.nodes.n_stage8_block0_ffn.down.weight`
- `graph.models.generator.network.nodes.n_stage16_block0_ffn.down.weight`
- `graph.models.generator.network.nodes.n_stage16_block1_ffn.down.weight`

Group B is every other owned generator parameter. Discriminator, prior, pretrained weights, and buffers are in neither group. Empty B, a missing path, a non-owned path, an alias, or a duplicate stops the probe. Coordinates `(sA, sB)` mean `before + sA * delta` on A and `before + sB * delta` on B. `(1, 1)` is the recorded first G step. One group multiplier scales that group's whole recorded displacement; it does not identify per-tensor rates inside the group.

At step 0, before the native update, copy the prior-stream state into one local generator. The monitor bank is the first `batch()` and the first sample from that generator, including particle ids, hashed with the same content hash the benchmark stores as `bank_sha256`. The fitting banks are the next two `batch()` calls and the next two samples from that same generator. Restore the step-0 snapshot, including both streams, before the native update. Record `monitor`, `fitting_0`, and `fitting_1`. If any two of those hashes coincide, abstain `bank_identity`. Do not fit, rank, or score on the monitor bank.

## Confirmation

On each fitting bank, run the existing four-projection estimator (`measure_function_space`, projections = 4, CPU seed 0, generator pixels, actual full G displacement). Do not add projections if the bar fails. Estimated squared response of a tensor is the mean of its four squared projection scalars, the square of the stored RMS. Rank by that value, descending; break ties by parameter-path string, ascending.

| Test, both fitting banks | Bar | Failure |
| --- | --- | --- |
| Membership of the top 4 | The top 4 paths are exactly group A | `group_not_confirmed` |
| Share of the sum of per-tensor estimated squared responses | Group A's share ≥ 0.70 | `group_not_confirmed` |

Shares are computed per bank, not averaged. The denominator is the sum over owned generator tensors of estimated squared responses, not the squared norm of the total output change. The 128px shares were 0.9154 and 0.9347, with a bank gap of 0.0193. The floor stays 0.70. Raising it toward 0.91, or adding the 128px fourth-to-fifth squared-response gap (about 9 and 12), would require that 1e-4 / 128px magnitude to reappear at 2e-4 / 64px. This revalidation does not assume that. Those four tensors also carried about 65% of the matching-bank first-order G loss decrease at 128px; the confirmation rule does not use slope share, and a large response is not a reason to shrink A.

If confirmation fails, stop. Do not fit the stencil, do not form another group from these banks, and do not inspect the monitor bank for a substitute.

## Loss stencil

The loss is the configured generator phase objective at anchor step 1, including its generator tail, with the anchor opponent, prior, buffers, and RNG. It is not a discriminator loss and not a pixel loss. The same six fitting points determine the coefficients exactly. There is no least-squares residual and no ridge.

`L00 + aA sA + aB sB + (kA/2) sA^2 + (kB/2) sB^2 + kAB sA sB`

| Point | `(sA, sB)` | Role |
| --- | --- | --- |
| `origin` | `(0, 0)` | Fit. One shared backward; split `g · delta` into A and B |
| `A_minus`, `A_plus` | `(-1, 0)`, `(1, 0)` | Fit. `A_plus` also supplies `dA` |
| `B_minus`, `B_plus` | `(0, -1)`, `(0, 1)` | Fit. `B_plus` also supplies `dB` |
| `full_step` | `(1, 1)` | In the fit. Identifies `kAB`. Also supplies `dAB` |
| `opposite_mix` | `(1, -1)` | Unused. Not in the fit |

`aA = (L(1,0) - L(-1,0)) / 2` and `kA = L(1,0) + L(-1,0) - 2 L00`, and likewise for B. Then `kAB = L(1,1) - L00 - aA - aB - kA/2 - kB/2`. The unused prediction is `L00 + aA - aB + kA/2 + kB/2 - kAB`. Opposite mix is the unused point because `full_step` is already fitted and the opposite mixed sign flips the cross term.

Count, if the stencil runs: 7 phase-loss evaluations per bank, 14 total, plus 2 origin backwards. Hook pixel outputs inside `origin`, `A_plus`, `B_plus`, and `full_step`; that adds no output forward. A separated re-forward is not authorized. Confirmation, whether or not the stencil runs, costs 8 projection backwards and 4 output forwards. Bank construction and the one native update are not output forwards. A nonfinite loss, slope, or output abstains `model_unresolved` with `failed_gate = nonfinite`. Do not drop the point.

## When the model is unresolved

Fit only the six fitting points. Score only `opposite_mix`. Apply every gate on both banks. The first failure in this order wins; do not shop for a later gate that passes.

| Order | Gate | Resolved only if |
| --- | --- | --- |
| 1 | Curvature | `kA > 0` and `kB > 0`. Nonpositive curvature is unresolved. Do not take `abs(k)` |
| 2 | Hessian | `kA kB - kAB^2 > 0`. Otherwise there is no joint Newton minimum, even if both diagonals are positive. Record `k_full = kA + kB + 2 kAB`; positive-definiteness already implies `k_full > 0` |
| 3 | Slopes | Symmetric `a` and the exact group dot are both finite and strictly negative. A missing per-tensor gradient in the group is unresolved, not zero |
| 4 | Separable factors | Symmetric `s* = -a/k` and exact `s* = -a_exact/k` are both finite and in `(0, 1]`. Outside that interval, abstain. Do not clip to 1, do not take the smaller of the two slope definitions, and do not switch to the exact slope |
| 5 | Banks | The two banks' symmetric `s*` differ by at most 0.05. A larger gap abstains. Do not average, and do not fall back to the smaller factor |
| 6 | Cross term | With `det = kA kB - kAB^2`, `sA_joint = (-kB aA + kAB aB) / det` and `sB_joint = (kAB aA - kA aB) / det`. Both are finite and in `(0, 1]`, and each lies within 0.05 of that bank's symmetric `s*`. Otherwise the cross term is unresolved. Do not emit the joint point |
| 7 | Unused point | Let `d = L(1,-1) - L00`. Finite `d ≠ 0` and `|predicted - actual| ≤ 0.5 |d|`. A zero change makes the ratio undefined and is unresolved |
| 8 | Adam | The epsilon comparison below passes for A and for B |

The bank bound 0.05 is the lazy-D pair's gap of 0.0150 (0.67395 against 0.68894) plus 0.035. It does not require the 128px banks to agree that closely again. A larger gap is unresolved rather than a smaller step. The same bound is reused for the cross term because the 128px report has no loss-Hessian cross term to calibrate; the large 128px output cross terms are a different object and are not this tolerance. Exact-versus-symmetric factors are not required to lie within 0.05: the lazy-D gaps were about 0.15 and 0.14, and that report refused those fits because the unused-point relative errors were 0.604 and 0.600. Both factors must still sit in `(0, 1]`. The unused-point fraction 0.5 rejects those 0.60 errors. It is not tightened to the first-G half-step ratios 0.412 and 0.394, which that report did not separate from negative curvature.

Adam, from the step-1 state, with configured betas `(0.5, 0.999)`, `eps = 1e-8`, and weight decay 0. Reconstruct `g = exp_avg / (1 - beta1)` and `predicted = -lr * g / (|g| + eps)` at `lr = 2e-4`. Per group, `RMS(delta)` and `sum |g_i delta_i|` must both be finite and positive; the relative RMS of `delta - predicted` must be ≤ 1e-4 (formula agreement, not a rate floor); and at most 0.01 of that absolute slope mass may sit on elements with `|g_i| ≤ 10^3 eps`. That factor is the declared meaning of `|g| >> eps`: those elements are within 0.1% of a pure `±lr` step. Any other step, weight decay, AMSGrad, zero mass, nonfinite state, or failed comparison abstains `adam_step_not_proportional`. A stationary factor is then not a learning-rate multiplier. Do not delete the contaminated coordinates and refit. A constant raw-gradient scale is not an allowed substitute: under Adam it can cancel.

The emitted symmetric factor for a group is the smaller bank's `-a/k`, and only after gate 5 has already passed. If that value is exactly 1, omit the group. Do not treat a rounded near-1 as 1.

## Output check, separate from the fit

On the same banks, using the hooked pixels, in float64:

`residual = dAB - dA - dB`, with `dA = out(1,0) - out(0,0)` and likewise for `dB` and `dAB`.

Independent multipliers are not a response decomposition on a bank when `RMS(dAB)` is not finite and positive, or when `RMS(residual) > 0.5 RMS(dAB)`. Either bank forbids the claim. Record both RMS values and `ms(dAB) - ms(dA) - ms(dB)`. This vetoes an interpretation. It does not by itself abstain a loss proposal, authorize a smaller multiplier, or justify shrinking A. A passing decomposition is also not permission to shrink A.

## What a passing model may emit

`decision = propose`. Keys are `schema_version`, `layer_lr_multipliers`, and `evidence` only. Global G and D rates stay 2e-4 and are omitted. No init scale is emitted. Nothing above 1 is emitted.

Each written rule is an exact parameter path and the group's emitted `s*`. No glob. Group A, when its emitted factor is not exactly 1, is the four paths above, each with that same factor. Group B, when its factor is not exactly 1, is one exact path per recorded complement tensor, each with B's factor. Omitting B is how a factor of 1 is represented. If both factors are exactly 1, abstain `no_change` instead of writing no-op rules or spending the benchmark on a copy of the source run.

## Evidence cost

Every `evidence` object carries these fields, including after `propose()` loads a saved probe JSON:

| Field | Contents |
| --- | --- |
| `native_updates` | 1 once the anchor update ran; otherwise 0 |
| `loss_evaluations` | 14, or 0 if the stencil did not run |
| `projection_backwards` | 8 once confirmation ran; otherwise 0 |
| `output_forwards` | 4 from the confirmation estimator; hooked stencil pixels add 0 |
| `elapsed_seconds` | Wall time of the probe that produced the JSON |
| `bank_hashes` | `monitor`, `fitting_0`, `fitting_1` |

Also store the raw per-bank losses, group slopes, projection scalars, Adam residuals, output RMS values, and the resolved A/B path lists. Gates are pass/fail records. There is no composite score.

## Non-claims

This probe does not establish useful GAN learning. The single later 64px benchmark trial, if a proposal is emitted, uses the monitor bank and was not a fitting bank. It still has to show saturation, between-sample diversity, and the signed DINO polynomial MMD together. The 64px screen records feature protocol `online_dinov3_block11_spatial_mean_candidate_only_poly3_64px_4x4`. This contract does not change that measurement. A proposal that only preserves initialization is a failure mode of that trial, not a success. Fixed-opponent descent, a small local loss, or a confirmed response ranking does not replace those three observations. No rate floor, ideal response profile, or further candidate is defined here.
