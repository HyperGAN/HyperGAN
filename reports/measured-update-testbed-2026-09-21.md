# Measured-update startup calibration: implementation and first testbed result

Implementation: `ff79496c`. PR #382 remains open, unmerged, with auto-merge disabled.
The [condensed measurements](measured-update-testbed-2026-09-21.json) include the
raw report path and SHA-256. This replaces the earlier initialization-scale,
D-half, and G-retention selectors with one measured-update pipeline.

**Result: the implementation completed, but the new selector did not fix this
configuration.** Its proposed G/D rates lowered both players' held-out losses;
the actual coupled replay still failed the startup guards. It rejected the
whole proposal, restored initialization, and retained the source rates. Do not
interpret this smoke test as successful training calibration.

## Implemented behavior

`hypergan train CONFIG --run-dir RUN --tune` now:

1. Runs eight disposable native updates and observes actual Adam parameter
   displacements at updates one and eight, separately for G, D, and the prior.
2. Measures G output change at fixed latent values. At the eighth-update D
   anchor, measures the change in its adversarial image gradient using the same
   detached image, real context, and scoring randomness.
3. Fits each player's complete configured phase loss at displacement factors
   zero, one-half, and one on two reserved banks. G's anchor includes the D
   update that preceded it. The opponent and other parameter groups stay fixed.
4. Derives at most one reduction-only pair from resolved descending,
   positive-curvature fits. It checks changed players on two additional banks,
   then permits one eight-update coupled replay. A failed pair has no fallback.
5. Restores all disposable training state before step zero. Only accepted G/D
   rates persist in run-local overrides. Source config, initialization, pretrained
   tensors/buffers, and the prior's configured absolute rate are unchanged.

The weight-scale search and new-run `--tune-warmup-steps` option are removed.
Selected rates are held with the source annealing schedule; they do not ramp
back to the unvalidated source G rate. Older checkpoints retain their recorded
warmup when resumed. Read-only signal diagnosis remains available.

The UI/console expose measure, fit, validate, and replay stages. `--tune` remains
opt-in. No new regularizer, ongoing adaptation, rate grid, or tuning loop was added.
The retained 25% diversity/transmission guards are failure screens; they no
longer generate learning-rate factors and are not universal quality thresholds.
No arbitrary functional-response target was introduced.

## Testbed and controls

Original configuration:
`~/dev/hypergan/training-runs/logos-transgan-dinov3-multidepth-128-init-v2/transgan-dinov3-multidepth.toml`.

- TransGAN generator, 128px, batch 64, owned pixel/feature heads and frozen DINOv3.
- Fused Adam, source G/D rates 0.0002, learned-prior rate 0.002.
- Seed 25002 and prior initialization seed 25003, unchanged. This was one
  baseline versus one derived intervention, not a seed experiment.
- Original independent phase draws, backend settings, EMA, and lazy `b_cap`.
  Normal training still uses `lazy_k=8`; tuning did not make the penalty eager.
  The eighth-update phase-loss stencil includes its active penalty and ×8 scaling.
- No initialization rescaling and no warmup.
- Physical GPU 1: `GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce`.

Run:
`/mnt/ml7tb/hypergan-signal-research/transgan-update-response-smoke-v1`.
Log is the adjacent `transgan-update-response-smoke-v1.log`.
The normal CLI ran with `--tune --no-server --no-previews --stop-after-steps 1`.
It completed tuning and stopped after one retained update, with full checkpoints
at steps zero and one. No experiment was continued toward the known 350–500-step
failure window because the startup candidate had already failed.

The process began while the implementation was uncommitted. The executed
production code was committed during the run as `ff79496c`; only trailing blank
lines in `signal_structure.py` changed between launch and that commit. The
published report records `ff79496c`, clean, at publication time.

## Measurements and decision

| Measurement | Configured baseline | Proposed pair replay |
| --- | ---: | ---: |
| G factor | 1 | 0.9562161194 |
| D factor | 1 | 0.5903215819 |
| First G update: parameter relative L2 displacement | 0.00816 | 0.00781 |
| First G update: fixed-latent image displacement RMS | 0.90821 | 0.84843 |
| First G update: displacement / preceding image RMS | 1.46277 | 1.36650 |
| End sample-diversity retention, two banks | 0.17721 / 0.17641 | 0.19235 / 0.19049 |
| End first-layer structural cotangent retention | 0.10604 / 0.10602 | 0.11711 / 0.11607 |
| End pixel-channel fraction with absolute value > 0.99 | 88.57% / 88.51% | 85.67% / 85.65% |

Initial saturation on the reserved banks was approximately 1%. The final
pre-tanh affine activation RMS rose from approximately 1.0 to 7.11 in baseline
and 6.55 in replay. The structural cotangent is independent of D's delivered
objective derivative; these are transmission measurements, not a score of
whether D gave useful instructions.

G's two fitted factors were 0.95622 and 0.95714. D's unconstrained factors were
1.74237 and 0.59032; the reduction-only rule and conservative two-bank minimum
produced D ×0.59032. Both banks resolved descent and positive curvature, but D's
curvatures differed substantially (0.27198 versus 4.95132). This is not evidence
for a universally meaningful precision on its rate estimate.

Held-out G losses decreased by 0.11089 and 0.10985. Held-out D losses decreased
by 0.02868 and 0.03740. These were resolved decreases at the recorded phase-local
state. Nevertheless, both replay banks failed both 0.25 retention guards.
Outcome: **unresolved**; selected factors remained G=1 and D=1.

At the baseline eighth D update, the fixed-image gradient norm ratios were
0.72720 and 1.15809, with cosine similarities 0.99712 and 0.92769. D weakened the
image-gradient magnitude on one bank and strengthened it on the other. This
measurement does not support a uniform disappearance of D's signal at that
anchor. It does not isolate earlier or cumulative D effects.

**Inference:** curvature measured after the early transient is insufficient to
select a suitable startup rate here. The first G update already produces a
large functional change, despite a small parameter displacement. A locally
helpful step in the state reached at update eight can still be too large when
used from initialization. This run does not establish which component causes
the transient, nor does it establish a universal allowed output-change target.

Stable output RMS also does not imply a small update: at update eight the
baseline output RMS changed by only about 0.1%, while its fixed-latent output
displacement RMS was 0.37158. Saturated outputs can still move substantially.
We did not measure per-layer directional output contributions, preactivation
displacement, or the full G-plus-prior functional response; prior parameter
motion is reported separately. Those limitations should remain explicit.

## Cost and state audit

Calibration took **122.994 seconds** on this testbed:

| Executed operation | Count |
| --- | ---: |
| Complete disposable D/G updates | 16 |
| Fitting phase-loss evaluations | 12 |
| Held-out phase-loss evaluations | 8 |
| G finite-response forwards | 8 |
| Fixed-image generation forwards | 2 |
| D image-gradient backwards | 4 |
| Structural guard evaluations | 6 |
| Adversarial signal guard evaluations | 6 |
| Extra fitting-bank gradient/slope evaluations | 0 |

This includes snapshot/restore and hashing overhead. Penalty-active loss
measurement can require differentiation; the loss evaluations are not all
forward-only. The update counter counts completed D/G updates, not a partially
completed phase if a numerical failure interrupts an update.

The protected-state hash matched before, during completion, and after restoration:
`01d54beef4f137b8e502c67dc7896e6a619eef9fe2c1e67c0820aa906e49fa9c`.
The complete trainer-state equality and existing-gradient identity/content checks
passed. The step-zero checkpoint has empty optimizer states, G=0.0002,
D=0.0002, prior=0.002, and no warmup. Step one retains those rates and reports
`dynamics_outcome=unresolved`. No pretrained calibration occurred.

## Validation and next research boundary

Focused suites passed:

- 125 tuner, numerical, persistence, legacy recovery, and lifecycle tests, plus
  10 phase-probe tests. These include actual Adam displacement, held-out rejection,
  one-pair replay, reused data-buffer isolation, storage alias checks, protected
  mutation detection, and rollback after observer failures.
- 27 native update/oracle, objective binding, and phase-policy tests; one heavy
  test deselected. Normal lazy training still matches the previous native oracle.
- 14 Chromium tests, 17 console tests, 37 lifecycle/web-service tests; frontend
  build and bundled-asset verification passed. Some suites overlap.
- Separate recovery/diagnostic checks passed. Step-zero/one checkpoint metadata
  and rates were inspected read-only on CPU after the GPU process stopped.

A user launcher was created without starting a long experiment:
`~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-128-tuned-response.sh`.
Its default separate run folder is `transgan-tuned-response-live`; it shows the
UI, previews/checkpoints every 100, and stops after 1200 retained updates. A
fresh run of this configuration may repeat the unresolved decision and train
at source rates. The launcher is a way to observe behavior, not evidence of a fix.

The immediate research problem is how to calibrate the initial functional
transient with a cheap, defensible bound or measurement. Merely accepting the
held-out loss improvement would have hidden the observed failure. No ad-hoc
response cap, first-step alternative, extra search, or new production fallback
was added to make this test pass. The PR remains unmerged for further work.
