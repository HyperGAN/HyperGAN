# Generator depth on ParticleGAN 100gaussians

This CPU research adapter asks whether increasing generator depth breaks the
existing 100-Gaussian GAN while its task and training recipe stay fixed. It is
not a reconstruction task. There are no assigned latent/target pairs.

The first declared comparison is **3, 8, and 16 hidden layers, 7,000 updates**,
with the original seed 1234. Width remains 128; activations remain LeakyReLU(.2),
with an unbounded linear output. The native Fourier-2 discriminator, learned
20,000-particle prior, RpGAN logistic objective, exact every-step b-cap,
optimizers, batch 256, EMA, and delayed annealing are retained. The native
example uses a particle table, not the image recipe's noisy MoG prior.

The original three layers and output projection are copied exactly into each
deeper generator. Extra layers share their initial tensors across the deeper
cases. Critic and prior initialization and all training data/particle streams
are unchanged. Initial generator **functions differ**: this is ordinary depth
scaling, not function-preserving insertion or a parameter-count-matched study.
Additional-layer initialization streams are fixed; no seed-only experiments.

No GPU, image renderer, or production training configuration is involved.
The adapter imports ParticleGAN and its original toy models/sampler from an
explicit local checkout; it does not edit that checkout. Provenance records
source commits and hashes of the runner and imported source files.

## Run and tail

Requires Torch, NumPy, and a local ParticleGAN checkout with its research `lib/`.
The `100gaussians` example and research helpers are not all in the installed wheel.
Use a fresh output directory for each declared experiment; existing outputs are
never replaced. Defaults are the full three-depth comparison:

```sh
python -u research/depth_toy/run.py \
  --particlegan-root /path/to/ParticleGAN \
  --output /path/to/new-output > /path/to/depth-toy.log 2>&1
tail -f /path/to/depth-toy.log
```

`--depths`, `--steps`, and `--observe-every` allow explicit structural/horizon
changes. Changing steps also changes the native annealing horizon: a shortened
run is not a prefix of the 7k recipe. There is deliberately no seed option.
Torch uses one CPU thread and deterministic algorithms.

## Measurements

Each case writes configuration/initialization identities, append-only
`metrics.jsonl`, and a final `report.json`. The root contains source provenance
and a summary updated after each completed case. Logs flush after every
observation: steps 0, 1, 8, 32, 128, every 500, and the final step by default.

Distribution measurements use 20,000 fixed particle IDs and one independent
fixed real sample bank, shared across cases. Online G/current prior, EMA G/EMA
prior, and online G/fixed initial latent coordinates are reported separately:

- Native coverage definition: count modes with >=10 samples within .09 (3 sigma)
  of a center. HQ is the fraction of samples within that radius.
- Sliced W1 over 64 fixed evenly spaced directions; nearest-center RMS and the
  full nearest-mode mass histogram. These complement coverage/HQ, which alone
  can reward variance compression. This is not exact W1 or within-mode calibration.
- Output RMS, between-sample RMS, and shared energy fraction
  `mean(mean_batch(x)^2) / mean(x^2)`; particle coordinate standard deviation.

At each recorded **actual training update**, record the image-space request
`-dL/dX` and the movement from the actual Adam G step at the same latents, before
the prior update. Record their cosine, norm ratio, and separate shared energy
fractions. Then record combined G/prior movement on the same particle IDs.
The gradient includes native batch-mean loss scaling; gain has corresponding
units and is only comparable under the same batch/loss protocol.

Hidden activations and their backward gradients get RMS/variation/shared-energy
measurements; every parameter gets gradient and actual-update RMS. These are
observations, not a Jacobian-spectrum estimate or an optimizer proposal. Changes
in normalized shared fraction alone do not establish increased absolute shared
movement. Common movement can be useful; check distribution progress alongside it.

## Interpretation gates

1. Establish that the original shallow control learns in this run. The native
   recipe historically reaches 100 modes and HQ >=.9, but that is not assumed.
2. If deeper models fail, determine whether contraction exists at initialization,
   grows during updates, and whether requests or realized movements become shared.
   Depth changes parameter count and forward scale as well as backward geometry.
3. If all depths learn, report that this toy has **not reproduced** image collapse.
   Do not manufacture failure by changing rates or seeds. A separately declared
   bridge can add the image model's residual/RMSNorm blocks or bounded output.
4. Even a positive toy result needs a matching intervention on the image recipe.
   This MLP has neither attention nor tanh; it cannot establish the image cause.

## Validation

```sh
PARTICLEGAN_ROOT=/path/to/ParticleGAN \
  python -m pytest tests/research/test_depth_toy.py -q
```

Tests compare all final G/D/prior/EMA tensors against the original example's
loop across four updates including its annealing transition. They also verify
bitwise training identity with/without instrumentation, shared initialization,
and metrics that distinguish shared motion and mode collapse from high HQ.
Plotting and native logging evaluation are disabled only in the parity test;
neither participates in native training.
