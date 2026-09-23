# Corrected projected critic still collapses

Read-only inspection of `train-transgan-projected-dinov3-128-scoremean`, run
`826e2b789a854910827bee2a6892987c`. The original run remains running on GPU 1;
this inspection launches no inference on that GPU and does not change its state.
The saved preview evidence is [inspection.json](inspection.json); the sampled
training trace through step 500 is [training-trace.json](training-trace.json).

| Step | EMA saturation | Recorded diversity / real | G adversarial | D adversarial |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.199% | 78.624% | — | — |
| 100 | 68.364% | 24.068% | 2.150 | 0.207 |
| 200 | 95.115% | 10.931% | 1.693 | 0.275 |
| 300 | 98.568% | 5.412% | 1.914 | 0.205 |
| 400 | 99.627% | 1.301% | 2.000 | 0.191 |
| 500 | 99.984% | 0.071% | 1.499 | 0.306 |

Saturation counts channel values with `abs(byte / 127.5 - 1) > .99` in
checksummed PNGs. Diversity ratios come from the trainer's unquantized preview
measurements. Unpacked PNG population spread, multiplied by `sqrt(32/15)` for
16 samples, agrees with recorded pairwise RMS within 0.001 at every saved step.
PNG rounding and EMA lag apply; these are not online gradients or a quality
score. Sample IDs and preview seed are fixed, while EMA G/prior evolve and real
preview batches change. The audit helper now also handles manifests without
the optional `initialization_tuning` key, as produced by `--no-tune` runs.

The 12 published active penalty values through step 500 range from 0.03099
to 2.44120. Metrics publish every 10 steps; lazy penalty applies every 8 steps,
so this only covers their intersection every 40 steps. It does not establish
a maximum over unlogged applications. The old 5k–22k published spikes are
absent from this trace, but saturation and collapse persist. Neither moderate
G loss nor a nonzero D adversarial loss certifies a functioning generator.

## What is established

The earlier critic swaps, optimizer swaps, and this corrected projected critic
all fail with the large logos generator. The working DCGAN/logos controls make
a general dataset or training-loop failure less likely. The small adversarial
CIFAR TransGAN shows that an encoder/reconstruction objective is not required
for every TransGAN adaptation. It does not isolate network width: dataset,
resolution, upsampling, prior, and other recipe details differ too.

Earlier matched checkpoint probes show intermediate residual activation growth,
large pre-tanh activations, and attenuated backward signal through tanh while
the discriminator still supplies an image gradient. Four early FFN down-weight
tensors dominate the measured initial response. This supports a hypothesis
about architecture and optimizer step scale; dominance alone is not evidence
that those tensors should be suppressed. The grouped curvature probe still
abstains and emits no multipliers.

The new, separately declared
[32-update ablation](../../configs/transgan-128-projected-ffn-screen.json)
compares the corrected projected recipe with a tenfold reduction of only those
four weight-tensor Adam rates. It is an empirical intervention, not a quadratic
optimum or production recommendation. It uses GPU 0 and the existing recipe
seed, with identical initial parameters and monitor inputs checked in the
reports. No seed sweep, source training-runs TOML edit, or PR merge is involved.
The legacy frozen-feature observer requires a gray-context scalar critic, so
it is explicitly disabled for this candidate-only spatial critic. This screen
can diagnose output contraction, not independent quality improvement.

## Matched intervention result: mitigates, still fails

Both screens completed on GPU 0 (`GPU-ed080e41-3193-3755-6756-f3d46c433331`)
from implementation `ce4bb9fc`. Total: 64 disposable native updates, 171.73
seconds including diagnostics, no retained training checkpoint. The working
tree additionally contained this report and the testbed README correction;
no training implementation changed during the screens. Reports and complete
resolved configurations are under [ffn-ablation](ffn-ablation/).

| Step | Source saturation | FFN tenth saturation | Source diversity RMS | FFN tenth diversity RMS |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.010% | 1.010% | 0.551861 | 0.551861 |
| 1 | 8.405% | 1.698% | 0.431923 | 0.524984 |
| 8 | 89.852% | 72.288% | 0.085696 | 0.128191 |
| 16 | 96.820% | 79.502% | 0.048102 | 0.122429 |
| 32 | 97.825% | 89.202% | 0.038219 | 0.086118 |

These are online, unquantized measurements on 64 fixed particle IDs/noise,
with the evolving prior. Diversity is population spread, not the EMA table's
pairwise RMS. The source retains 6.93% of initial diversity; the intervention
retains 15.61%. Pre-tanh RMS grows from 0.999 to 11.773 (source) or 6.569
(intervention). Mean tanh derivative falls from 0.6125 to 0.00572 or 0.02793.
The outputs lose latent-dependent variation while activations grow into tanh's
flat regions. Both are failures by saturation and diversity, regardless of
apparently moderate adversarial losses.

Initial and prepared parameter hashes match between cases, as do bank,
measurement RNG, and prior RNG hashes. Step-zero measurements are identical.
Exactly the four named generator weights have effective rate 0.00002; all other
G weights stay at 0.0002, D at 0.0002, and prior at 0.002. Both full trainer-state
restorations passed, all protected parameter/buffer hashes match, and source
configurations are unchanged. Configured TF32/nondeterministic kernels remain
enabled, so this is one matched intervention, not a statistical replication.

Fixed-initial-latent final saturation is 97.824% or 89.189%, closely matching
the evolving-prior measurements. Prior motion at the final generator does not
explain this contraction. This does not rule out prior effects on the trajectory.

The intervention establishes a contribution of these four updates to rapid
saturation in this setup, but not that they are the only cause or that 4096-wide
FFNs are intrinsically invalid. Reducing their step scale is insufficient.
There is no successful candidate to promote, and the earlier negative-curvature
abstention is unchanged.

The complete every-update screen also exposes lazy penalties hidden by the
live run's logging cadence: source active penalties at steps 8/16/24/32 are
0.152/8.836/75.563/1.236; intervention values are 0.362/9.072/22.392/6.378.
The installed corrected implementation is captured in
[penalty-implementation.json](penalty-implementation.json). Do not describe
the live run's sampled maximum of 2.441 as its actual all-step maximum.

Next architectural isolation should stay on logos at 128px with this same
critic/optimizer/prior and change only the early FFN hidden width, retaining
shared-shaped parameter values when possible. That would test width directly;
this rate ablation tests update scale. A narrower candidate is not implemented
or claimed to work here. The separate 32→64px DINO-input upsampling idea addresses
CIFAR critic spatial resolution, and cannot by itself explain the 128px
generator's collapse against multiple critics.

Validation: 17 existing CPU tests for proposal application/restoration and
the projected critic passed. Preview checksums, PNG diversity reconstruction,
matched experiment identities, actual rate assignments, and restoration audits
were checked. GPU 1's job was neither stopped nor modified. PR #382 remains
unmerged.
