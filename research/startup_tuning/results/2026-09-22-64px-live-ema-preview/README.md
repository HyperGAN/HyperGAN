# Live 64px EMA previews already saturate

Inspection of the user's running configuration at
`/home/martyn/dev/hypergan/training-runs/train-transgan-dinov3-multidepth-64-init-v2`,
run `6c49528376ad425db3ee97ee12deeb57`, source commit `6bbdf60e`, seed 25002,
G/D rates 0.0002, prior rate 0.002, `--no-tune`. This is not a startup-benchmark
row. The manifest was still marked `running` at step 166, with
`possible_lost_steps` 166 and no compute process on GPU 1. The last durable
checkpoint is step 0. Do not treat this partial run as the reproducible screen.

## What was measured

Saved EMA previews of 16 samples, seed 35002. Diversity below is the trainer's
recorded pairwise RMS (`generated_rms`). Saturation is reconstructed from the
checksummed preview PNGs by `reports/preview_saturation_probe.py`:
`byte/127.5 - 1`, then the fraction of channel values with absolute value
above 0.99. PNG quantization can move a value across that threshold by about
0.004. Unpacking was checked by converting the PNG population RMS back to the
pairwise definition, `sqrt(2 * N / (N - 1))` with N=16, which matches the
recorded diversity at both steps.

| Step | Generated saturation | Recorded diversity RMS | Retention | Real-preview saturation | G loss | D adversarial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.04% | 0.83090 | 1 | 20.47% | — | — |
| 100 | 80.47% | 0.21621 | 0.260 | 22.34% | 8.760 | 0.0003 |

Real previews stay near 20–22% saturated. Generated saturation rises by 77
percentage points while between-sample diversity keeps about a quarter of its
initial value. Logged G loss climbs from 4.56 at step 10 to 10.10 at step 160.
D adversarial falls from 0.061 to about 0.0001. The lazy penalty is nonzero at
steps 40, 80, 120, and 160, so the penalty schedule is executing. Throughput is
about 0.89 steps/second.

Low saturation is not the success criterion here. The same previews show both
extreme outputs and lost sample variation, while the critic's adversarial loss
has collapsed. Fixed-opponent descent is not available from these logs and
would not overturn that pair of measurements.

Numbers are in [inspection.json](inspection.json). The reproducible online
screen is a separate 64px benchmark manifest and is not this EMA preview.
