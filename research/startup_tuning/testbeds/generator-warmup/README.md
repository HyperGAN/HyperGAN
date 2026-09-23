# Finite generator-heavy startup experiment

Research runner: `research/startup_tuning/generator_warmup_screen.py`.
Uses the unchanged `transgan-resnet128/transgan-resnet.toml`, original seed25002,
pretrained ResNet, logos128 data, G/D rates2e-4 and betas[.5,.999]. No architecture
change, seed sweep, source training-TOML edit, or production --tune change.

Initial candidate: 32 rounds of 1D:4G, then96 rounds of native1D:1G. A round means
one original D/G/prior update, followed by three extra G-only updates during
warmup. Thus128 rounds contain224 G updates,128 D updates and128 prior updates.
Extra G updates use fresh real/latent draws and the native adversarial objective;
prior parameters/moments, critic parameters/buffers and penalty RNG stay fixed.
G EMA follows G updates; prior and critic EMA, learning-rate annealing and lazy
D penalties follow rounds. Training draws therefore diverge from baseline after
the first native update; initial weights and monitoring bank must match exactly.

Existing512-round source result supplies the baseline at128; no repeat is needed.
Observe saturation, pixel/spatial/pooled diversity versus the same real bank,
fixed-latent variation and pre-tanh amplitude. The first candidate retains the
source G learning rate. If it fails, a distinct diagnostic may use four smaller
G updates (rate/4 during warmup only), returning to source rates at round33.
Report that as a separate hypothesis, not as the requested unchanged-rate test.

The128-round horizon tests early behavior and the transition, not final quality
or Nash equilibrium. A collapsed128 result need not prove permanent failure:
working CIFAR previously recovered after a severe early saturation transient.

GPU0 only, after checking availability. GPU1 stays reserved. Use worktree
PYTHONPATH=src and the transgan-128-env Python; never python -I. Example arguments:
`--ratio 4 --warmup-rounds 32 --steps 128 --output /path/to/new-case`.
The runner saves source/config/code snapshots and restores full trainer state.
No training checkpoint is retained. PR382 remains open and unmerged.

## User-directed interpolation follow-up

`--extra-penalty e_interp` applies `(norm(grad D(x_interp)) - 1)^2` to D
before each extra G update. Interpolation samples uniformly on straight lines
between matched real/fake rows. G then uses the same fresh real/latent draw.
This penalizes gradient magnitude, not direction or semantic distance.

The coefficient remains1, applied each extra step without the native8x lazy
multiplier. The normal D update retains its original endpoint b_cap penalty
every8 rounds. Interpolation and extra G updates both end after round32.
The extra D penalty uses a separate Adam with source D rate/betas and fresh
moments, so no adversarial momentum masquerades as a penalty-only update.
Zero penalty skips its optimizer step. Count adversarial D and penalty-only D
updates separately (128 versus up to96). Prior still updates128 times.

This is a changed startup regularization objective as well as a schedule test.
It follows the user's explicit selection of real–fake interpolation rather
than merely applying the inactive existing cap more often. No ongoing controller.
