# Diagnosing 256px colorization collapse

PR: https://github.com/HyperGAN/HyperGAN/pull/362

The original random-prior GAN path collapsed independently of encoder routing.
The selected fix adds learned RGB features and a nonlinear shared head to the
frozen DINOv3 critic. The example and owner launcher now select
`DINOv3ProjectedDiscriminator(head="conv", pixel_width=32)`. These experiments keep the latent-only generator, fixed-sigma 4096-particle prior,
encoder-only grayscale reconstruction, data, and GAN update ordering fixed while
changing the discriminator. The earlier eight-image reconstruction and fixed-fake
classification probes established limited capacity, not adversarial stability.

## DINOv3 wiring audit

The real pinned ViT-S/16 backbone passes a CPU audit against its upstream API:
input conversion from [-1,1] to [0,1] and ImageNet normalization agree exactly;
256 patch tokens exclude the CLS token and four storage tokens; the normalized
final patch map agrees exactly with `get_intermediate_layers(...)[-1]`. The frozen
backbone retains finite nonzero derivatives with respect to image pixels.
The four sampled transformer depths all have 16x16 spatial resolution.

There is no detected DINOv2-to-v3 token or normalization bug. The existing
projected discriminator uses only the last map, two frozen linear convolutions,
attention, and a linear output: 6,145 trainable parameters. This is much smaller
than Projected GAN's nonlinear multiscale discriminators.

## Controlled comparisons

The bounded GPU 1 harness uses the same initial G/E/prior and shared discriminator
weights, independent phase RNG streams, and deterministic CUDA policy. Evaluation
uses 64 fixed hash-ordered held-out images and 64 fixed random-prior samples.
Evaluation restores the training RNG and does not update spectral-normalization
buffers. These raw-model panels are diagnostics, not representative quality
estimates or checkpoint selection by a learned perceptual metric.

The convolutional DINO head retains the frozen backbone/projection and attention,
then applies spectrally normalized nonlinear convolutions at 16, 8, and 4 pixels.
The DCGAN control takes RGB through six convolutional stages with spectral
normalization and no batch normalization; it has no pretrained features.
Both accept only the candidate image, D(X) or D(G(z)).

Pairwise RMS measures spread; pooled pairwise RMS suppresses fine pixel noise;
nearest-neighbor RMS detects repeated outputs. None establishes logo quality.
The current D's AUC measures its own discrimination, not independent fidelity.
Spectral-normalization power iteration also occurs on training-mode forwards with
frozen parameters, so penalty/no-penalty comparisons can differ in buffer updates.

Machine evidence, reproducible scripts, JSON results, training logs, and checkpoints:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-21-colorization-collapse/`.

## Initial discriminator controls

| Discriminator | b-cap | Updates at evaluation | Random pairwise RMS | Pooled random RMS | D AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Original linear DINO head | yes | 350 | 0.1035 | 0.1014 | 0.000 |
| Original linear DINO head | no | 350 | 0.2251 | 0.2093 | 0.726 |
| Convolutional DINO head | yes | 350 | 0.0293 | 0.0244 | 0.000 |
| Convolutional DINO head | no | 1000 | 0.4812 | 0.4735 | 0.999 |
| DCGAN control | yes | 1000 | 0.7836 | 0.7310 | 0.407 |
| Multiscale DINO | yes | 300 | 0.0561 | 0.0510 | 0.000 |
| Multiscale DINO | no | 400 | 0.9121 | 0.8423 | 1.000 |

Reference pairwise RMS is 1.0493; pooled reference RMS is 0.9794. The multiscale
runs were interrupted to pause for compaction; their shorter results do not
qualify stability beyond the original failure horizon. Completed arms have full
model/optimizer/EMA/RNG checkpoints. Interrupted arms retain logs and metrics,
but the harness only checkpoints on completion and therefore cannot resume them.

The **DCGAN control changed only D**. Its generator, encoder, prior, losses,
b-cap, learning rates, and reconstruction ownership stayed fixed. It avoided the
near-constant-output failure through 1,000 updates. This implicates the original
D setup; it does not identify a DINOv3 token bug or prove longer-run stability.
A fixed-latent CPU probe also finds preserved generator sensitivity after DCGAN
training, while the collapsed DINO models progressively suppress variation in
later upsampling blocks. No generator modification was justified by this audit.

A separate exploratory joint-L2 arm was stopped after the user clarified that
mode collapse, rather than image quality/reconstruction, is the immediate task.
It is not evidence for the discriminator-only result and is not the selected
formulation. A second DCGAN seed was also stopped before its first 100-update
evaluation so effort could return to DINO. No image-quality metric run was done.

## Selected discriminator

The working CIFAR discriminator combines a learned pixel path and three
pretrained ResNet feature stages. Its success does not establish that a
pretrained-only DINO critic should work. The selected critic restores local RGB
information while retaining one DINO call and one shared attention/output head:

```text
X or G(z)
  -> [frozen DINOv3 projected features, learned RGB stem]
  -> concatenate at 16x16
  -> shared attention -> shared convolutional head -> scalar D
```

Use `DINOv3ProjectedDiscriminator(head="conv", pixel_width=32)`. The learned
RGB stem uses four stride-two convolutions; there is no additional grayscale
input, separate pixel scalar loss, or generator/encoder change. The default
`pixel_width=0` preserves older architectures and checkpoint keys. The original b-cap settings are retained in the selected recipe, as are
G, E, prior, losses, learning rates, and reconstruction ownership. The RGB critic
has passed 1,500 controlled updates without the earlier near-constant failure.
An otherwise identical no-cap comparison also retained variation. Second-seed
runs were stopped at the user's request; they are not used to qualify the fix.
These are bounded collapse diagnostics, not a claim of useful logo quality or
unlimited training stability.

## Completed RGB-feature comparison

| Discriminator | Updates | Random pairwise RMS | Pooled RMS | Nearest RMS | D AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINO + RGB, original b-cap | 1500 | 0.8106 | 0.7783 | 0.5085 | 0.603 |
| DINO + RGB, no b-cap | 1500 | 0.3291 | 0.3131 | 0.1589 | 1.000 |

The selected critic's pooled spread at step 1,500 is 79.5% of the reference,
compared with 2.5% for the collapsed conv-only control at step 350. Its nearest
sample distance is 0.5085 versus reference 0.5643, so high aggregate spread is
not explained merely by one outlier among near-identical outputs. Variation
fluctuated during training, including a temporary dip at step 800, then recovered.
The no-cap comparison also avoided constant output but weakened late in the run;
its pooled spread fell to 0.1790 at step 1,400 before recovering to 0.3131.
These findings support retaining b-cap for the selected hybrid architecture.

Both first-seed runs completed normally and saved full checkpoints. The second
seeds were interrupted on request (last evaluations at 400 capped / 100 uncapped)
and are not presented as replication evidence. No second-seed qualification is
required by the user. All bounded training jobs are now stopped.

## Gradient-penalty audit

The installed ParticleGAN implementation penalizes excess L2 image-gradient norm
above kappa 1, multiplied by the lazy interval 8. This includes derivatives
through the frozen DINO backbone and normalization. In the collapsed conv-only
control, the first two scheduled penalties were 1,897.49 and 2,148.02, compared
with adversarial losses 0.0583 and 0.6314. The DCGAN control had zero penalty at
all 125 scheduled applications. Identical settings therefore imposed very
different interventions on the two architectures.

A saved-checkpoint probe, with spectral vectors calibrated on in-memory copies,
found mean fake-image gradient norm 16.35 for the uncapped DINO conv critic versus
0.091 for DCGAN. Spectral normalization on the head cannot bound the frozen
backbone's input Jacobian. The cap can push the head to suppress that sensitivity;
this is a mechanistic interpretation, not a measured parameter-gradient ratio
or proof that the penalty alone caused collapse. The selected RGB critic works
with the existing penalty, so no penalty change is included in the owner recipe.

Evidence scripts and receipts: `summarize-bcap-intervention.py`,
`bcap-intervention-summary.json`, `input-gradient-probe.py`, and
`input-gradients-*.json` under the machine evidence directory above.

## Validation and owner state

Independent review found no blocking implementation issue. A fresh installed
wheel passed **928 fast tests**, with all **186 heavy tests deselected**.
The discriminator-specific selection also passed 50 CPU tests, including
input gradients, frozen weights, double backward, sample independence, and
strict state reload. CI's Foundation and Repository integrity checks pass;
heavy CI jobs remain skipped on develop PRs. No prerelease heavy suite ran.

The validated wheel is installed in `training-runs/colorization-critic-env`.
An isolated real-DINO RGB-critic run completed eight updates, including a lazy
b-cap update, checkpoint, and previews. A second invocation resumed that
checkpoint and completed updates 9–16. A 256-sample `random_spread` snapshot
completed with `generated_binding="generated"`; its value is an execution check,
not a trained-model result. Logs and the wheel receipt are in the evidence folder.

`training-runs/start-color.sh` selects the new machine configuration
`logos-colorization-critic-256/colorization.toml`, this installed environment,
and fresh `train-color-critic` directory on physical GPU 1. The owner run is left
unstarted. Earlier random and projected runs/configurations/launchers are
preserved. The example also adds unconditional spread and chroma metrics without
changing the conditional X/B/X_hat preview or conditional metric bindings.
