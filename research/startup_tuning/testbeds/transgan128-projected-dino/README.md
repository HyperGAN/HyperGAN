# Logos 128 TransGAN, DINOv3 Projected GAN critic

This is the Sauer et al. Projected GAN critic (ICLR 2022,
https://arxiv.org/abs/2111.01007) on the logos 128 adversarial recipe.
The feature network is frozen DINOv3 ViT-S/16, readout `multidepth`,
blocks 2, 5, 8, and 11. The candidate is ImageNet-normalized and passed
through that backbone once, at batch B. Each of the four maps gets its own
frozen Kaiming 1×1 cross-channel mix (384→384, no bias, no activation) and
its own spectral discriminator. Those heads see an 8×8 map and emit 4×4
logits. The module concatenates them to `[B, 4, 4, 4]`; the relativistic
logistic loss and b-cap penalty reduce that tensor. There is no gray
context, no second image, and no pixel/RGB discriminator. The paper's
ablation found that an RGB branch hurt.

The one adaptation is that there is no CSM. Every DINO map is 8×8. ViT
features are not a CNN pyramid, so there is no higher-resolution feature
map to skip-add and no CSM U-Net.

The generator is the 128px TransGAN. The logos manifest, batch, seeds,
logistic relativistic loss, lazy b-cap penalty, prior, and Adam rates match
the previous 128px TransGAN adversarial recipe. There is no encoder and
there are no objectives. Tuning is off.

The b-cap penalty averages each image's logits before differentiating, so
this 4×4×4 map does not multiply the input gradient by 64. A one-logit critic
is unchanged. The first projected run, stopped at step 810, used the old sum
and is not resumed.

The corrected run has been started. Its step-500 EMA preview is 99.984%
saturated, with only 0.071% of real-preview pairwise diversity. The published
active penalty values through step 500 range from 0.031 to 2.441; collapse
persists after the logit-mean penalty correction. See the
[measured audit](../../results/2026-09-22-projected-scoremean-audit/README.md).
The run directory is
`~/dev/hypergan/training-runs/train-transgan-projected-dinov3-128-scoremean`.

```bash
~/dev/hypergan/training-runs/start-transgan-projected-dinov3-128.sh
```
