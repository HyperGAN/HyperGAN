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

It has not been started. The fresh run directory is
`~/dev/hypergan/training-runs/train-transgan-projected-dinov3-128`.

```bash
~/dev/hypergan/training-runs/start-transgan-projected-dinov3-128.sh
```
