# Logos 128 TransGAN with frozen pretrained ResNet18

Uses the existing `examples/networks/resnet18-multiscale-discriminator-128.hndl`
unchanged: frozen ImageNet ResNet18 layer1/2/3 maps at 32/16/8px, trainable
feature heads, fixed gray context, and the trainable pixel branch. The backbone
checkpoint is local and SHA256-pinned; BatchNorm stays in evaluation mode.
Input gradients pass through the frozen backbone. This is the 128px counterpart
of the CIFAR critic family, not a newly designed projected critic.

The generator is byte-identical to the projected-DINO control, with its original
rates (no FFN rate reduction). Data, prior, loss, penalty, optimizer, batch,
seeds, and schedule are unchanged. Only the discriminator component and recipe
name differ in the resolved configuration. Comparing with the rebuilt projected
DINO critic changes the whole critic, including heads and pixel/context paths;
it does not isolate the pretrained backbone alone.

The initial bounded diagnostic uses GPU 0 and
`../../configs/transgan-128-resnet-screen.json`: 32 disposable updates with
online saturation, diversity, pre-tanh activations, and fixed-latent checks.
The DINO-specific feature observer is disabled. No seed sweep and no startup
tuning. Saturation and diversity determine whether collapse persists; G/D
losses alone do not establish success.

The separate ordinary training launcher is installed at:

```bash
~/dev/hypergan/training-runs/start-transgan-resnet-multiscale-128.sh
```

It uses GPU 0, this worktree's Python sources, `transgan-128-env`, `--no-tune`,
and the fresh run directory `train-transgan-resnet-multiscale-128`. Check GPU
availability before starting. The bounded diagnostic does not start that
ordinary run or create a resumable checkpoint.
