ParticleGAN 0.9 integration was exercised on the historical CIFAR architecture:
two transformer blocks at 64 spatial tokens, convolutional upsampling to 32px,
and a frozen ResNet18 feature critic with three trainable heads. Batch 64,
latent dimension 128 and a 65,536-row particle prior were retained.

This is a new seed-0 cohort with the public deterministic initializer,
independent phase draws, random CIFAR batches with horizontal flips and isolated
checkpointed RNG streams. Both trainer candidates use the complete selected
BISPF recipe (G .012, D .018, prior .03), FP32, TF32 disabled and deterministic
kernels on one RTX A6000. Only the convolution rule changes. Twenty complete
updates include data sampling/transfer, both backward passes, the prior and
EMA updates; five warmups are excluded. Preview, FID and checkpoint I/O are
separate. GPU occupancy was not independently certified.

| Convolution rule | Median update | Images/sec | Peak allocated |
| --- | ---: | ---: | ---: |
| Native polar | 389.576 ms | 164.28 | 1,594,270,720 bytes |
| Quintic | 294.698 ms | 217.17 | 1,594,270,720 bytes |

The quintic run passes the predeclared 300ms gate, with a narrow margin, and is
1.322× faster in this observation. All measured losses/gradients remained finite;
frozen parameters and buffers stayed identical. A three-update continuation
matches every model tensor, optimizer/controller field and RNG stream exactly.
The profile still attributes approximately 100ms per update to dense matrix SVD;
the released acceleration applies to convolution matrices only.

Integration exposed a missing checkpoint contract: normalized BISPF used the
base critic observer, which omitted learned residual and data-scale EMAs.
Resuming reset the image penalty's effective threshold even when RNG streams
matched. ParticleGAN now persists those fields and thermal state while preserving
the uninterrupted update law. A second issue was an import-time replicated
backend monkeypatch that replaced the native recipe with legacy KA2. Native
recipe construction is now canonical; unsupported replicated presets fail
explicitly.

Two failed diagnostic runs are retained unchanged outside Git. The first lacked
the serial autograd host scope; adding that scope alone did not repair resume.
Two independently restored trainers then matched exactly after one update,
isolating the omitted observer state rather than CUDA randomness. The corrected
run passed after the observer fix. The new default also binds module ownership,
prior sampled-row ownership, protected-loss backward and the recipe's SBNP
factory. SBNP is inactive by its declared minimum-row rule at batch 64/z128;
the benchmark does not change its sampling law to activate it.

Reproduction source: `scripts/benchmark_particlegan_cifar.py`. Local bulk archive:
`/mnt/ml7tb/hypergan-training-runs/bispf-integration-smoke/` contains immutable
`quintic-v1`, `quintic-serial-v2`, `quintic-checkpoint-v3` and
`polar-checkpoint-v3` directories, with resolved configs, events, profiles,
checkpoints and exact-resume diagnostics. The valid run used the ParticleGAN API
worktree with checkpoint fix `351ee0536`; later observer fixes preserve that
normalized path's numerical law. These performance and numerical checks do not
establish CIFAR image quality. The 200k training run evaluates that separately.
