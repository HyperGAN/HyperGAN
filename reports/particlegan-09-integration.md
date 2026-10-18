ParticleGAN 0.9 integration was exercised on the historical CIFAR architecture:
two transformer blocks at 64 spatial tokens, convolutional upsampling to 32px,
and a frozen ResNet18 feature critic with three trainable heads. Batch 64,
latent dimension 128 and a 65,536-row particle prior were retained.

This is a new seed-0 cohort with the public deterministic initializer,
independent phase draws, random CIFAR batches with horizontal flips and isolated
checkpointed RNG streams. Both historical trainer candidates used the selected
ParticleGAN recipe (G .012, D .018, prior .03), FP32, TF32 disabled and deterministic
kernels on one RTX A6000. Only a dependency optimizer setting changed; its
implementation belongs to ParticleGAN. Twenty complete updates include data sampling/transfer, both backward passes, the prior and
EMA updates; five warmups are excluded. Preview, FID and checkpoint I/O are
separate. GPU occupancy was not independently certified.

| Historical candidate | Median update | Images/sec | Peak allocated |
| --- | ---: | ---: | ---: |
| Earlier candidate | 389.576 ms | 164.28 | 1,594,270,720 bytes |
| Selected candidate | 294.698 ms | 217.17 | 1,594,270,720 bytes |

The selected run passes the predeclared 300ms gate, with a narrow margin, and is
1.322× faster in this observation. All measured losses/gradients remained finite;
frozen parameters and buffers stayed identical. A three-update continuation
matches every model tensor, optimizer/controller field and RNG stream exactly.
The profile still attributes approximately 100ms per update to dense matrix SVD;
the released acceleration applies to convolution matrices only.

Integration exposed a missing checkpoint contract: the ParticleGAN optimizer used the
base critic observer, which omitted learned residual and data-scale EMAs.
Resuming reset the image penalty's effective threshold even when RNG streams
matched. ParticleGAN now persists those fields and thermal state while preserving
the uninterrupted update law. A second issue was an import-time replicated
backend monkeypatch that replaced the native recipe with the legacy dependency
preset. Native recipe construction is now canonical; unsupported replicated presets fail
explicitly.

Two failed diagnostic runs are retained unchanged outside Git. The first lacked
the serial autograd host scope; adding that scope alone did not repair resume.
Two independently restored trainers then matched exactly after one update,
isolating the omitted observer state rather than CUDA randomness. The corrected
run passed after the observer fix. The new default also binds module ownership,
prior sampled-row ownership, protected-loss backward and the recipe's regularizer
factory. The dependency regularizer was inactive for this batch/latent shape;
the benchmark does not change its sampling law to activate it.

These are historical measurements of dependency settings selected at that time.
The current `scripts/benchmark_particlegan_cifar.py` harness measures only the
released default; it no longer selects or compares internal optimizer rules.
Bulk configurations, events, profiles, checkpoints and continuation diagnostics
remain in the historical integration archive under
`/mnt/ml7tb/hypergan-training-runs/`.
The valid run used the ParticleGAN API
worktree with checkpoint fix `351ee0536`; later observer fixes preserve that
normalized path's numerical law. These performance and numerical checks do not
establish CIFAR image quality. The 200k training run evaluates that separately.
