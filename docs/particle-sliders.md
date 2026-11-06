# Particle LoRAs and frozen-model hosts

HyperGAN can supply the training loop for a particle slider through its
[custom PyTorch module interface](custom-pytorch.md). The application supplies
the frozen base model, adapter, teacher responses and inference binding.
HyperGAN does not ship a pretrained particle-slider catalog or a turnkey loader
for every image, audio or language model.

A routed adapter can share one trainable FP32 particle bank of shape `[128, 4]`
across selected frozen linear projections. One concrete architecture uses a
rank-8 down projection, a router that queries the bank, a nonlinear residual
network and a rank-8 up projection:

```text
h       = down_8(x)
weights = softmax(router(h) @ particles.T / sqrt(4))
z       = weights @ particles
delta   = up_8(residual_network(concat(h, z)))
output  = frozen_projection(x) + strength * delta
```

The zero-initialized up projection starts at the frozen model's output. A
strength of zero should bypass the adapter exactly. This is an input-dependent
particle LoRA: its router and residual network prevent treating it as a single
constant matrix update that can always be merged into the base weights.

## Training ownership

Bind the adapter's bank as the **same parameter object** used by the ParticleGAN
prior; a second, disconnected bank does not regularize the routed adapter.
Register the remaining adapter parameters once as generator parameters and
the critic parameters separately. Keep the frozen host outside optimizer groups
and checkpoint copies while retaining gradients through its operations to the
adapter. HNDL graphs can implement the adapter and critic alongside a Python
host; see [HNDL integration](hndl-upstream-requirements.md).

For paired response training, the generator can expose the adapted neutral
response minus a frozen positive response, normalized per response coordinate.
The application owns the pairing, normalization, shared noise and response
surface. Preserve that surface when defining the critic; a synthetic projection
test alone does not qualify training on the full pretrained model.

ParticleGAN owns the adversarial recipe, critic penalty, prior regularizer,
optimizers and schedules; see [the integration boundary](particlegan.md).
Using ParticleGAN 0.9.4 does not by itself select its current `bispf_optimized`
preset. In particular, the local P-H3 custom-host prototype uses explicit
`Recipe(...)` restoration settings with **KA2, VICReg and cosine scheduling**.
It is a separate formulation from the current BISPF preset, and its CPU
integration evidence does not establish pretrained GPU training or image quality.
Use `defaults = "particlegan"` for the released preset in ordinary HyperGAN
configurations; retain the entire resolved recipe for a custom host as well.

## Checkpoints and inference

A resumable training checkpoint needs generator/router weights, the shared
bank, critic, optimizer and controller state, EMA, RNG streams and completed
step. Bind it to the exact frozen-model identity, adapter target names, cache,
normalization, recipe and runtime/source fingerprints. An adapter-only weight
file is an inference artifact, not a complete training checkpoint. Custom hosts
must also restore their externally owned state and reattach their callbacks;
see [recovery](recovery.md).

For inference, export the selected raw or EMA adapter **and its corresponding
particle bank**, plus the architecture, target projections and frozen-model
identity. Reconstruct the routed adapter on that base model and apply the saved
strength semantics. An external host captured through Python callbacks needs
an application-owned loader; the built-in [standalone export](deployment.md)
does not automatically capture that host. Verify export/reload equivalence and
strength-zero identity before treating an adapter as deployable.

Report source compatibility, native-model training and published trained
controls separately. Passing CPU integration or checkpoint tests establishes
the first of these, not the other two.
