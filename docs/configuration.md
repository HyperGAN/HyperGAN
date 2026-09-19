# Recipe configuration

A project is a directory containing `config.toml`, or a TOML file passed directly to `hypergan validate` and `hypergan train`. Run `hypergan new demo` to obtain the complete default configuration. Defaults are resolved into the run manifest so experiments do not depend on hidden CLI settings.

The initial runtime is a bounded CPU numerical reference. It has a configurable component/data interface, not a qualified folder-of-images recipe or a distributed strategy.

## Components and I/O

`components.generator` and `components.discriminator` are required. Other named components can represent an encoder, feature extractor or auxiliary transform. Each component has a `factory`, constructor `args`, explicit forward keyword `inputs`, and a `trainable` flag.

Factories are built-in identifiers such as `mlp`, `linear` and `identity`, or explicit `module:object` paths available in the Python environment. Custom model factories must construct a `torch.nn.Module`. Unknown constructor arguments fail instead of being silently discarded.

Bindings refer to `latent`, `batch.<field>`, `components.<name>` and their nested outputs. Generator output is available as `generated`; discriminator candidate input is `candidate`. Auxiliary components execute when an input or objective requires them. Cyclic or missing bindings fail. Dictionary/list outputs can be selected through dotted keys/indices.

The runnable [paired synthetic example](../examples/paired-linear.toml) exercises a trainable encoder, conditional generator/discriminator and weighted reconstruction loss:

```sh
hypergan train examples/paired-linear.toml --run-dir runs/paired
hypergan sample runs/paired --count 8 --output paired-samples.json
```

CLI sampling reuses saved example conditions and records that provenance in the JSON output. The Python `hypergan.artifacts.sample(..., inputs={...})` interface accepts fresh batched conditions. This is an I/O fixture, not a colorization or super-resolution model.

For conditional generation, bind a generator keyword to `batch.condition` or an encoder output and bind the discriminator to the corresponding condition. The data factory supplies both the condition and real target. This can express paired tasks, but an image architecture, preprocessing and task evaluation still need their own implementation and qualification.

Trainable auxiliary components belong to the generator optimizer. The discriminator has its own optimizer; discriminator conditioning is detached. Frozen components stay in evaluation mode. Separate encoder optimizers or arbitrary alternating schedules are outside this initial runtime and must not be implied by an encoder's presence.

## Numerical recipe

The configuration separates `prior`, `adversarial`, `gradient_penalty`, `prior_regularizer`, `objectives`, `optimizer`, `training` and `sampling`. The generated default uses a particle prior, paired relativistic logistic loss, b-cap and VICReg. The exact resolved parameters are saved with each run.

The discriminator penalty uses ParticleGAN arm names: `b_cap` is the default; `a_r1r2` denotes its paired zero-centered R1/R2 alternative. Changing an arm produces a custom, unqualified configuration. Do not treat these names as interchangeable with other papers' formulations.

Additional objectives select a loss factory, input bindings and weight. Reconstruction objectives such as MSE or L1 can connect generated output and paired targets. Custom task losses can be imported through the same factory mechanism. The runtime's supported update ownership is explicit; it does not infer a new training algorithm from component names.

Changing objective or regularizer settings may change batch/distributed semantics. No configuration in this checkpoint is approved for multi-GPU or cluster execution. Unsupported device/execution settings must fail rather than silently run a different profile.

## Validation and qualification

`hypergan validate demo` checks the configuration without importing torch or executing custom Python factories. Training constructs components and validates their actual interfaces. Configuration errors, unavailable bindings and runtime failures return a nonzero exit status.

The built-in configuration is labelled reference-only, with its runtime recorded as not certified. It does not receive an application approval stamp. Future qualification must apply to an exact resolved configuration, component versions and tested execution profile. Custom values or implementations receive an unqualified warning and may run when compatible. Numerical-reference qualification does not establish image quality or application readiness. Review the recorded qualification and configuration in the run manifest rather than relying on a recipe's name.

Sample artifacts preserve inference state. Complete optimizer/RNG/data-position checkpoint recovery belongs to the next milestone; changing a seed or loading an inference artifact is not exact training resume.
