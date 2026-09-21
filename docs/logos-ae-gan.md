# RGB logo AE-GAN at 256×256

`examples/logos-ae-gan-256.toml` separates the ordinary autoencoder task from
colorization. It uses the same 256px generator, 4,096 particles and logo inventory,
with the existing DINOv3 projected convolutional discriminator plus RGB stem.
This is a new run: do not resume a colorization checkpoint with this configuration.

```text
X = RGB logo
E(X) -> (query, offset)
k = nearest particle to query
z_X = particle[k] + sigma * 3*tanh(offset/3)
G(z_X) -> X_hat
L2(X_hat, X)                        -> update E, G, particle means

z ~ uniform particle + sigma * Gaussian noise
D(X), D(G(z))                      -> adversarial updates to D, G, particle means
```

The encoder implements ParticleGAN's deterministic `ae_gan` / `particle_ae`
encoding. There is no reconstruction noise, KL term, grayscale objective or
parameter freeze on the reconstruction path. Hard routing has a soft
straight-through query gradient; selected centers receive reconstruction
gradients, including the prior's existing table standardization. Sigma is
calibrated once and fixed. The offset head starts at zero.

The reference is ParticleGAN commit
`d2a6450985282047d2d27c36eb80ff9f5200f8c9`:
[recipe and caller-owned loop](https://github.com/255BITS/ParticleGAN/blob/d2a6450985282047d2d27c36eb80ff9f5200f8c9/docs/particle-autoencoders.md),
[encoding](https://github.com/255BITS/ParticleGAN/blob/d2a6450985282047d2d27c36eb80ff9f5200f8c9/particlegan/autoencoder.py),
and [image encoder](https://github.com/255BITS/ParticleGAN/blob/d2a6450985282047d2d27c36eb80ff9f5200f8c9/lib/image_particle_autoencoder.py).
The package presets describe composable settings, not a complete tested 256px
architecture. This reproduces the AE formulation; logo quality is unproven.

The image adaptation uses layer-normalized queries, mean squared routing distance
and temperature .125. Other explicit departures from the toy `ae_gan` defaults
are latent dimension 128, 4,096 particles, batch 16, 200,000 updates, these image
networks, GPU backend and checkpointed RNG streams. The recipe retains RGB MSE
weight 1, logistic Rp, prior spread weight 1, fixed sigma_rel .025, constant
G/E LR .0003, D LR .00045, prior LR .003, Adam betas (0,.999) and prior betas
(.5,.999), and EMA .995. B-cap uses coefficient 1, kappa 1 and the AE recipe's
lazy interval **4** (the previous colorization trial used 8).

The generator is exactly `ColorizationGenerator(z_dim=128,width=32)`: its name
is historical; it takes only a latent and produces RGB. The discriminator is
exactly the previously tested `DINOv3ProjectedDiscriminator(feature_width=64,
head="conv",pixel_width=32)`. Frozen DINOv3 ViT-S/16 features and a learned RGB
stem feed the same attention/convolution head. It sees only the RGB candidate.
No generator or discriminator architecture changes accompany the AE objective.

The existing dataset adapter still computes an unused grayscale tensor. All
model inputs, reconstruction targets and paired previews use RGB. Reuse the
manifest and local hash-pinned DINO source/weights from the colorization setup;
there are no downloads or new dataset splits. See [data preparation](colorization.md#data-preparation).

Previews have one example per row with `X | X_hat` columns and a separate random
sample shelf. AE reconstructions are deterministic for a fixed model/input;
repeated-condition diversity is therefore removed. Existing held-out chroma
and edge metrics describe reconstruction. Random-prior spread and chroma
metrics separately monitor the GAN. Edge error is related to reconstruction,
and spread can remain high for saturated or nonsensical images: none of these
alone establishes quality or absence of collapse. Training also reports the
RGB reconstruction loss directly.

On the prepared machine:

```sh
bash ~/dev/hypergan/training-runs/start-ae.sh
```

This pins physical GPU1 and uses `logos-ae-env`, `logos-ae-gan-256/ae.toml`, and
a fresh `train-ae-dino` directory. Ctrl-C saves a recoverable boundary; repeat
the command to resume. `start-color.sh` continues to select the previous ResNet
colorization trial. The AE smoke run is separate from the owner run, which is
left unstarted for manual control.
