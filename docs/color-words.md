# Eight-color words

`examples/color-words.toml` is a CPU proof, not a qualified image model. It claims only this: eight noisy solid colors, one generator, one shared particle codebook, and three critics declared as adversarial terms. There is no CLIP tower, no second image generator, and no token decoder.

Labels are fixed in this order: red, green, blue, yellow, cyan, magenta, white, black. `hypergan.color_words:ColorWords` draws them with the caller-owned generator. Images are 16×16 RGB. The canonical colors sit on the cube corners in `[-1, 1]` — red `(1,-1,-1)`, green `(-1,1,-1)`, blue `(-1,-1,1)`, yellow `(1,1,-1)`, cyan `(-1,1,1)`, magenta `(1,-1,1)`, white `(1,1,1)`, black `(-1,-1,-1)`. Gaussian noise of standard deviation 0.05 is added and clamped back into `[-1, 1]`, which truncates noise that would leave a corner. The factory reads no files and downloads nothing. `ColorWords.resume_stateless` is true, so checkpoints treat the source as stateless.

Both encoders read the same MoG table (`prior.means`, `prior.sigma`) and the same `ColorGenerator` decodes every latent. The text encoder's vector is a learned embedding of the eight labels, not a pretrained text tower. `ZEmbedding` is a linear map from the prior latent into that embedding size; it is the fake sample for the text critic, not a token generator.

The router does not layer-normalize the query and does not call the 256px DINO encoder. Distance is mean squared Euclidean distance to detached means (sum of squares divided by the latent size). Responsibilities are a softmax at temperature 0.125. The straight-through center is `means[ids] + (soft @ fixed - (soft @ fixed).detach())`. The code is `center + sigma * 3 * tanh(offset/3)`. Offset maps are zero-initialized, so a new encoder sits on the selected center. Routing samples no noise. Sigma is the prior's fixed value 0.05.

`[adversarial]` is the image marginal on `G(z)`: the image critic scores `batch.real` and `generated`, with `[gradient_penalty]` applied to that term only. Every term uses ParticleGAN 0.8's paired (RpGAN) loss. The recipe used the vanilla loss before 0.8 so that a shared condition was not subtracted out of both scores; with pairing, a score component that depends only on the shared condition cancels. The three `[[adversarial_terms]]` entries are:

- `image-from-text`: the same image critic on `batch.real` versus `G(E(text))`, with no penalty, so the image critic is not penalized twice.
- `text-marginal`: the text critic on `E(text)` versus `ZEmbedding(z)`, penalty coefficient 0.1.
- `joint`: the joint critic on the real image versus `G(E(text))`, conditioned on the text embedding, penalty coefficient 0.1.

Loss type and mode are omitted on those entries and inherit `[adversarial]`. Conditioning stays detached on critic forwards. Encoder parameters are not given an ALI-style path through the critic. They train on the generator step from RGB reconstruction, the latent MSE (image side detached), and attached fakes.

The CI gate is the gradient and optimizer-ownership tests in `tests/reference/test_color_words.py`, at the recipe seed, batch 8, and one update. They record critic-phase gradients before `opt_d.step` and generator-phase gradients before `opt_g.step`. The image critic must see both the legacy `G(z)` sample and `G(E(text))`. The text and joint critics must receive a critic-phase gradient. The joint condition is the text embedding, and that tensor does not require grad on the critic forward. After the critic step, image, text, and joint critic parameters have changed and the generator, both encoders, and z-embedding have not. After the generator step, that ownership reverses. A weight of 0 on `latent-text` must still run the latent forward. `validate_replicated_recipe` rejects this resolved recipe; `resolve_config({})` does not. Replicated execution is not qualified for reused modules or prior bindings.

Metrics are computed from tensors by `color_word_metrics`. Label recovery is the fraction of `G(E(text))` spatial means whose nearest canonical color matches `label`. Chance is 0.125. Recovery near 0.125 means that run needs investigation. It does not get another seed. The same function reports mean absolute RGB error of `G(E(text))` and of `G(E(image))` against `real`, the fraction of rows whose image and text particle ids match, and the count of distinct ids in the union of those rows. Agreement without that count is not success. The joint swap gap scores `(real, true embedding)` minus `(real, embedding of (label + 1) % 8)`. The swapped label is forced; it is not a shuffle.

The checked training length is 200 steps at seed 25021, batch 16, on CPU, with constant learning rate (`lr_floor` 1). That test is marked `heavy` and only asserts that the four measurements are finite. It does not assert convergence. The default suite does not run it:

```sh
PYTHONPATH=src python -m pytest tests/reference/test_color_words.py
PYTHONPATH=src python -m pytest tests/reference/test_color_words.py -m heavy
```

One 200-step run at seed 25021, measured on the last batch of 16, gave label recovery 0.1875 (chance 0.125), `G(E(text))` mean absolute RGB error 0.811, reconstruction error 0.790, particle agreement 0.1875 across 7 distinct particles, and joint swap gap 0.251. Recovery is one image above chance on that batch. The mechanism tests passed, including a critic-phase gradient and a critic-optimizer step for each critic, so this is not a disconnected term. Two hundred steps did not move recovery off chance. That run does not get another seed.
