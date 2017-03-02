# Config examples

Here are a few configurations we've found interesting.  New configs are welcome (open a pull request off develop).

## improved-gan.json

A noisy discriminator combined with the original GAN formula.  The discriminator has label smoothing and adds instance noise but does not include minibatch.

> https://arxiv.org/abs/1606.03498

## wgan.json

Value clipping and dcgan.

https://arxiv.org/abs/1701.07875

> discussion at
> https://www.reddit.com/r/MachineLearning/comments/5qxoaz/r_170107875_wasserstein_gan/

## wgan-and-standard.json

Combines wgan with standard gan as two separate discriminators.  No value clipping.

> Standard GAN and wgan are compatible as different losses on different discriminators in the same GAN architecture.

## multi-wgan.json

Combines multiple wgan discriminators with multiple losses.

> This shows that you can have multiple discriminators/losses with wgan powering each.
