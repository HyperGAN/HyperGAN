# Wasserstein Loss

* Source: [/losses/wasserstein\_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/wasserstein_loss.py)

```python
d_loss = -d_real + d_fake
g_loss = -d_fake
```

## examples

* Configurations: [/losses/wasserstein\_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/wasserstein_loss/)

```javascript
{
  "class": "function:hypergan.losses.wasserstein_loss.WassersteinLoss",
  "kl": true
}
```

```javascript
{
  "class": "function:hypergan.losses.wasserstein_loss.WassersteinLoss",
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| kl | [https://arxiv.org/abs/1910.09779](https://arxiv.org/abs/1910.09779) Defaults to `false` | boolean \(optional\) |

