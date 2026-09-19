# Softmax Loss

* Adapted from [https://arxiv.org/abs/1704.06191](https://arxiv.org/abs/1704.06191)
* Source: [/losses/softmax\_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/softmax_loss.py)

```python
ln_zb = (((-d_real).exp().sum()+(-d_fake).exp().sum())+1e-12).log()

d_target = 1.0 / d_real.shape[0]
g_target = d_target / 2.0

g_loss = g_target * (d_fake.sum() + d_real.sum()) + ln_zb
d_loss = d_target * d_real.sum() + ln_zb
```

## examples

* Configurations: [/losses/softmax\_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/softmax_loss/)

```javascript
{                                                                                       
  "class": "function:hypergan.losses.softmax_loss.SoftmaxLoss",
}
```

