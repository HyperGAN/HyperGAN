---
description: 'https://arxiv.org/abs/1811.07296'
---

# QP Loss

* From [https://arxiv.org/abs/1811.07296](https://arxiv.org/abs/1811.07296)
* Source: [/losses/qp\_loss\_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/qp_loss_loss.py)

```python
lam = 10.0/(reduce(lambda x,y:x*y, gan.output_shape()))
dist = (gan.generator.sample - self.gan.inputs.sample).abs().mean()

dl = - d_real + d_fake
d_norm = 10 * dist
d_loss = ( dl + 0.5 * dl**2 / d_norm).mean()

g_loss = d_real - d_fake
```

## examples

```javascript
{                                                                                       
  "class": "function:hypergan.losses.qp_loss.QPLoss"
}
```

