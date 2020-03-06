# ALI Loss

* From [https://ishmaelbelghazi.github.io/ALI/](https://ishmaelbelghazi.github.io/ALI/)
* Source: [/losses/ali_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/ali_loss.py)

```python
g_loss = criterion(d_fake, torch.ones_like(d_fake)) + criterion(d_real, torch.zeros_like(d_real))
d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
```

## examples

* Configurations: [/losses/ali_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/ali_loss/)

```javascript
{                                                                                       
  "class": "function:hypergan.losses.ali_loss.AliLoss"
}
```

