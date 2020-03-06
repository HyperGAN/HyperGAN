# Standard Loss

* Source: [/losses/standard_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/standard_loss.py)

```python
criterion = torch.nn.BCEWithLogitsLoss()
g_loss = criterion(d_fake, torch.ones_like(d_fake))
d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
```

## examples

* Configurations: [/losses/standard_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/standard_loss/)

```javascript
{
  "class": "class:hypergan.losses.standard_loss.StandardLoss"
}
```

