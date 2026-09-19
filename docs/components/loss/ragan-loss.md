# RAGAN Loss

* Adapted from [https://arxiv.org/abs/1807.00734](https://arxiv.org/abs/1807.00734)
* Source: [/losses/ragan\_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/ragan_loss.py)

```python
# wasserstein type
cr = torch.mean(d_real,0)
cf = torch.mean(d_fake,0)
d_loss = -(d_real-cf) + (d_fake-cr)
g_loss = -(d_fake-cr)
```

## examples

* Configurations: [/losses/ragan\_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/ragan_loss/)

```javascript
{
  "class": "function:hypergan.losses.ragan_loss.RaganLoss",
  "type": "hinge"
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| type | `least_squares`,`hinge`,`wasserstein` or `standard`.  Defaults to `standard` | string \(optional\) |
| rgan | rgan does not average over batch.  Defaults to `false` | boolean \(optional\) |
| labels | \[a,b,c\].  Defaults to `[-1,1,1]`.  Only used in `least_squares` type | array of floats \(optional\) |

