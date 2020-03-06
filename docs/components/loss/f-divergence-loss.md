# F Divergence Loss

* From [https://arxiv.org/abs/1606.00709](https://arxiv.org/abs/1606.00709)
* Source: [/losses/f_divergence_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/f_divergence_loss.py)
* Configurations: [/losses/f_divergence_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/f_divergence_loss/)

```javascript
{
  "class": "class:hypergan.losses.f_divergence_loss.FDivergenceLoss",
  "type": "js",
  "g_loss_type": "js",
  "regularizer": "js"
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| type | supported types `js`,`js_weighted`,`gan`,`reverse_kl`,`pearson`,`jeffrey`,`alpha1`,`alpha2`,`squared_hellinger`,`neyman`,`total_variation`,`alpha1`.  Defaults to `gan`  | string \(optional\) |
| g_loss_type | Defaults to type's value | string \(optional\) |
| regularizer | Defaults to none.  Same options as type | string \(optional\) |

