# Realness Loss

* Adapted from [https://github.com/kam1107/RealnessGAN/](https://github.com/kam1107/RealnessGAN/)
* Source: [/losses/realness\_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/realness_loss.py)

## examples

* Configurations: [/losses/realness\_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/realness_loss/)

```javascript
{
  "class": "class:hypergan.losses.realness_loss.RealnessLoss",
  "skew": [-0.1, 0.1]
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| skew |  | array of floats |

