# Least Squares Loss

* From [https://arxiv.org/abs/1611.04076](https://arxiv.org/abs/1611.04076)
* Source: [/losses/least_squares_loss.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/losses/least_squares_loss.py)

```python
a,b,c = (config.labels or [-1,1,1])
d_loss = 0.5*((d_real - b)**2) + 0.5*((d_fake - a)**2)
g_loss = 0.5*((d_s```

## examples

* Configurations: [/losses/least_squares_loss/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/losses/least_squares_loss/)

```javascript
{
  "class": "function:hypergan.losses.least_squares_loss.LeastSquaresLoss"
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| labels | [a,b,c].  Defaults to `[-1,1,1]` | array of floats \(optional\) |

