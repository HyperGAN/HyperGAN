# Accumulate Gradient Trainer

* Source: [/trainers/accumulate_gradient_trainer.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/trainers/accumulate_gradient_trainer.py)

```python
d_grads, g_grads = self.calculate_gradients()
if accumulated_count == config.accumulate:
    self.train_g(average_g_grads)
    accumulated_count = 0
else:
    train_d(d_grads)
    average_g_grads += g_grads / config.accumulate
    accumulated_count += 1
```

## examples

* Configurations: [/trainers/accumulate_gradient_trainer/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/trainers/accumulate_gradient_trainer/)

```javascript
{
  "class": "class:hypergan.trainers.accumulate_gradient_trainer.AccumulateGradientTrainer",
  "accumulate": 10,
  "d_optimizer": {
    "class": "class:torch.optim.Adam",
    "lr": 1e-4,
    "betas":[0.0,0.999]
  },
  "g_optimizer": {
    "class": "class:torch.optim.Adam",
    "lr": 1e-4,
    "betas":[0.0,0.999]
  },
  "hooks": [
    {
      "class": "function:hypergan.train_hooks.gradient_norm_train_hook.GradientNormTrainHook",
      "gamma": 1e3,
      "loss": ["d"]
    }
  ]
}
```
## options

| attribute | description | type |
| :--- | :--- | :--- |
| g_optimizer | Optimizer configuration for G | Config \(required\) |
| d_optimizer | Optimizer configuration for D | Config \(required\) |
| hooks | Train Hooks | Array of configs \(optional\) |
| accumulate | Amount of steps to accumulate G.  Defaults to 3 | Integer \(optional\) |

