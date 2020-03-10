# Balanced Trainer

* Source: [/trainers/balanced_trainer.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/trainers/balanced_trainer.py)

```python
fake, real = self.gan.forward_discriminator()
if d_real < (fake+config.imbalance):
  self.train_d()
else:
  self.train_g()
```

## examples

* Configurations: [/trainers/balanced_trainer/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/trainers/balanced_trainer/)

```javascript
{
  "class": "class:hypergan.trainers.balanced_trainer.BalancedTrainer",
  "imbalance": 0.06,
  "pretrain_d": 1000,
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
      "gamma": 2e4,
      "loss": ["d"]
    },
    {
      "class": "function:hypergan.train_hooks.initialize_as_autoencoder.InitializeAsAutoencoder",
      "steps": 10000,
      "optimizer": {
        "class": "class:torch.optim.Adam",
        "lr": 1e-4,
        "betas":[0.9,0.999]
      },
      "encoder": {
        "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
        "layers":[
          "conv 32 stride=1", "adaptive_avg_pool", "relu",
          "conv 64 stride=1", "adaptive_avg_pool", "relu",
          "conv 128 stride=1", "adaptive_avg_pool", "relu",
          "conv 256 stride=1", "adaptive_avg_pool", "relu",
          "conv 512 stride=1", "adaptive_avg_pool", "relu",
          "conv 512 stride=1", "adaptive_avg_pool", "relu",
          "flatten",
          "linear 256 bias=false", "tanh"
        ]
      }
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
| pretrain_d | First N steps only trains D | Integer \(optional\) |
| imbalance | Threshold distance for G training.  Defaults to 0.1 | Float \(optional\)  |
| d_fake_balance | Changes conditional to `d_fake(t) > d_fake(t-1)` | Boolean \(optional\)  |

