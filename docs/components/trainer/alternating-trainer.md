# Alternating Trainer

* Source: [/trainers/alternating.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/trainers/alternating_trainer.py)

```python
d_grads = self.calculate_gradients(D)
self.train_d(d_grads)
g_grads = self.calculate_gradients(G)
self.train_g(g_grads)
```

## examples

* Configurations: [/trainers/alternating\_trainer/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/trainers/alternating_trainer/)

```javascript
"trainer": {
  "class": "class:hypergan.trainers.alternating_trainer.AlternatingTrainer",
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
      "class": "function:hypergan.train_hooks.adversarial_norm_train_hook.AdversarialNormTrainHook",
      "gamma": 1e3,
      "loss": ["d"]
    }
  ]
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |
| g\_optimizer | Optimizer configuration for G | Config \(required\) |
| d\_optimizer | Optimizer configuration for D | Config \(required\) |
| hooks | Train Hooks | Array of configs \(optional\) |
| train\_d\_every | train D every N steps | Integer \(optional\) |
| train\_g\_every | train G every N steps | Integer \(optional\) |
| pretrain\_d | First N steps only trains D | Integer \(optional\) |

