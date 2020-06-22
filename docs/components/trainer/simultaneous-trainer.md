# Simultaneous Trainer

* Source: [/trainers/simultaneous_trainer.py](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/trainers/simultaneous_trainer.py)

```python
d_grads,g_grads = self.calculate_gradients(D, G)
self.train_d(d_grads)
self.train_g(g_grads)
```

## examples

* Configurations: [/trainers/simultaneous_trainer/](https://github.com/HyperGAN/HyperGAN/tree/pytorch/hypergan/configurations/components/trainers/simultaneous_trainer/)

```javascript
{
  "class": "function:hypergan.trainers.simultaneous_trainer.SimultaneousTrainer",
  "optimizer": {
    "class": "function:torch.optim.Adam",
    "lr": 1e-4,
    "betas":[0.0,0.999]
  },
  "hooks": [
    {
      "class": "function:hypergan.train_hooks.adversarial_norm_train_hook.AdversarialNormTrainHook",
      "gamma": 100,
      "loss": ["d"]
    },
    {
      "class": "function:hypergan.train_hooks.negative_momentum_train_hook.NegativeMomentumTrainHook",
      "gamma": 0.33
    }
  ]
}
```
## options

| attribute | description | type |
| :--- | :--- | :--- |
| optimizer | Optimizer configuration | Config \(required\) |
| hooks | Train Hooks | Array of configs \(optional\) |


