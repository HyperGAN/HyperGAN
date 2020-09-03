---
description: Custom research
---

# Adversarial norm

## examples

```javascript
    {
        "class": "function:hypergan.train_hooks.adversarial_norm_train_hook.AdversarialNormTrainHook",
        "gammas": [-1e12, 1e12],
        "offset": 1.0,
        "loss": [
          "dg"
        ],
        "mode": "fake"
      },
      {
        "class": "function:hypergan.train_hooks.adversarial_norm_train_hook.AdversarialNormTrainHook",
        "gamma": -1e12,
        "offset": 1.0,
        "loss": [
          "d"
        ],
        "mode": "real"
      }

```
