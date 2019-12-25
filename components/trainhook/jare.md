---
description: 'https://arxiv.org/abs/1806.09235'
---

# JARE

## examples

```javascript
{
    "class": "function:hypergan.train_hooks.jare_train_hook.JARETrainHook",                                       
    "gamma": "anneal(0.9 multiplier=2.5 metric=jare)"
}
```

```javascript
{
    "class": "function:hypergan.train_hooks.jare_train_hook.JARETrainHook",                                       
    "d_gamma": 2.0,
    "g_gamma": 0
}
```

{% embed url="https://github.com/weilinie/JARE/blob/master/src/ops.py\#L226" %}

## options

| attribute | description | type |
| :---: | :---: | :---: |
| gamma | loss multiplier. | float |
| d\_gamma | loss multiplier for discriminator.  overrides `gamma` | float |
| g\_gamma | loss multiplier for generator.  overides `gamma` | float |

{% hint style="info" %}
Floats are [configurable parameters](../../configuration/configurable-parameters.md)
{% endhint %}

