---
description: 'https://arxiv.org/abs/1912.00144'
---

# Learning Rate Dropout

## examples

```javascript
{
  "class": "function:hypergan.train_hooks.learning_rate_dropout_train_hook.LearningRateDropoutTrainHook",
  "dropout": 0.01,
  "ones": 1e12,
  "zeros": 0.0,
  "skip_d": true
}
```

## options

<table>
  <thead>
    <tr>
      <th style="text-align:center">attribute</th>
      <th style="text-align:center">description</th>
      <th style="text-align:center">type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">dropout</td>
      <td style="text-align:center">0-1 dropout ratio. Defaults to <code>0.5</code>
      </td>
      <td style="text-align:center">float</td>
    </tr>
    <tr>
      <td style="text-align:center">ones</td>
      <td style="text-align:center">The gradient multiplier when not dropped out.
        <br />Defaults to <code>0.1</code>
      </td>
      <td style="text-align:center">float</td>
    </tr>
    <tr>
      <td style="text-align:center">zeros</td>
      <td style="text-align:center">The gradient multiplier when dropped out. Defaults to <code>0.0</code>
      </td>
      <td style="text-align:center">float</td>
    </tr>
    <tr>
      <td style="text-align:center">skip_d</td>
      <td style="text-align:center">
        <p>skip d gradients</p>
        <p>Defaults to <code>false</code>
        </p>
      </td>
      <td style="text-align:center">boolean</td>
    </tr>
    <tr>
      <td style="text-align:center">skip_g</td>
      <td style="text-align:center">
        <p>skip g gradients</p>
        <p>Defaults to <code>false</code>
        </p>
      </td>
      <td style="text-align:center">boolean</td>
    </tr>
  </tbody>
</table>{% hint style="info" %}
Floats are [configurable parameters](../../configuration/configurable-parameters.md)
{% endhint %}

