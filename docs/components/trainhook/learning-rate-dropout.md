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

| attribute | description | type |
| :--- | :--- | :--- |


| dropout | 0-1 dropout ratio. Defaults to `0.5` | float |
| :--- | :--- | :--- |


| ones | The gradient multiplier when not dropped out. Defaults to `0.1` | float |
| :--- | :--- | :--- |


| zeros | The gradient multiplier when dropped out. Defaults to `0.0` | float |
| :--- | :--- | :--- |


<table>
  <thead>
    <tr>
      <th style="text-align:left">skip_d</th>
      <th style="text-align:left">
        <p>skip d gradients</p>
        <p>Defaults to <code>false</code>
        </p>
      </th>
      <th style="text-align:left">boolean</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<table>
  <thead>
    <tr>
      <th style="text-align:left">skip_g</th>
      <th style="text-align:left">
        <p>skip g gradients</p>
        <p>Defaults to <code>false</code>
        </p>
      </th>
      <th style="text-align:left">boolean</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

{% hint style="info" %}
Floats are [configurable parameters](../../configuration/configurable-parameters.md)
{% endhint %}

