---
description: 'https://arxiv.org/pdf/1704.00028.pdf'
---

# Gradient Penalty

$$
lambda * relu(||gradients(target, components)||_2 - flex) ^2
$$

## examples

```javascript
{                                                                                       
  "class": "function:hypergan.train_hooks.gradient_penalty_train_hook.GradientPenaltyTra
  "lambda": 1.00,                                                                       
  "flex": 1.0,                                                                          
  "components": ["discriminator"],                                                       
  "target": "discriminator"
}
```

## options

| attribute | description | type |
| :--- | :--- | :--- |


<table>
  <thead>
    <tr>
      <th style="text-align:left">target</th>
      <th style="text-align:left">
        <p>Used in gradients(target, components)</p>
        <p>defaults to <code>discriminator</code>
        </p>
      </th>
      <th style="text-align:left">string (optional)</th>
    </tr>
  </thead>
  <tbody></tbody>
</table><table>
  <thead>
    <tr>
      <th style="text-align:left">lambda</th>
      <th style="text-align:left">
        <p>Loss multiple</p>
        <p>defaults to <code>1.0</code>
        </p>
      </th>
      <th style="text-align:left">float</th>
    </tr>
  </thead>
  <tbody></tbody>
</table><table>
  <thead>
    <tr>
      <th style="text-align:left">components</th>
      <th style="text-align:left">
        <p>Used in gradients(target, components)</p>
        <p>defaults to all components</p>
      </th>
      <th style="text-align:left">array of strings</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>| flex | Max amount of gradient before penalty | float |
| :--- | :--- | :--- |


<table>
  <thead>
    <tr>
      <th style="text-align:left">flex</th>
      <th style="text-align:left">
        <p>Can also be a list for separate X/G flex.</p>
        <p>example: <code>[0.0, 10.0]</code>
        </p>
      </th>
      <th style="text-align:left">array of float</th>
    </tr>
  </thead>
  <tbody></tbody>
</table><table>
  <thead>
    <tr>
      <th style="text-align:left">loss</th>
      <th style="text-align:left">
        <p>Side loss is added to: <code>g_loss</code> or <code>d_loss</code>
        </p>
        <p>defaults to <code>g_loss</code>
        </p>
      </th>
      <th style="text-align:left">string</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>{% hint style="info" %}
Floats are [configurable parameters](../../configuration/configurable-parameters.md)
{% endhint %}

