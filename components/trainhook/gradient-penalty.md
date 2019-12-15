---
description: 'https://arxiv.org/pdf/1704.00028.pdf'
---

# Gradient Penalty

$$
lambda * || relu(abs(gradients(target, components)) - flex) ||_2
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
      <td style="text-align:center">target</td>
      <td style="text-align:center">
        <p>Used in gradients(target, components)</p>
        <p>defaults to <code>discriminator</code>
        </p>
      </td>
      <td style="text-align:center">string (optional)</td>
    </tr>
    <tr>
      <td style="text-align:center">lambda</td>
      <td style="text-align:center">
        <p>Loss multiple</p>
        <p>defaults to <code>1.0</code>
        </p>
      </td>
      <td style="text-align:center">float</td>
    </tr>
    <tr>
      <td style="text-align:center">components</td>
      <td style="text-align:center">
        <p>Used in gradients(target, components)</p>
        <p>defaults to all components</p>
      </td>
      <td style="text-align:center">array of strings</td>
    </tr>
    <tr>
      <td style="text-align:center">flex</td>
      <td style="text-align:center">
        <p>Max amount of gradient before penalty</p>
        <p></p>
      </td>
      <td style="text-align:center">float</td>
    </tr>
    <tr>
      <td style="text-align:center">flex</td>
      <td style="text-align:center">
        <p>Can also be a list for separate X/G flex.</p>
        <p>example: <code>[0.0, 10.0]</code>
        </p>
      </td>
      <td style="text-align:center">array of float</td>
    </tr>
    <tr>
      <td style="text-align:center">loss</td>
      <td style="text-align:center">
        <p>Side loss is added to: <code>g_loss</code> or <code>d_loss</code>
        </p>
        <p>defaults to <code>g_loss</code>
        </p>
      </td>
      <td style="text-align:center">string</td>
    </tr>
  </tbody>
</table>{% hint style="info" %}
Floats are [configurable parameters](../../configuration/configurable-parameters.md)
{% endhint %}



