---
description: 'http://stylegan.xyz/paper'
---

# Logistic Loss

```python
d_loss = self.softplus(-d_real) + self.softplus(d_fake)
g_loss = self.softplus(-d_fake)
```

## examples

```javascript
{                                                                                       
  "class": "function:hypergan.losses.logistic_loss.LogisticLoss"
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
      <td style="text-align:center">beta</td>
      <td style="text-align:center">
        <p>https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softplus</code>
        </p>
      </td>
      <td style="text-align:center">float (optional)</td>
    </tr>
    <tr>
      <td style="text-align:center">threshold</td>
      <td style="text-align:center">
        <p>https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#Softplus</code>
      </td>
      <td style="text-align:center">float (optional)</td>
    </tr>
  </tbody>
</table>



