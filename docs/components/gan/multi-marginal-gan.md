---
description: 'https://arxiv.org/abs/1911.00888'
---

# Multi Marginal GAN

### examples

```javascript

{
  "discriminator": 
  {
    ...
  },
  "latent":
  {
    ...
  },
  "generator": {
    ...
  },
  "encoder": {
    ...
  },
 "loss": {
  ...
  },
  "trainer": {
  ...
  }
  "d1_lambda": 0.01,
  "shared_encoder": true,
  "class": "class:hypergan.gans.multi_marginal_gan.MultiMarginalGAN"
}
```



## options

| attribute | description | type |
| :---: | :---: | :---: |
| shared\_encoder | Defaults to `false` | Boolean |
| l1\_loss | If set, adds x-x\_hat loss. | Float |

