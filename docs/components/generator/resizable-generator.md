# Resizable Generator

## examples

```javascript
"generator": {
    "class": "function:hypergan.generators.resizable_generator.ResizableGenerator",
    "defaults":{
      "initializer": "random_normal",
      "activation": "relu"
    },
    "final_activation": "clamped_unit",
    "final_depth": 32,
    "initial_dimensions": [32, 32],
    "depth_increase": 32,
    "block": "deconv"
  }
```

![netron visualization](../../.gitbook/assets/smallgenerator%20%281%29.png)

