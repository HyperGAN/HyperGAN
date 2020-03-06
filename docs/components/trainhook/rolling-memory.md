---
description: (no paper)
---

# Rolling Memory

Rolling memory is a type of experience replay. Each training step, a memory is replaced with the top scoring batch item.

Each `types` pairing becomes a discriminator that is added to the loss.

## examples

```javascript
{
    "class": "function:hypergan.train_hooks.experimental.rolling_memory_2_train_hook.RollingMemoryTrainHook",
    "types": ["mx-/g(mz-)"]
}
```

mx- is a memory of x that gets updated each training step. g\(mz-\) is a memory of z that gets run through a generator and updated each trainng step.

A discriminator `d(mx-, g(mz-))` is created and added to the gan loss.

## options

| attribute | description | type |
| :---: | :---: | :---: |
| types | What memories and how they are paired.  See **memory types** below | array of strings |
| top\_k | How many memory items to replace per frame.  Defaults to `1` | integer |
| only | Overrides all other losses when this is set. Defaults to `false` | boolean |

### memory types

| memory | description |
| :--- | :--- |
| mx- | x reverse sorted by d\_real |
| mx+ | x sorted by d\_real |
| mg- | memory of g reverse sorted by d\_fake |
| mg+ | memory of g sorted by d\_fake |
| g\(mz-\) | generator of memory of z reverse sorted by d\_fake |
| g\(mz+\) | generator of memory of z sorted by d\_fake |
| x | gan.inputs.x |
| g | gan.generator.sample |

