---
description: Create the domain object map and connect all components.
---

# GAN

## Creation

```python
import hypergan as hg
configuration = {...}
gan = hg.GAN(configuration)
```

## Actions

```python
gan.step(feed_dict)
gan.save(file)
gan.load(file)
gan.initialize_variables()
gan.configurable_param(string)
```

## Properties

```python
gan.batch_size()
gan.channels()
gan.width()
gan.height()
gan.output_shape()

gan.components
gan.inputs
gan.steps

gan.trainable_variables()
gan.parameter_count()
```

## Create components

```python
gan.create_component({...}, *args, **kwargs)
```

This will create and attach a GANComponent to `gan.components`

