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
gan.save(file)
gan.load(file)
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

gan.parameters()
```

## Create components

```python
gan.create_component({...}, *args, **kwargs)
```

This will create and attach a GANComponent to `gan.components`
