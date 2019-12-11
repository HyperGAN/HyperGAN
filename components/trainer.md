---
description: Trains the GAN object.
---

# Trainer

## Creation

The trainer creates the optimizer, and any associated train hooks.

```python
trainer_config = {...}
gan.create_component(trainer_config)
```

## Access

```text
gan.trainer
```

## Actions

```python
gan.trainer.step(feed_dict) # Step forward
```

## Events

```python
trainer.before_step(step, feed_dict)
trainer.after_step(step, feed_dict)
```



