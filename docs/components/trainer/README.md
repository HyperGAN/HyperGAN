---
description: Trains the GAN object.
---

# Trainer

## Creation

The trainer creates the optimizer, and any associated train hooks.

```python
trainable_gan = TrainableGAN(gan)
```

## Access

```text
trainable_gan.trainer
```

## Actions

```python
trainer.step(feed_dict) # Step forward
```

## Events

```python
trainer.before_step(step, feed_dict)
trainer.after_step(step, feed_dict)
```

