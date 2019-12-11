---
description: Trains the GAN object.
---

# Trainer

## Component creation

The trainer creates the optimizer, and any associated train hooks.

```python
trainer_config = {...}
gan.create_component(trainer_config)
```

{% hint style="info" %}
Trainers are setup by the GAN objects during initialization and are available as `gan.trainer`
{% endhint %}

## Training step

```python
gan.trainer.step(feed_dict)
```

## Events

```python
trainer.before_step(step, feed_dict)
trainer.after_step(step, feed_dict)
```



