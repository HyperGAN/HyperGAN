---
description: Train hooks provide training events and loss modification to trainers.
---

# Train Hook

[https://github.com/HyperGAN/HyperGAN/tree/master/hypergan/train\_hooks](https://github.com/HyperGAN/HyperGAN/tree/master/hypergan/train_hooks)

## Access

```python
gan.trainer.train_hooks # => [...]
```

Train hooks are setup and invoked by the trainer.

## Events

Override these methods to change the train loop

```python
before_step(step, feed_dict)
after_step(step, feed_dict)
after_create()
gradients(d_grads, g_grads)
```

### before\_step\(feed\_dict\)

### after\_step\(feed\_dict\)

Executed before/after the step takes place. `feed_dict` is what is being sent to the graph during the training step.

### after\_create\(\)

Ran after the trainer is created.

### gradients\(d\_grads, g\_grads\)

Refines the gradients before they are applied to the optimizer.

