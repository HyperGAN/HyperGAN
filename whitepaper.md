# HyperGAN Technical White Paper WIP DRAFT

**September 25 2018**

**Abstract:" The hypergan software uses machine learning to generate cross-platform models.  Deployed components can be run in real time on consumer hardware.


<!-- MarkdownTOC depth=4 autolink=true bracket=round list_bullets="-*+" -->

- [Background](#background)
- [Requirements for training models]
	* [Network designs](#networkdesigns)
  * [Transfer learning](#transferlearning)
- [Requirements for running models]
- [Original research](#originalresearch)
  * Next frame prediction
- [Conclusion](#conclusion)
- [References](#references)

<!-- /MarkdownTOC -->


# Background

GANs were introduced in <year>.

While a number of machine learning frameworks.

GANs have incredible use cases in blockchain, 

# Requirements for training

Training a GAN is much more resource intensive than running it.  When running a neural network, you only must do a forward pass through a single component(generator, for instance).  When training, you must run back prop through multiple components.

## Network designs

Hypergan networks are defined in a JSON file.  This allows anyone to become a network architect.

## Transfer learning

hypergan uses a form of transfer learning called `optimistic loading`.  If you load a save in a different network configuration, any matching named variables will be transferred.  Matching named variables with different sizes will keep as much weight information as possible.


# Original research

## Next frame prediction

Next frame prediction in GANs is an attractive use case for GANs.  The hypergan implementation of next frame is defined below:

TODO explain and reference

# References

TODO - ali
TODO - guided
TODO - bengio c prior
