Examples
--------

Each example has 3 actions:

* train
  
  trains a new example

* sample

  samples from a trained example

* search
  
  randomly searches for a configuration and outputs JSON / metrics


2d distribution
===================

Trains a generator to output 2d points (pixels) matching a known distribution.

![](https://j.gifs.com/NxRKnD.gif)

Colorizer 
=========

Colorizer feeds a black and white version of the input into the generator.

State: working

Alignment
=========

Align two different datasets along features.

State: running not converging

Needs search.

AlignedRandomSearch

Autoencode
==========

Reconstruct input images using AutoencoderGAN

State: working

Not done

CharGAN and Sequence (experimental)
===================================

Character based GANs

State: running not converging

Not done

Classification
==============

Classify MNIST by generating label distributions.  G(x) = label

State: working

Not done

Static
======

Memorize X and Z values then test against them.

State: working

Not done

Inpainting (pending)
==========

Hides a random part of the image from the discriminator and the generator.

Not present

Super Resolution (pending)
================

Provides a low resolution image to the generator.

Not present
