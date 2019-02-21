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

Search:  2d-distance measure from generator batch to known distribution

Colorizer 
=========

Colorizer feeds a black and white version of the input into the generator.

State: working

Search:  Distance from black and white image to black and white version of generated output

Alignment
=========

Align images and black and white versions of those images.

State: working

Search:  Distance from Gab(Xa),black_and_white(Xa)

Autoencode
==========

Reconstruct input images using AutoencoderGAN

State: working

Search: Reconstruction cost

Static
======

Memorize X and Z values then test against them.

State: working

Search: Reconstruction
