# HyperGAN 0.10.0-alpha1

[![CircleCI](https://circleci.com/gh/255BITS/HyperGAN/tree/master.svg?style=svg)](https://circleci.com/gh/255BITS/HyperGAN/tree/master)
[![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF)

A composable GAN API and CLI.  Built for developers, researchers, and artists.

HyperGAN is currently in open beta.

![Colorizer 0.9 1](https://s3.amazonaws.com/hypergan-apidocs/0.9.0-images/colorizer-2.gif)

_Logos generated with [examples/colorizer](#examples),  AlphaGAN, and the RandomWalk sampler_


# Table of contents

* [About](#about)
* [Showcase](#showcase)
* [Documentation](#documentation)
* [Changelog](#changelog)
* [Quick start](#quick-start)
  * [Minimum Requirements](#minimum-requirements)
  * [Create a new project](#create-a-new-project)
  * [Install](#install)
  * [Train](#train)
  * [Increasing Performance](#increasing-performance)
  * [Development Mode](#development-mode)
  * [Running on CPU](#running-on-cpu)
* [The pip package hypergan](#the-pip-package-hypergan)
 * [Training](#training)
 * [Sampling](#sampling)
* [API](API)
  * [Examples](#examples)
  * [Search](#search)
* [Configuration](#configuration)
  * [Usage](#usage)
  * [Architecture](#architecture)
  * [GANComponent](#GANComponent)
  * [Generator](#generator)
  * [Distributions](#distributions)
  * [Discriminators](#discriminators)
  * [Losses](#losses)
   * [WGAN](#wgan)
   * [LS-GAN](#ls-gan)
   * [Standard GAN and Improved GAN](#standard-gan-and-improved-gan)
   * [Categories](#categorical-loss)
   * [Supervised](#supervised-loss)
  * [Trainers](#trainers)
* [Datasets](#datasets)
  * [Unsupervised learning](#unsupervised-learning)
  * [Supervised learning](#supervised-learning)
  * [Creating a Dataset](#creating-a-dataset)
  * [Downloadable Datasets](#downloadable-datasets)
* [Contributing](#contributing)
* [Versioning](#Versioning)
* [Sources](#sources)
* [Papers](#papers)
* [Citation](#citation)

# About

Generative Adversarial Networks consist of 2 learning systems that learn together.  HyperGAN implements these learning systems in Tensorflow with deep learning, using json files to reproduce architectures from white papers.

For an introduction, see here [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)


HyperGAN is a community project.  GANs are a very new and active field of research.  Join the community [discord](https://discord.gg/t4WWBPF).

## Features

* 100% community project
* Reproducible architectures using json configurations
* Domain Specific Language to define custom architectures
* API
* Builds graphs that can run on consumer hardware, like phones and web browsers
* Trainable on consumer hardware
* Dataset agnostic
* Combine different components to build your own GAN
* Transfer learning
* Optimistic loading allows for network changes at train time


# Showcase

Coming... eventually

# Documentation

## API Documentation

 * [0.10.0](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/index.html)
 * [0.9.x](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
 * [Test coverage](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/coverage/index.html)

# Changelog

See the full changelog here:
[Changelog.md](Changelog.md)

# Quick start

## Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better.  32x32 can be run on lower-end GPUs.
2. CPU training is _extremely_ slow.  Use a GPU if you can!
3. Python3


## Install


### Install hypergan:

```bash
  pip3 install hypergan --upgrade
```

### Optional `virtualenv`:

If you use virtualenv:

```bash
  virtualenv --system-site-packages -p python3 hypergan
  source hypergan/bin/activate
```
### Dependencies:

If installation fails try this.

```bash
  pip3 install numpy tensorflow-gpu hyperchamber pillow pygame
```

### Dependency help

If the above step fails see the dependency documentation:

* tensorflow - https://www.tensorflow.org/install/
* pygame  - http://www.pygame.org/wiki/GettingStarted


## Create a new project

```bash
  hypergan new mymodel
```

This will create a mymodel.json based off the default configuration.  You can change configuration templates with the `-c` flag.  

## List configuration templates

```bash
  hypergan new mymodel -l
```

See all configuration templates with `--list-templates` or `-l`.

## Train

```bash
  # Train a 32x32 gan with batch size 32 on a folder of folders of pngs, resizing images as necessary
  hypergan train folder/ -s 32x32x3 -f png -c mymodel --resize
```

### Increasing performance

On ubuntu `sudo apt-get install libgoogle-perftools4` and make sure to include this environment variable before training

```bash
  LD_PRELOAD="/usr/lib/libtcmalloc.so.4" hypergan train my_dataset
```

HyperGAN does not cache image data in memory. Images are loaded every time they're needed, so you can increase performance by pre-processing your inputs, especially by resampling large inputs to the output resolution. e.g. with ImageMagick:

```bash
  convert image1.jpg -resize '128x128^' -gravity Center -crop 128x128+0+0 image1.png
```

## Development mode

If you wish to modify hypergan

```bash
git clone https://github.com/255BITS/hypergan
cd hypergan
python3 setup.py develop
```


## Running on CPU

Make sure to include the following 2 arguments:

```bash
CUDA_VISIBLE_DEVICES= hypergan --device '/cpu:0'
```
Don't train on CPU!  It's too slow.


# The pip package hypergan

```bash
 hypergan -h
```

## Training

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name]
```

## Sampling

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name] --sampler static_batch --sample_every 5 --save_samples
```

By default hypergan will not save samples to disk.  To change this, use `--save_samples`.

One way a network learns:

[![Demo CountPages alpha](https://j.gifs.com/58KmzA.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg&index=1)


To create videos:

```bash
  ffmpeg -i samples/%06d.png -vcodec libx264 -crf 22 -threads 0 gan.mp4
```
## Arguments

To see a detailed list, run 
```bash
  hypergan -h
```

# API

See the API documentation at https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html

```python3
  import hypergan as hg
```

Examples
--------

See the example documentation https://github.com/255BITS/HyperGAN/tree/master/examples

## Search

Each example is capable of random search.  You can search along any set of parameters, including loss functions, trainers, generators, etc.

# Datasets

To build a new network you need a dataset.  Your data should be structured like:

``` 
  [folder]/[directory]/*.png
```

## Creating a Dataset

Datasets in HyperGAN are meant to be simple to create.  Just use a folder of images.

```
 [folder]/*.png
```

For jpg(pass `-f jpg`)

# Configuration

Configuration in HyperGAN uses JSON files.  You can create a new config with the default template by running `hypergan new mymodel`.

You can see all templates with `hypergan new mymodel -l`.


## Architecture

A hypergan configuration contains all hyperparameters for reproducing the full GAN.

In the original DCGAN you will have one of the following components:

* Distribution(latent space)
* Generator
* Discriminator
* Loss
* Trainer


Other architectures may differ.  See the configuration templates.

## GANComponent

A base class for each of the component types listed below.

## Generator

A generator is responsible for projecting an encoding (sometimes called *z space*) to an output (normally an image).  A single GAN object from HyperGAN has one generator.

## Distributions

Sometimes referred to as the `z-space` representation or `latent space`.  In `dcgan` the 'distribution' is random uniform noise.

Can be thought of as input to the `generator`.


### Uniform Distribution

| attribute   | description | type
|:----------:|:------------:|:----:|
| z | The dimensions of random uniform noise inputs | int > 0
| min | Lower bound of the random uniform noise | int
| max | Upper bound of the random uniform noise | int > min
| projections | See more about projections below | [f(config, gan, net):net, ...]
| modes | If using modes, the number of modes to have per dimension | int > 0


### Projections

This distribution takes a random uniform value and outputs it as many possible types.  The primary idea is that you are able to query Z as a random uniform distribution, even if the gan is using a spherical representation.

Some projection types are listed below.

#### "identity" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-linear.png'/>

#### "sphere" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-sphere.png'/>

#### "gaussian" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-gaussian.png'/>

#### "modal" projection

One of many

#### "binary" projection

On/Off


### Category Distribution

Uses categorical prior to choose 'one-of-many' options.


## Discriminators

A discriminator's main purpose(sometimes called a critic) is to separate out G from X, and to give the Generator
a useful error signal to learn from.


## DSL

Each component in the GAN can be specified with a flexible DSL inside the JSON file.

## Losses

## WGAN

Wasserstein Loss is simply:

```python
 d_loss = d_real - d_fake
 g_loss = d_fake
```

d_loss and g_loss can be reversed as well - just add a '-' sign.

## Least-Squares GAN

```python
 d_loss = (d_real-b)**2 - (d_fake-a)**2
 g_loss = (d_fake-c)**2
```

a, b, and c are all hyperparameters.

### Standard GAN and Improved GAN

Includes support for Improved GAN.  See `hypergan/losses/standard_gan_loss.py` for details.

### Boundary Equilibrium Loss

Use with the `AutoencoderDiscriminator`.

See the `began` configuration template.

### Loss configuration

| attribute   | description | type
|:----------:|:------------:|:----:|
| batch_norm | batch_norm_1, layer_norm_1, or None | f(batch_size, name)(net):net
| create | Called during graph creation | f(config, gan, net):net
| discriminator |  Set to restrict this loss to a single discriminator(defaults to all) | int >= 0 or None
| label_smooth | improved gan - Label smoothing. | float > 0
| labels | lsgan - A triplet of values containing (a,b,c) terms. | [a,b,c] floats
| reduce | Reduces the output before applying loss | f(net):net
| reverse | Reverses the loss terms, if applicable | boolean

## Trainers

Determined by the GAN implementation.  These variables are the same across all trainers.

### Consensus

Consensus trainers trains G and D at the same time.  Resize Conv is known to not work with this technique(PR welcome).

#### Configuration

| attribute   | description | type
|:----------:|:------------:|:----:|
| learn_rate | Learning rate for the generator | float >= 0
| beta1 | (adam) | float >= 0
| beta2 | (adam)  | float >= 0
| epsilon | (adam)  | float >= 0
| decay | (rmsprop)  | float >= 0
| momentum | (rmsprop)  | float >= 0


### Alternating

Original GAN training.  Locks generator weights while training the discriminator, and vice-versa.

#### Configuration

| attribute   | description | type
|:----------:|:------------:|:----:|
| g_learn_rate | Learning rate for the generator | float >= 0
| g_beta1 | (adam) | float >= 0
| g_beta2 | (adam)  | float >= 0
| g_epsilon | (adam)  | float >= 0
| g_decay | (rmsprop)  | float >= 0
| g_momentum | (rmsprop)  | float >= 0
| d_learn_rate | Learning rate for the discriminator | float >= 0
| d_beta1 | (adam) | float >= 0
| d_beta2 | (adam)  | float >= 0
| d_epsilon | (adam)  | float >= 0
| d_decay | (rmsprop)  | float >= 0
| d_momentum | (rmsprop)  | float >= 0
| clipped_gradients | If set, gradients will be clipped to this value. | float > 0 or None
| d_clipped_weights | If set, the discriminator will be clipped by value. |float > 0 or None


### Fitness

Only trains on good z candidates.

### Curriculum

Train on a schedule.

### Gang

An evolution based trainer that plays a subgame between multiple generators/discriminators.


## Downloadable datasets

* CelebA aligned faces http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* MS Coco http://mscoco.org/
* ImageNet http://image-net.org/
* youtube-dl (see [examples/Readme.md](examples/Readme.md))

# Contributing

Contributions are welcome and appreciated!  We have many open issues in the *Issues* tab.

See <a href='CONTRIBUTING.md'>how to contribute.</a>

# Versioning

HyperGAN uses semantic versioning.  http://semver.org/

TLDR: *x.y.z*

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  *z* changes will be able to reload a saved graph.

## Papers

* GAN - https://arxiv.org/abs/1406.2661
* DCGAN - https://arxiv.org/abs/1511.06434
* InfoGAN - https://arxiv.org/abs/1606.03657
* Improved GAN - https://arxiv.org/abs/1606.03498
* Adversarial Inference - https://arxiv.org/abs/1606.00704
* Energy-based Generative Adversarial Network - https://arxiv.org/abs/1609.03126
* Wasserstein GAN - https://arxiv.org/abs/1701.07875
* Least Squares GAN - https://arxiv.org/pdf/1611.04076v2.pdf
* Boundary Equilibrium GAN - https://arxiv.org/abs/1703.10717
* Self-Normalizing Neural Networks - https://arxiv.org/abs/1706.02515
* Variational Approaches for Auto-Encoding
Generative Adversarial Networks - https://arxiv.org/pdf/1706.04987.pdf
* CycleGAN - https://junyanz.github.io/CycleGAN/
* DiscoGAN - https://arxiv.org/pdf/1703.05192.pdf
* Softmax GAN - https://arxiv.org/abs/1704.06191
* The Cramer Distance as a Solution to Biased Wasserstein Gradients - https://arxiv.org/abs/1705.10743
* Improved Training of Wasserstein GANs - https://arxiv.org/abs/1704.00028
* More...

## Sources

* DCGAN - https://github.com/carpedm20/DCGAN-tensorflow
* InfoGAN - https://github.com/openai/InfoGAN
* Improved GAN - https://github.com/openai/improved-gan
* Hyperchamber - https://github.com/255bits/hyperchamber

# Citation

If you wish to cite this project, do so like this:

```
  HyperGAN Community
  HyperGAN, (2017-2018+), 
  GitHub repository, 
  https://github.com/255BITS/HyperGAN
```

HyperGAN comes with no warranty or support.
