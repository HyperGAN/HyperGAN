# HyperGAN
A versatile GAN(generative adversarial network) implementation focused on scalability and ease-of-use.

![hypergan logo 1](https://raw.githubusercontent.com/255BITS/HyperGAN/develop/doc/hypergan-logo1.jpg)
![hypergan logo 2](https://raw.githubusercontent.com/255BITS/HyperGAN/develop/doc/hypergan-logo2.jpg)
![hypergan logo 4](https://raw.githubusercontent.com/255BITS/HyperGAN/develop/doc/hypergan-logo4.jpg)

_Logos generated with [examples/colorizer](#colorizer)_

# Table of contents

* [Changelog](#changelog)
* [Quick start](#quick-start)
  * [Minimum Requirements](#minimum-requirements)
  * [Install](#install)
  * [Train](#train)
  * [Increasing Performance](#increasing-performance)
  * [Development Mode](#development-mode)
  * [Running on CPU](#running-on-cpu)
* [Configuration](#configuration)
  * [Usage](#usage)
  * [Architecture](#architecture)
  * [Generator](#generator)
  * [Encoders](#encoders)
  * [Discriminators](#discriminators)
  * [Losses](#losses)
   * [WGAN](#wgan)
   * [LS-GAN](#ls-gan)
   * [Standard GAN and Improved GAN](#standard-gan-and-improved-gan)
   * [Categories](#categorical-loss)
   * [Supervised](#supervised-loss)
  * [Trainers](#trainers)
* [The pip package hypergan](#the-pip-package-hypergan)
 * [Training](#training)
 * [Sampling](#sampling)
 * [Web Server](#web-server)
* [API](API)
  * [Examples](#examples)
  * [GAN object](#gan-object)
* [Datasets](#datasets)
 * [Supervised learning](#supervised-learning)
 * [Unsupervised learning](#unsupervised-learning)
 * [Creating a Dataset](#creating-a-dataset)
 * [Downloadable Datasets](#downloadable-datasets)
* [Contributing](#contributing)
  * [Our process](#our-process)
  * [Branches](#branches)
  * [Showcase](#showcase)
  * [Notable Configurations](#notable-configurations)
* [About](#about)
* [Sources](#sources)
* [Papers](#papers)
* [Citation](#citation)

# Changelog

## 0.8 ~ "GAN API"

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/develop/doc/face-manifold-0.8-64x64.png'/>

* Tensorflow 1.0 support
* New configuration format and refactored api.
* New loss function based on least squared GAN.  See <a href="#lsgan">lsgan implementation</a>.
* API example `2d-test` - tests a trainer/encoder/loss combination against a known distribution.
* API example `2d-measure` - measure and report the above test by randomly combining options.
* And more

## 0.7 ~ "WGAN API"

* New loss function based on `wgan` :.  Fixes many classes of mode collapse!  See <a href="#wgan">wgan implementation</a>
* Initial Public API Release
* API example: `colorizer` - re-colorize an image!
* API example: `inpainter` - remove a section of an image and have your GAN repaint it
* API example: `super-resolution` - zoom in and enhance.  We've caught the bad guy!
* 4 *new* samplers.  `--sampler` flag.  Valid options are: `batch`,`progressive`,`static_batch`,`grid`. 

## 0.6 ~ "MultiGAN"

* New encoders
* Support for multiple discriminators
* Support for discriminators on different image resolutions

## 0.5 ~ "FaceGAN"

### 0.5.x

* fixed configuration save/load
* cleaner cli output
* documentation cleanup

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold-0-5-6.png'/>

### 0.5.0
* pip package released!
* Better defaults.  Good variance.  256x256.  The broken images showed up after training for 5 days.

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold.png'/>

### 0.1-0.4
* Initial private release

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1-2.png'/>


# Quick start


## Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better.  32x32 can be run on lower-end GPUs.
2. CPU mode is _extremely_ slow.  Never train with it!
3. Python3


## Install


```bash
  pip3 install hypergan --upgrade
```


## Train

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32
```

### Increasing performance

On ubuntu `sudo apt-get install libgoogle-perftools4` and make sure to include this environment variable before training

```bash
  LD_PRELOAD="/usr/lib/libtcmalloc.so.4" hypergan train my_dataset
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

# Configuration

Configuration in HyperGAN uses JSON files.  You can create a new config by running `hypergan train`.  By default, configurations are randomly generated using [Hyperchamber](https://github.com/255BITS/hyperchamber).

Configurations are located in:

```bash
  ~/.hypergan/configs/
```


## Usage

```bash
  --config [name]
```

Naming a configuration during training required.

## Architecture

A hypergan configuration contains multiple encoders, multiple discriminators, multiple loss functions, and a single generator.

## Generator

A generator is responsible for projecting an encoding (sometimes called *z space*) to an output (normally an image).  A single GAN object from HyperGAN has one generator.

### Resize Conv

Resize conv pseudo code looks like this
```python
 1.  net = linear(z, z_projection_depth)
 2.  net = resize net to max(output width/height, double input width/height)
 3.  add layer filter if defined
 4.  convolution block
 5.  If at output size: 
 6.  Else add first 3 layers to progressive enhancement output and go to 2
```

| attribute   | description | type
|:----------:|:------------:|:----:|
| create | Called during graph creation | f(config, gan, net):net
| z_projection_depth | The output size of the linear layer before the resize-conv stack. | int > 0
| activation |  Activations to use.  See <a href='#configuration-activations'>activations</a> | f(net):net
| final_activation | Final activation to use.  This is usually set to tanh to squash the output range. | f(net):net
| depth_reduction | Reduces the filter sizes on each convolution by this multiple. | float > 0
| layer_filter | On each resize of G, we call this method.  Anything returned from this method is added to the graph before the next convolution block.  See <a href='#configuration-layer-filters'>common layer filters</a> | f(net):net
| layer_regularizer | This "regularizes" each layer of the generator with a type.  See <a href='#layer-regularizers'>layer regularizers</a>| f(name)(net):net

## Encoders

You can combine multiple encoders into a single GAN.

### Linear Encoder

| attribute   | description | type
|:----------:|:------------:|:----:|
| create | Called during graph creation | f(config, gan, net):net
| z | The dimensions of random uniform noise inputs | int > 0
| min | Lower bound of the random uniform noise | int > 0
| max | Upper bound of the random uniform noise | int > min
| projections | See more about projections below | [f(config, gan, net):net, ...]
| modes | If using modes, the number of modes to have per dimension | int > 0


### Projections

This encoder takes a random uniform value and outputs it as many possible types.  The primary idea is that you are able to query Z as a random uniform distribution, even if the gan is using a spherical representation.

Some projection types are listed below.

#### "linear" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-linear.png'/>

#### "sphere" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-sphere.png'/>

#### "gaussian" projection

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-gaussian.png'/>

#### "modal" projection

One of many

#### "binary" projection

On/Off


### Category Encoder

Uses categorical prior to choose 'one-of-many' options.  Can be paired with Categorical Loss.


## Discriminators

You can combine multiple discriminators in a single GAN.  This type of ensembling can be useful, but by default only 1 is enabled.

### Pyramid Discriminator

| attribute   | description | type
|:----------:|:------------:|:----:|
| create | Called during graph creation | f(config, gan, net):net
| activation |  Activations to use.  See <a href='#configuration-activations'>activations</a> | f(net):net
| depth_increase | Increases the filter sizes on each convolution by this multiple. | float > 0
| final_activation | Final activation to use.  This is usually set to tanh to squash the output range. | f(net):net
| layers | The number of convolution layers | int > 0
| layer_filter | Append information to each layer of the discriminator | f(config, net):net
| layer_regularizer | batch_norm_1, layer_norm_1, or None | f(batch_size, name)(net):net
| fc_layer_size | The size of the linear layers at the end of this network(if any). | int > 0
| fc_layers | fully connected layers at the end of the discriminator(standard dcgan is 0) | int >= 0
| noise | Instance noise.  Can be added to the input X | float >= 0
| progressive_enhancement | If true, enable [progressive enhancement](#progressive-enhancement) | boolean


### progressive enhancement

If true, each layer of the discriminator gets a resized version of X and additional outputs from G.

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/progressive-enhancement.png'/>

## Losses

## WGAN

Our implementation of WGAN is based off the paper.  WGAN loss in Tensorflow can look like:

```python
 d_loss = d_real - d_fake
 g_loss = d_fake
```

d_loss and g_loss can be reversed as well - just add a '-' sign.


## LS-GAN

```python
 d_loss = (d_real-b)**2 - (d_fake-a)**2
 g_loss = (d_fake-c)**2
```

a, b, and c are all hyperparameters.

### Standard GAN and Improved GAN

Includes support for Improved GAN.  See `hypergan/losses/standard_gan_loss.py` for details.


### Supervised loss

Supervised loss is for labeled datasets.  This uses a standard softmax loss function on the outputs of the discriminator.

### Categorical loss

This is currently untested.

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

### RMSProp

Uses RMSProp on G and D



### Adam

Uses Adam on G and D

### SGD

Uses SGD on G and D


### Configuration

| attribute   | description | type
|:----------:|:------------:|:----:|
| create | Called during graph creation | f(config, gan, net):net
| run | Steps forward once in training. | f(gan):[d_cost, g_cost]
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


# The pip package hypergan

```bash
 hypergan -h
```

## Training

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name]
```
## Sampling

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name] --sampler static_batch --sample_every 5
```

One way a network learns:

[![Demo CountPages alpha](https://j.gifs.com/58KmzA.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg&index=1)


To create videos:

```bash
  ffmpeg -i samples/%06d.png -vcodec libx264 -crf 22 -threads 0 gan.mp4
```


## Web Server

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan serve [folder] -s 32x32x3 -f png -b 32 --config [name]
```

Serve starts a flask server.  You can then access:

[http://localhost:5000/sample.png?type=batch](http://localhost:5000/sample.png?type=batch)

To prevent the GPU from allocating space, see <a href='#qs-runoncpu'>Running on CPU</a>.

## hypergan build

Build takes the same arguments as train and builds a generator.  It's required for serve.

Building does 2 things:

* Loads the training model, which include the discriminator
* Saves into a ckpt model containing only the generator

## Saves

Saves are stored in `~/.hypergan/saves/`

They can be large.

## Formats

```bash
--format <type>
```

Type can be one of:
* jpg
* png

## Arguments

To see a detailed list, run 
```bash
  hypergan -h
```

* -s, --size, optional(default 64x64x3), the size of your data in the form 'width'x'height'x'channels'
* -f, --format, optional(default png), file format of the images.  Only supports jpg and png for now.


# API

```python3
  import hypergan as hg
```

## Examples
API is currently under development.  The best reference are the examples in the `examples` directory.

Examples
--------

2d test
=======

Runs a 2d toy problem for a given configuration.  Can be sampled to show how a given configuration learns.

![](https://j.gifs.com/NxRKnD.gif)

2d measure accuracy
===================

Applies a batch accuracy (nearest neighbor) measurement to the 2d toy problem.

Colorizer 
=========

Colorizer feeds a black and white version of the input into the generator.

Inpainting
==========

Hides a random part of the image from the discriminator and the generator.

Super Resolution
================

Provides a low resolution image to the generator.

Constant inpainting
===================

Applies a constant mask over part of the image.  An easier problem than general inpainting.


## GAN object

The `GAN` object consists of:

* The `config`(configuration) used
* The `graph` - specific named Tensors in the Tensorflow graph
* The tensorflow `sess`(session)

### Constructor

```python
hg.GAN(config, initial_graph, graph_type='full', device='/gpu:0')
```

When a GAN constructor is called, the Tensorflow graph will be constructed.

#### Arguments

* config - The graph configuration.  See examples or the CLI tool for usage.
* initial_graph - a Dictionary consisting of any variables used by the GAN
* graph_type - Either 'full' or 'generator'
* device - Tensorflow device id

###  Properties
| property   | type       | description |
|:----------:|:----------:|:---------------------:|
| gan.graph|Dictionary|Maps names to tensors |
| gan.config|Dictionary|Maps names to options(from the json) |
| gan.sess|tf.Session|The tensorflow session |

### Methods

#### save

```python
 gan.save(save_file)
```

save_file - a string designating the save path

Saves the GAN

#### sample_to_file

```python
 gan.sample_to_file(name, sampler=grid_sampler.sample)
```

* name - the name of the file to sample to
* sampler - the sampler method to use

Sample to a specified path.

#### train

```python
 gan.train()
```

Steps the gan forward in training once.  Trains the D and G according to your specified `trainer`.

# Datasets

To build a new network you need a dataset.  Your data should be structured like:

``` 
  [folder]/[directory]/*.png
```

## Creating a Dataset


## Supervised learning

Training with labels allows you to train a `classifier`.

Each directory in your dataset represents a classification.  

Example:  Dataset setup for classification of apple and orange images:
```
 /dataset/apples
 /dataset/oranges
```


## Unsupervised learning

You can still build a GAN if your dataset is unlabelled.  Just make sure your folder is formatted like

```
 [folder]/[directory]/*.png
```
where all files are in 1 directory.


## Downloadable datasets

* CelebA aligned faces http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* MS Coco http://mscoco.org/
* ImageNet http://image-net.org/

# Contributing

Contributions are welcome and appreciated!  We have many open issues in the *Issues* tab that have the label *Help Wanted*.


## Our process

HyperGAN uses semantic versioning.  http://semver.org/

TLDR: *x.y.z*

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  *z* changes will be able to reload a saved graph.

## Branches

The branches are:

* `master` contains the best GAN we've found as default.  It aims to *just work* for most use cases.
* `develop` contains the latest and can be in a broken state.

*Bug fixes* and *showcases* can be merged into `master`

*Configuration changes*, *new architectures*, and generally anything experimental belongs in `develop`.

## Showcase

If you create something cool with this let us know!  Open a pull request and add your links, and screenshots here!

In case you are interested, our pivotal board is here: https://www.pivotaltracker.com/n/projects/1886395

## Notable Configurations

Notable configurations are stored in `example/configs` Feel free to submit additional ones.


# About

Generative Adversarial Networks consist of 2 learning systems that learn together.  HyperGAN implements these learning systems in Tensorflow with deep learning.

The `discriminator` learns the difference between real and fake data.  The `generator` learns to create fake data.

For a more in-depth introduction, see here [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

A single fully trained `GAN` consists of the following useful networks:

* `generator` - Generates content that fools the `discriminator`.  If using supervised learning mode, can generate data on a specific classification.
* `discriminator` - The discriminator learns how to identify real data and how to detect fake data from the generator.
* `classifier` - Only available when using supervised learning.  Classifies an image by type.  Some examples of possible datasets are 'apple/orange', 'cat/dog/squirrel'.  See <a href='#createdataset'>Creating a Dataset</a>.

HyperGAN is currently in open beta.


## Papers

* GAN - https://arxiv.org/abs/1406.2661
* DCGAN - https://arxiv.org/abs/1511.06434
* InfoGAN - https://arxiv.org/abs/1606.03657
* Improved GAN - https://arxiv.org/abs/1606.03498
* Adversarial Inference - https://arxiv.org/abs/1606.00704
* WGAN - https://arxiv.org/abs/1701.07875
* LS-GAN - https://arxiv.org/pdf/1611.04076v2.pdf

## Sources

* DCGAN - https://github.com/carpedm20/DCGAN-tensorflow
* InfoGAN - https://github.com/openai/InfoGAN
* Improved GAN - https://github.com/openai/improved-gan
* Hyperchamber - https://github.com/255bits/hyperchamber

# Citation

If you wish to cite this project, do so like this:

```
  255bits (M. Garcia),
  HyperGAN, (2017), 
  GitHub repository, 
  https://github.com/255BITS/HyperGAN
```

