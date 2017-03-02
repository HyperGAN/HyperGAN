# HyperGAN
A versatile GAN(generative adversarial network) implementation focused on scalability and ease-of-use.

# Table of contents

* <a href="#changelog">Changelog</a>
* <a href="#quickstart">Quick start</a>
  * <a href="#minreqs">Minimum Requirements</a>
  * <a href="#qs-install">Install</a>
  * <a href="#qs-train">Train</a>
  * <a href="#qs-increase">Increasing Performance</a>
  * <a href="#qs-devmode">Development Mode</a>
  * <a href="#qs-runoncpu">Running on CPU</a>
* <a href="#configuration">Configuration</a>
  * <a href="#configuration-usage">Usage</a>
  * <a href="#configuration-architecture">Architecture</a>
  * <a href="#configuration-generator">Generator</a>
  * <a href="#configuration-encoders">Encoders</a>
  * <a href="#configuration-discriminators">Discriminators</a>
  * <a href='#api-losses'>Losses</a>
   * <a href="#wgan">WGAN</a>
   * <a href="#lsgan">LS-GAN</a>
   * <a href="#standard-gan">Standard GAN/Improved GAN</a>
* <a href="#cli">The pip package `hypergan`</a>
 * <a href="#cli-train">Training</a>
 * <a href="#cli-sample">Sampling</a>
 * <a href="#cli-serving">Web Server</a>
* <a href="#api">API</a>
  * <a href="#api-examples">Examples</a>
  * <a href="#api-gan">GAN object</a>
* <a href="#datasets">Datasets</a>
 * <a href="#supervised-learning">Supervised learning</a>
 * <a href="#unsupervised-learning">Unsupervised learning</a>
 * <a href="#createdataset">Creating a Dataset</a>
 * <a href='#downloadabledatasets'>Downloadable Datasets</a>
* <a href="#contributing">Contributing</a>
  * <a href="#our-process">Our process</a>
  * <a href="#branches">Branches</a>
  * <a href="#showcase">Showcase</a>
  * <a href="#notable-configurations">Notable Configurations</a>
* <a href="#about">About</a>

<div id="changelog"></div>
## Changelog

## 0.8 ~ "GAN API"

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


<div id='quickstart'/>
# Quick start


<div id='minreqs'/>
## Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better.  32x32 can be run on lower-end GPUs.
2. CPU mode is _extremely_ slow.  Never train with it!
3. Python3


<div id='qs-install'/>
## Install hypergan


```bash
  pip3 install hypergan --upgrade
```


<div id='qs-train'/>
## Train

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32
```

<div id='qs-increase'/>
### Increasing performance

On ubuntu `sudo apt-get install libgoogle-perftools4` and make sure to include this environment variable before training

```bash
  LD_PRELOAD="/usr/lib/libtcmalloc.so.4" hypergan train my_dataset
```


<div id='qs-devmode'/>
## Development mode

If you wish to modify hypergan

```bash
git clone https://github.com/255BITS/hypergan
cd hypergan
python3 setup.py develop
```


<div id='qs-runoncpu'/>
## Running on CPU

Make sure to include the following 2 arguments:

```bash
CUDA_VISIBLE_DEVICES= hypergan --device '/cpu:0'
```
Don't train on CPU!  It's too slow.

<div id='configuration'/>
# Configuration

Configuration in HyperGAN uses JSON files.  You can create a new config by running `hypergan train`.  By default, configurations are randomly generated using [Hyperchamber](https://github.com/255BITS/hyperchamber).

Configurations are located in:

```bash
  ~/.hypergan/configs/
```


<div id='configuration-usage'/>
## Usage

```bash
  --config [name]
```

Naming a configuration during training required.

During beta, the best source of configuration documentation is the source code.  When in doubt, use a configuration in the <a href='#notable-configurations'>notable configurations</a> section.


<div id="configuration-architecture"></div>
## Architecture

A hypergan configuration contains multiple encoders, multiple discriminators, multiple loss functions, and a single generator.

<div id="configuration-generator"></div>
## The Generator

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
| create | a method that will be called at the beginning of graph creation | f(config, gan, net):net
| z_projection_depth | The output size of the linear layer before the resize-conv stack. | int > 0
| activation |  Activations to use.  See <a href='#configuration-activations'>activations</a> | f(net):net
| final_activation | Final activation to use.  This is usually set to tanh to squash the output range. | f(net):net
| depth_reduction | Reduces the filter sizes on each convolution by this multiple. | f(net):net
| layer_filter | On each resize of G, we call this method.  Anything returned from this method is added to the graph before the next convolution block.  See <a href='#configuration-layer-filters'>common layer filters</a> | f(net):net
| layer_regularizer | This "regularizes" each layer of the generator with a type.  See <a href='#layer-regularizers'>layer regularizers</a>| f(name)(net):net

<div id="configuration-encoders"></div>
## Encoders

Choose any number of encoders!

TODO table of common options

### Linear Encoder

Standard DCGan uses this.

### Category Encoder

Uses categorical prior to choose 'one-of-many' options.  Can be paired with Categorical Loss.

<div id="configuration-discriminators"></div>
## Discriminators

TODO common options

Progressive enhancement is enabled by default:

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/progressive-enhancement.png'/>

### Pyramid Discriminator

TODO table of options

## Losses

TODO common options

<div id="wgan"/>
## Wasserstein GAN in Tensorflow

Our implementation of WGAN is based off the paper.  WGAN loss in Tensorflow can look like:

```python
 d_fake = tf.reduce_mean(d_fake,axis=1)
 d_real = tf.reduce_mean(d_real,axis=1)
 d_loss = d_real - d_fake
 g_loss = d_fake
```

d_loss and g_loss can be reversed as well - just add a '-' sign.


<div id="lsgan"/>
## LS-GAN in Tensorflow

```python
 d_loss = (d_real-b)**2 - (d_fake-a)**2
 g_loss = (d_fake-c)**2
```

a, b, and c are all hyperparameters.


### Standard GAN Loss

Includes support for Improved GAN.  See `hypergan/losses/standard_gan_loss.py` for details.

### Categorical loss

This is currently untested.

<div id='#cli'/>
# CLI

```bash
 hypergan -h
```

<div id='#cli-train'/>
## Training

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name]
```


<div id='#cli-sampling'/>
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


<div id='#cli-serving'/>
## Web Server

```bash
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan serve [folder] -s 32x32x3 -f png -b 32 --config [name]
```

To prevent the GPU from allocating space, see <a href='#qs-runoncpu'>Running on CPU</a>.

<div id="api"/>
# API

```python3
  import hypergan as hg
```

<div id='api-examples'>
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


<div id="api-gan">
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

<div id="datasets"/>
# Datasets

To build a new network you need a dataset.  Your data should be structured like:

``` 
  [folder]/[directory]/*.png
```

<div id="createdataset"/>
## Creating a Dataset

<div id='supervised-learning'/>

## Supervised learning

Training with labels allows you to train a `classifier`.

Each directory in your dataset represents a classification.  

Example:  Dataset setup for classification of apple and orange images:
```
 /dataset/apples
 /dataset/oranges
```

<div id='unsupervised-learning'/>

## Unsupervised learning

You can still build a GAN if your dataset is unlabelled.  Just make sure your folder is formatted like

```
 [folder]/[directory]/*.png
```
where all files are in 1 directory.

<div id='downloadabledatasets'/>

## Downloadable datasets

* CelebA aligned faces http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* MS Coco http://mscoco.org/
* ImageNet http://image-net.org/

# Building

## hypergan build

Build takes the same arguments as train and builds a generator.  It's required for serve.

Building does 2 things:

* Loads the training model, which include the discriminator
* Saves into a ckpt model containing only the generator

# Server mode

## hypergan serve

Serve starts a flask server.  You can then access:

[http://localhost:5000/sample.png?type=batch](http://localhost:5000/sample.png?type=batch)

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

<div id="contributing"/>
# Contributing

Contributions are welcome and appreciated!  We have many open issues in the *Issues* tab that have the label *Help Wanted*.


<div id="our-process"/>
## Our process

HyperGAN uses semantic versioning.  http://semver.org/

TLDR: *x.y.z*

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  *z* changes will be able to reload a saved graph.

<div id="branches"/>
## Branches

The branches are:

* `master` contains the best GAN we've found as default.  It aims to *just work* for most use cases.
* `develop` contains the latest and can be in a broken state.

*Bug fixes* and *showcases* can be merged into `master`

*Configuration changes*, *new architectures*, and generally anything experimental belongs in `develop`.

<div id="showcase"/>
## Showcase

If you create something cool with this let us know!  Open a pull request and add your links, and screenshots here!

In case you are interested, our pivotal board is here: https://www.pivotaltracker.com/n/projects/1886395

<div id="notable-configurations"/>
## Notable Configurations

Notable configurations are stored in `example/configs` Feel free to submit additional ones.

<div id='about'/>

# About

Generative Adversarial Networks consist of 2 learning systems that learn together.  HyperGAN implements these learning systems in Tensorflow with deep learning.

The `discriminator` learns the difference between real and fake data.  The `generator` learns to create fake data.

For a more in-depth introduction, see here [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

A single fully trained `GAN` consists of the following useful networks:

* `generator` - Generates content that fools the `discriminator`.  If using supervised learning mode, can generate data on a specific classification.
* `discriminator` - The discriminator learns how to identify real data and how to detect fake data from the generator.
* `classifier` - Only available when using supervised learning.  Classifies an image by type.  Some examples of possible datasets are 'apple/orange', 'cat/dog/squirrel'.  See <a href='#createdataset'>Creating a Dataset</a>.

HyperGAN is currently in open beta.

## Discriminators

The discriminators job is to tell if a piece of data is real or fake.  In hypergan, a discriminator can also be a classifier by adding an additional loss.

You can combine multiple discriminators in a single GAN.  This type of ensembling can be useful, but by default only 1 is enabled.

### progressive enhancement

If true, each layer of the discriminator gets a resized version of X and additional outputs from G.

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/progressive-enhancement.png'/>

## Encoders

### LinearEncoder

This encoder takes a random uniform value and outputs it as many possible types.  The primary idea is that you are able to query Z as a random uniform distribution, even if the gan is using a spherical representation.

#### Linear

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-linear.png'/>

#### Spherical

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-sphere.png'/>

#### Gaussian

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/sphere/doc/encoder-linear-gaussian.png'/>

#### Modal

One of many

#### Binary

On/Off


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

