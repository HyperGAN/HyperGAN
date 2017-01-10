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

* <a href="#training">Training</a>
  * <a href="#configuration">Configuration</a>
* <a href="#about">About</a>

<div id="changelog"></div>

## Changelog

### 0.6.0 ~ "MultiGAN"

* 3 new encoders
* New discriminator: `densenet` - based loosely on https://arxiv.org/abs/1608.06993
* Updated discriminator: `pyramid_no_stride` - `conv` and `avg_pool` together
* New generator: `dense_resize_conv` - original type of generator that seems to work well
* Updated generator: `resize_conv` - standard resize-conv generator.  This works much better than `deconv`, which is not supported.
* Several quality of life improvements
* Support for multiple discriminators
* Support for discriminators on different image resolutions

### 0.5.x

* fixed configuration save/load
* cleaner cli output
* documentation cleanup

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold-0-5-6.png'/>

### 0.5.0 ~ "FaceGAN"
* pip package released!
* Better defaults.  Good variance.  256x256.  The broken images showed up after training for 5 days.

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold.png'/>

### 0.1-0.4
* Initial private release

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1-2.png'/>


<div id="samples"/>


<div id='quickstart'/>

## Quick start

<div id='minreqs'/>

### Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better.
2. CPU mode is _extremely_ slow.  Never train with it!


<div id='qs-install'/>

### Install hypergan

```
  pip install hypergan
```

<div id='qs-train'/>

### Train

```
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32
```

<div id='qs-increase'/>

### Increasing performance

On ubuntu `sudo apt-get install libgoogle-perftools4` and make sure to include this environment variable before training

```
  LD_PRELOAD="/usr/lib/libtcmalloc.so.4" hypergan train my_dataset
```

<div id='qs-devmode'/>
### Development mode

If you wish to modify hypergan

```
git clone https://github.com/255BITS/hypergan
cd hypergan
python3 setup.py develop
```

<div id='qs-runoncpu'/>
### Running on CPU

Make sure to include the following 2 arguments:

```
CUDA_VISIBLE_DEVICES= hypergan --device '/cpu:0'
```

# API

```python
  import hypergan
```

## Runtime API

## Loading discriminators

```
  d = hypergan.Discriminator.load('name')
  d.graph # the tensorflow graph
```

## Loading generators
```
  g = hypergan.Generator.load('name')
  g.graph # the tensorflow graph
```

## Examples

The api is still being actively developed.  Right now the best reference will be in the `examples` directory.

# Training

## hypergan train

To build a new network you need a dataset.  Your data should be structured like:

``` 
  [folder]/[directory]/*.png
```

If you don't have a dataset, you can use [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

```
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name]
```

Configs and saves are located in:

```
  ~/.hypergan/
```

# Configuration

Configuration in HyperGAN uses JSON files.  You can create a new config by running `hypergan train`.  By default, configurations are randomly generated using [Hyperchamber](https://github.com/255BITS/hyperchamber).

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

```
--format <type>
```

Type can be one of:
* jpg
* png

## Arguments

To see a detailed list, run 
```
  hypergan -h
```

* -s, --size, optional(default 64x64x3), the size of your data in the form 'width'x'height'x'channels'
* -f, --format, optional(default png), file format of the images.  Only supports jpg and png for now.


## Discriminators

The discriminators job is to tell if a piece of data is real or fake.  In hypergan, a discriminator can also be a classifier.

You can combine multiple discriminators in a single GAN. 

### pyramid_stride

### pyramid_nostride

Default.

### densenet

### resnet

## Encoders

### Vae

### RandomCombo

Default

### RandomNormal

## Generators

### resize-conv

Default.

### dense-resize-conv

Inspired by densenet.

## Trainers

### Adam

Default.

### Slowdown

Experimental.

# Debugging a generator

## Visualizing learning

One way a network learns:

[![Demo CountPages alpha](https://j.gifs.com/58KmzA.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg&index=1)

To create your own visualizations, you can use the flag:

``` 
  --frame_sample grid 
```

To turn these images into a video:

```
  ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 gan.mp4
```

NOTE: z_dims must equal 2 and batch size must equal 32 to work.

<div id='about'/>

# About

Generative Adversarial Networks(2) consist of (at least) two neural networks that learn together over many epochs.
The discriminator learns the difference between real and fake data.  The generator learns to create fake data.

For a more depth introduction, see here [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)


## Papers

* GAN - https://arxiv.org/abs/1406.2661
* DCGAN - https://arxiv.org/abs/1511.06434
* InfoGAN - https://arxiv.org/abs/1606.03657
* Improved GAN - https://arxiv.org/abs/1606.03498
* Adversarial Inference - https://arxiv.org/abs/1606.00704

## Sources

* DCGAN - https://github.com/carpedm20/DCGAN-tensorflow
* InfoGAN - https://github.com/openai/InfoGAN
* Improved GAN - https://github.com/openai/improved-gan
* Hyperchamber - https://github.com/255bits/hyperchamber

# Contributing

Our pivotal board is here: https://www.pivotaltracker.com/n/projects/1886395

Contributions are welcome and appreciated.  To help out, just issue a pull request.

Also, if you create something cool with this let us know!

# Citation

If you wish to cite this project, do so like this:

```
  255bits (M. Garcia),
  HyperGAN, (2017), 
  GitHub repository, 
  https://github.com/255BITS/HyperGAN
```

