# HyperGAN 0.10

[![CircleCI](https://circleci.com/gh/HyperGAN/HyperGAN.svg?style=svg)](https://circleci.com/gh/HyperGAN/HyperGAN)
[![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF)
[![Twitter](https://img.shields.io/badge/twitter-follow-blue.svg)](https://twitter.com/hypergan)

A composable GAN API and CLI.  Built for developers, researchers, and artists. 

 0.10 is now available in pip.  Installation instructions and support are available in our [discord](https://discord.gg/t4WWBPF)

HyperGAN is in open beta.

![Colorizer 0.9 1](https://s3.amazonaws.com/hypergan-apidocs/0.9.0-images/colorizer-2.gif)

_Logos generated with [examples/colorizer](#examples)_

See more on the [hypergan youtube](https://www.youtube.com/channel/UCU33XvBbMnS8002_NB7JSvA)

# Table of contents

* [About](#about)
* [Showcase](#showcase)
* [Documentation](#documentation)
* [Changelog](#changelog)
* [Quick start](#quick-start)
  * [Requirements](#requirements)
  * [Install](#install)
  * [Testing install](#testing-install)
  * [Train](#train)
  * [Development Mode](#development-mode)
  * [Running on CPU](#running-on-cpu)
* [The pip package hypergan](#the-pip-package-hypergan)
 * [Training](#training)
 * [Sampling](#sampling)
* [API](API)
  * [Examples](#examples)
* [Datasets](#datasets)
  * [Creating a Dataset](#creating-a-dataset)
  * [Downloadable Datasets](#downloadable-datasets)
  * [Cleaning up data](#cleaning-up-data)
* [Contributing](#contributing)
* [Versioning](#Versioning)
* [Sources](#sources)
* [Papers](#papers)
* [Citation](#citation)

# About

Generative Adversarial Networks consist of 2 learning systems that learn together.  HyperGAN implements these learning systems in Tensorflow with deep learning.

For an introduction to GANs, see [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

HyperGAN is a community project.  GANs are a very new and active field of research.  Join the community [discord](https://discord.gg/t4WWBPF).

## Features

* Community project
* Unsupervised learning
* Transfer learning
* Online learning
* Dataset agnostic
* Reproducible architectures using json configurations
* Domain Specific Language to define custom architectures
* GUI(pygame and tk)
* API
* CLI

# Showcase

See the [![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF)

# Documentation

 * [Model author JSON reference](json.md)
 * [Model author tutorial 1](tutorial1.md)
 * [0.10.x](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/index.html)
 * [0.9.x](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
 * [Test coverage](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/coverage/index.html)

# Changelog

See the full changelog here:
[Changelog.md](Changelog.md)

# Quick start

## Requirements

Recommended: GTX 1080+

## Install

### Install hypergan:

```bash
  pip3 install hypergan --upgrade
```


### Testing install

To see that tensorflow and hypergan are installed correctly and have access to devices, please run:

```
  hypergan test
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


## Create a new model

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

## Development mode

If you wish to modify hypergan

```bash
git clone https://github.com/hypergan/hypergan
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

Examples
--------

See the example documentation https://github.com/hypergan/HyperGAN/tree/master/examples

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

## Downloadable datasets

* Loose images of any kind can be used
* CelebA aligned faces http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* MS Coco http://mscoco.org/
* ImageNet http://image-net.org/
* youtube-dl (see [examples/Readme.md](examples/Readme.md))

## Cleaning up data

To convert and resize your data for processing, you can use imagemagick

```
for i in *.jpg; do convert $i  -resize "300x256" -gravity north   -extent 256x256 -format png -crop 256x256+0+0 +repage $i-256x256.png;done

```

# Contributing

Contributions are welcome and appreciated!  We have many open issues in the *Issues* tab.  Join the discord.

See <a href='CONTRIBUTING.md'>how to contribute.</a>

# Versioning

HyperGAN uses semantic versioning.  http://semver.org/

TLDR: *x.y.z*

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  *z* changes will be able to reload a saved graph.


## Other GAN projects
* [StyleGAN](https://github.com/NVlabs/stylegan)
* https://github.com/LMescheder/GAN_stability
* Add yours with a pull request

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

# Citation

```
  HyperGAN Community
  HyperGAN, (2016-2019+), 
  GitHub repository, 
  https://github.com/HyperGAN/HyperGAN
```

HyperGAN comes with no warranty or support.
