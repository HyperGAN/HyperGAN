# README

## HyperGAN 1.0

[![docs](https://img.shields.io/badge/gitbook-docs-yellowgreen)](https://hypergan.gitbook.io/hypergan/) [![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF) [![Twitter](https://img.shields.io/badge/twitter-follow-blue.svg)](https://twitter.com/hypergan)

A composable GAN built for developers, researchers, and artists.

HyperGAN is in pre-release and open beta.

![Colorizer 0.9 1](https://s3.amazonaws.com/hypergan-apidocs/0.9.0-images/colorizer-2.gif)

_Logos generated with_ [_examples/colorizer_](./#examples/colorizer.py)

See more on the [hypergan youtube](https://www.youtube.com/channel/UCU33XvBbMnS8002_NB7JSvA)

## Table of contents

* [About](#about)
* [Documentation](https://hypergan.gitbook.io/hypergan/)
* [Changelog](./changelog.md)
* [Quick start](#quick-start)
  * [Requirements](#requirements)
  * [Install](#install)
  * [Troubleshooting](#troubleshooting)
  * [Testing install](#testing-install)
  * [Train](#train)
  * [Development Mode](#development-mode)
  * [Running on CPU](#running-on-cpu)
* [The pip package hypergan](#the-pip-package-hypergan)
  * [Training](#training)
  * [Sampling](#sampling)
* [API](#api)
  * [Using a trained hypergan model](#using-a-trained-hypergan-model)
  * [Training a gan](#training-a-gan)
  * [Examples](#examples)
  * [Tutorials](#tutorials)
* [Datasets](#datasets)
  * [Creating a Dataset](#creating-a-dataset)
  * [Downloadable Datasets](#downloadable-datasets)
  * [Cleaning up data](#cleaning-up-data)
* [Features](#features)
* [Showcase](#showcase)
* [Sponsors](#sponsors)
* [Contributing](./#contributing.md)
* [Versioning](#Versioning)
* [Citation](#citation)

## About

HyperGAN builds generative adversarial networks in pytorch and makes them easy to train and share.

For a general introduction to GANs see [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

Join the community [discord](https://discord.gg/t4WWBPF).

## Documentation

* [Gitbook documentation](https://hypergan.gitbook.io/)

## Changelog

See the full changelog here: [Changelog.md](changelog.md)

## Quick start

### Requirements

Nvidia GPU

OS: Windows, OSX, Linux

GPU: NVIDIA, GTX 1080+ recommended

### Install

1. Install HyperGAN
  For users: `pip3 install hypergan`

  For developers: Download this repo and run `python3 setup.py develop`

2. Test it out
  * `hypergan train preset:celeba -s 128x128x3`

3. Join the community
  * Once you've made something cool, be sure to share it on the Discord \([https://discord.gg/t4WWBPF](https://discord.gg/t4WWBPF)\).

### Troubleshooting

Make sure that your cuda, nvidia drivers, pillow, pytorch, and pytorch vision are the latest version.

Check the discord for help.

### Create a new model

```bash
  hypergan new mymodel
```

This will create a mymodel.json based off the default configuration. You can change configuration templates with the `-c` flag.

### List configuration templates

```bash
  hypergan new mymodel -l
```

See all configuration templates with `--list-templates` or `-l`.

### Train

```bash
  # Train a 32x32 gan with batch size 32 on a folder of folders of pngs, resizing images as necessary
  hypergan train folder/ -s 32x32x3 -c mymodel --resize
```

### Running on CPU

Don't train on CPU! It's too slow.

## The pip package hypergan

```bash
 hypergan -h
```

### Training

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -b 32 --config [name]
```

### Sampling

```bash
  hypergan sample [folder] -s 32x32x3 -b 32 --config [name] --sampler batch_walk --sample_every 5 --save_samples
```

By default hypergan will not save samples to disk. To change this, use `--save_samples`.

One way a network learns:

![https://hypergan.s3-us-west-1.amazonaws.com/1.0/output.gif](https://hypergan.s3-us-west-1.amazonaws.com/1.0/output.gif)

To create videos:

```bash
  ffmpeg -i samples/%06d.png video.mp4
```

### Arguments

To see a detailed list, run

```bash
  hypergan -h
```

## API

```python
import hypergan as hg
```

Note this API is currently under work in 1.0. If you are reading this before 1.0 is released check the examples.

See the [gitbook documentation](https://hypergan.gitbook.io/) for more details.

### Using a trained hypergan model

```python
    my_gan = hg.GAN('model.hypergan')
    batch_sample = my_gan.sample()
```

### Training a gan

```python
    gan = hg.GAN("default.json", inputs=hg.inputs.ImageLoader(...))
    trainable_gan = hg.TrainableGAN(gan)
    for step in trainable_gan.train():
        print("I'm on step ", step)
```

### Examples

See the examples [https://github.com/hypergan/HyperGAN/tree/master/examples](https://github.com/hypergan/HyperGAN/tree/master/examples)

### Tutorials

See the tutorials [https://hypergan.gitbook.io/hypergan/tutorials](https://hypergan.gitbook.io/hypergan/tutorials)

## Datasets

To build a new network you need a dataset. Your data should be structured like:

```text
  [folder]/[directory]/*.png
```

### Creating a Dataset

Datasets in HyperGAN are meant to be simple to create. Just use a folder of images.

### Cleaning up data

HyperGAN is built to be resilient to all types of unclean data. If your images are too large you have the choice of `--crop`, `--random_crop`, or `--resize`

`--crop` first resizes to include as much as the image as possible(center cropped).

## Features

A list of features in the 1.0 release:

* API
* CLI
* Viewer - an electron app to explore and create models
* Cross platform - Windows, OSX, Linux
* Inference - Add AI content generation to your project
* Training - Train custom models using accelerated parallel training backends
* Sharing - Share built models with each other. Use them in python projects as hypergan models, or in any project as onxx models
* Customizable - Define custom architectures in the json, or replace any component with your own pytorch creation
* Data - Built to work on unclean data and multiple data types
* Unsupervised learning
* Unsupervised alignment - Align one distribution to another or discover new novel distributions.
* Transfer learning
* Online learning

## Showcase

### 1.0 models are still training

Submit your showcase with a pull request!

For more, see the \#showcase room in [![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF)

## Sponsors

We are now accepting financial sponsors. Sponsor to (optionally) be listed here.

https://github.com/sponsors/hypergan

## Contributing

Contributions are welcome and appreciated! We have many open issues in the _Issues_ tab. Join the discord.

See [how to contribute.](./)

## Versioning

HyperGAN uses semantic versioning. [http://semver.org/](http://semver.org/)

TLDR: _x.y.z_

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  _z_ changes will be able to reload a saved graph.

## Citation

```text
  HyperGAN Community
  HyperGAN, (2016-2020+), 
  GitHub repository, 
  https://github.com/HyperGAN/HyperGAN
```

HyperGAN comes with no warranty or support.

