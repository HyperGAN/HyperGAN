---
description: >-
  HyperGAN is a composable GAN API and CLI. Built for developers, researchers,
  and artists.
---

# About

* [Documentation](./#documentation)
* [Quick start](./#quick-start)
* * [Requirements](./#requirements)
  * [Install](./#install)
  * [Testing install](./#testing-install)
  * [Train](./#train)
  * [Development Mode](./#development-mode)
  * [Running on CPU](./#running-on-cpu)
* [The pip package hypergan](./#the-pip-package-hypergan)
  * [Training](./#training)
  * [Sampling](./#sampling)

## About

Generative Adversarial Networks consist of 2 learning systems that learn together. HyperGAN implements these learning systems in Tensorflow with deep learning.

For an introduction to GANs, see [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

HyperGAN is a community project. GANs are a very new and active field of research. Join the community [discord](https://discord.gg/t4WWBPF).

### Features

* Community project
* Unsupervised learning
* Transfer learning
* Online learning
* Dataset agnostic
* Reproducible architectures using json configurations
* Domain Specific Language to define custom architectures
* GUI\(pygame and tk\)
* API
* CLI

## Documentation

* [Model author JSON reference](configuration/)
* [Model author tutorial 1](tutorials/training.md)
* [0.10.x](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/index.html)
* [0.9.x](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
* [Test coverage](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/coverage/index.html)

## Changelog

See the full changelog here: [Changelog.md](changelog.md)

## Quick start

### Requirements

Recommended: GTX 1080+

### Install

#### Install hypergan:

```bash
  pip3 install hypergan --upgrade
```

#### Testing install

To see that tensorflow and hypergan are installed correctly and have access to devices, please run:

```text
  hypergan test
```

#### Optional `virtualenv`:

If you use virtualenv:

```bash
  virtualenv --system-site-packages -p python3 hypergan
  source hypergan/bin/activate
```

#### Dependencies:

If installation fails try this.

```bash
  pip3 install numpy tensorflow-gpu hyperchamber pillow pygame
```

#### Dependency help

If the above step fails see the dependency documentation:

* tensorflow - [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)
* pygame  - [http://www.pygame.org/wiki/GettingStarted](http://www.pygame.org/wiki/GettingStarted)

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
  hypergan train folder/ -s 32x32x3 -f png -c mymodel --resize
```

### Development mode

If you wish to modify hypergan

```bash
git clone https://github.com/hypergan/hypergan
cd hypergan
python3 setup.py develop
```

### Running on CPU

Make sure to include the following 2 arguments:

```bash
CUDA_VISIBLE_DEVICES= hypergan --device '/cpu:0'
```

Don't train on CPU! It's too slow.

## The pip package hypergan

```bash
 hypergan -h
```

### Training

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -f png -b 32 --config [name]
```

### Sampling

```bash
  hypergan sample [folder] -s 32x32x3 -f png -b 32 --config [name] --sampler batch_walk --sample_every 5 --save_samples
```

By default hypergan will not save samples to disk. To change this, use `--save_samples`.

To create videos:

```bash
  ffmpeg -i samples/%06d.png -vcodec libx264 -crf 22 -threads 0 gan.mp4
```

### Arguments

To see a detailed list, run

```bash
  hypergan -h
```

### Examples

See the example documentation [https://github.com/hypergan/HyperGAN/tree/master/examples](https://github.com/hypergan/HyperGAN/tree/master/examples)

## Contributing

Contributions are welcome and appreciated! We have many open issues in the _Issues_ tab. Join the discord.

## Versioning

HyperGAN uses semantic versioning. [http://semver.org/](http://semver.org/)

TLDR: _x.y.z_

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  _z_ changes will be able to reload a saved graph.

## Citation

```text
  HyperGAN Community
  HyperGAN, (2016-2019+), 
  GitHub repository, 
  https://github.com/HyperGAN/HyperGAN
```

HyperGAN comes with no warranty or support.
