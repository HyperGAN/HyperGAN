# README

## HyperGAN 0.10

[![docs](https://img.shields.io/badge/gitbook-docs-yellowgreen)](https://hypergan.gitbook.io/hypergan/) [![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF) [![Twitter](https://img.shields.io/badge/twitter-follow-blue.svg)](https://twitter.com/hypergan)

A composable GAN API and CLI. Built for developers, researchers, and artists.

0.10 is now available in pip. Installation instructions and support are available in our [discord](https://discord.gg/t4WWBPF)

HyperGAN is in open beta.

![Colorizer 0.9 1](https://s3.amazonaws.com/hypergan-apidocs/0.9.0-images/colorizer-2.gif)

_Logos generated with_ [_examples/colorizer_](./#examples)

See more on the [hypergan youtube](https://www.youtube.com/channel/UCU33XvBbMnS8002_NB7JSvA)

## Table of contents

* [About](./#about)
* [Showcase](./#showcase)
* [Documentation](./#documentation)
* [Changelog](./#changelog)
* [Quick start](./#quick-start)
  * [Requirements](./#requirements)
  * [Install](./#install)
  * [Testing install](./#testing-install)
  * [Train](./#train)
  * [Development Mode](./#development-mode)
  * [Running on CPU](./#running-on-cpu)
* [The pip package hypergan](./#the-pip-package-hypergan)
  * [Training](./#training)
  * [Sampling](./#sampling)
* [API](https://github.com/HyperGAN/HyperGAN/tree/2170598c5c299e7b8d600a6e9a7db0f1d0bf36b6/API/README.md)
  * [Examples](./#examples)
* [Datasets](./#datasets)
  * [Creating a Dataset](./#creating-a-dataset)
  * [Downloadable Datasets](./#downloadable-datasets)
  * [Cleaning up data](./#cleaning-up-data)
* [Contributing](./#contributing)
* [Versioning](./#Versioning)
* [Sources](./#sources)
* [Papers](./#papers)
* [Citation](./#citation)

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

## Showcase

![Hypergan Mobile released!](https://miro.medium.com/max/1404/1*uJmzGUvoP0WdaQPkkT8s-Q.jpeg)

Run trained models with HyperGAN on your android device!

Submit your showcase with a pull request!

For more, see the \#showcase room in [![Discord](https://img.shields.io/badge/discord-join%20chat-brightgreen.svg)](https://discord.gg/t4WWBPF)

## Documentation

* [Model author JSON reference](json.md)
* [Model author tutorial 1](tutorial1.md)
* [0.10.x](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/index.html)
* [0.9.x](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
* [Test coverage](https://s3.amazonaws.com/hypergan-apidocs/0.10.0/coverage/index.html)

## Changelog

See the full changelog here: [Changelog.md](changelog.md)

## Quick start

### Requirements

Recommended: GTX 1080+

### Install

#### Comprehensive HyperGAN Installation Tutorial

1. Notes
   * The point of this guide is to install HyperGAN with GPU support.
   * Installation tested and working on ElementaryOS 5.0 Juno \(equivilant to Ubuntu 18.04\), NVIDIA GeForce 970M.
   * Some restarts might be unnecessary, but do them, just to be sure.
   * If you follow these instructions and need further help, please visit the Discord.
   * Written 10.29.2019.

     0.5. Disabling Secure Boot

   * From the "GPU Support" page, www.tensorflow.org - 

     "Secure Boot complicates installation of the NVIDIA driver and is beyond the scope of these instructions."

   * A quick Google search such as "disable secure boot {motherboard}" will get you more detailed instructions for this step.
   * After disabling secure boot, restart your computer.
2. Installing the proper GPU drivers 
   * We're going to need to update our drivers to be above 410.x to run HyperGAN.
   * To check what installation you need, use the command `ubuntu-driver devices`.
   * The name of your graphics card should pop up, with a list of drivers after.
   * You're going to want to choose the one that states "third party free reccommended".
   * Go ahead and run the command 'sudo apt-get install nvidia-driver-xxx\` with the correct numbers to update your drivers.
   * After install, restart your computer.
   * When rebooted, make sure your drivers are installed properly by running `nvidia-smi`.
   * The output should show your graphics card model and driver version.
3. Installing tensorflow-gpu Dependencies
   * HyperGAN requires the use of Google's TensorFlow to run.
   * In addition, TensorFlow needs NVIDIA's CUDA toolkit and cuDNN \(NVIDIA CUDA® Deep Neural Network library\).
   * We're going to be installing 2 things in this section: the CUDA toolkit and the cuDNN.
     * CUDA toolkit:
       * [https://developer.nvidia.com/cuda-10.0-download-archive](https://developer.nvidia.com/cuda-10.0-download-archive)
       * \(It is important that you download the 10.0 version of the toolkit, linked above.\)
       * Click the buttons to narrow down your target platform.
       * Once you've selected your OS version, select deb\(local\).
       * Download this file and follow the instructions on the site to complete installation.
       * After you have finished following those instructions, restart your computer.
     * cuDNN:
       * [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download)
       * \(It is important that you download v7.6.4, compatible with the 10.0 version of the CUDA toolkit.\)
       * To download the cuDNN, you're going to have to sign up for a NVIDIA account.
       * Create an account \(or sign in with Google\) and log in.
       * Download cuDNN v7.6.4 for CUDA 10.0. Choose the "cuDNN Runtime Library for UbuntuXX.XX \(Deb\)".
       * Navigate to where you downloaded the program and use `sudo dpkg -i` followed by the .deb file name to finish installation.
       * After the package is installed, reboot your computer.
4. Installing HyperGAN 
   * Now it is time to install HyperGAN and it's dependencies.
   * If you haven't installed pip3, install it using `sudo apt-get install python3-pip`.
   * Run the command \`pip3 install hypergan tensorflow-gpu hyperchamber pillow pygame natsort nashpy'
   * The newest version of numpy spits out a ton of non-pretty warnings, so we'll install an older version using \`pip3 install numpy==1.16.4'.
   * Reboot your computer, for the last time.
5. Checking HyperGAN installation
   * Test that HyperGAN and TensorFlow are correctly installed on your computer by running the command `hypergan test`.
   * If you're all good to go and have followed these instructions, a message should be returned saying \`Congratulations! Tensorflow and hypergan both look installed correctly'.
   * If your installation has been completed, jump to step 6.
   * If something has gone wrong, continue to step 5.
6. Troubleshooting
   * "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
     * Something has gone wrong with the installation of your drivers.
     * Run the command \`sudo apt-get purge nvidia-\* && sudo apt-get autoremove'.
     * Go back to step 1 and retry the driver installation.
   * "FutureWarning: Passing \(type, 1\) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as \(type, \(1,\)\) / '\(1,\)type'."
     * This is a warning message. If you want to remove it, downgrade numpy.
     * Run the command \`pip3 uninstall numpy && pip3 install numpy==1.16.4'.
   * "ImportError: libcudart.so.10.0: cannot open shared object file: No such file or directory"
     * This error means that there was a problem installing the CUDA toolkit.
     * Please reinstall that step of the instructions, making sure you download the 10.0 version.
   * "ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory"
     * This error means that there was a problem installing the cuDNN.
     * Please reinstall that step of the instructions, making sure you download cuDNN v7.6.4 for CUDA 10.0.
7. Conclusion 
   * Congrats! Now you're ready to start using HyperGAN.
   * Once you've made something cool, be sure to share it on the Discord \([https://discord.gg/t4WWBPF](https://discord.gg/t4WWBPF)\).

#### Optional `virtualenv`:

If you use virtualenv:

```bash
  virtualenv --system-site-packages -p python3 hypergan
  source hypergan/bin/activate
```

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

One way a network learns:

[![Demo CountPages alpha](https://j.gifs.com/58KmzA.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg&index=1)

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

## Datasets

To build a new network you need a dataset. Your data should be structured like:

```text
  [folder]/[directory]/*.png
```

### Creating a Dataset

Datasets in HyperGAN are meant to be simple to create. Just use a folder of images.

```text
 [folder]/*.png
```

For jpg\(pass `-f jpg`\)

### Downloadable datasets

* Loose images of any kind can be used
* CelebA aligned faces [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* MS Coco [http://mscoco.org/](http://mscoco.org/)
* ImageNet [http://image-net.org/](http://image-net.org/)
* youtube-dl \(see [examples/Readme.md](examples/examples.md)\)

### Cleaning up data

To convert and resize your data for processing, you can use imagemagick

```text
for i in *.jpg; do convert $i  -resize "300x256" -gravity north   -extent 256x256 -format png -crop 256x256+0+0 +repage $i-256x256.png;done
```

## Contributing

Contributions are welcome and appreciated! We have many open issues in the _Issues_ tab. Join the discord.

See [how to contribute.](./)

## Versioning

HyperGAN uses semantic versioning. [http://semver.org/](http://semver.org/)

TLDR: _x.y.z_

* _x_ is incremented on stable public releases.
* _y_ is incremented on API breaking changes.  This includes configuration file changes and graph construction changes.
* _z_ is incremented on non-API breaking changes.  _z_ changes will be able to reload a saved graph.

### Other GAN projects

* [StyleGAN](https://github.com/NVlabs/stylegan)
* [https://github.com/LMescheder/GAN\_stability](https://github.com/LMescheder/GAN_stability)
* Add yours with a pull request

### Papers

* GAN - [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
* DCGAN - [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
* InfoGAN - [https://arxiv.org/abs/1606.03657](https://arxiv.org/abs/1606.03657)
* Improved GAN - [https://arxiv.org/abs/1606.03498](https://arxiv.org/abs/1606.03498)
* Adversarial Inference - [https://arxiv.org/abs/1606.00704](https://arxiv.org/abs/1606.00704)
* Energy-based Generative Adversarial Network - [https://arxiv.org/abs/1609.03126](https://arxiv.org/abs/1609.03126)
* Wasserstein GAN - [https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)
* Least Squares GAN - [https://arxiv.org/pdf/1611.04076v2.pdf](https://arxiv.org/pdf/1611.04076v2.pdf)
* Boundary Equilibrium GAN - [https://arxiv.org/abs/1703.10717](https://arxiv.org/abs/1703.10717)
* Self-Normalizing Neural Networks - [https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515)
* Variational Approaches for Auto-Encoding

  Generative Adversarial Networks - [https://arxiv.org/pdf/1706.04987.pdf](https://arxiv.org/pdf/1706.04987.pdf)

* CycleGAN - [https://junyanz.github.io/CycleGAN/](https://junyanz.github.io/CycleGAN/)
* DiscoGAN - [https://arxiv.org/pdf/1703.05192.pdf](https://arxiv.org/pdf/1703.05192.pdf)
* Softmax GAN - [https://arxiv.org/abs/1704.06191](https://arxiv.org/abs/1704.06191)
* The Cramer Distance as a Solution to Biased Wasserstein Gradients - [https://arxiv.org/abs/1705.10743](https://arxiv.org/abs/1705.10743)
* Improved Training of Wasserstein GANs - [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
* More...

### Sources

* DCGAN - [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
* InfoGAN - [https://github.com/openai/InfoGAN](https://github.com/openai/InfoGAN)
* Improved GAN - [https://github.com/openai/improved-gan](https://github.com/openai/improved-gan)

## Citation

```text
  HyperGAN Community
  HyperGAN, (2016-2019+), 
  GitHub repository, 
  https://github.com/HyperGAN/HyperGAN
```

HyperGAN comes with no warranty or support.

