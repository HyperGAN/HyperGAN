# HyperGAN
A versatile GAN(generative adversarial network) implementation focused on scalability and ease-of-use.

## hypergan is in pre-release state.  The documentation is a WIP and there is no pip package released yet.

## Changelog

### 0.5('faces' release)rc1
* Initial release of new refactored gan

## Goals

HyperGAN is focused on making GANs easy to train and run.

It is currently in an open beta state, and contributions are welcome.

* Easy to use and deploy
* Fast
* Extensible

## Quick start

### Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better
2. For smaller sizes, you can use an older GPU. 
3. For debugging syntax errors, CPU use is fine.  Otherwise use a GPU.

### Install hypergan

```
  pip install hypergan
```

### Run hypergan

```
  # Train a 256x256 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 256x256 -f png -b 32
```

### Increasing performance

On ubuntu `sudo apt-get install libgoogle-perftools4` and make sure to include this environment variable before training

```
  LD_PRELOAD="/usr/lib/libtcmalloc.so.4" hypergan train my_dataset
```

## CLI

### hypergan train

### hypergan serve

Runs a trained s

Our server is a small flask server which contains the following endpoints:

* /sample.png

Returns a random sample

### hypergan build

The trained generator can now be built for deployment.  Building does 2 things:

* Loads the training model, which include the discriminator
* Saves into a ckpt model containing only the generator

## Saves

~/.hypergan/saves/
~/.hypergan/samples/


### List models

```
  hypergan models
```

### Building a generator

```
  hypergan build [model]
```

### Server mode

```
  hypergan serve [model]
```

TODO: api docs/routes

## Development mode

```
git clone https://github.com/255BITS/hypergan
cd hypergan
python3 hypergan.py # with usual arguments
```


## Architecture

Building a GAN involves making a lot of choices.  

Choices like:  Should I use variational inference?  Adversarial encoding?  What should the size of my z dimensions be? Categorical loss?

Some of these choices will vary by dataset.

hypergan is a flexible GAN framework that lets us easy explore complex GANs by just making declarative choices.

hypergan makes it easy to replace the discriminator, use a different training technique, switch data types, use a different loss function, change z size, and much more.
```
```

## Discriminators

The discriminators job is to tell if a piece of data is real or fake.  In hypergan, a discriminator can also be a classifier.

If the discriminator is a classifier, we treat this part of the network as a softmax classifier.

To put this as an example, if we were to classify the difference between apples and oranges, most classifiers would classify a pear as an apple, having never seen a pear before.
A classifier trained with a GAN will include additional information - a discriminator which could identify the pear as a fake image(in the context of worlds consisting of only apples and oranges).

At a high level our discriminator does the following:

```
  create_discriminator:
    graph = x #our input data
    graph.apply 'discriminator.pre.regularizers' - this could be just gaussian noise
    graph.apply 'discriminator'
    graph.apply 'discriminator.post.regularizers' -
```

### Options

Implemented discriminators: TODO

## Generators

Generators generate data.  Any real valued data theoretically.


### Generating audio

Experimental.  So experimental that you'll have to dig through to figure out how to even run it right now.

### Generating images

```
  --format png
```

Future goals include being able to generate discrete data.  Sequence GAN and other reinforcement learning techniques seem very promising.

### Debugging a generator

## Visualizing learning

Different GAN configurations learn differently, and it's sometimes useful to visualize how they learn.

[![HyperGAN visualization](https://j.gifs.com/mwNKjn.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&index=1&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg)

To create your own visualizations, you can use the flag:

``` 
  --frame_sample grid 
```

This will create a sample using samples/grid_sampler to iterate over the first two z dimensions.

To turn these images into a video, use the following:

```
  ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 gan.mp4
```

TODO: z_dims must equal 2 and batch size must equal 24 to work.  This is temporary

### Trainers

## Server mode

```
  hypergan server=True
```
TODO 

## Formats

* jpg
* png
* wav(experimental)
* mp3(experimental)

## Features

* Efficient GAN implementation
* Semi-supervised or unsupervised learning(works with and without labels)
* Variational methods
* InfoGAN-inspired categories
* Minibatch normalization
* Adversarial inference
* Flask server mode

## Arguments

To see a detailed list, run 
```
  hypergan -h
```

* -s, --size, optional(default 64x64x3), the size of your data in the form 'width'x'height'x'channels'
* -f, --format, optional(default png), file format of the images.  Only supports jpg and png for now.

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

## Contributing

Contributions are welcome and appreciated.  To help out, just add something you wish existed.  

Also, if you create something cool with this let us know!

# Citation

If you wish to cite this project, do so like this:

```
  TODO
```
