# HyperGAN
A versatile GAN(generative adversarial network) implementation focused on scalability and ease-of-use.

## hypergan is in pre-release state.  The documentation is a WIP and there is no pip package released yet.

## Changelog


## Examples

### Best samplings:

TODO IMAGE

These are the best images we have been able to obtain from a GAN.  They are hand picked.

### Random batch, same network:

TODO IMAGE

This shows what a random batch looks.  They are not hand picked.

### Progressive enhancement:

TODO IMAGE

Somewhat inspired by LAPGAN, we feed resized images across the layers of D.  Our generator learns to render at multiple
resolutions.  

### 0.5.0
* pip package released!
* Better defaults.  Good variance.  The broken images showed up after training for 5 days.

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/face-manifold.png'/>

### 0.1-0.4
* Initial private release

<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472503244410-fcc6b07b-ec8f-44f6-aa2b-937a6ca755dc'/>
<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472511234866-6123711b-229c-436b-a337-19e35bb79457'/>

## Goals

HyperGAN is focused on making GANs easy to train and run.

It is currently in an open beta state, and contributions are welcome.

* Easy to use and deploy
* Fast
* Extensible

## Quick start

### Minimum requirements

1. For 256x256, we recommend a GTX 1080 or better.
2. For smaller sizes, you can use an older GPU. 
3. CPU works slowly, it is useful for server mode. 
4. CPU is _extremely_ discouraged for use in training.  Use a GPU to train.
5. We train on nvidia titan Xs: [https://www.nvidia.com/en-us/geforce/products/10series/titan-x-pascal/](https://www.nvidia.com/en-us/geforce/products/10series/titan-x-pascal/)

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

Sample from a trained generator.

Our server is a small flask server which contains the following endpoints:

* /sample.png

Returns a random sample.

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

Once your model is loaded and starts, you can hit the following paths:

```
/sample.png?type=batch
```

This creates a random sampling of generated images.

## Development mode

```
git clone https://github.com/255BITS/hypergan
cd hypergan
python3 hypergan.py # with usual arguments
```


## Architecture

Generative Adversarial Networks(2) consist of (at least) two neural networks that learn together over many epochs.
The discriminator learns the difference between real and fake data.  The generator learns to create fake data.

For a more depth introduction, see here [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)

Building a GAN involves making a lot of choices with largely unexplored consequences.

Some common questions: 

* Should I use variational inference?  
* Should I add adversarial encoding?
* What should the size of my z dimensions be? 
* Do I add categorical loss to model discrete values?
* Which loss functions are important?

Some of these choices may vary by dataset.

hypergan is a flexible GAN framework that lets us easy explore complex GANs by just making declarative choices.

hypergan aims to make things easy.  You can easily test out combinations of:

* learning rates & training technique
* Use many different loss functions
* change z size and type
* and so much more.

## Discriminators

The discriminators job is to tell if a piece of data is real or fake.  In hypergan, a discriminator can also be a classifier.

To put this as an example, if we were to classify the difference between apples and oranges, most classifiers would classify a pear as an apple, having never seen a pear before.
A classifier trained with a GAN will include additional information - a discriminator which could identify the pear as a fake image(in the context of worlds consisting of only apples and oranges).

At a high level our discriminator does the following:

```
  create_discriminator:
    graph = x #our input data
    graph.apply 'discriminator.pre.regularizers' - this could be just gaussian noise
    graph.apply 'discriminator'
    graph.apply 'discriminator.post.regularizers' - commonly this is minibatch
```

## Generators

Generators generate data.  Any real valued data theoretically.  HyperGAN is currently focused on images, but other data types are on our horizon.

Specifically we are running experiments with audio.

### Debugging a generator

## Visualizing learning

Different GAN configurations learn differently, and it's sometimes useful to visualize how they learn.

[![Demo CountPages alpha](https://j.gifs.com/58KmzA.gif)](https://www.youtube.com/watch?v=tj3ZLNfcJFo&list=PLWW3WtkBA3MuSnAVS__D0FkENZzuTbHFg&index=1)

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

```
--format <type>
```

Type can be one of:
* jpg
* png

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

Our pivotal board is here: https://www.pivotaltracker.com/n/projects/1886395

Contributions are welcome and appreciated.  To help out, just issue a pull request.

Also, if you create something cool with this let us know!

# Citation

If you wish to cite this project, do so like this:

```
  255bits (M. Garcia), HyperGAN, (2017), GitHub repository, https://github.com/255BITS/HyperGAN
```

