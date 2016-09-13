# HyperGAN
A versatile GAN(generative adversarial network) implementation focused on scalability and ease-of-use.

## Samples

The following are a small sample of the manifolds that our generators have learned.

Randomly chosen:

* Card game

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/magic-1.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/magic-2.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/magic-3.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/magic-4.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/magic-5.png'/>

* People

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/decent-1.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/decent-2.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/decent-3.png'/>


* Fonts (the letter 'r')

<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/font-1.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/font-2.png'/>
<img src='https://raw.githubusercontent.com/255BITS/HyperGAN/master/samples/font-3.png'/>

# References


## Features

* Efficient GAN implementation
* Semi-supervised or unsupervised learning(works with and without labels)
* Variational methods
* InfoGAN-inspired categories
* Minibatch normalization
* Adversarial inference

## Arguments

* --directory, required, specifies which parent directory to use.  Each subdirectory is a different classification.  For example if you have 'data/a' and 'data/b', files in 'data/a' will be represented internally as the one-hot vector [1,0].
* --channels, optional(default 3), the number of channels in your images.  Black and white images typically only have 1.
* --width, optional(default 64), the width of each image.  If smaller than this width, the image will be centered and cropped.
* --height, optional(default 64), the height of each image.  If smaller than this height, the image will be centered and cropped.
* --format, optional(default png), file format of the images.  Only supports jpg and png for now.
* --epoch, optional(default 10), number of epochs to run before exiting
* --load_config, optional(default None), the config uuid from hyperchamber to run
* --save_every, optional(default 0), after this many epochs, the network weights/checkpoint are saved into the 'saves' directory.

## Running

First, choose a dataset.  If you use one of the standard datasets, you can find a config that will work with hyperchamber.  If using a new/different dataset, start with a 'Search', detailed below.

For example, the following command will run a configuration with id [https://hyperchamber.255bits.com/ba3051e944601e2d98b601f3347df0b1/40k_overfit_3:1.2/samples/25e15455541d84bb76dcc4c8f8dce5a1](7543d9f4cc6746b68b3247e7c258a50e)

Note: Requires a (free) account at hyperchamber to run.

```
  python3 directory-gan.py --directory test/ --channels 3 --width 64 --height 80 --format jpg --epoch 10000 --load_config 6d09d839d3b74a208a168dc3189c3f59 --save_every 10
```


## Searching

````
  python3 directory-gan.py --directory test/ --channels 3 --width 64 --height 80 --format jpg --epoch 1
```

You can review the results on hyperchamber.

## How to use

Fair warning, this is all still alpha.  Even so, email us if you have issues running this on your dataset.

To use on any data:

* Run config sweep with hyperchamber
* Run your favorite config for longer


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

