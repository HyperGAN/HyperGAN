# hyperchamber-gan
A DC-GAN(Convolutional Generative Adversarial Network) that you can run from the command line.  Integrates with hyperchamber to find the best GAN for your dataset.


## Results

Note, these results are live from the latest samples of each running model.  They update over time, so check back often(or see more at hyperchamber):

### MTG cards
<img src="https://hyperchamber.255bits.com/api/v1/sample/martyn/magic:0.2/latest/0.jpg"/>

### Comics
<img src="https://hyperchamber.255bits.com/api/v1/sample/martyn/magic:0.2/latest/0.jpg"/>

### MNIST
<img src="https://hyperchamber.255bits.com/api/v1/sample/martyn/magic:0.2/latest/0.jpg"/>

### CIFAR-10
<img src="https://hyperchamber.255bits.com/api/v1/sample/martyn/magic:0.2/latest/0.jpg"/>

## Features

* Efficient GAN implementation
* Softmax semi-supervised and sigmoid unsupervised learning(works with and without labels)
* Variational methods
* InfoGAN-inspired self sorting categories
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

For example, the following command will run a configuration with id [hyperchamber link](6d09d839d3b74a208a168dc3189c3f59).

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
To use on any data:

* Run config sweep with hyperchamber
* Run your favorite config for longer

## Join our community

Interested in work like this?  Need help?  Come join us at hyperchamber's slack room.  Sign up and request an invite at hyperchamber.

# References

## Papers

* GAN
* DCGAN
* InfoGAN
* Improved GAN
* Adversarial Inference

## Sources

* DCGAN
* InfoGAN
* Ian's GAN
