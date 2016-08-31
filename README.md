# hyperchamber-gan
A GAN(generative adversarial network) that you can run from the command line.  Integrates with hyperchamber to find the best GAN for your dataset.

## Screenshots

<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472598314113-ad31fc0a-b7aa-4447-8ec4-fafd8ae72df1'/>
<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472598316062-fe406955-0527-4fe5-a2c8-cfdb96567f79'/>
<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472598314858-464c1d85-e8ba-47f3-bd73-b112117d2f37'/>
<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472595854675-96b0b0df-c727-45c5-a13d-b714916e188f'/>
<img src='https://hyperchamber.s3.amazonaws.com/samples/images-1472595283436-3f54384a-382d-40af-bd78-65deb917120f'/>

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
To use on any data:

* Run config sweep with hyperchamber
* Run your favorite config for longer


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
