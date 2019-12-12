# Changelog

## 0.11 ~ Usability improvements+

* Documentation update \(gitbook\)
* New train hooks
* Mobile AI explorer released
* Multithreaded UI - no more slowdowns
* More...

## 0.10 ~ Community Project release

* HyperGAN 0.10 released as a community project
* Configurable network architectures using a simple DSL
* Lots of new regularizers and losses
* Evolution based GangTrainer
* Curriculum trainer with progressive growing
* Optimistic loading
* New next-frame example
* More...

## 0.9 ~ Refactorings and optimizations

* API Documentation - [https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html](https://s3.amazonaws.com/hypergan-apidocs/0.9.0/index.html)
* Prepackaged configurations
* New viewer front-end!
* Examples, including the ability to randomly search for good configurations

See more here [https://github.com/255BITS/HyperGAN/pull/66](https://github.com/255BITS/HyperGAN/pull/66)

## 0.8 ~ Least Squares GAN and API examples

* Tensorflow 1.0 support
* New configuration format and refactored api.
* New loss based on least squared GAN.  See [least squares GAN implementation](changelog.md#Least-Squares-GAN).
* API example `2d-test` - tests a trainer/encoder/loss combination against a known distribution.
* API example `2d-measure` - measure and report the above test by randomly combining options.
* Updated default configuration.
* More

## 0.7 ~ WGAN & API

* New loss `wgan`
* Initial Public API Release
* API example: `colorizer` - re-colorize an image!
* API example: `inpainter` - remove a section of an image and have your GAN repaint it
* API example: `super-resolution` - zoom in and enhance.  We've caught the bad guy!
* 4 _new_ samplers.  `--sampler` flag.  Valid options are: `batch`,`progressive`,`static_batch`,`grid`. 

## 0.5 / 0.6

* pip package released

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold-0-5-6.png) ![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/face-manifold.png)

### 0.1-0.4

* Initial private release

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1.png) ![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/legacy-0.1-2.png)

