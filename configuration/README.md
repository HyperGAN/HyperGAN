# Configurations

## Hypergan model author JSON reference

### ConfigurableComponent

Hypergan comes with a simple DSL to design your own network architectures. A ConfigurableComponent can be used to create custom designs when building your models.

### ConfigurableDiscriminator

For example, a discriminator component can be defined as:

```javascript
...
  "discriminator": 
  {
      "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
      "defaults":{
        "activation": "tanh",
        "initializer": "he_normal",
        "filter": [3,3],
        "stride": [1,1],
        "avg_pool": [2,2]
      },
      "layers":[
        "conv 32",
        "conv 64 ", 
        "conv 128",
        "conv 256",
        "linear 1 activation=null"
      ]

  }
```

This means to create a network composed of 4 convolution layers that decrease along stride and increase filter size, ending in a linear without activation. Losses expect `activation=null` on the last layer.

We can modify this to contain `resnet` instead of `conv` blocks to increase model capacity\(though its not clear by how much\).

### ConfigurableGenerator

```javascript
  "generator": {
    "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
    "defaults": {
      "activation": "tanh",
      "initializer": "he_normal",
      "filter": [3,3],
      "stride": [1,1],
      "avg_pool": [1,1]
    },
    "layers": [
      "linear 128",
      "reshape -1 name=w",
      "const 1*1*128",
      "adaptive_instance_norm",
      "resize_conv 256",
      "adaptive_instance_norm",
      "resize_conv 256",
      "adaptive_instance_norm",
      "resize_conv 256",
      "adaptive_instance_norm",
      "resize_conv 256",
      "adaptive_instance_norm",
      "resize_conv 256",
      "adaptive_instance_norm",
      "resize_conv 128",
      "adaptive_instance_norm",
      "resize_conv 64",
      "adaptive_instance_norm",
      "resize_conv 32",
      "adaptive_instance_norm",
      "resize_conv 3 activation=null"
    ]

  }
```

This is a generator. A generator takes in a latent space and returns a data type that matches the input.

`adaptive_instance_norm` looks up the layer named 'w' and uses it to perform the adaptive instance norm.

Our output range depends on our input loader. In the case of images, our output range is covered by both `tanh` and `null`.

## Layers

Common layer types:

### linear

`linear [outputs] (options)`

Creates a linear layer.

### conv

`conv [filters] (options)`

A convolution. Average pool and stride are applied if they are set. For example: `conv [filters] filter=5 avg_pool=2 stride=1` will run set stride to 1 and run avg\_pool of 2 with a filter size of 5.

### deconv

`deconv [filters] (options)`

Set stride=2 to double the width and height of your tensor.

### resize\_conv

`resize_conv [output filters] (options)`

### reshape

`reshape [size]`

size can be:

* -1
* `*` delimited dimensions.

### const

`const [size]`

size is `*` delimited dimensions

### attention

`attention (options)`

Allows the model to correlate feature vectors. Expects a 4 dimensional tensor as input.

Usage:

`attention`

### crop

`crop [w h d]`

Crops the network to the output size.

Defaults to the input resolution if no arguments are specified.

Output: A tensor with the form \[batch\_size, w, h, d\]

### resize\_images

`resize_images [w h] (method=1...4)`

Resizes the network to the output size.

Defaults to the input resolution if no arguments are specified.

Output: A tensor with the form \[batch\_size, w, h, channels\]

See [https://www.tensorflow.org/api\_docs/python/tf/image/resize\_images](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)

## Configuration

Configuration in HyperGAN uses JSON files. You can create a new config with the default template by running `hypergan new mymodel`.

You can see all templates with `hypergan new mymodel -l`.

### Architecture

A hypergan configuration contains all hyperparameters for reproducing the full GAN.

In the original DCGAN you will have one of the following components:

* Distribution\(latent space\)
* Generator
* Discriminator
* Loss
* Trainer

Other architectures may differ. See the configuration templates.

### GANComponent

A base class for each of the component types listed below.

### Generator

A generator is responsible for projecting an encoding \(sometimes called _z space_\) to an output \(normally an image\). A single GAN object from HyperGAN has one generator.

### 

### Discriminators

A discriminator's main purpose\(sometimes called a critic\) is to separate out G from X, and to give the Generator a useful error signal to learn from.

### DSL

Each component in the GAN can be specified with a flexible DSL inside the JSON file.

### Losses

### WGAN

Wasserstein Loss is simply:

```python
 d_loss = d_real - d_fake
 g_loss = d_fake
```

d\_loss and g\_loss can be reversed as well - just add a '-' sign.

### Least-Squares GAN

```python
 d_loss = (d_real-b)**2 - (d_fake-a)**2
 g_loss = (d_fake-c)**2
```

a, b, and c are all hyperparameters.

#### Standard GAN and Improved GAN

Includes support for Improved GAN. See `hypergan/losses/standard_gan_loss.py` for details.

#### Boundary Equilibrium Loss

Use with the `AutoencoderDiscriminator`.

See the `began` configuration template.

#### Loss configuration

| attribute | description | type |
| :---: | :---: | :---: |
| batch\_norm | batch\_norm\_1, layer\_norm\_1, or None | f\(batch\_size, name\)\(net\):net |
| create | Called during graph creation | f\(config, gan, net\):net |
| discriminator | Set to restrict this loss to a single discriminator\(defaults to all\) | int &gt;= 0 or None |
| label\_smooth | improved gan - Label smoothing. | float &gt; 0 |
| labels | lsgan - A triplet of values containing \(a,b,c\) terms. | \[a,b,c\] floats |
| reduce | Reduces the output before applying loss | f\(net\):net |
| reverse | Reverses the loss terms, if applicable | boolean |

### Trainers

Determined by the GAN implementation. These variables are the same across all trainers.

#### Consensus

Consensus trainers trains G and D at the same time. Resize Conv is known to not work with this technique\(PR welcome\).

**Configuration**

| attribute | description | type |
| :---: | :---: | :---: |
| learn\_rate | Learning rate for the generator | float &gt;= 0 |
| beta1 | \(adam\) | float &gt;= 0 |
| beta2 | \(adam\) | float &gt;= 0 |
| epsilon | \(adam\) | float &gt;= 0 |
| decay | \(rmsprop\) | float &gt;= 0 |
| momentum | \(rmsprop\) | float &gt;= 0 |

#### Alternating

Original GAN training. Locks generator weights while training the discriminator, and vice-versa.

**Configuration**

| attribute | description | type |
| :---: | :---: | :---: |
| g\_learn\_rate | Learning rate for the generator | float &gt;= 0 |
| g\_beta1 | \(adam\) | float &gt;= 0 |
| g\_beta2 | \(adam\) | float &gt;= 0 |
| g\_epsilon | \(adam\) | float &gt;= 0 |
| g\_decay | \(rmsprop\) | float &gt;= 0 |
| g\_momentum | \(rmsprop\) | float &gt;= 0 |
| d\_learn\_rate | Learning rate for the discriminator | float &gt;= 0 |
| d\_beta1 | \(adam\) | float &gt;= 0 |
| d\_beta2 | \(adam\) | float &gt;= 0 |
| d\_epsilon | \(adam\) | float &gt;= 0 |
| d\_decay | \(rmsprop\) | float &gt;= 0 |
| d\_momentum | \(rmsprop\) | float &gt;= 0 |
| clipped\_gradients | If set, gradients will be clipped to this value. | float &gt; 0 or None |
| d\_clipped\_weights | If set, the discriminator will be clipped by value. | float &gt; 0 or None |

#### Fitness

Only trains on good z candidates.

#### Curriculum

Train on a schedule.

#### Gang

An evolution based trainer that plays a subgame between multiple generators/discriminators.

