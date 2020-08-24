# Getting started

HyperGAN is currently in pre-release and open beta.

Everyone will have different goals when using hypergan. Here are some common uses:

## Training a network

HyperGAN is currently beta. We are still searching for a default cross data-set configuration. Various papers are implemented by listing \`hypergan new modelname -l'

## Deploying a model

## Building datasets

### Building image datasets

## Using search

Each of the examples support search. Automated search can help find good configurations.

If you are unsure, you can start with the 2d-distribution.py. Check out random\_search.py for possibilities, you'll likely want to modify it.

Please consider sharing in a PR if you find goood configurations!

## Manual search

It's faster to do manual search if you are actively looking at results. For manual search here are some tips:

* create a baseline json configuration
* start with a working configuration
* run baseline every time
* each phase make the best performer the new baseline
* the examples are capable of \(sometimes\) finding a good trainer, like 2d-distribution.  Mixing and matching components seems to work.

## Creating a new configuration

Hyperparameter tuning, or doing something like trying softmax loss with gradient penalty, can all be done in the json configuration file.

## Implementing a paper

Each paper is a combination of json file and code.

HyperGAN has tried to make it easy to add a new component. Here are some basic classification rules on where paper implementations should go:

| paper type | proposed implementation |
| :--- | :--- |
| new type of encoder | new encoder class |
| new type of input distribution encoding | either new encoder class or projection of uniform\_encoder |
| new type of generator | create a new generator class |
| feature that could possibly apply to all generators | BaseGenerator |
| new type of discriminator | create a new discriminator class |
| applies to all discriminators | add it to all discriminators with BaseDiscriminator |
| new type of loss | create a new loss class |
| new type of gradient penalty | add it to all losses with BaseLoss |
| new gan design | new GAN |
| new trainer | configuration change |
| new activation | add it to ops.py.  Be sure to include it in the huge conditional, search for prelu and you'll see |
| something else | Probably easiest to hack it in as a GAN, even if you have to build the graph up yourself in \#create. |

When implementing a new paper, feature-gate any breaking code so that old configurations still work\(and you have something to compare\).

## Custom research

HyperGAN is meant to support custom research as well. You can replace any part of the GAN with the json file, or just create a new GAN altogether.

Also, try to save refactoring until after it works.

## Using hyperGAN in an app

`hypergan build`

This builds a standard pytorch onxx file. You can then quantize and/or deploy the model.

