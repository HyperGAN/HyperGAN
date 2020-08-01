# Uniform Distribution

## Uniform Distribution

| attribute | description | type |
| :---: | :---: | :---: |
| z | The dimensions of random uniform noise inputs | int &gt; 0 |
| min | Lower bound of the random uniform noise | int |
| max | Upper bound of the random uniform noise | int &gt; min |
| projections | See more about projections below | \[f\(config, gan, net\):net, ...\] |
| modes | If using modes, the number of modes to have per dimension | int &gt; 0 |

## Projections

This distribution takes a random uniform value and outputs it as many possible types. The primary idea is that you are able to query Z as a random uniform distribution, even if the gan is using a spherical representation.

Some projection types are listed below.

**"identity" projection**

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-linear.png)

**"sphere" projection**

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-sphere.png)

**"gaussian" projection**

![](https://raw.githubusercontent.com/255BITS/HyperGAN/master/doc/encoder-linear-gaussian.png)

**"modal" projection**

One of many

**"binary" projection**

On/Off

## Category Distribution

Uses categorical prior to choose 'one-of-many' options.

