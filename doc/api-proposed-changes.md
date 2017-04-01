#API v2

# Better abstractions

## Exmaples
### discogan

```python
xa = hg.input.ImageDirectory("path/to/horses")
xa = hg.input.ImageDirectory("path/to/zebras")

options1 = {
  encoder:xa,
  input: xb
}
g1 = Generator(options1).build()

options2 = {
  encoder: xb,
  input: xa
}
g2 = Generator(options2).build()

gan=GAN(
  generator=[g1,g2],
  losses=[hg.losses.LeastSquares(), hg.losses.UnsupervisedAlignment(g1, g2)]
)

gan.train_for(10000)

```
This could also be a implemented in HG CLI as --align 0,1

# stackgan


```python
options1 = {
  encoder:standard_encoder,
  output_resolution: [64,64,3]
}
g1 = Generator(options1).build()

options2 = {
  encoder: g1,
  output_resolution: [256,256,3]
}
g2 = Generator(options2).build()

gan1=GAN(
  generator=g1
)

gan1.train_for(10000)

# optionally remove the discriminator of gan1 from graph?

gan2 = GAN(
  generator=g2
)

gan2.train_for(10000)

```

# Internal APi


One future goal of hypergan is platform agnostic behavior.  Mobile and desktop can be targeted with pbgraph, and treated as runtimes.  Both the discriminator and the generator could prove to be useful.  Javascript
should be considered a target platform as well.

This would involve parsing the config file and constructing a Backend that serves the config.

```python
backend.conv()
backend.activation()
backend.linear()
backend.layer_regularizer()
backend...
```
It could potentially be huge in scope, if you consider all the emerging research.  And we don't want to have to 'green light' functions for HG devs to use.  There is hopefully tooling we can leverage here.
