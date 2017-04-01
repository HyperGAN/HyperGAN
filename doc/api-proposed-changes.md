#API v2

# Better abstractions

# example scripts

# discogan

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

# stackgan


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


# Component API

Is there a standard way that would let us integrate with things like Keras?


# Internal APi


One future goal of hypergan is platform agnostic behavior.  Mobile and desktop can be targeted with pbgraph, and treated as runtimes.  Both the discriminator and the generator could prove to be useful.  Javascript
should be considered a target platform as well.

This would involve parsing the config file and constructing a Backend that serves the config.

backend.conv()
backend.activation()
backend.linear()
backend...

It could potentially be huge in scope, if you consider all the emerging research.  And we don't want to have to 'green light' functions for HG devs to use.  There is hopefully tooling we can leverage here.
