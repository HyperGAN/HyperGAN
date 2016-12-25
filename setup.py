from distutils.core import setup
setup(
  name = 'hyperchamber',
  packages = ['hyperchamber'], # this must be the same as the name above
  version = '0.1',
  description = 'A tool for searching hyperparameters in models.',
  author = 'Mikkel Garcia',
  author_email = 'mikkel@255bits.com',
  url = 'https://github.com/255BITS/hyperchamber', # use the URL to the github repo
  download_url = 'https://github.com/255BITS/hyperchamber/tarball/0.1', # I'll explain this in a second
  keywords = ['hyperparameter','neural network tuning','random parameter search', 'grid search'], # arbitrary keywords
  scripts = ['bin/hypergan'],
  classifiers = [],
)
