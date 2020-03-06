# Creating an image dataset

To build a new network you need a dataset. Your data should be structured like:

```text
  [folder]/[directory]/*.png
```

### Creating a Dataset

Datasets in HyperGAN are meant to be simple to create. Just use a folder of images.

```text
 [folder]/*.png
```

For jpg\(pass `-f jpg`\)

### Downloadable datasets

* Loose images of any kind can be used
* CelebA aligned faces [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* MS Coco [http://mscoco.org/](http://mscoco.org/)
* ImageNet [http://image-net.org/](http://image-net.org/)
* youtube-dl \(see [examples/Readme.md](../examples-1/2d.md)\)

### Cleaning up data

To convert and resize your data for processing, you can use imagemagick

```text
for i in *.jpg; do convert $i  -resize "300x256" -gravity north   -extent 256x256 -format png -crop 256x256+0+0 +repage $i-256x256.png;done
```

