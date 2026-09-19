# CLI guide

The cli is available with a `pip install hypergan`

## Using `virtualenv`:

If you use virtualenv:

```bash
  virtualenv --system-site-packages -p python3 hypergan
  source hypergan/bin/activate
```

```bash
 hypergan -h
```

### Training

```bash
  # Train a 32x32 gan with batch size 32 on a folder of pngs
  hypergan train [folder] -s 32x32x3 -b 32 --config [name]
```

### Sampling

```bash
  hypergan sample [folder] -s 32x32x3 -b 32 --config [name] --sampler batch_walk --sample_every 5 --save_samples
```


