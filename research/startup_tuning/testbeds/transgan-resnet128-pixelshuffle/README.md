# Pixel shuffle from the first transition

Compared with the original pretrained-ResNet recipe, replace the 8->16 bicubic
transition with pixel shuffle and propagate the resulting channel reduction.
Stage channels are 1024/256/64/16/4, FFN widths 4096/1024/256/64/16, with the
same two blocks and four attention heads per stage. Initial stage8 is unchanged.
This tests the channel schedule as a whole: it does not isolate interpolation
from width, capacity, head dimensions or initialization changes. It is separate
from the four-FFN narrowing ablation, not stacked on top of it.

The paired screen is `research/startup_tuning/architecture_screen.py --case
pixelshuffle --device cuda:0 --output-root OUTPUT`, with CUDA_VISIBLE_DEVICES
set explicitly to idle GPU 0. Matching tensors, including the whole critic,
are copied from the source initialization. Reduced tensors take leading slices;
Linear tensors are rescaled for the new fan-in (Xavier for the RGB weight),
and position tables keep their original distribution. The same seed, data,
prior and original optimizer settings apply. This screen declares 32 updates.
