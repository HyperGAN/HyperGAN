# Four early FFNs narrowed to 1024

The frozen pretrained ResNet18 control with only the four stage8/stage16 FFN
hidden widths reduced from 4096 to 1024. Their down-projection bias initializer
bounds change from +/-0.015625 to +/-0.03125 with the new fan-in. Channel widths,
upsampling, attention, later FFNs, data, optimizer and prior are unchanged.

Run the paired research screen using `research/startup_tuning/architecture_screen.py`
with `--case source` and then `--case narrow`, the same output root, and an
explicit free GPU. Each performs 32 native updates with stage measurements.
The source replay adds stage measurements absent from the previous screen.
The narrow case copies unchanged tensors from the original initialization,
takes the first 1024 hidden units of each FFN, and rescales down weights/biases
by two to preserve their initialization distributions. This is a paired
architecture ablation, not a seed experiment. Ordinary training from this TOML
alone will not reproduce this explicit cross-architecture weight alignment.

Result: the 32-update paired screen failed, with 93.67% saturation
and 10.81% initial diversity retained.
See [evidence](../../results/2026-09-22-generator-architecture/README.md).
