# Function-preserving FFN width diagnostic

Run only through `research/startup_tuning/replicated_ffn_screen.py`. Loading the
TOML alone initializes a different experiment. This is a causal diagnostic,
not a production recipe or an identified fix for the logos generator.

Starting from the unchanged adversarial CIFAR recipe, replicate hidden FFN
units 4 times at 8px and 16 times at 16px, producing four 4096-wide FFNs.
Copy up weights/biases and divide each replicated down-weight column by its
replication count. Down biases, all other generator tensors, the full critic,
prior, seeds, data and player rates remain exactly those of the source.
The initial generator function is preserved in exact arithmetic; the runner
checks complete-generator agreement in CPU float64 to 1e-9 tolerance.

This initialization deliberately duplicates units. It does not add independent
representational capacity or follow the HNDL's default independent initialization.
It isolates parameterization: vanilla Adam adds each duplicate down-weight
update, changing the function update despite starting from the same function.

`--compensated` divides each duplicated down weight's LR by its replication
factor and divides the corresponding up weights'/biases' Adam epsilon by that
factor. This recovers the source functional Adam trajectory in exact arithmetic
while duplicates stay tied. The epsilon correction accounts for the up-gradient
being divided by that factor. A float64 regression verifies initial equivalence,
eight compensated updates, and uncompensated first-update divergence.

Each screen lasts 512 updates with fixed/evolving-prior stage diagnostics,
original source rates and schedule, full restoration and pretrained-state audits.
CUDA/TF32 reduction-order differences can accumulate; functional equivalence in
exact arithmetic is not a promise of identical long GPU trajectories.

The comparison is against the unchanged CIFAR control in
`results/2026-09-22-healthy-control/`, using the same source seed, not a seed sweep.
