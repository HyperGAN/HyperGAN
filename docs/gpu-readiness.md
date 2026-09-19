# Local two-GPU NCCL readiness

The two local RTX A6000s pass a small NCCL communication and gradient diagnostic. This does not qualify HyperGAN's full replicated GAN update, checkpoint recovery or cluster execution on CUDA. The [readiness report](../reports/gpu-readiness-2026-09-19.md) records exact hardware, results, failures and remaining implementation work.

The standalone diagnostic lives in [tests/cuda/nccl_smoke.py](../tests/cuda/nccl_smoke.py). It requires a CUDA-enabled PyTorch interpreter, two visible CUDA devices and `nvidia-smi`. It imports no HyperGAN training implementation and does not download data. Run from a checkout or the source distribution test inventory; the tool is not a public training command or an execution profile. The explicit `python -m pytest tests/cuda -q` hardware gate includes this check and requires two GPUs; it never skips unavailable hardware. Its NCCL wrapper uses a 35-second per-job supervisor deadline plus cleanup grace.

```sh
/path/to/cuda/python tests/cuda/nccl_smoke.py \
  --devices 0,1 --output /tmp/nccl-check/receipt.json
```

The receipt includes GPU UUIDs, CUDA/NCCL versions, peer access, collective results, differentiable-gather derivatives, averaged-gradient Adam parity and managed child exit status. Each output path and its sibling `.artifacts` directory must be new; per-rank logs remain available for diagnosis. Device indices are relative to `CUDA_VISIBLE_DEVICES` if set. The script uses small tensors and one linear Adam update, and does not stop other GPU processes.

An optional missing-peer test deliberately prevents one rank from entering a collective. On the tested Torch build, NCCL diagnostic collection delayed process exit beyond the collective timeout. The following tested configuration bounds that diagnostic wait, while the parent still enforces its own independent deadline:

```sh
TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=1000 \
/path/to/cuda/python tests/cuda/nccl_smoke.py \
  --devices 0,1 --fault-timeout --collective-timeout 5 --timeout 45 \
  --output /tmp/nccl-fault-check/receipt.json
```

A fault-test pass requires both ranks to have entered the intended fault, a logged NCCL timeout, a rank failure before the outer deadline, and successful reaping. A supervisor-forced stop remains explicit in the receipt and returns failure for this stronger prompt-abort check. The initial default configuration failed that check while successfully reaping both ranks at the outer deadline; do not interpret a five-second collective timeout as a five-second process-exit guarantee.

The diagnostic parent terminates, kills if needed, and joins only its own ranks. Its cleanup can add four seconds of join grace beyond the job deadline. It has no independent coordinator-death broker: do not use it as a training launcher or kill the parent with SIGKILL. Production GPU integration must retain the existing broker's coordinator-death protection, strict complete-update boundaries and parent checkpoint publication authority.
