# HyperGAN 2.0.0 release readiness

## Release outcome on November 18, 2026

**HyperGAN 2.0.0 shipped** from `master` commit `0f256ba8c2479b4a4bd365f3ca4d630c80246af0`. All 34 jobs passed on the release PR and again on the merged commit. The annotated tag points to that commit, all 37 tag-workflow jobs passed, and GitHub marks the stable release **Latest**. The a4 publish also completed and its PyPI and GitHub files matched. [Release PR](https://github.com/HyperGAN/HyperGAN/pull/434) · [Master CI](https://github.com/HyperGAN/HyperGAN/actions/runs/44251301926) · [Tagged publish run](https://github.com/HyperGAN/HyperGAN/actions/runs/44255259821)

Both public distributions contain version `2.0.0`, the stable classifier, and the tagged source revision with `dirty: false`. The files downloaded separately from PyPI and GitHub were byte-identical. [PyPI files](https://pypi.org/project/hypergan/2.0.0/#files) · [GitHub release](https://github.com/HyperGAN/HyperGAN/releases/tag/v2.0.0)

| Distribution | SHA-256 on both services |
| --- | --- |
| `hypergan-2.0.0-py3-none-any.whl` | `9d5bf9901a14d8000f3fd5658ee0e2670641f8f69721bcdf7e2cec7e126bb34d` |
| `hypergan-2.0.0.tar.gz` | `177dfbf5dd2e342e2e7b8ef9b729fa411e506b075b622aed0e44d295ac2e1d4f` |

- **Public installs.** Fresh Python 3.12 environments installed `hypergan[all]==2.0.0` from PyPI with Torch 2.14.0 and TorchVision 0.29.0 from the CPU and CUDA 13.0 indexes. Both passed `pip check`, reported `hypergan 2.0.0`, and sampled a Gaussian bundle on their respective devices. The public CPU wheel also scaffolded, validated, trained and resumed a run to step 5.
- **Serving and upgrade.** Each documented format loaded in a separate serving profile. ONNX Runtime's four-row output matched the Python bundle within 1.5e-8. The PT2 example accepted a single row and enforced the batch guard. SafeTensors exposed weights and configuration. A PyPI a2 run resumed in its original environment and its exported bundle reproduced four reference samples exactly under GA. [Serving instructions](../docs/deployment.md) · [Upgrade path](../docs/migration.md)
- **Hardware.** The GA candidate saved and resumed the packaged CIFAR tiny transformer to a durable step-3 checkpoint on two RTX A6000s (run `95bb5366`). On the exact master wheel with Torch 2.14.0+cu130, 65 CUDA cases passed and two skipped. The CIFAR sharding case then passed with its dataset path pointed at the local cache. The other case requires an unavailable archived training run. Multi-host execution retains its experimental status. [Hardware scope](../docs/multigpu-readiness.md)

## Assessment before release on November 17, 2026

**Hold the GA tag.** The candidate builds, trains, resumes its own checkpoints, and passes a two-GPU smoke. Before tagging, correct the deployment and upgrade instructions, put the GA version and notes into the distributions, and pass the release CI on `master`. The existing CIFAR runs already provide evidence for the two packaged architectures.

**Snapshot at 22:31 MST, November 17, 2026.** `origin/develop` and `origin/master` had reached `54e7387bd48fbfa89ca05a953ed9217233ed9a0a`, tagged `v2.0.0a4`. Its Git tree is identical to the tested develop commit `71054b88d3ced227a46e9b4b18d7769e72a9edc1`. Develop CI passed on that source. The a4 release workflow had started, and its PyPI files were still absent. The public index's newest 2.x files were the **a2** wheel and sdist. GitHub's a3 prerelease has no attached distributions, and the a3 publish job was skipped after CI failed. Cover the changes since a2 for installed users, even if a4 publishes before GA. [Audited source](https://github.com/HyperGAN/HyperGAN/tree/54e7387bd48fbfa89ca05a953ed9217233ed9a0a) · [a4 publish run](https://github.com/HyperGAN/HyperGAN/actions/runs/44244816880) · [a3 publish run](https://github.com/HyperGAN/HyperGAN/actions/runs/42767113002) · [PyPI file index](https://pypi.org/simple/hypergan/)

## Work before tagging

1. **Pass the release-only test suites and build the distributions.** The Windows registry test that failed across Python 3.11–3.14 now passes on all four versions. Foundation CI passed on `71054b88` with the heavy suites and release distribution build skipped. The a4 tag provides the first opportunity since a3 to check the fixes through publication.

   Verify the a4 heavy jobs, distribution provenance, and delivery to PyPI and GitHub. For GA, require the full matrix on the actual `2.0.0` commit and verify the resulting artifacts. [Previous Windows failure](https://github.com/HyperGAN/HyperGAN/actions/runs/44242326476/job/134592938318) · [Passing develop CI](https://github.com/HyperGAN/HyperGAN/actions/runs/44243705797) · [a4 master CI](https://github.com/HyperGAN/HyperGAN/actions/runs/44244815217)

2. **Publish a working serving recipe for each export format.** The guide promises Torch-only serving for `.pt`. In a fresh environment with HyperGAN, Torch and NumPy, `hg.load_bundle("toy.pt")` raised `ModuleNotFoundError: hndl` rebuilding the packaged Gaussian generator. Installing HNDL 0.11.0 made the same bundle sample successfully with ParticleGAN absent. The default `.pt2` export accepted a batch of one and rejected four with `latent.size()[0] == 1`.

   Give each format a tested install, input shape, and load command. For HNDL bundles, document the HNDL dependency or change serialization. Document the PT2 batch constraint or export a dynamic batch. Describe SafeTensors as weights and metadata and identify the consumer that constructs the graph. Exercise the documented serving environment separately from `[train]`. [Deployment guide](../docs/deployment.md) · [Export and load implementation](../src/hypergan/artifacts.py)

3. **Give a2 users an upgrade path.** A CPU `gaussian-grid` checkpoint written at step 1 by PyPI a2 with ParticleGAN 0.9.7 was rejected by the candidate with 0.10.1. After matching NumPy, the error named only `particlegan: saved "0.9.7", current "0.10.1"`. The original a2 environment resumed that checkpoint to step 5. A bundle exported there produced identical outputs on the candidate for the same four draws. The migration guide mainly addresses 1.x.

   Add a 2.x-alpha upgrade section. Tell users with active runs which environment to retain for exact resume, how to export before upgrading, and how to start a fresh GA run. Claim cross-version continuation only for a version pair and checkpoint layout that pass a restore test. Preserve the existing runtime identity check. [Migration guide](../docs/migration.md) · [Recovery contract](../docs/recovery.md)

4. **Prepare the GA distributions.** The version in `src/hypergan/__init__.py`, the Alpha classifier, the README's two a4 install pins, and the dated a4 changelog heading describe an alpha. Tag validation requires a matching version and nonempty release notes.

   Set `__version__ = "2.0.0"`, select the stable classifier, update the install and tagging examples, and add `## 2.0.0 — YYYY-MM-DD` with the actual release date. Summarize the user-visible changes since a2 and retain the top `Unreleased` heading and historical notes. Check the built wheel's metadata and README. [Package metadata](../pyproject.toml) · [Changelog](../CHANGELOG.md) · [Tag validator](../.github/workflows/release.yml)

## Recipe evidence and support claims

These measurements come from existing run manifests and evaluation summaries on the audit host. The run IDs and source revisions identify the evidence independently of the local directory layout. All four rows used two RTX A6000s.

| Recipe and run ID | HyperGAN / recorded ParticleGAN | Measured result |
| --- | --- | --- |
| CIFAR tiny transformer + ResNet-18, `117ab950` | `0b6d6cfb` / 0.9.7 BISPF | Best FID-50k **14.56** at step 15,000 of 25,000 observed. This is the figure in the CIFAR guide. |
| Same architecture, `3e046470` | `66ba975d` / 0.10.0 EMPF | Best FID-50k **15.16** at 12,500. Latest **16.06** at 25,000. |
| CIFAR scratch TransGAN, `f2cce6c4` | `66ba975d` / 0.10.0 EMPF | Best FID-50k **51.30** at 22,500. Reached 25,000 steps. |
| 256px DINOv3 logo colorization, `d1eef023` | `c02dc4bf` / 0.9.7 BISPF | Held-out FID-10k **6.37** at 2,500. Latest chroma **0.01353** at 4,000, structure **2.61e-7** at 4,012, diversity **0.09285** at 4,019, random spread **0.9940** at 4,027. Operator interrupt at 4,447. |

Label the CIFAR figures by preset and version. Both EMPF manifests record ParticleGAN commit `af0764b1`. With the public 0.10.1 wheel and batch 256, both packaged models resolve to the same serialized ParticleGAN recipe fields, components, optimizer, and training settings as those manifests. That supports keeping the measurements with their original provenance. The changelog currently attributes them to the 0.10.1 upgrade. [CIFAR guide](../docs/cifar-recipe.md) · [ParticleGAN integration](../docs/particlegan.md) · [a4 notes](../CHANGELOG.md)

Choose how to ship 256px colorization. The wheel contains five named models, including `colorization-128`. The 256px DINOv3 recipe lives in `examples/archive/` and requires a user-supplied dataset manifest, digest, and backbone weights. The README calls both resolutions packaged production recipes. Either document the 256px source workflow and its measured BISPF scope, or include a portable model and verify its setup from the wheel. The existing five metrics above address the outstanding local colorization review. [Packaged models](../src/hypergan/models) · [256px example](../examples/archive/logos-colorization-dinov3-256.toml) · [README](../README.md)

Align the hardware claims in the README and readiness guide. The published 2.20× speedup names a particular two-A6000, batch-256 benchmark. The two-host Modal smoke used an older dependency stack. For GA, identify the tested topology and recovery conditions beside each claim. The ROCm guide already labels AMD hardware qualification as experimental. [Scaling measurements](../docs/scaling-benchmark.md) · [Multi-GPU readiness](../docs/multigpu-readiness.md) · [ROCm scope](../docs/rocm.md)

## Verification performed

- **Distributions and CLI.** Built the sdist and its wheel in a clean worktree. Both passed `twine check --strict` and contained the tested commit `71054b88` with `dirty: false`. On Linux/Python 3.12.14 the installed wheel passed **1,323** fast foundation tests (3 optional skips, 12 heavy deselections) and **12/12** heavy foundation tests. A core-only install can scaffold and validate the Gaussian project. Fresh image projects using `defaults = "particlegan"` require the train extra for preset metadata during validation. State that dependency in the source-install quickstart. [Test tiers](../tests/README.md) · [Install instructions](../README.md)
- **CPU runtime and export.** Installed the wheel with `[all,test]`, CPU Torch 2.14.0, TorchVision 0.29.0, HNDL 0.11.0 and public ParticleGAN 0.10.1. All **74** selected export, released-recipe and ParticleGAN integration/restoration tests passed. A fixed Gaussian fixture stopped after step 1, resumed to step 5 and sampled. Its ONNX export passed the ONNX checker. ONNX Runtime accepted four inputs in a batch and agreed with the Python bundle to **1.5e-8** maximum absolute error on that input. All **9** heavy CPU cases that failed on the a3 publish run also passed in a targeted rerun on the audited commit. The initial lifecycle smoke used `f834b14f`. The only diff to the audited commit is the Windows test fixture.
- **CUDA.** On the audited wheel, `hypergan qualify --suite cuda_replicated_nccl --cuda-world-size 2` passed two updates with consensus on both RTX A6000s (Python 3.14.7, Torch 2.14.1+cu130, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`). Using public ParticleGAN 0.10.1 and HNDL 0.11.0, the packaged `cifar10-tiny-transformer` also ran offline on both GPUs at global batch 256 with its original seed. It saved at step 2, resumed in fresh workers, and wrote a durable step-3 checkpoint (run `248d8fab`, preset `empf`). The full CUDA suite and a fresh multi-host run remain unverified on this commit. [Qualification guide](../docs/multigpu-readiness.md)

## Release sequence

1. Land the fixes, support decisions and GA notes on `develop`. Run the documented install, resume and serving examples against the GA wheel in clean environments.
2. Open the release PR to `master`. Require repository integrity and all release CI jobs, then verify the merged `master` SHA. Confirm its wheel and sdist both carry version `2.0.0` and that clean SHA before tagging it `v2.0.0`. The tag validator checks the version and changelog. Verify the tagged commit belongs to `master` as a separate step. [Release procedure](../docs/releasing.md) · [CI job definitions](../.github/workflows/ci.yml)
3. After Trusted Publishing, verify both files on PyPI, install `hypergan[all]==2.0.0` with the documented Torch wheels, and confirm the GitHub release is **Latest** with the same two SHA-256 digests. This last comparison matters on retries because the PyPI step skips existing files and the GitHub upload can replace assets. [Release workflow](../.github/workflows/release.yml)
