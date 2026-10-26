# Contributing

The next HyperGAN release integrates on `develop`; `master` holds releases. Create a short-lived branch from current `develop` and target your pull request at `develop`.

## Set up

Use Python 3.12 or newer for the training runtime (lightweight commands support 3.11–3.14):

```sh
git clone https://github.com/HyperGAN/HyperGAN.git
cd HyperGAN
python -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
python -m pip install -e '.[dev,train,image,cifar,web]'
```

## Run the tests

```sh
python -m pytest            # fast suite: CPU and lightweight tests, heavy tests deselected
python -m pytest -m heavy   # tests that start subprocesses, worker services or multi-rank jobs
```

[tests/README.md](tests/README.md) describes every test directory, the markers, and the tiers that need extra dependencies or GPUs. CI runs the same selections against the built wheel; see [CI and PyPI releases](docs/releasing.md).

## Open a pull request

Keep PRs focused on a working user outcome or a bounded correctness improvement. Describe the behavior changed, how you validated it, and what remains limited. Every bug fix needs a regression test that fails without the fix. Passing required CI and coordinator review are required before integration. If you run several agents in parallel, give each its own worktree outside the repository; agents also follow [AGENTS.md](AGENTS.md).

## Design rules

Recipe configuration supports custom Python components. Document each component's inputs, outputs, parameters and numerical assumptions. Unknown configurations should receive an honest qualification warning; invalid inputs and unsupported capabilities must fail clearly. Do not silently ignore parameters, replace components, or imply a successful run demonstrates image quality or distributed correctness.

Research enters the qualified catalog only after reproducible evidence and product validation. Historical implementations remain accessible through archive tags; avoid reintroducing the legacy catalog or its dependencies wholesale.

Lightweight commands must work without the training dependencies. CPU tests precede local GPU qualification; paid cluster checks are deliberately scheduled and are not automatic PR jobs.

Guides for users and developers are indexed in [docs/README.md](docs/README.md); terms are defined in [GLOSSARY.md](GLOSSARY.md). Dated design notes and experiment reports live in [reports/](reports/), with older ones in `reports/archive/`.
