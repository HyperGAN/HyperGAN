# CI and PyPI releases

HyperGAN is published to PyPI as `hypergan`. The package version is declared in
`src/hypergan/__init__.py` (`__version__`) and read dynamically by
`pyproject.toml`.

`develop` holds upcoming work; `master` reflects the source published to PyPI.
Prepare the version, `CHANGELOG.md` (move **Unreleased** entries under a new
`## <version> — YYYY-MM-DD` heading while keeping an empty `## Unreleased`
section at the top, as verified by `test_repository_consistency.py`), and
`README.md` on `develop`, verify CI, then merge into `master` to publish the version
from that exact commit.

## CI

[Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/workflows/ci.yml)
runs on pull requests and pushes to `develop` and `master`, manual dispatches,
and release workflow calls against an immutable source SHA:

- **Lightweight (`ubuntu-latest`, `windows-latest`, `macos-latest`, Python 3.11–3.14):**
  Builds the wheel and source distribution, verifies a Torch-free and
  ParticleGAN-free environment, runs `tests/foundation` against the installed
  wheel outside the checkout, and verifies clean sdist rebuilds.
- **CPU reference (`ubuntu-latest`, Python 3.12):**
  Installs the tested CPU runtime (`[train,image,cifar,fid,web,test]`) and runs
  `tests/reference`, `tests/unit`, `tests/playbooks`, and `tests/e2e` outside the
  checkout.
- **Heavy lightweight & Heavy CPU reference:**
  Run the subprocess, worker-service, and multi-rank tests marked `heavy` on
  pushes/PRs targeting `master` and on every release workflow call.
- **Shared reducer & Viewer API (`ubuntu-latest`, `windows-latest`, `macos-latest`):**
  Verify the Foldlite DB shared WASM reducer (`tests/reducers`,
  `tests/e2e_integration`), Chromium Playwright live browser continuation
  (`tests/browser`), reproducible offline frontend bundle checks (`frontend`),
  and Web Observatory API (`tests/web`).
- **Distributions (`ubuntu-latest`, Python 3.12):**
  After `Foundation CI` passes, builds the wheel and source archive from the
  exact tested commit, verifies that no research/docs/worktree artifacts leak
  into distributions, verifies that `_build_provenance.json` stamps the exact
  commit with `dirty: false`, runs `python -m twine check --strict dist/*`, and
  uploads the `python-package-distributions` artifact.

## Trusted Publishing

PyPI publication uses GitHub OIDC Trusted Publishing rather than a stored API
token. Configure the PyPI Trusted Publisher at
`https://pypi.org/manage/project/hypergan/settings/publishing/` (or
`https://pypi.org/manage/account/publishing/`) with:

| Field | Value |
| --- | --- |
| PyPI project | `hypergan` |
| GitHub owner/repository | `HyperGAN/HyperGAN` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

## Publishing

1. Merge reviewed changes into `develop`, including `__version__` in
   `src/hypergan/__init__.py`, `CHANGELOG.md` (`## <version> — YYYY-MM-DD`), and
   `README.md`.
2. Merge `develop` into `master`. `release.yml` checks whether PyPI already
   has files for the version. If the version is unpublished, it extracts the
   matching notes from `CHANGELOG.md` and runs Foundation CI and the
   distribution build against the exact merge commit.
3. Monitor the release run. It creates or verifies `v<version>` at the tested
   commit, publishes the tested wheel and sdist to PyPI, and attaches those
   same files and the notes to the GitHub Release. If an upload or GitHub
   Release sync needs a retry after the tag exists, use
   `gh workflow run release.yml -f tag=v<version>`. Pushing a matching tag
   also triggers the existing release path.

4. Compare the wheel and source archive SHA-256 digests on PyPI and GitHub.
   Install the public wheel into a fresh environment with the README's tested
   Torch wheels, check `hypergan --version`, and confirm that GitHub's Latest
   release points to the stable tag.
