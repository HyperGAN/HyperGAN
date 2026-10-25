# CI and PyPI releases

HyperGAN is published to PyPI as `hypergan`. The package version is declared in
`src/hypergan/__init__.py` (`__version__`) and read dynamically by
`pyproject.toml`.

`develop` holds upcoming work; `master` reflects the source published to PyPI.
Prepare the version, `changelog.md` (rename its **Unreleased** section to the
new version), and `README.md` on `develop`, verify CI, merge into `master`, and publish from the verified
`master` commit.

## CI

[Foundation CI](https://github.com/HyperGAN/HyperGAN/actions/workflows/ci.yml)
runs on pull requests and pushes to `develop` and `master`, manual dispatches,
and release workflow calls against an immutable tagged source SHA:

- **Lightweight (`ubuntu-latest`, `windows-latest`, `macos-latest`, Python 3.11–3.14):**
  Builds the wheel and source distribution, verifies a Torch-free and
  ParticleGAN-free environment, runs `tests/foundation` against the installed
  wheel outside the checkout, and verifies clean sdist rebuilds.
- **CPU reference (`ubuntu-latest`, Python 3.12):**
  Installs the tested CPU runtime (`[train,image,cifar,web,test]`) and runs
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
   `src/hypergan/__init__.py`, `changelog.md`, and `README.md`.
2. Merge `develop` into `master` and wait for `Foundation CI` (including heavy
   suites and `Build release distributions`) to pass on `master`.
3. Create and publish the GitHub release with a `v<version>` tag targeting the
   verified `master` commit:

   ```sh
   RELEASE_SHA=$(git rev-parse master)
   gh release create v2.0.0a1 --repo HyperGAN/HyperGAN \
     --prerelease \
     --target "$RELEASE_SHA" --title "HyperGAN 2.0.0a1" \
     --notes-file /tmp/hypergan-2.0.0a1-release-notes.md
   ```

   Publishing the release triggers `release.yml`, which validates that the tag
   matches `v<__version__>`, runs `ci.yml` against the immutable `source_sha`,
   downloads the verified `python-package-distributions` artifact, and publishes
   to PyPI via Trusted Publishing.
