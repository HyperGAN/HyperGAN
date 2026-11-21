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
runs on `develop` pushes, PRs between this repository's `develop` and `master`
branches, manual dispatches, and release workflow calls against an immutable
source SHA. `release.yml` owns the checks on `master` pushes. Feature PRs keep
the required Repository integrity check and can dispatch Foundation CI with
`gh workflow run ci.yml --ref <branch>`. Add `-f full=true` to include heavy
tests and release distribution verification.

- **Lightweight.** One job per version from Python 3.11 through 3.14 across
  Linux, Windows, and macOS. Builds the wheel and source distribution, verifies a Torch-free and
  ParticleGAN-free environment, runs `tests/foundation` against the installed
  wheel outside the checkout, and verifies clean sdist rebuilds.
- **CPU reference (`ubuntu-latest`, Python 3.12).**
  Installs the tested CPU runtime (`[train,image,cifar,fid,web,test]`) and runs
  `tests/reference`, `tests/unit`, `tests/playbooks`, and `tests/e2e` outside the
  checkout across four file shards.
- **Heavy tests.** Lightweight and CPU reference jobs reuse their installed
  wheels for the `heavy` selection on `master` checks, release calls and full
  manual runs.
- **Shared reducer & Viewer API (`ubuntu-latest`, `windows-latest`, `macos-latest`):**
  Verify the Foldlite DB shared WASM reducer (`tests/reducers`,
  `tests/e2e_integration`), Chromium Playwright live browser continuation
  (`tests/browser`), reproducible offline frontend bundle checks (`frontend`),
  and Web Observatory API (`tests/web`).
- **Distributions (`ubuntu-latest`, Python 3.12).**
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
2. Merge `develop` into `master`. `release.yml` runs Foundation CI and the
   distribution build against the exact merge commit. If PyPI has no files for
   the version, the workflow extracts the matching notes from `CHANGELOG.md`
   and proceeds to publish after the checks pass.
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

## GitBook preview

The [GitBook preview workflow](https://github.com/HyperGAN/HyperGAN/blob/69116e009d81ebefa5a8253784650f83f809d77b/.github/workflows/gitbook-preview.yml)
builds the tracked `docs/` pages and assets. It pins links outside the book to
the build's GitHub commit. Once the workflow reaches `master`, documentation
merges advance `gitbook-preview` with the generated book and its
`SOURCE_REVISION`. Use the GitHub source for edits to the generated book.

1. Open a separate preview space as a GitBook owner. Connect the GitBook
   GitHub app to `HyperGAN/HyperGAN` and select `gitbook-preview`, with
   GitHub to GitBook as the initial sync direction.
2. Check the table of contents, images, and the installation and CLI links.
   Confirm the current preview matches `SOURCE_REVISION`, then verify the
   next reviewed docs merge updates the preview through Git Sync.
3. Keep the legacy space and its URLs available. Before publishing the new
   preview as a site version, verify its navigation and the legacy routes.
   Revert the relevant source changes on `master` to restore preview content.

[Git Sync setup](https://gitbook.com/docs/guides/editing-and-publishing-documentation/import-or-migrate-your-content-to-gitbook-with-git-sync)
