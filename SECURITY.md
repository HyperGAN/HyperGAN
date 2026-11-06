# Security Policy

## Reporting a Vulnerability

Please do not report security problems in public issues, discussions or pull requests.

Report them privately through GitHub's private vulnerability reporting: open the repository's **Security** tab, choose **Report a vulnerability**, and fill in the advisory form (direct link: <https://github.com/HyperGAN/HyperGAN/security/advisories/new>). Include the HyperGAN version or commit, the command or API call, and the smallest reproduction you have.

The maintainers will acknowledge the report, work on a fix in a private advisory, and credit you when the advisory is published unless you ask otherwise.

## Supported Versions

Fixes land on `develop` and ship in the next release. Only the latest release on PyPI receives security fixes.

## Scope Notes

The local viewer (`hypergan view`, and the viewer `hypergan train` starts) serves run data and run controls over HTTP. Treat it as a local development tool and bind it to `127.0.0.1` (`--host 127.0.0.1`) on any machine other hosts can reach. See [Local browser and streaming API](docs/local-web.md).
