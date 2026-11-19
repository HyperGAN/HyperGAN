# Versioned API docs and assets on S3

The release workflow uses `master`, the existing HyperGAN release branch. It
publishes reviewed files to a new `releases/<version>/` prefix. Keep the API
docs in `hypergan-apidocs` in `us-east-1` and the assets in `hypergan` in
`us-west-1`. Each target uses a separate short-lived AWS role.

## Prepare a release

Build and qualify the package at the intended HyperGAN version. Check the
rendered API docs and links, or verify the intended downloadable assets and
their redistribution rights. Put the qualified public files in
`release-assets/<api-docs|assets>/<version>/payload/`. Add a neighboring
`manifest.json` that records the SHA-256, byte length, and MIME type of each
relative file. For API docs, include the `text/html` root `index.html`.

```json
{
  "files": {
    "index.html": {
      "sha256": "<sha256 of the reviewed file>",
      "bytes": 42,
      "content_type": "text/html"
    }
  }
}
```

Open the package PR against `develop`. Checks validate the complete payload.
Publish it through the normal reviewed promotion to `master`. The workflow
reserves the version with the exact manifest, uses S3 conditional uploads, and
reads every object back anonymously to compare all bytes and MIME types.
It publishes the docs index after other files, and `release.json` after all
files pass. A failed attempt can resume with the same manifest and bytes.
Use a fresh version for a correction. Consumers can keep using the preceding
version URL during recovery.

## Configure the production roles

- Use the GitHub OIDC provider in the 255bits AWS account. Create one role for
  `hypergan-s3-api-docs` and one for `hypergan-s3-assets`. Trust the exact
  GitHub OIDC subject for the named environment in `HyperGAN/HyperGAN` and
  the `sts.amazonaws.com` audience. Restrict each environment to `master`
  and set its `AWS_S3_RELEASE_ROLE_ARN` variable to its exact role ARN.
- Grant each role `s3:PutObject` on `releases/*` in its own bucket only, with
  `StringEquals` on `s3:if-none-match` set to `*`. The workflow requires neither
  delete permission nor object ACL changes. Add the public `s3:GetObject`
  policy for this release prefix using the bucket's existing access controls.
- Test the first reviewed version against both the new public `release.json`
  and expected route or downloadable files. Check the legacy URLs as part of
  activation. IAM and bucket policies can be reverted independently of the
  immutable published bytes.

[AWS conditional write policy examples](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes-enforce.html)
