"""Validate and append a reviewed HyperGAN artifact release to its S3 bucket."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

import boto3
from botocore.exceptions import ClientError


SOURCE = "HyperGAN/HyperGAN"
OWNER = "563240432511"
TARGETS = {"api-docs": ("hypergan-apidocs", "us-east-1"), "assets": ("hypergan", "us-west-1")}
RELEASES = Path("release-assets")
SHA256 = re.compile(r"[0-9a-f]{64}")
OID = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[.-][A-Za-z0-9][A-Za-z0-9.-]*)?")
# Keep paths portable between Git and the public S3 endpoints.
PATH_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
CONTENT_TYPE = re.compile(r"[a-z0-9.+-]+/[a-z0-9.+-]+")


def manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.keys() != {"files"} or not isinstance(data["files"], dict) or not data["files"]:
        raise ValueError(f"Expected a nonempty files object in {path}")
    target, version = path.parent.parent.name, path.parent.name
    if target not in TARGETS or not VERSION.fullmatch(version):
        raise ValueError(f"Invalid release target or version: {path}")
    root = (path.parent / "payload").resolve(strict=True)
    if root != path.parent.absolute() / "payload" or path.is_symlink():
        raise ValueError("Use regular files in the release directory")
    for name, entry in data["files"].items():
        parts = PurePosixPath(name)
        if (not parts.parts or str(parts) != name or any(not PATH_PART.fullmatch(p) for p in parts.parts)
                or name in {"release.json", "release-manifest.json"}):
            raise ValueError(f"Unsafe or reserved object path: {name!r}")
        source = root / name
        if source.resolve(strict=True) != source or not source.is_file():
            raise ValueError(f"Expected a regular payload file: {name}")
        if (not isinstance(entry, dict) or entry.keys() != {"sha256", "bytes", "content_type"}
                or not isinstance(entry["sha256"], str) or not SHA256.fullmatch(entry["sha256"])
                or type(entry["bytes"]) is not int or not 0 <= entry["bytes"] <= 5 * 1024**3
                or not isinstance(entry["content_type"], str) or not CONTENT_TYPE.fullmatch(entry["content_type"])):
            raise ValueError(f"Invalid object metadata: {name}")
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != entry["sha256"] or source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Payload does not match its reviewed checksum and size: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != data["files"].keys():
        raise ValueError("The manifest must cover the complete payload directory")
    if target == "api-docs" and data["files"].get("index.html", {}).get("content_type") != "text/html":
        raise ValueError("The API documentation requires a text/html index.html")
    return target, version, root, data["files"]


def public_get(bucket, region, key):
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key)}"
    return urlopen(url, timeout=30)


def write_object(client, bucket, region, key, handle, sha, size, content_type):
    try:
        client.put_object(Bucket=bucket, Key=key, Body=handle, ContentType=content_type,
                          ContentLength=size, IfNoneMatch="*", ExpectedBucketOwner=OWNER,
                          ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode("ascii"))
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
            raise
    with public_get(bucket, region, key) as response:
        digest = hashlib.file_digest(response, "sha256").hexdigest()
        if (digest != sha or int(response.headers["Content-Length"]) != size
                or response.headers.get_content_type() != content_type):
            raise ValueError(f"Public object conflicts with the release manifest: {key}")


def publish(client, target, version, root, files):
    bucket, region = TARGETS[target]
    prefix = f"releases/{version}"
    canonical = json.dumps({"target": target, "version": version, "files": files},
                           sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"

    def metadata_file(name, payload):
        write_object(client, bucket, region, f"{prefix}/{name}", payload,
                     hashlib.sha256(payload).hexdigest(), len(payload), "application/json")

    # Claim the version using the exact reviewed manifest before sending payloads.
    metadata_file("release-manifest.json", canonical)
    for name in sorted(files, key=lambda name: (name == "index.html", name)):
        entry = files[name]
        with (root / name).open("rb") as handle:
            write_object(client, bucket, region, f"{prefix}/{name}", handle,
                         entry["sha256"], entry["bytes"], entry["content_type"])
    receipt = json.dumps({"manifest_sha256": hashlib.sha256(canonical).hexdigest()}, sort_keys=True).encode() + b"\n"
    metadata_file("release.json", receipt)
    return f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}/release.json"


def changed_releases(before):
    if not OID.fullmatch(before):
        raise ValueError("Expected the previous Git commit SHA")
    names = subprocess.check_output(["git", "diff", "--no-renames", "--name-only", "-z", before, "HEAD", "--", RELEASES])
    paths = set()
    for name in names.decode().split("\0"):
        parts = PurePosixPath(name).parts
        if not parts:
            continue
        if len(parts) < 4:
            raise ValueError(f"Put each package in release-assets/<target>/<version>: {name}")
        path = Path(*parts[:3]) / "manifest.json"
        if path.parent.is_dir():
            paths.add(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    releases = [manifest(path) for path in changed_releases(args.before)]
    if not args.publish:
        targets = {target for target, _, _, _ in releases}
        matrix = {"include": [{"target": target, "region": TARGETS[target][1]} for target in sorted(targets)]}
        print(json.dumps(matrix, separators=(",", ":")))
        return
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (args.target is None or os.getenv("GITHUB_REPOSITORY") != SOURCE or os.getenv("GITHUB_REF") != "refs/heads/master"
            or os.getenv("GITHUB_EVENT_NAME") != "push" or os.getenv("GITHUB_SHA") != sha):
        raise ValueError("Publication requires the canonical master push and its exact checkout")
    client = boto3.client("s3", region_name=TARGETS[args.target][1])
    for release in releases:
        if release[0] == args.target:
            print(f"{sha} published {publish(client, *release)}", flush=True)


if __name__ == "__main__":
    main()
