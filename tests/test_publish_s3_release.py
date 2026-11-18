from email.message import Message
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError

source = Path(__file__).resolve().parents[1] / "scripts" / "publish_s3_release.py"
spec = importlib.util.spec_from_file_location("publish_s3_release", source)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Bucket:
    def __init__(self):
        self.objects = {}
        self.requests = []

    def put_object(self, **request):
        key = request["Key"]
        self.requests.append(request)
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        body = request["Body"]
        payload = body if isinstance(body, bytes) else body.read()
        self.objects[key] = payload, request["ContentType"]

    def open(self, bucket, region, key):
        payload, mime = self.objects[key]
        handle = BytesIO(payload)
        handle.headers = Message()
        handle.headers["Content-Type"] = mime
        handle.headers["Content-Length"] = str(len(payload))
        return handle


class PublishS3Tests(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name) / "api-docs" / "2.0.1"
        root = self.directory / "payload"
        root.mkdir(parents=True)
        self.metadata = {"files": {}}
        self.path = self.directory / "manifest.json"
        for name, body in (("index.html", b"<a href='module.html'>module</a>"),
                           ("module.html", b"<h1>Module</h1>")):
            (root / name).write_bytes(body)
            self.metadata["files"][name] = {
                "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "content_type": "text/html",
            }

    def manifest(self):
        self.path.write_text(json.dumps(self.metadata))
        return release.manifest(self.path)

    def test_complete_manifest_and_index_are_required(self):
        self.assertEqual(self.manifest()[0:2], ("api-docs", "2.0.1"))
        self.metadata["files"]["index.html"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "reviewed checksum"):
            self.manifest()

    def test_surplus_file_and_symlink_are_rejected(self):
        payload = self.directory / "payload"
        (payload / "extra.txt").touch()
        with self.assertRaisesRegex(ValueError, "complete payload"):
            self.manifest()
        (payload / "extra.txt").unlink()
        (payload / "module.html").unlink()
        (payload / "module.html").symlink_to(payload / "index.html")
        with self.assertRaisesRegex(ValueError, "regular payload"):
            self.manifest()

    def test_verified_upload_reserves_the_version_and_is_retryable(self):
        args = self.manifest()
        bucket = Bucket()
        with patch.object(release, "public_get", bucket.open):
            receipt = release.publish(bucket, *args)
            self.assertTrue(receipt.endswith("/releases/2.0.1/release.json"))
            self.assertEqual(receipt, release.publish(bucket, *args))
        names = [Path(r["Key"]).name for r in bucket.requests]
        self.assertEqual(names, ["release-manifest.json", "module.html", "index.html", "release.json"] * 2)
        self.assertTrue(all(r["IfNoneMatch"] == "*" and r["ExpectedBucketOwner"] == release.OWNER
                            and r["ChecksumSHA256"] for r in bucket.requests))

    def test_existing_different_object_cannot_be_replaced_or_marked_complete(self):
        bucket = Bucket()
        key = "releases/2.0.1/module.html"
        bucket.objects[key] = (b"different", "text/html")
        with patch.object(release, "public_get", bucket.open), self.assertRaisesRegex(ValueError, "conflicts"):
            release.publish(bucket, *self.manifest())
        self.assertEqual(bucket.objects[key][0], b"different")
        self.assertNotIn("releases/2.0.1/release.json", bucket.objects)
        self.assertNotIn("releases/2.0.1/index.html", bucket.objects)


if __name__ == "__main__":
    unittest.main()
