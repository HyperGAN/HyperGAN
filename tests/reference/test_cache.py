"""Training-runtime integration with the external artifact cache."""
import hashlib

from hypergan.image_components import _verified_file


def test_image_components_resnet_default_download(tmp_path, monkeypatch):
    """Verify image_components._verified_file delegates default ResNet-18 to cache."""
    cache_dir = tmp_path / "cache"
    payload = b"fake resnet weights"
    sha = hashlib.sha256(payload).hexdigest()

    cached_file = cache_dir / "resnet18-f37072fd.pth"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(payload)

    monkeypatch.setattr("hypergan.image_components.RESNET18_SHA256", sha)
    monkeypatch.setattr("hypergan.image_components.RESNET18_URL", "https://download.pytorch.org/models/resnet18-f37072fd.pth")
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))

    resolved = _verified_file(None, expected=sha)
    assert resolved == cached_file.resolve()
