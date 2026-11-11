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


def test_hndl_network_resolves_https_weights_path_via_cache(tmp_path, monkeypatch):
    """Verify HNDLNetwork resolves HTTPS weights_path template parameters via HYPERGAN_CACHE_DIR."""
    from pathlib import Path
    import torch
    from torchvision.models import resnet18
    from hypergan.hndl_networks import HNDLNetwork

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "resnet18-f37072fd.pth"
    with torch.random.fork_rng(devices=[]):
        torch.save(resnet18(weights=None).state_dict(), cached_file)
    sha = hashlib.sha256(cached_file.read_bytes()).hexdigest()

    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HYPERGAN_OFFLINE", "1")

    source = (Path(__file__).parents[2] / "examples/networks/resnet18-features-discriminator-32.hndl").read_text()
    params = {
        "weights_path": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "weights_sha256": sha,
    }
    net = HNDLNetwork(source=source, input_shape=["B", 3, 32, 32], output_shape=["B", 1], parameters=params)
    assert params["weights_path"] == "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    out = net(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 1)
