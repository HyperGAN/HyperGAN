"""Local checkpoint fixtures exercising HNDL's real pretrained loader/readouts."""
import hashlib
import torch

from hypergan import pretrained_providers


def dinov3_assets(monkeypatch, tmp_path, builder):
    checkpoint = tmp_path / 'fixture-dinov3.pth'
    torch.save(builder().state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def build_on_context():
        device = torch.empty(0).device
        return builder().to(device)
    monkeypatch.setattr(pretrained_providers, '_dinov3_builder', lambda *args: build_on_context)
    return ('unused', '0' * 40, str(checkpoint), digest)


def backbone_model(model):
    return model.backbone['backbone'].model
