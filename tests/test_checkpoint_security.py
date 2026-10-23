"""Exercise the checkpoint method without importing optional GAN backends."""
import ast
from pathlib import Path

import torch


def checkpoint_loader():
    source = Path(__file__).parents[1] / 'hypergan/gans/base_gan.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseGAN')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_load')
    namespace = {'torch': torch, 'Path': Path}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_load']


def test_tensor_state_dict_loads_on_cpu(tmp_path):
    model = torch.nn.Linear(2, 1)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    torch.save(expected, tmp_path / 'linear.save')
    restored = torch.nn.Linear(2, 1)
    assert checkpoint_loader()(None, str(tmp_path), 'linear', restored)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_pickle_payload_is_rejected_even_with_unsafe_default(tmp_path, monkeypatch):
    import os
    marker = tmp_path / 'pickle-executed'

    class Payload:
        def __reduce__(self):
            return os.system, (f'touch {marker}',)

    torch.save(Payload(), tmp_path / 'hostile.save')
    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    assert not checkpoint_loader()(None, str(tmp_path), 'hostile', torch.nn.Linear(2, 1))
    assert not marker.exists()
