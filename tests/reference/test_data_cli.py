"""Image preflight works without the training runtime."""
import json
import subprocess
import sys


def test_image_data_check_has_no_training_imports_and_refuses_overwrite(tmp_path):
    from PIL import Image
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 3), color="red").save(images / "red.png")
    config = tmp_path / "image.toml"
    config.write_text('[data]\nfactory="image_folder"\n[data.args]\n'
                      f'root={json.dumps(str(images))}\nheight=3\nwidth=4\n', encoding="utf-8")
    output = tmp_path / "inventory.json"
    code = """
import importlib.abc
import sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'particlegan'}:
            raise AssertionError('data-check imported training runtime')
sys.meta_path.insert(0, NoTraining())
from hypergan.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    args = [sys.executable, "-I", "-c", code, "data-check", str(config), "--output", str(output)]
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    assert manifest["inventory"]["accepted"] == 1
    assert manifest["data"]["preprocessing"]["layout"] == "NCHW"
    duplicate = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert duplicate.returncode != 0
    assert json.loads(output.read_text()) == manifest
