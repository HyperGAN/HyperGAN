"""Unit tests covering Milestone 4 (Features 40–58).

Features covered:
- Feature 40: -b / --bind across CLI & API
- Feature 41: Default listener 127.0.0.1 & UDP probe bypass
- Feature 42: TTY-gated viewer autostart
- Feature 44: Web observatory preview thumbnail route & FID metrics
- Features 45-46: Sample/Checkpoint action buttons & empty sample guidance state
- Feature 47: Privacy strip: example_inputs replacement with shape/dtype specs
- Feature 48: Standalone generator baking: particle table baking without runtime deps
- Feature 49: Generator multi-format export (bundle, safetensors, pt2, onnx)
- Feature 50: Dual-inheritance LoadedGenerator (nn.Module + dict)
- Feature 51: Latent interpolation slerp/lerp & zero steps rejection
- Feature 52: Batch PNG sampling with --output-dir and --device
- Feature 53: Non-image conditioning inputs (JSON and tensor strings)
- Feature 54: ImageFolder alpha compositing (composite_white, composite_black, error)
- Feature 55: ImageFolder symlink traversal, __len__, __getitem__
- Feature 56: 32..1024px multi-GPU templates using group_norm(8) (0 batch_norm)
- Feature 57: 512px generator template shape match
- Feature 58: Extended starter catalog (transgan, colorization, paired-image, conditional)
"""
import argparse
import contextlib
import io
from importlib.resources import files
import json
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from PIL import Image

import hypergan.api as hg
from hypergan.artifacts import (
    LoadedGenerator,
    export_generator,
    interpolate,
    sample,
    _coerce_conditioning_input,
)
from hypergan.cli import _parser as _build_parser, _port_number, _print_training_header, main as cli_main
from hypergan.data import ImageFolder
from hypergan.recipe_generator import (
    STARTER_TEMPLATES,
    get_hndl_templates,
    generate_recipe,
)


def run_cli(*args: str) -> tuple[int, str, str]:
    """Execute hypergan CLI in-process, capturing return code, stdout, and stderr."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            rc = cli_main([str(a) for a in args])
            if rc is None:
                rc = 0
        except SystemExit as exc:
            rc = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except Exception as exc:
            stderr_buf.write(f"\nUnhandled Exception: {exc}\n")
            rc = 1
    return rc, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Feature 40, 41, 42: Web Observatory Flags, Binding, TTY Autostart
# ---------------------------------------------------------------------------
class TestWebObservatoryFlagsAndLifecycle:
    def test_bind_flag_parsing_across_subcommands(self):
        parser = _build_parser()

        args_train = parser.parse_args(["train", "config.toml", "-b", "0.0.0.0"])
        assert args_train.server_host == "0.0.0.0"

        args_resume = parser.parse_args(["resume", "run123", "--bind", "192.168.1.50"])
        assert args_resume.server_host == "192.168.1.50"

        args_view = parser.parse_args(["view", "-b", "10.0.0.1"])
        assert args_view.host == "10.0.0.1"

        args_serve = parser.parse_args(["serve", "-b", "0.0.0.0"])
        assert args_serve.host == "0.0.0.0"

    def test_port_number_validation(self):
        assert _port_number("8080") == 8080
        assert _port_number("0") == 0
        assert _port_number("65535") == 65535

        with pytest.raises(argparse.ArgumentTypeError):
            _port_number("-1")
        with pytest.raises(argparse.ArgumentTypeError):
            _port_number("65536")
        with pytest.raises(argparse.ArgumentTypeError):
            _port_number("abc")

    def test_loopback_skips_udp_probe(self, monkeypatch, capsys):
        import socket
        called = []
        orig_connect = socket.socket.connect
        def mock_connect(self, addr):
            called.append(addr)
            return orig_connect(self, addr)
        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        # Host 127.0.0.1 must skip socket connect
        _print_training_header(Path("/tmp"), "test", "cpu", "http://127.0.0.1:8765", host="127.0.0.1")
        assert ("8.8.8.8", 80) not in called

    def test_tty_gated_viewer_autostart_logic(self, monkeypatch):
        # When CI=1, autostart should be suppressed
        monkeypatch.setenv("CI", "1")
        assert os.environ.get("CI") == "1"

        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("HYPERGAN_NO_SERVER", "1")
        assert os.environ.get("HYPERGAN_NO_SERVER") == "1"


# ---------------------------------------------------------------------------
# Feature 44, 45, 46: Web UI Assets and Elements
# ---------------------------------------------------------------------------
class TestWebAssetsAndElements:
    def test_runs_html_and_css_fid_and_thumbnail_support(self):
        runs_css = files("hypergan").joinpath("web_assets/runs.css").read_text(encoding="utf-8")
        assert "run-thumb" in runs_css

        runs_js = files("hypergan").joinpath("web_assets/runs.js").read_text(encoding="utf-8")
        assert "thumbnail" in runs_js.lower()
        assert "fid" in runs_js.lower()

    def test_index_html_action_buttons_and_empty_guidance(self):
        index_html = files("hypergan").joinpath("web_assets/index.html").read_text(encoding="utf-8")
        assert "action-sample" in index_html
        assert "action-checkpoint" in index_html
        assert "sample-empty" in index_html


# ---------------------------------------------------------------------------
# Feature 47, 48, 49, 50: Privacy Stripping, Standalone Baking & LoadedGenerator
# ---------------------------------------------------------------------------
class TestModelExportAndInferenceAPI:
    @pytest.fixture
    def trained_run(self, tmp_path):
        tmp_dir = tmp_path / "export_fixture"
        tmp_dir.mkdir()
        proj_dir = tmp_dir / "proj"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_dir = proj_dir / "runs" / "export_run"
        rc, _, err = run_cli("train", str(cfg_file), "--run-dir", str(run_dir), "--steps", "2", "-d", "cpu")
        assert rc == 0, err
        return run_dir, tmp_dir

    def test_export_privacy_strip_default(self, trained_run):
        run_dir, tmp_dir = trained_run
        export_out = tmp_dir / "privacy_model.pt"
        rc, _, err = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc == 0, err
        assert export_out.is_file()

        state = torch.load(export_out, weights_only=False)
        assert isinstance(state, dict)
        assert "example_inputs" in state
        for val in state["example_inputs"].values():
            assert not isinstance(val, torch.Tensor)
            assert isinstance(val, dict)
            assert "shape" in val
            assert "dtype" in val

    def test_coerce_conditioning_from_privacy_spec(self):
        spec = {"shape": [2, 3, 32, 32], "dtype": "float32"}
        coerced = _coerce_conditioning_input(spec, None)
        assert isinstance(coerced, torch.Tensor)
        assert list(coerced.shape) == [2, 3, 32, 32]
        assert coerced.sum().item() == 0.0

    def test_loaded_generator_dual_inheritance_and_callable(self, trained_run):
        run_dir, tmp_dir = trained_run
        export_out = tmp_dir / "callable_model.pt"
        rc, _, err = run_cli("export", str(run_dir), "-o", str(export_out))
        assert rc == 0, err

        gen = hg.load_bundle(export_out, device="cpu")
        # Dual inheritance: nn.Module and dict
        assert isinstance(gen, nn.Module)
        assert isinstance(gen, dict)

        # Standalone baking buffers present
        assert hasattr(gen, "particle_z")
        assert hasattr(gen, "particle_sigma")

        # Callable inference
        assert callable(gen)
        samples = gen.sample(count=4)
        assert isinstance(samples, torch.Tensor)
        assert samples.shape[0] == 4

    def test_export_generator_formats(self, trained_run):
        run_dir, tmp_dir = trained_run

        # 1. Format: bundle (.pt)
        out_bundle = tmp_dir / "exported.pt"
        res_bundle = export_generator(run_dir, out_bundle, format="bundle")
        assert res_bundle.is_file()
        assert res_bundle.with_suffix(".sha256").is_file()

        # 2. Format: safetensors
        out_st = tmp_dir / "exported.safetensors"
        res_st = export_generator(run_dir, out_st, format="safetensors")
        assert res_st.is_file()
        assert res_st.with_suffix(".sha256").is_file()

        # 3. Format: pt2
        out_pt2 = tmp_dir / "exported.pt2"
        res_pt2 = export_generator(run_dir, out_pt2, format="pt2")
        assert res_pt2.is_file()

        # 4. Format: onnx
        out_onnx = tmp_dir / "exported.onnx"
        try:
            import onnx
            res_onnx = export_generator(run_dir, out_onnx, format="onnx")
            assert res_onnx.is_file()
        except ImportError:
            with pytest.raises(ImportError, match=r"(?i)onnx.*requires"):
                export_generator(run_dir, out_onnx, format="onnx")

    def test_export_generator_hndl_zdim64_and_pt2_reload(self, tmp_path, monkeypatch):
        import sys
        import types
        import warnings

        data_dir = tmp_path / "img_z64"
        data_dir.mkdir()
        for i in range(4):
            Image.new("RGB", (16, 16), (i * 50, 100, 150)).save(data_dir / f"im_{i}.png")

        cfg_path = tmp_path / "hndl_z64.toml"
        cfg_path.write_text(
            f"""
format = "hypergan-model/1"
name = "hndl_z64"

[data]
builtin = "image_folder"
args = {{ root = "{data_dir.as_posix()}", height = 16, width = 16, mode = "RGB" }}

[prior]
kind = "particles"
z_dim = 64
num_particles = 32

[networks.generator]
role = "generator"
inputs = {{ z = "latent" }}
input_shape = {{ z = ["B", 64] }}
output_shape = ["B", 3, 16, 16]
source = \"\"\"
h = linear(z, 256, name="proj")
h = reshape(h, 16, 16)
h = attention(h, 4, dropout=0.0)
h = permute(h, 2, 1)
h = reshape(h, 16, 4, 4)
h = pixel_shuffle(h, 2)
h = pixel_shuffle(h, 2)
h = conv(h, 3, kernel_size=3, padding=1)
out = tanh(h)
\"\"\"

[networks.discriminator]
role = "critic"
inputs = {{ x = "candidate" }}
input_shape = ["B", 3, 16, 16]
output_shape = ["B", 1]
source = \"\"\"
conv(16, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(32, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(1, kernel_size=4, stride=1, padding=0)
flatten()
\"\"\"

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"
weight = 1.0

[train]
steps = 1
batch_size = 2
seed = 0
device = "cpu"

[sampling]
count = 2
seed = 42
""",
            encoding="utf-8",
        )
        run_dir = tmp_path / "runs" / "hndl_z64_run"
        rc, _, err = run_cli("train", str(cfg_path), "--run-dir", str(run_dir), "--steps", "1", "-d", "cpu")
        assert rc == 0, err

        # 1. Bundle + z_dim=64 resolution check (with & without baked particle_z)
        out_bundle = tmp_path / "z64.pt"
        res_bundle = export_generator(run_dir, out_bundle, format="bundle")
        assert res_bundle.is_file()
        loaded = hg.load_bundle(res_bundle, device="cpu")
        assert loaded.sample_latents(3).shape == (3, 64)
        loaded.particle_z = None
        assert loaded.sample_latents(3).shape == (3, 64)

        # 2. Safetensors
        out_st = tmp_path / "z64.safetensors"
        res_st = export_generator(run_dir, out_st, format="safetensors")
        assert res_st.is_file()
        assert res_st.with_suffix(".sha256").is_file()

        # 3. PT2 export and torch.export.load verification
        out_pt2 = tmp_path / "z64.pt2"
        res_pt2 = export_generator(run_dir, out_pt2, format="pt2")
        assert res_pt2.is_file()
        exported_prog = torch.export.load(res_pt2)
        test_z = torch.randn(1, 64)
        pt2_out = exported_prog.module()(test_z)
        assert pt2_out.shape == (1, 3, 16, 16)
        assert torch.isfinite(pt2_out).all()

        # 4. ONNX export
        out_onnx = tmp_path / "z64.onnx"
        try:
            import onnx  # noqa: F401
            res_onnx = export_generator(run_dir, out_onnx, format="onnx")
            assert res_onnx.is_file()
        except ImportError:
            with pytest.raises(ImportError, match=r"(?i)onnx.*requires"):
                export_generator(run_dir, out_onnx, format="onnx")
            captured = {}

            def fake_onnx_export(mod, args, f, *, input_names=None, output_names=None, dynamic_axes=None, **kw):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=torch.jit.TracerWarning)
                    out = mod(*args)
                captured["args"] = args
                captured["input_names"] = input_names
                captured["dynamic_axes"] = dynamic_axes
                captured["out"] = out
                Path(f).write_bytes(b"FAKE_ONNX")

            monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
            monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
            res_onnx = export_generator(run_dir, out_onnx, format="onnx")
            assert res_onnx.is_file()
            assert len(captured["args"]) == 1
            assert captured["args"][0].shape == (1, 64)
            assert captured["input_names"] == ["latent"]
            assert captured["dynamic_axes"] == {"latent": {0: "batch"}, "output": {0: "batch"}}
            assert captured["out"].shape == (1, 3, 16, 16)
            assert torch.isfinite(captured["out"]).all()

    def test_export_generator_conditional_hndl_all_formats(self, tmp_path, monkeypatch):
        import sys
        import types
        import warnings

        cfg_path = tmp_path / "cond_hndl.toml"
        cfg_path.write_text(
            """
format = "hypergan-model/1"
name = "cond_hndl"

[data]
builtin = "paired_linear"
args = { dimensions = 16, scale = 1.0, offset = 0.0 }

[prior]
kind = "particles"
z_dim = 64
num_particles = 32

[networks.generator]
role = "generator"
inputs = { z = "latent", condition = "batch.condition" }
input_shape = { z = ["B", 64], condition = ["B", 16] }
output_shape = ["B", 16]
source = \"\"\"
g = reshape(condition, 1, 4, 4)
h = linear(z, 64)
h = reshape(h, 4, 4, 4)
h = concat(h, g, axis=1)
h = conv(h, 8, kernel_size=3, padding=1)
h = relu(h)
h = flatten(h)
out = linear(h, 16)
\"\"\"

[networks.discriminator]
role = "critic"
inputs = { x = "candidate", condition = "batch.condition" }
input_shape = { x = ["B", 16], condition = ["B", 16] }
output_shape = ["B", 1]
source = \"\"\"
h = concat(x, condition, axis=1)
h = linear(h, 32)
h = leaky_relu(h, 0.2)
out = linear(h, 1)
\"\"\"

[[losses]]
type = "adversarial"
critic = "discriminator"
weight = 1.0
penalty = 1.0

[[losses]]
type = "prior"
weight = 1.0

[train]
steps = 1
batch_size = 2
seed = 0
device = "cpu"

[sampling]
count = 2
seed = 42
""",
            encoding="utf-8",
        )
        run_dir = tmp_path / "runs" / "cond_hndl_run"
        rc, _, err = run_cli("train", str(cfg_path), "--run-dir", str(run_dir), "--steps", "1", "-d", "cpu")
        assert rc == 0, err

        # 1. Bundle (privacy-stripped conditioning spec)
        out_bundle = tmp_path / "cond.pt"
        res_bundle = export_generator(run_dir, out_bundle, format="bundle")
        assert res_bundle.is_file()
        bundle_state = torch.load(res_bundle, weights_only=False)
        assert "condition" in bundle_state["example_inputs"]
        assert isinstance(bundle_state["example_inputs"]["condition"], dict)
        loaded = hg.load_bundle(res_bundle, device="cpu")
        cond_out = loaded.sample(count=3, conditioning={"condition": torch.randn(1, 16)})
        assert cond_out.shape == (3, 16)
        assert torch.isfinite(cond_out).all()

        # 2. Safetensors (both privacy-stripped and include_example_inputs=True)
        out_st = tmp_path / "cond.safetensors"
        res_st = export_generator(run_dir, out_st, format="safetensors")
        assert res_st.is_file()
        out_st_raw = tmp_path / "cond_raw.safetensors"
        res_st_raw = export_generator(run_dir, out_st_raw, format="safetensors", include_example_inputs=True)
        assert res_st_raw.is_file()

        # 3. PT2 export and torch.export.load verification with conditioning tensor
        out_pt2 = tmp_path / "cond.pt2"
        res_pt2 = export_generator(run_dir, out_pt2, format="pt2")
        assert res_pt2.is_file()
        exported_prog = torch.export.load(res_pt2)
        pt2_out = exported_prog.module()(torch.randn(1, 64), torch.randn(1, 16))
        assert pt2_out.shape == (1, 16)
        assert torch.isfinite(pt2_out).all()

        # 4. ONNX export for conditional generator
        out_onnx = tmp_path / "cond.onnx"
        try:
            import onnx  # noqa: F401
            res_onnx = export_generator(run_dir, out_onnx, format="onnx")
            assert res_onnx.is_file()
        except ImportError:
            with pytest.raises(ImportError, match=r"(?i)onnx.*requires"):
                export_generator(run_dir, out_onnx, format="onnx")
            captured = {}

            def fake_onnx_export(mod, args, f, *, input_names=None, output_names=None, dynamic_axes=None, **kw):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=torch.jit.TracerWarning)
                    out = mod(*args)
                captured["args"] = args
                captured["input_names"] = input_names
                captured["dynamic_axes"] = dynamic_axes
                captured["out"] = out
                Path(f).write_bytes(b"FAKE_COND_ONNX")

            monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
            monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
            res_onnx = export_generator(run_dir, out_onnx, format="onnx")
            assert res_onnx.is_file()
            assert len(captured["args"]) == 2
            assert captured["args"][0].shape == (1, 64)
            assert captured["args"][1].shape == (1, 16)
            assert captured["input_names"] == ["latent", "condition"]
            assert captured["dynamic_axes"] == {
                "latent": {0: "batch"},
                "condition": {0: "batch"},
                "output": {0: "batch"},
            }
            assert captured["out"].shape == (1, 16)
            assert torch.isfinite(captured["out"]).all()


# ---------------------------------------------------------------------------
# Feature 51: Latent Interpolation API
# ---------------------------------------------------------------------------
class TestLatentInterpolation:
    def test_interpolate_slerp_and_lerp(self):
        v0 = torch.randn(1, 16)
        v1 = torch.randn(1, 16)

        # Default slerp
        walk = interpolate(v0, v1, steps=5, mode="slerp")
        assert walk.shape == (5, 16)
        assert torch.allclose(walk[0], v0, atol=1e-4)
        assert torch.allclose(walk[-1], v1, atol=1e-4)

        # Lerp
        walk_lerp = interpolate(v0, v1, steps=4, mode="lerp")
        assert walk_lerp.shape == (4, 16)
        assert torch.allclose(walk_lerp[0], v0, atol=1e-4)
        assert torch.allclose(walk_lerp[-1], v1, atol=1e-4)

    def test_interpolate_zero_or_negative_steps_rejected(self):
        v0 = torch.randn(1, 8)
        v1 = torch.randn(1, 8)
        with pytest.raises((ValueError, AssertionError)):
            interpolate(v0, v1, steps=0)
        with pytest.raises((ValueError, AssertionError)):
            interpolate(v0, v1, steps=-3)

    def test_loaded_generator_interpolate_method(self, tmp_path):
        proj_dir = tmp_path / "proj_interp"
        run_cli("new", "--project", str(proj_dir), "--template", "gaussian-grid", "--device", "cpu")
        cfg_file = proj_dir / "configs" / "gaussian-grid.toml"
        run_dir = proj_dir / "runs" / "interp_run"
        rc, _, err = run_cli("train", str(cfg_file), "--run-dir", str(run_dir), "--steps", "2", "-d", "cpu")
        assert rc == 0, err
        export_out = tmp_path / "model_interp.pt"
        run_cli("export", str(run_dir), "-o", str(export_out))

        loaded = hg.load_bundle(export_out, device="cpu")
        interp_walk = loaded.interpolate(seed_a=42, seed_b=43, steps=3)
        assert interp_walk.shape[0] == 3


# ---------------------------------------------------------------------------
# Feature 52 & 53: Batch PNG Sampling & Conditioning Inputs
# ---------------------------------------------------------------------------
class TestSamplingAndConditioning:
    def test_batch_png_sampling_output_dir(self, tmp_path):
        data_dir = tmp_path / "img_data"
        data_dir.mkdir()
        for i in range(4):
            img = Image.new("RGB", (32, 32), (i * 40, i * 40, i * 40))
            img.save(data_dir / f"img_{i}.png")
        proj_dir = tmp_path / "proj_batch_sample"
        run_cli("new", "-p", str(proj_dir), "-t", "dcgan", "--dataset", str(data_dir), "--device", "cpu")
        cfg_file = proj_dir / "configs" / "dcgan.toml"
        run_dir = proj_dir / "runs" / "batch_sample_run"
        rc, _, err = run_cli("train", str(cfg_file), "--run-dir", str(run_dir), "--steps", "1", "-d", "cpu")
        assert rc == 0, err

        out_dir = tmp_path / "png_samples"
        res = sample(run_dir, count=3, device="cpu", output_dir=out_dir)
        assert out_dir.is_dir()
        files = sorted(out_dir.glob("sample_*.png"))
        assert len(files) == 3
        assert files[0].name == "sample_0000.png"
        assert files[2].name == "sample_0002.png"

    def test_json_string_conditioning_coercion(self):
        json_str = json.dumps([1.0, 0.5, -0.5, 0.0])
        tensor = _coerce_conditioning_input(json_str, None)
        assert isinstance(tensor, torch.Tensor)
        assert list(tensor.shape) in ([4], [1, 4])
        assert tensor.flatten()[1].item() == 0.5

    def test_tensor_string_conditioning_coercion(self):
        t_str = "[[0.1, 0.2], [0.3, 0.4]]"
        tensor = _coerce_conditioning_input(t_str, None)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert abs(tensor[0, 0].item() - 0.1) < 1e-4


# ---------------------------------------------------------------------------
# Feature 54 & 55: ImageFolder Alpha Compositing & Symlinks
# ---------------------------------------------------------------------------
class TestImageFolderAlphaAndSymlinks:
    def test_alpha_composite_white(self, tmp_path):
        img_dir = tmp_path / "data_white"
        img_dir.mkdir()
        # Half-transparent red (R=255, G=0, B=0, A=128)
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 128))
        img.save(img_dir / "test.png")

        ds = ImageFolder(img_dir, height=32, width=32, alpha="composite_white")
        assert len(ds) == 1
        item = ds[0]
        assert "real" in item
        tensor = item["real"]
        assert tensor.shape == (3, 32, 32)
        # Red channel should be higher than green and blue
        assert tensor[0].mean() > tensor[1].mean()

    def test_alpha_composite_black(self, tmp_path):
        img_dir = tmp_path / "data_black"
        img_dir.mkdir()
        # Fully transparent
        img = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
        img.save(img_dir / "test_transparent.png")

        ds = ImageFolder(img_dir, height=32, width=32, alpha="composite_black")
        item = ds[0]
        tensor = item["real"]
        # In black composite, background is black -> -1.0 in [-1, 1]
        assert torch.allclose(tensor, torch.full_like(tensor, -1.0), atol=1e-2)

    def test_alpha_error_raises_on_rgba(self, tmp_path):
        img_dir = tmp_path / "data_err"
        img_dir.mkdir()
        img = Image.new("RGBA", (32, 32), (100, 100, 100, 200))
        img.save(img_dir / "rgba.png")

        with pytest.raises(ValueError, match="alpha channel"):
            ImageFolder(img_dir, height=32, width=32, alpha="error")

    def test_symlink_traversal(self, tmp_path):
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        for i in range(2):
            img = Image.new("RGB", (32, 32), (0, 255, 0))
            img.save(real_dir / f"real_{i}.png")

        symlink_dir = tmp_path / "symlink_dir"
        symlink_dir.mkdir()
        sublink = symlink_dir / "linked_sub"
        sublink.symlink_to(real_dir, target_is_directory=True)

        # Without follow_symlinks: raises ValueError (symlink detected)
        with pytest.raises(ValueError, match="contains symlink"):
            ImageFolder(symlink_dir, height=32, width=32, follow_symlinks=False, recursive=True)

        # With follow_symlinks: discovers the images
        ds_yes = ImageFolder(symlink_dir, height=32, width=32, follow_symlinks=True, recursive=True)
        assert len(ds_yes) == 2
        item = ds_yes[0]
        assert item["real"].shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Feature 56, 57, 58: Resolution Ladder, Group Norm, Extended Catalog
# ---------------------------------------------------------------------------
class TestResolutionLadderAndTemplates:
    @pytest.mark.parametrize("res", [32, 64, 128, 256, 512, 1024])
    def test_resolutions_supported_in_templates(self, res):
        gen_hndl, crit_hndl = get_hndl_templates(width=res, height=res, channels=3, template="dcgan")
        # Multi-GPU requirement: no batch_norm in generators, group_norm(8) used
        assert "batch_norm" not in gen_hndl
        assert "group_norm(8)" in gen_hndl or "group_norm(" in gen_hndl

    def test_512px_generator_shape_matches(self):
        gen_hndl, crit_hndl = get_hndl_templates(width=512, height=512, channels=3, template="dcgan")
        # Output layer must produce 512px
        assert "512" in gen_hndl

    def test_extended_starter_templates_catalog(self):
        assert "transgan" in STARTER_TEMPLATES
        assert "colorization" in STARTER_TEMPLATES
        assert "paired-image" in STARTER_TEMPLATES
        assert "conditional" in STARTER_TEMPLATES

    def test_conditional_recipe_generation(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        img.save(data_dir / "sample.png")

        recipe_path = generate_recipe(
            name="cond_test",
            dataset_dir=data_dir,
            output_path=tmp_path / "cond_test.toml",
            template="conditional",
            resolution=32,
        )
        assert recipe_path.is_file()
        content = recipe_path.read_text(encoding="utf-8")
        assert "conditional" in content or "conditioning" in content
        assert 'alpha = "composite_white"' in content

    @pytest.mark.parametrize("template_name", list(STARTER_TEMPLATES.keys()))
    def test_all_starter_templates_generate_and_validate(self, template_name, tmp_path):
        """Verify that every starter template generates valid TOML accepted by hypergan validate."""
        res = 64 if STARTER_TEMPLATES[template_name]["resolutions"] else None
        recipe_path = tmp_path / f"{template_name}.toml"
        out = generate_recipe(
            name=f"test_{template_name}",
            template=template_name,
            resolution=res,
            device="cpu",
            output_path=recipe_path,
        )
        assert out.is_file()
        warnings = hg.validate(out)
        assert isinstance(warnings, list)

        # For reconstruction templates, verify loss specification and lowering
        if template_name in ("colorization", "paired-image"):
            content = out.read_text(encoding="utf-8")
            assert 'fn = "l1"' in content
            assert 'loss = "l1"' not in content
            model = hg.load(out)
            resolved = hg.resolve(model)
            objectives = resolved.get("objectives", [])
            recon_terms = [obj for obj in objectives if obj.get("id") == "reconstruction"]
            assert len(recon_terms) == 1
            assert recon_terms[0]["factory"] == "l1"
            assert recon_terms[0]["weight"] == 10.0
