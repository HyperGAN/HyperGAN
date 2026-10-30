"""Empirical adversarial challenge suite for Milestone 4 (Features 40–46, 54–58).

Author: Challenger M4-2 (Empirical Challenger)
Focus:
  - Feature 54: ImageFolder alpha compositing (composite_white, composite_black, error)
                across pure RGBA, palette with transparency (P mode), and LA images.
  - Feature 55: ImageFolder symlink traversal (directory symlinks, file symlinks,
                len, indexing, follow_symlinks=False rejection).
  - Feature 56 & 57: 32..1024px multi-GPU templates (ZERO batch_norm, presence of group_norm(8),
                     512px dataset generating 512px network, 1024px dataset generating 1024px network).
  - Feature 58: Extended starter template catalog (transgan, colorization, paired-image,
                --conditional) and template validation / schema integrity.
  - Features 40–42: Web Observatory lifecycle (-b / --bind across train/resume/view/serve/api,
                    invalid port rejection, loopback default listener, TTY-gated autostart).
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import pytest
from PIL import Image
import torch
import torch.nn as nn

import hypergan.api as hg
from hypergan.cli import _build_parser, _port_number, _print_training_header, _training_viewer
from hypergan.data import ImageFolder, ImageDataError
from hypergan.hndl_networks import build_network
from hypergan.model_file import ModelFileError
from hypergan.recipe_generator import (
    STARTER_TEMPLATES,
    get_hndl_templates,
    generate_recipe,
    inspect_image_dataset,
)
from tests.e2e.harness_v2 import run_cli


# ===========================================================================
# 1. Feature 54: ImageFolder Alpha Compositing Stress Tests
# ===========================================================================
class TestImageFolderAlphaCompositingAdversarial:
    """Stress-test ImageFolder alpha compositing modes across image representations."""

    @pytest.fixture
    def image_fixtures(self, tmp_path):
        """Generate test images with transparency: RGBA, P (palette with transparency), and LA."""
        img_dir = tmp_path / "alpha_fixtures"
        img_dir.mkdir()

        # 1. Pure RGBA with semi-transparent red (R=255, G=0, B=0, A=128)
        rgba_img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        rgba_path = img_dir / "rgba_semi.png"
        rgba_img.save(rgba_path)

        # 2. Pure RGBA fully transparent (A=0)
        rgba_trans = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
        rgba_trans_path = img_dir / "rgba_trans.png"
        rgba_trans.save(rgba_trans_path)

        # 3. Palette mode (P) with transparency
        # Palette: index 0 is red (255,0,0), index 1 is blue (0,0,255)
        p_img = Image.new("P", (16, 16), 0)
        p_img.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        p_path = img_dir / "palette_trans.png"
        p_img.save(p_path, transparency=0)

        # 4. Luminance + Alpha (LA) image: L=128, A=64
        la_img = Image.new("LA", (16, 16), (128, 64))
        la_path = img_dir / "la_trans.png"
        la_img.save(la_path)

        # 5. Opaque RGB control image
        rgb_img = Image.new("RGB", (16, 16), (100, 150, 200))
        rgb_path = img_dir / "rgb_opaque.png"
        rgb_img.save(rgb_path)

        return {
            "root": img_dir,
            "rgba_semi": rgba_path,
            "rgba_trans": rgba_trans_path,
            "palette_trans": p_path,
            "la_trans": la_path,
            "rgb_opaque": rgb_path,
        }

    def test_alpha_composite_white_exact_pixel_blending(self, tmp_path):
        """Verify composite_white blends semi-transparent pixels onto white (1.0)."""
        single_dir = tmp_path / "single_rgba"
        single_dir.mkdir()
        # Half-opacity red (255, 0, 0, 128)
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        img.save(single_dir / "test.png")

        ds = ImageFolder(single_dir, height=16, width=16, alpha="composite_white")
        tensor = ds[0]["real"]
        # In RGB [-1, 1] range:
        # Expected composite:
        # out_R = (255*128 + 255*127)/255 = 255 -> 255/127.5 - 1 = +1.0
        # out_G = (0*128 + 255*127)/255 = 127 -> 127/127.5 - 1 = -0.00392
        # out_B = (0*128 + 255*127)/255 = 127 -> 127/127.5 - 1 = -0.00392
        assert abs(tensor[0, 0, 0].item() - 1.0) < 1e-4
        assert abs(tensor[1, 0, 0].item() - (-0.00392)) < 1e-3
        assert abs(tensor[2, 0, 0].item() - (-0.00392)) < 1e-3

    def test_alpha_composite_black_exact_pixel_blending(self, tmp_path):
        """Verify composite_black blends semi-transparent pixels onto black (-1.0)."""
        single_dir = tmp_path / "single_rgba_black"
        single_dir.mkdir()
        # Half-opacity red (255, 0, 0, 128)
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
        img.save(single_dir / "test.png")

        ds = ImageFolder(single_dir, height=16, width=16, alpha="composite_black")
        tensor = ds[0]["real"]
        # Expected composite:
        # out_R = (255*128 + 0*127)/255 = 128 -> 128/127.5 - 1 = +0.00392
        # out_G = 0 -> 0/127.5 - 1 = -1.0
        # out_B = 0 -> 0/127.5 - 1 = -1.0
        assert abs(tensor[0, 0, 0].item() - 0.00392) < 1e-3
        assert abs(tensor[1, 0, 0].item() - (-1.0)) < 1e-4
        assert abs(tensor[2, 0, 0].item() - (-1.0)) < 1e-4

    def test_alpha_compositing_palette_with_transparency(self, tmp_path):
        """Verify palette (P mode) images with transparency composite properly."""
        p_dir = tmp_path / "p_dir"
        p_dir.mkdir()
        img = Image.new("P", (16, 16), 0)
        img.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
        img.save(p_dir / "palette.png", transparency=0)

        # White composite: transparent pixels become white
        ds_white = ImageFolder(p_dir, height=16, width=16, alpha="composite_white")
        t_w = ds_white[0]["real"]
        assert torch.allclose(t_w, torch.ones_like(t_w), atol=1e-2)

        # Black composite: transparent pixels become black
        ds_black = ImageFolder(p_dir, height=16, width=16, alpha="composite_black")
        t_b = ds_black[0]["real"]
        assert torch.allclose(t_b, -torch.ones_like(t_b), atol=1e-2)

    def test_alpha_compositing_la_mode(self, tmp_path):
        """Verify Luminance + Alpha (LA) images composite properly under RGB and L modes."""
        la_dir = tmp_path / "la_dir"
        la_dir.mkdir()
        # Fully transparent LA image
        Image.new("LA", (16, 16), (128, 0)).save(la_dir / "trans_la.png")

        # White composite in RGB mode
        ds_w = ImageFolder(la_dir, height=16, width=16, mode="RGB", alpha="composite_white")
        assert torch.allclose(ds_w[0]["real"], torch.ones_like(ds_w[0]["real"]), atol=1e-2)

        # Black composite in L mode
        ds_b = ImageFolder(la_dir, height=16, width=16, mode="L", alpha="composite_black")
        assert ds_b[0]["real"].shape == (1, 16, 16)
        assert torch.allclose(ds_b[0]["real"], -torch.ones_like(ds_b[0]["real"]), atol=1e-2)

    def test_alpha_error_mode_raises_value_error_across_all_alpha_formats(self, tmp_path):
        """Verify alpha='error' raises ValueError on RGBA, LA, and P-with-transparency."""
        # 1. RGBA error
        d_rgba = tmp_path / "err_rgba"
        d_rgba.mkdir()
        Image.new("RGBA", (16, 16), (255, 0, 0, 100)).save(d_rgba / "rgba.png")
        with pytest.raises(ValueError, match="alpha channel"):
            ImageFolder(d_rgba, height=16, width=16, alpha="error")

        # 2. LA error
        d_la = tmp_path / "err_la"
        d_la.mkdir()
        Image.new("LA", (16, 16), (200, 100)).save(d_la / "la.png")
        with pytest.raises(ValueError, match="alpha channel"):
            ImageFolder(d_la, height=16, width=16, alpha="error")

        # 3. P with transparency error
        d_p = tmp_path / "err_p"
        d_p.mkdir()
        img_p = Image.new("P", (16, 16), 0)
        img_p.putpalette([100, 100, 100] + [0] * 765)
        img_p.save(d_p / "p.png", transparency=0)
        with pytest.raises(ValueError, match="alpha channel"):
            ImageFolder(d_p, height=16, width=16, alpha="error")

    def test_opaque_rgb_and_l_unaffected_by_alpha_error(self, tmp_path):
        """Control: opaque RGB and L images load identically across all alpha modes."""
        ctrl_dir = tmp_path / "ctrl"
        ctrl_dir.mkdir()
        Image.new("RGB", (16, 16), (80, 90, 100)).save(ctrl_dir / "rgb.png")

        ds_w = ImageFolder(ctrl_dir, height=16, width=16, alpha="composite_white")
        ds_b = ImageFolder(ctrl_dir, height=16, width=16, alpha="composite_black")
        ds_e = ImageFolder(ctrl_dir, height=16, width=16, alpha="error")

        assert torch.equal(ds_w[0]["real"], ds_b[0]["real"])
        assert torch.equal(ds_w[0]["real"], ds_e[0]["real"])


# ===========================================================================
# 2. Feature 55: ImageFolder Symlink Traversal & Integrity
# ===========================================================================
class TestImageFolderSymlinkTraversalAdversarial:
    """Stress-test ImageFolder symlink traversal and reject rules."""

    @pytest.fixture
    def symlink_dataset(self, tmp_path):
        """Create a dataset with nested directory symlinks and file symlinks."""
        ext_dir = tmp_path / "external_store"
        ext_dir.mkdir()
        # Images outside dataset root
        Image.new("RGB", (16, 16), (10, 10, 10)).save(ext_dir / "ext1.png")
        Image.new("RGB", (16, 16), (20, 20, 20)).save(ext_dir / "ext2.png")

        sub_ext = ext_dir / "sub_store"
        sub_ext.mkdir()
        Image.new("RGB", (16, 16), (30, 30, 30)).save(sub_ext / "ext3.png")

        # Dataset root
        data_root = tmp_path / "data_root"
        data_root.mkdir()
        # Ordinary image
        Image.new("RGB", (16, 16), (40, 40, 40)).save(data_root / "local.png")

        # Directory symlink: data_root/linked_dir -> ext_dir
        (data_root / "linked_dir").symlink_to(ext_dir, target_is_directory=True)

        # File symlink: data_root/linked_file.png -> ext_dir/ext1.png
        (data_root / "linked_file.png").symlink_to(ext_dir / "ext1.png")

        return {
            "root": data_root,
            "ext": ext_dir,
        }

    def test_follow_symlinks_discovers_dir_and_file_symlinks(self, symlink_dataset):
        """When follow_symlinks=True, all images through directory and file symlinks are indexed."""
        root = symlink_dataset["root"]
        ds = ImageFolder(root, height=16, width=16, follow_symlinks=True, recursive=True)
        # Expected: local.png, linked_file.png, linked_dir/ext1.png, linked_dir/ext2.png, linked_dir/sub_store/ext3.png
        assert len(ds) == 5
        # Verify indexing across all elements
        for i in range(len(ds)):
            item = ds[i]
            assert "real" in item
            assert item["real"].shape == (3, 16, 16)

    def test_follow_symlinks_false_strictly_rejects_directory_symlink(self, tmp_path):
        """When follow_symlinks=False, directory symlink raises ValueError immediately."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        Image.new("RGB", (16, 16), (1, 2, 3)).save(real_dir / "a.png")

        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        (ds_dir / "symlink_dir").symlink_to(real_dir, target_is_directory=True)

        with pytest.raises(ValueError, match="contains symlink"):
            ImageFolder(ds_dir, height=16, width=16, follow_symlinks=False, recursive=True)

    def test_follow_symlinks_false_strictly_rejects_file_symlink(self, tmp_path):
        """When follow_symlinks=False, file symlink raises ValueError immediately."""
        real_file = tmp_path / "real.png"
        Image.new("RGB", (16, 16), (1, 2, 3)).save(real_file)

        ds_dir = tmp_path / "ds_file"
        ds_dir.mkdir()
        (ds_dir / "symlink_file.png").symlink_to(real_file)

        with pytest.raises(ValueError, match="contains symlink"):
            ImageFolder(ds_dir, height=16, width=16, follow_symlinks=False)

    def test_dangling_symlink_under_follow_symlinks_false(self, tmp_path):
        """A dangling symlink under follow_symlinks=False is caught as an illegal symlink."""
        ds_dir = tmp_path / "dangling_ds"
        ds_dir.mkdir()
        Image.new("RGB", (16, 16)).save(ds_dir / "valid.png")
        (ds_dir / "ghost.png").symlink_to(tmp_path / "nonexistent.png")

        with pytest.raises(ValueError, match="contains symlink"):
            ImageFolder(ds_dir, height=16, width=16, follow_symlinks=False)


# ===========================================================================
# 3. Features 56 & 57: 32..1024px Resolution Ladder & GroupNorm(8)
# ===========================================================================
class TestResolutionLadderAndNormalizationAdversarial:
    """Stress-test 32..1024px template generation, zero batch_norm, and shape correctness."""

    RESOLUTIONS = [32, 64, 128, 256, 512, 1024]

    @pytest.mark.parametrize("res", RESOLUTIONS)
    def test_zero_batch_norm_and_presence_of_group_norm(self, res):
        """Verify generator and critic across 32..1024px contain ZERO batch_norm and use group_norm(8)."""
        gen_hndl, disc_hndl = get_hndl_templates(width=res, height=res, channels=3, template="dcgan")

        # Zero instances of batch_norm in both networks
        assert "batch_norm" not in gen_hndl.lower()
        assert "batch_norm" not in disc_hndl.lower()

        # Presence of group_norm(8) in generator
        assert "group_norm(8)" in gen_hndl

    @pytest.mark.parametrize("res", RESOLUTIONS)
    def test_compile_generator_and_discriminator_architectures(self, res):
        """Compile each generated HNDL template with build_network to prove structural correctness."""
        gen_hndl, disc_hndl = get_hndl_templates(width=res, height=res, channels=3, template="dcgan")

        # Compile Generator: [B, 128] -> [B, 3, res, res]
        gen_net = build_network(gen_hndl, input_shape=["B", 128], output_shape=["B", 3, res, res])
        assert gen_net is not None

        # Compile Discriminator: [B, 3, res, res] -> [B, 1]
        disc_net = build_network(disc_hndl, input_shape=["B", 3, res, res], output_shape=["B", 1])
        assert disc_net is not None

    def test_512px_dataset_generates_512px_network_fixing_truncation(self, tmp_path):
        """Confirm that a 512px dataset generates a 512px network (fixing previous 256px truncation)."""
        data_512 = tmp_path / "data_512"
        data_512.mkdir()
        Image.new("RGB", (512, 512), (10, 20, 30)).save(data_512 / "sample_512.png")

        info = inspect_image_dataset(data_512)
        assert info["preset_res"] == 512
        assert info["width"] == 512
        assert info["height"] == 512

        recipe_path = generate_recipe(
            name="test_dcgan_512",
            dataset_dir=data_512,
            output_path=tmp_path / "dcgan_512.toml",
            template="dcgan",
        )
        content = recipe_path.read_text(encoding="utf-8")
        assert "height = 512" in content
        assert "width = 512" in content
        assert "batch_norm" not in content
        assert "group_norm(8)" in content

    def test_1024px_dataset_generates_1024px_network(self, tmp_path):
        """Confirm that a 1024px dataset generates a 1024px network."""
        data_1024 = tmp_path / "data_1024"
        data_1024.mkdir()
        Image.new("RGB", (1024, 1024), (10, 20, 30)).save(data_1024 / "sample_1024.png")

        info = inspect_image_dataset(data_1024)
        assert info["preset_res"] == 1024

        recipe_path = generate_recipe(
            name="test_dcgan_1024",
            dataset_dir=data_1024,
            output_path=tmp_path / "dcgan_1024.toml",
            template="dcgan",
        )
        content = recipe_path.read_text(encoding="utf-8")
        assert "height = 1024" in content
        assert "width = 1024" in content
        assert "group_norm(8)" in content


# ===========================================================================
# 4. Feature 58: Extended Starter Templates & Schema Integrity
# ===========================================================================
class TestExtendedStarterTemplatesAdversarial:
    """Stress-test hypergan new extended templates and identify schema boundaries."""

    TEMPLATES = ["transgan", "colorization", "paired-image"]

    def test_extended_templates_catalog_metadata(self):
        """All four extended templates are registered in STARTER_TEMPLATES."""
        assert "transgan" in STARTER_TEMPLATES
        assert "colorization" in STARTER_TEMPLATES
        assert "paired-image" in STARTER_TEMPLATES
        assert "conditional" in STARTER_TEMPLATES

    def test_cli_new_with_extended_templates(self, tmp_path):
        """Test hypergan new CLI generation with transgan, colorization, paired-image, and --conditional."""
        # 1. transgan
        out_transgan = tmp_path / "new_transgan.toml"
        rc, _, err = run_cli("new", str(out_transgan), "-t", "transgan", "-r", "64", "--device", "cpu")
        assert rc == 0, err
        assert out_transgan.is_file()
        content_trans = out_transgan.read_text()
        assert "rms_norm" in content_trans
        assert "attention" in content_trans

        # 2. conditional via flag
        out_cond = tmp_path / "new_cond.toml"
        rc, _, err = run_cli("new", str(out_cond), "--conditional", "-r", "64", "--device", "cpu")
        assert rc == 0, err
        assert out_cond.is_file()
        content_cond = out_cond.read_text()
        assert "condition" in content_cond

        # 3. colorization
        out_color = tmp_path / "new_color.toml"
        rc, _, err = run_cli("new", str(out_color), "-t", "colorization", "-r", "64", "--device", "cpu")
        assert rc == 0, err
        assert out_color.is_file()
        content_color = out_color.read_text()
        assert "gray" in content_color

        # 4. paired-image
        out_paired = tmp_path / "new_paired.toml"
        rc, _, err = run_cli("new", str(out_paired), "-t", "paired-image", "-r", "64", "--device", "cpu")
        assert rc == 0, err
        assert out_paired.is_file()
        content_paired = out_paired.read_text()
        assert "condition" in content_paired

    def test_transgan_and_conditional_pass_validation(self, tmp_path):
        """Verify generated transgan and conditional recipes pass hypergan validate."""
        out_transgan = tmp_path / "valid_transgan.toml"
        run_cli("new", str(out_transgan), "-t", "transgan", "-r", "64", "--device", "cpu")
        rc_t, out_t, err_t = run_cli("validate", str(out_transgan))
        assert rc_t == 0, f"transgan validation failed: {err_t}"

        out_cond = tmp_path / "valid_cond.toml"
        run_cli("new", str(out_cond), "--conditional", "-r", "64", "--device", "cpu")
        rc_c, out_c, err_c = run_cli("validate", str(out_cond))
        assert rc_c == 0, f"conditional validation failed: {err_c}"

    def test_colorization_and_paired_image_loss_schema_remediated(self, tmp_path):
        """Verify colorization and paired-image generated recipes use 'fn' instead of 'loss' and validate cleanly."""
        for template in ("colorization", "paired-image"):
            out_file = tmp_path / f"valid_{template}.toml"
            rc, _, err = run_cli("new", str(out_file), "-t", template, "-r", "64", "--device", "cpu")
            assert rc == 0, err
            content = out_file.read_text()
            assert 'fn = "l1"' in content, f"Expected 'fn = \"l1\"' key in generated {template} recipe"
            assert 'loss = "l1"' not in content, f"Defective 'loss = \"l1\"' key still found in {template} recipe"

            # Validation succeeds cleanly
            warnings = hg.validate(out_file)
            assert isinstance(warnings, list)
            rc_val, _, err_val = run_cli("validate", str(out_file))
            assert rc_val == 0, f"{template} failed validation: {err_val}"



# ===========================================================================
# 5. Features 40–42: Web Observatory Lifecycle & CLI Flags
# ===========================================================================
class TestWebObservatoryLifecycleAdversarial:
    """Stress-test Web Observatory CLI flags, invalid port rejection, and TTY autostart gates."""

    def test_bind_flag_parsing_across_all_subcommands(self):
        """Verify -b and --bind across train, resume, view, and serve."""
        parser = _build_parser()

        # train
        args1 = parser.parse_args(["train", "m.toml", "-b", "0.0.0.0"])
        assert args1.server_host == "0.0.0.0"
        args2 = parser.parse_args(["train", "m.toml", "--bind", "127.0.0.1"])
        assert args2.server_host == "127.0.0.1"

        # resume
        args3 = parser.parse_args(["resume", "run_id", "-b", "0.0.0.0"])
        assert args3.server_host == "0.0.0.0"
        args4 = parser.parse_args(["resume", "run_id", "--bind", "10.0.0.1"])
        assert args4.server_host == "10.0.0.1"

        # view
        args5 = parser.parse_args(["view", "-b", "0.0.0.0"])
        assert args5.host == "0.0.0.0"
        args6 = parser.parse_args(["view", "--bind", "192.168.1.10"])
        assert args6.host == "192.168.1.10"

        # serve
        args7 = parser.parse_args(["serve", "-b", "0.0.0.0"])
        assert args7.host == "0.0.0.0"
        args8 = parser.parse_args(["serve", "--bind", "127.0.0.1"])
        assert args8.host == "127.0.0.1"

    def test_loopback_default_listener_across_commands(self):
        """Verify view and serve default to 127.0.0.1, and train/resume default server_host is None."""
        parser = _build_parser()

        args_v = parser.parse_args(["view"])
        assert args_v.host == "127.0.0.1"

        args_s = parser.parse_args(["serve"])
        assert args_s.host == "127.0.0.1"

        args_t = parser.parse_args(["train", "m.toml"])
        assert args_t.server_host is None  # Defaults to 127.0.0.1 at runtime

    @pytest.mark.parametrize("invalid_port", ["-1", "-8080", "65536", "70000", "99999", "abc", "80.5", ""])
    def test_invalid_port_numbers_rejected_by_validator(self, invalid_port):
        """_port_number strictly raises ArgumentTypeError on negative, >65535, and non-integer inputs."""
        with pytest.raises(argparse.ArgumentTypeError):
            _port_number(invalid_port)

    @pytest.mark.parametrize("valid_port", [0, 80, 8080, 8765, 65535])
    def test_valid_port_numbers_accepted(self, valid_port):
        """Boundary ports 0, 80, 65535 are accepted without error."""
        assert _port_number(str(valid_port)) == valid_port

    def test_cli_rejects_invalid_ports(self):
        """CLI subcommands exit with non-zero exit code when given invalid ports."""
        rc, _, err = run_cli("view", "--port", "-1")
        assert rc != 0
        assert "range" in err.lower() or "integer" in err.lower()

        rc2, _, err2 = run_cli("serve", "--port", "65536")
        assert rc2 != 0
        assert "range" in err2.lower() or "integer" in err2.lower()

    def test_loopback_listener_skips_external_udp_probe(self, monkeypatch):
        """_print_training_header skips socket connect to 8.8.8.8:80 when host is loopback."""
        called = []
        orig_connect = socket.socket.connect

        def mock_connect(self, addr):
            called.append(addr)
            return orig_connect(self, addr)

        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        # Loopback: must skip
        _print_training_header(Path("/tmp"), "test", "cpu", "http://127.0.0.1:8765", host="127.0.0.1")
        assert ("8.8.8.8", 80) not in called

    def test_tty_gated_viewer_autostart_suppression(self, monkeypatch):
        """Verify autostart is suppressed in headless TTY, under CI=1, and under HYPERGAN_NO_SERVER=1."""
        class MockNonTTY:
            def isatty(self):
                return False
            def write(self, s):
                pass
            def flush(self):
                pass

        args = argparse.Namespace(
            run_dir=Path("/tmp"),
            no_server=False,
            server=False,
            open=False,
            server_port=None,
            server_host=None,
            auth=None,
            public_origin=None,
            viewer_dev=False,
        )

        # 1. Non-interactive (headless)
        monkeypatch.setattr(sys, "stdin", MockNonTTY())
        monkeypatch.setattr(sys, "stdout", MockNonTTY())
        ctx = _training_viewer(args, config={})
        assert isinstance(ctx, nullcontext)

        # 2. Under CI=1
        monkeypatch.setenv("CI", "1")
        ctx_ci = _training_viewer(args, config={})
        assert isinstance(ctx_ci, nullcontext)

        # 3. Under HYPERGAN_NO_SERVER=1
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("HYPERGAN_NO_SERVER", "1")
        ctx_noserver = _training_viewer(args, config={})
        assert isinstance(ctx_noserver, nullcontext)
