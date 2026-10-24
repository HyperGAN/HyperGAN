#!/usr/bin/env python3
"""Deterministically generate HyperGAN 2.0 'HG' logo assets using the trained 128x128 colorization model.

Pipeline:
1. Synthesize a crisp 128x128 multi-tone grayscale 'HG' monogram (`docs/assets/hg-bw-128.png`)
    via 4x supersampling (512x512 -> 128x128 Lanczos) with pure white background (255, gray = +1.0)
    and distinct mid-gray structural facets (56..154) across the 'H' and 'G' letterforms.
2. Quantitatively evaluate checkpoints (`step 3000` vs `step 7758` `latest`) from the trained
    128x128 colorization run (`logos-colorization-128`)
    on luminance lock precision, background white purity, foreground chroma saturation,
   intra-logo H-vs-G two-tone contrast, pairwise tile diversity, and 360-degree hue coverage.
3. From a single fixed prior draw (`seed = 10000`, 64 particles; no seed sweeps), deterministically
   select 16 maximally diverse, high-chroma particles and emit:
   - `docs/assets/hg-bw-128.png` (128x128 grayscale input monogram)
   - `docs/assets/hypergan-logo.png` (128x128 standout two-tone colorized 'HG' mark)
   - `docs/assets/hypergan-logo-grid.png` (512x512 4x4 grid: tile (0,0) = BW input, tiles 1..15 = colorized)
   - `docs/assets/hypergan-colorizer-banner.png` (736x512 side-by-side `[BW HG] -> [4x4 Colorized Grid]`)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

import hypergan.api as hg
from hypergan.checkpoints import extract_ema_bundle, resolve_checkpoint
from hypergan.recipes import ComponentGraph, make_prior


DEFAULT_RUN_TARGET = "logos-colorization-128"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
FIXED_SEED = 10000
POOL_SIZE = 64
GRID_TILES = 16


def build_hg_monogram(size: int = 128, supersample: int = 4) -> np.ndarray:
    """Construct a 128x128 multi-tone grayscale 'HG' monogram with pure white (255) background."""
    s = size * supersample  # 512x512 supersampled canvas
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)

    # Bold geometric 'H' (left half: x in [32, 244], y in [84, 428])
    h_left = (xx >= 32) & (xx <= 114) & (yy >= 84) & (yy <= 428)
    h_right = (xx >= 162) & (xx <= 244) & (yy >= 84) & (yy <= 428)
    h_bar = (xx >= 90) & (xx <= 186) & (yy >= 214) & (yy <= 298)
    h_mask = h_left | h_right | h_bar

    # Bold geometric 'G' (right half: superellipse C-arc + crossbar spur + lower stem)
    gx0, gy0 = 374.0, 256.0
    rx_out, ry_out = 110.0, 176.0
    rx_in, ry_in = 42.0, 100.0
    dx = (xx - gx0) / rx_out
    dy = (yy - gy0) / ry_out
    r_out = (np.abs(dx) ** 2.2 + np.abs(dy) ** 2.2) ** (1.0 / 2.2)
    dx_in = (xx - gx0) / rx_in
    dy_in = (yy - gy0) / ry_in
    r_in = (np.abs(dx_in) ** 2.2 + np.abs(dy_in) ** 2.2) ** (1.0 / 2.2)

    g_ring = (r_out <= 1.0) & (r_in >= 1.0)
    g_mouth = (xx >= 368) & (yy >= 176) & (yy <= 244)
    g_ring = g_ring & (~g_mouth)
    g_spur = (xx >= 366) & (xx <= 484) & (yy >= 242) & (yy <= 312)
    g_stem = (xx >= 410) & (xx <= 484) & (yy >= 242) & (yy <= 428)
    g_mask = g_ring | g_spur | g_stem

    fg_mask = h_mask | g_mask
    grad_y = np.sin(np.clip((yy - 84.0) / 344.0, 0.0, 1.0) * math.pi)

    # Assign complementary mid-gray levels in [56, 158] (maximal RGB polytope chroma volume)
    # so the 16-channel GroupNorm spatial basis segments H, G, crossbar, and bevels cleanly.
    tone = np.full((s, s), 255.0, dtype=np.float32)
    tone = np.where(h_mask, 90.0 + 18.0 * grad_y, tone)
    tone = np.where(h_right, 108.0 + 18.0 * grad_y, tone)
    tone = np.where(h_bar & (~h_left) & (~h_right), 146.0 + 12.0 * grad_y, tone)
    tone = np.where(g_mask, 134.0 + 18.0 * grad_y, tone)
    tone = np.where(g_spur | g_stem, 96.0 + 18.0 * grad_y, tone)

    mask_pil = Image.fromarray((fg_mask.astype(np.uint8) * 255), mode="L")
    blur_small = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(radius=6.0)), dtype=np.float32) / 255.0
    blur_large = np.asarray(mask_pil.filter(ImageFilter.GaussianBlur(radius=16.0)), dtype=np.float32) / 255.0

    # Inner facets & outer structural bevel contour
    tone = np.where(h_mask & (blur_large > 0.86), 142.0, tone)
    tone = np.where(g_mask & (~g_spur) & (blur_large > 0.86), 94.0, tone)
    tone = np.where(fg_mask & (blur_small < 0.82), 66.0, tone)
    top_bot_rim = fg_mask & ((yy <= 102) | (yy >= 410))
    tone = np.where(top_bot_rim, 56.0, tone)

    img_512 = np.where(fg_mask, tone, 255.0)
    pil_512 = Image.fromarray(np.clip(img_512, 0, 255).astype(np.uint8), mode="L")
    pil_128 = pil_512.resize((size, size), Image.Resampling.LANCZOS)
    arr_128 = np.asarray(pil_128, dtype=np.float32)
    # Snap outer anti-aliased fringe (>245) to exact 255 so background is 100% pure white
    arr_128 = np.where(arr_128 > 245.0, 255.0, np.round(arr_128))
    return arr_128


def load_ema_colorizer(run_dir: Path, target: int | str) -> tuple[ComponentGraph, torch.nn.Module, dict]:
    """Load EMA ComponentGraph and 4096-particle ParticlePrior from a training checkpoint."""
    ckpt_dir = resolve_checkpoint(run_dir, target)
    bundle = extract_ema_bundle(ckpt_dir)
    graph = ComponentGraph(bundle["components"]).float().eval()
    for name, model in graph.models.items():
        model.load_state_dict(bundle["model_states"][name])
        for key, buf in model.named_buffers():
            if key in bundle["model_buffers"][name]:
                buf.copy_(bundle["model_buffers"][name][key])
    prior = make_prior(
        bundle["config"]["prior"],
        device="cpu",
        recipe=bundle["config"].get("recipe"),
    ).float().eval()
    prior.load_state_dict(bundle["prior"])
    for key, buf in prior.named_buffers():
        if key in bundle["prior_buffers"]:
            buf.copy_(bundle["prior_buffers"][key])
    return graph, prior, bundle


def _tile_hue_sectors(
    idx: int,
    h_rgb: torch.Tensor,
    g_rgb: torch.Tensor,
    fg_rgb_mean: torch.Tensor,
) -> set[int]:
    """Return 30-degree hue wheel sectors (0..11) covered by a tile's H, G, and overall foreground."""
    sectors: set[int] = set()
    for vec in (h_rgb[idx], g_rgb[idx], fg_rgb_mean[idx]):
        rr, gg, bb = vec.tolist()
        if max(rr, gg, bb) - min(rr, gg, bb) >= 20.0:
            hue = (math.atan2(math.sqrt(3.0) * (gg - bb), 2.0 * rr - gg - bb) * 180.0 / math.pi) % 360.0
            sectors.add(int(hue // 30.0))
    return sectors


def select_diverse_particles(
    rgb: torch.Tensor,
    ids: torch.Tensor,
    fg_mask: torch.Tensor,
    h_region: torch.Tensor,
    g_region: torch.Tensor,
    count: int = GRID_TILES,
) -> list[int]:
    """Deterministically select `count` high-chroma, color-wheel-diverse particles from the 64-particle draw."""
    chroma = (rgb.max(dim=1).values - rgb.min(dim=1).values)[:, fg_mask].mean(dim=-1)
    h_rgb = rgb[:, :, h_region].mean(dim=-1)
    g_rgb = rgb[:, :, g_region].mean(dim=-1)
    fg_rgb_mean = rgb[:, :, fg_mask].mean(dim=-1)
    hg_contrast = (h_rgb - g_rgb).abs().mean(dim=-1)

    # First particle: highest joint chroma + two-tone H-vs-G contrast
    scores = chroma + 0.5 * hg_contrast
    first_idx = int(scores.argmax().item())
    selected = [first_idx]
    covered_sectors = _tile_hue_sectors(first_idx, h_rgb, g_rgb, fg_rgb_mean)

    fg_flat = rgb[:, :, fg_mask].reshape(rgb.shape[0], -1)
    pool = [i for i in range(rgb.shape[0]) if chroma[i].item() >= 95.0]
    if len(pool) < count:
        pool = list(range(rgb.shape[0]))

    while len(selected) < count:
        best_cand, best_val = None, -1.0
        for cand in pool:
            if cand in selected:
                continue
            min_dist = min((fg_flat[cand] - fg_flat[s]).abs().mean().item() for s in selected)
            new_sectors = len(_tile_hue_sectors(cand, h_rgb, g_rgb, fg_rgb_mean) - covered_sectors)
            val = min_dist + 0.25 * chroma[cand].item() + 0.20 * hg_contrast[cand].item() + 32.0 * new_sectors
            if val > best_val:
                best_val = val
                best_cand = cand
        selected.append(int(best_cand))
        covered_sectors |= _tile_hue_sectors(int(best_cand), h_rgb, g_rgb, fg_rgb_mean)
    return selected


def evaluate_checkpoint_on_hg(
    run_dir: Path,
    step: int,
    luma_128: np.ndarray,
    seed: int = FIXED_SEED,
    pool_size: int = POOL_SIZE,
) -> dict:
    """Run deterministic colorization on `luma_128` and compute quantitative metrics."""
    graph, prior, bundle = load_ema_colorizer(run_dir, step)
    gray_t = torch.from_numpy(luma_128).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
    fg_mask = torch.from_numpy(luma_128 < 240.0)
    bg_mask = torch.from_numpy(luma_128 == 255.0)
    h_region = torch.zeros_like(fg_mask)
    h_region[:, :62] = fg_mask[:, :62]
    g_region = torch.zeros_like(fg_mask)
    g_region[:, 64:] = fg_mask[:, 64:]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    z, ids = prior.sample(pool_size, generator=gen)
    gray_batch = gray_t.expand(pool_size, -1, -1, -1)
    with torch.no_grad():
        out = graph.generate(z, {"gray": gray_batch}, prior=prior)["generated"]

    rgb = (out.clamp(-1.0, 1.0) + 1.0) * 127.5
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y_out = 0.299 * r + 0.587 * g + 0.114 * b
    y_in = (gray_batch[:, 0] + 1.0) * 127.5

    max_luma_err_255 = (y_out - y_in).abs().max().item()
    max_luma_err_unit = max_luma_err_255 / 127.5
    bg_white_purity = (rgb[:, :, bg_mask] >= 254.99).float().mean().item()

    chroma_per_tile = (rgb.max(dim=1).values - rgb.min(dim=1).values)[:, fg_mask].mean(dim=-1)
    h_rgb = rgb[:, :, h_region].mean(dim=-1)
    g_rgb = rgb[:, :, g_region].mean(dim=-1)
    hg_contrast = (h_rgb - g_rgb).abs().mean(dim=-1)

    selected = select_diverse_particles(rgb, ids, fg_mask, h_region, g_region, count=GRID_TILES)
    sel_rgb = rgb[selected]
    sel_fg = sel_rgb[:, :, fg_mask]  # [16, 3, N_fg]
    diffs = (sel_fg.unsqueeze(0) - sel_fg.unsqueeze(1)).abs().mean(dim=(2, 3))
    pw_l1_div_16 = diffs[torch.triu(torch.ones(GRID_TILES, GRID_TILES), diagonal=1) == 1].mean().item()

    # Hue angle coverage across 12 x 30-degree color wheel sectors
    fg_rgb_mean = rgb[:, :, fg_mask].mean(dim=-1)
    sectors_all_set: set[int] = set()
    for i in range(rgb.shape[0]):
        sectors_all_set |= _tile_hue_sectors(i, h_rgb, g_rgb, fg_rgb_mean)
    sectors_all = len(sectors_all_set)
    sel_sectors: set[int] = set()
    for idx in selected:
        sel_sectors |= _tile_hue_sectors(idx, h_rgb, g_rgb, fg_rgb_mean)

    return {
        "step": step,
        "fg_chroma_mean_pool64": float(chroma_per_tile.mean().item()),
        "fg_chroma_mean_selected16": float(chroma_per_tile[selected].mean().item()),
        "fg_chroma_max_selected16": float(chroma_per_tile[selected].max().item()),
        "pairwise_fg_rgb_l1_selected16": float(pw_l1_div_16),
        "hg_two_tone_contrast_mean_selected16": float(hg_contrast[selected].mean().item()),
        "hg_two_tone_contrast_max_selected16": float(hg_contrast[selected].max().item()),
        "hue_sectors_pool64": int(sectors_all),
        "hue_sectors_selected16": len(sel_sectors),
        "max_luma_error_255": float(max_luma_err_255),
        "max_luma_error_unit": float(max_luma_err_unit),
        "bg_white_purity": float(bg_white_purity),
        "selected_draw_indices": selected,
        "selected_particle_ids": [int(ids[i].item()) for i in selected],
        "rgb_selected": sel_rgb.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy(),
    }


def render_banner(bw_rgb: np.ndarray, tiles_rgb: np.ndarray) -> Image.Image:
    """Render a 736x512 banner showing `[128x128 BW HG Input]  ->  [512x512 4x4 Colorized Grid]`."""
    tile_h, tile_w, _ = bw_rgb.shape  # 128x128
    grid_side = 4 * tile_w  # 512
    left_w = 224
    banner_w = left_w + grid_side  # 736
    banner_h = grid_side  # 512

    banner = Image.new("RGB", (banner_w, banner_h), (255, 255, 255))
    draw = ImageDraw.Draw(banner)

    # Paste 128x128 grayscale input centered vertically on the left column with a subtle 1px border
    bx0 = 20
    by0 = (banner_h - tile_h) // 2
    draw.rectangle([bx0 - 2, by0 - 2, bx0 + tile_w + 1, by0 + tile_h + 1], outline=(210, 214, 220), width=1)
    banner.paste(Image.fromarray(bw_rgb, mode="RGB"), (bx0, by0))

    # Draw crisp geometric arrow `->` from x=164 to x=206 at y=256
    ay = banner_h // 2
    draw.line([(162, ay), (200, ay)], fill=(80, 86, 96), width=5)
    draw.polygon([(208, ay), (192, ay - 11), (192, ay + 11)], fill=(80, 86, 96))

    # Paste 4x4 grid of all 16 vibrant colorized tiles on the right (x=224..736, y=0..512)
    for idx in range(GRID_TILES):
        r, c = divmod(idx, 4)
        tx = left_w + c * tile_w
        ty = r * tile_h
        banner.paste(Image.fromarray(tiles_rgb[idx], mode="RGB"), (tx, ty))

    return banner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_TARGET, help="Run path, slug, or name of trained 128x128 colorizer run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated PNG assets")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help="Deterministic prior sampling seed")
    args = parser.parse_args()

    run_dir = hg.resolve_run_target(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build 128x128 grayscale 'HG' monogram
    luma_128 = build_hg_monogram(size=128, supersample=4)
    bw_uint8 = luma_128.astype(np.uint8)
    bw_rgb = np.stack([bw_uint8, bw_uint8, bw_uint8], axis=-1)

    bw_path = args.output_dir / "hg-bw-128.png"
    Image.fromarray(bw_uint8, mode="L").save(bw_path, format="PNG", optimize=True)

    # 2. Evaluate checkpoints (step 3000 vs step 7758 latest)
    res_3000 = evaluate_checkpoint_on_hg(run_dir, 3000, luma_128, seed=args.seed)
    res_7758 = evaluate_checkpoint_on_hg(run_dir, 7758, luma_128, seed=args.seed)

    # Pick checkpoint with higher foreground chroma + pairwise diversity
    chosen = (
        res_3000
        if (res_3000["fg_chroma_mean_selected16"] + res_3000["pairwise_fg_rgb_l1_selected16"])
        >= (res_7758["fg_chroma_mean_selected16"] + res_7758["pairwise_fg_rgb_l1_selected16"])
        else res_7758
    )
    tiles_rgb = chosen["rgb_selected"]  # [16, 128, 128, 3]

    # 3. Save single standout colorized logo (`docs/assets/hypergan-logo.png`, 128x128)
    logo_path = args.output_dir / "hypergan-logo.png"
    Image.fromarray(tiles_rgb[0], mode="RGB").save(logo_path, format="PNG", optimize=True)

    # 4. Save 512x512 4x4 grid (`docs/assets/hypergan-logo-grid.png`):
    #    Tile (0, 0) = grayscale 'HG' input; Tiles 1..15 = 15 vibrant colorizations (tiles_rgb[0..14])
    grid_img = Image.new("RGB", (512, 512), (255, 255, 255))
    grid_img.paste(Image.fromarray(bw_rgb, mode="RGB"), (0, 0))
    for cell_idx in range(1, GRID_TILES):
        r, c = divmod(cell_idx, 4)
        grid_img.paste(Image.fromarray(tiles_rgb[cell_idx - 1], mode="RGB"), (c * 128, r * 128))
    grid_path = args.output_dir / "hypergan-logo-grid.png"
    grid_img.save(grid_path, format="PNG", optimize=True)

    # 5. Save side-by-side workflow banner (`docs/assets/hypergan-colorizer-banner.png`, 736x512)
    banner_img = render_banner(bw_rgb, tiles_rgb)
    banner_path = args.output_dir / "hypergan-colorizer-banner.png"
    banner_img.save(banner_path, format="PNG", optimize=True)

    report = {
        "selected_checkpoint_step": chosen["step"],
        "seed": args.seed,
        "monogram": {
            "fg_fraction": float((luma_128 < 240.0).mean()),
            "bg_white_fraction": float((luma_128 == 255.0).mean()),
            "fg_luma_mean": float(luma_128[luma_128 < 240.0].mean()),
            "fg_luma_min": float(luma_128[luma_128 < 240.0].min()),
            "fg_luma_max": float(luma_128[luma_128 < 240.0].max()),
        },
        "checkpoints": {
            "step_3000": {k: v for k, v in res_3000.items() if k != "rgb_selected"},
            "step_7758": {k: v for k, v in res_7758.items() if k != "rgb_selected"},
        },
        "assets": {
            str(p.relative_to(args.output_dir.parent.parent)): {
                "size": Image.open(p).size,
                "mode": Image.open(p).mode,
                "bytes": p.stat().st_size,
            }
            for p in (bw_path, logo_path, grid_path, banner_path)
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
