"""HyperGAN bespoke 255labs-branded static site generator.

Builds the complete documentation, interactive reference catalog, examples gallery,
and client-side search index into a 100% relative-URL static directory suitable
for GitHub Pages (``https://hypergan.github.io/HyperGAN/``), local HTTP preview,
or any custom domain.

Usage:
    python site/build.py --out _site --strict
"""

from __future__ import annotations

import argparse
import importlib
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


SITE_DIR = Path(__file__).resolve().parent
if str(SITE_DIR) not in sys.path:
    sys.path.insert(0, str(SITE_DIR))
TEMPLATES_DIR = SITE_DIR / "templates"
STATIC_DIR = SITE_DIR / "static"

WEB_ASSET_FILES = (
    "255labs-tokens.css",
    "brand.css",
    "manrope-latin-variable.woff2",
    "ibm-plex-mono-latin-400.woff2",
    "255labs-lockup.svg",
    "logo.png",
)

# Verified landing-page showcase snippets (valid HyperGAN 2.1.1 usage).
LANDING_SNIPPETS: dict[str, dict[str, str]] = {
    "cli": {
        "label": "CLI Workflow",
        "lang": "sh",
        "filename": "terminal",
        "code": """# 1. Scaffold an image GAN project auto-sized from your image folder
hypergan new --project my_gan --dataset ./images

# 2. Validate configuration & HNDL shapes without importing PyTorch
hypergan validate my_gan/configs/dcgan.toml

# 3. Train with live Web Observatory telemetry and global run registration
hypergan train my_gan/configs/dcgan.toml --steps 1000 -n baseline-128

# 4. Monitor from any directory, sample EMA generator, and export standalone
hypergan status baseline-128
hypergan sample baseline-128 --count 16 --seed 42
hypergan export baseline-128 --checkpoint latest -o generator.pt""",
    },
    "python": {
        "label": "Python API",
        "lang": "python",
        "filename": "train_and_export.py",
        "code": """import hypergan as hg

def main():
    # Torch-free config loading, overrides, and static validation
    model = hg.load("my_gan/configs/dcgan.toml")
    model = hg.override(model, {"train.steps": 1000, "train.batch_size": 32})
    warnings = hg.validate(model)

    # Launch training, inspect typed RunStatus, and tag the run
    run = hg.train(model, "my_gan/runs/exp1", steps=1000)
    st = hg.status("exp1")
    hg.tag("exp1", "baseline-128")

    # Export and reload a standalone EMA generator bundle
    bundle = hg.export("baseline-128", checkpoint="latest")
    gen = hg.load_bundle(bundle, device="cuda")
    samples = gen.sample(count=16, seed=42)
    walk = gen.interpolate(seed_a=100, seed_b=200, steps=10, mode="slerp")

if __name__ == "__main__":
    main()""",
    },
    "toml": {
        "label": "TOML Model File",
        "lang": "toml",
        "filename": "colorization-128.toml",
        "code": """format = "hypergan-model/1"
name = "colorization-128"
defaults = "particlegan"

[data]
builtin = "image_folder"
args = { root = "./data", height = 128, width = 128, mode = "RGB", alpha = "composite_white" }

[prior]
kind = "particles"
z_dim = 128
num_particles = 10000

[networks.luma]
role = "auxiliary"
inputs = { x = "batch.real" }
input_shape = ["B", 3, 128, 128]
output_shape = ["B", 1, 128, 128]
source = \"\"\"
r, gb = split(x, 1)
g, b = split(gb, 1)
out = add(add(scale(r, 0.299), scale(g, 0.587)), scale(b, 0.114))
\"\"\"

[networks.generator]
role = "generator"
inputs = { z = "latent", gray = "luma" }
input_shape = { z = ["B", 128], gray = ["B", 1, 128, 128] }
output_shape = ["B", 3, 128, 128]
hndl = "examples/networks/colorization-generator-128.hndl" """,
    },
    "hndl": {
        "label": "HNDL Architecture",
        "lang": "hndl",
        "filename": "colorization-generator-128.hndl",
        "code": """# 1. Multi-scale residual grayscale U-Net encoder (128 -> 64 -> 32 -> 16 -> 8)
e128 = resblock(gray, 16, norm="group_norm", groups=4, name="enc128")
e64  = resblock(e128, 32, stride=2, norm="group_norm", groups=8, name="enc64")
e32  = resblock(e64,  64, stride=2, norm="group_norm", groups=8, name="enc32")
e16  = resblock(e32, 128, stride=2, norm="group_norm", groups=8, name="enc16")
e8   = resblock(e16, 256, stride=2, norm="group_norm", groups=8, name="enc8")

# 2. RoPE + SwiGLU Transformer bottleneck with particle palette cross-attention
palette_tokens = reshape(linear(z, 2048, name="palette_tokens"), 8, 256)
tokens = permute(reshape(e8, 256, 64), 2, 1)
for _ in range(2):
    tokens = transformer_block(tokens, 4, mlp_ratio=2, norm="layer_norm", rope=True, ffn_kind="swiglu", name="bottleneck_block")
    attended = cross_attention(layer_norm(tokens, name="xattn_norm"), palette_tokens, 4, name="palette_xattn")
    tokens = add(tokens, attended)

# 3. Sub-pixel U-Net skip decoder with shape-inferred adaptive_norm
h16 = pixel_shuffle(conv(reshape(permute(layer_norm(tokens, name="bottleneck_norm"), 2, 1), 256, 8, 8), 512, kernel_size=3, padding=1, name="up16_conv"), 2)
h16 = adaptive_norm(resblock(concat(h16, e16), 128, norm="group_norm", groups=8, name="dec16"), linear(z, name="style16"), name="adain16")""",
    },
}

QUICKSTART_SNIPPETS: dict[str, dict[str, str]] = {
    "install_cuda": {
        "lang": "sh",
        "code": """python -m venv .venv
. .venv/bin/activate
python -m pip install "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
python -m pip install "hypergan[all]=={version}"
hypergan --help
python -m hypergan --version""",
    },
    "install_cpu": {
        "lang": "sh",
        "code": """python -m venv .venv
. .venv/bin/activate
python -m pip install "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "hypergan[all]=={version}"
hypergan --help
python -m hypergan --version""",
    },
    "install_source": {
        "lang": "sh",
        "code": """git clone https://github.com/HyperGAN/HyperGAN.git
cd HyperGAN
python -m venv .venv
. .venv/bin/activate
python -m pip install "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[all]"
hypergan --help""",
    },
    "scaffold": {
        "lang": "sh",
        "code": """# Explore starter templates and packaged wheel recipes
hypergan new --list
hypergan recipes

# Auto-size a project from an image directory (reads headers only)
hypergan new --project my_gan --dataset ./images

# Or start from a fast CPU 2D reference fixture
hypergan new --project my_toy --template gaussian-grid --device cpu""",
    },
    "validate_train": {
        "lang": "sh",
        "code": """# Static validation (no PyTorch import)
hypergan validate my_gan/configs/dcgan.toml
hypergan validate my_gan/configs/dcgan.toml -s train.batch_size=32 --json

# Launch training with live telemetry
hypergan train my_gan/configs/dcgan.toml --steps 1000
hypergan train my_toy/configs/gaussian-grid.toml --steps 20 -d cpu --no-server""",
    },
    "manage_runs": {
        "lang": "sh",
        "code": """# Inspect active and historical runs from any directory
hypergan ps -a
hypergan status
hypergan logs -f

# Open the Web Observatory (default 127.0.0.1:8765)
hypergan view --open

# Tag, checkpoint, stop, and resume
hypergan tag fleet-hawk baseline-dcgan
hypergan checkpoint baseline-dcgan
hypergan stop baseline-dcgan
hypergan resume baseline-dcgan --steps 2000""",
    },
    "sample_export": {
        "lang": "sh",
        "code": """# Draw fresh samples from the EMA generator
hypergan sample baseline-dcgan --count 16 --seed 42

# Condition the 128x128 colorizer on a single grayscale input
hypergan sample colorization-128 --input docs/assets/hg-bw-128.png --count 16 --output hg-grid.png

# Run configured evaluations (FID, diversity, color moments)
hypergan evaluate baseline-dcgan

# Export standalone generator bundle (+ SHA-256 digest)
hypergan export baseline-dcgan --checkpoint latest -o exported_model.pt""",
    },
    "python_quickstart": {
        "lang": "python",
        "code": """import hypergan as hg

def main():
    info = hg.new(project="my_gan", template="dcgan", data_dir="./images", device="cpu")
    model = hg.load(info["recipe_path"])
    model = hg.override(model, {"train.steps": 100, "train.batch_size": 16})
    warnings = hg.validate(model)

    run = hg.train(model, "my_gan/runs/exp1", steps=100)
    st = hg.status("exp1")
    hg.tag("exp1", "my-best-gan")

    sample = hg.sample("my-best-gan", count=8, seed=42)
    bundle_path = hg.export("my-best-gan", checkpoint="latest")
    generator = hg.load_bundle(bundle_path, device="cuda")
    walk = generator.interpolate(seed_a=100, seed_b=200, steps=10, mode="slerp")

if __name__ == "__main__":
    main()""",
    },
}


def _load_sibling_module(name: str) -> ModuleType:
    """Import a sibling module inside ``site/`` without colliding with stdlib ``site``."""
    if str(SITE_DIR) not in sys.path:
        sys.path.insert(0, str(SITE_DIR))
    return importlib.import_module(name)


def normalize_output_path(path: str) -> str:
    """Normalize a site-relative output path into a clean POSIX HTML file path."""
    cleaned = path.replace("\\", "/").lstrip("/")
    if not cleaned:
        return "index.html"
    if cleaned.endswith("/"):
        cleaned = f"{cleaned}index.html"
    elif not cleaned.endswith(".html"):
        cleaned = f"{cleaned}/index.html"
    return posixpath.normpath(cleaned)


def root_prefix_for(output_path: str) -> str:
    """Return the relative prefix (``''``, ``'../'``, ``'../../'``, ...) from ``output_path`` to site root."""
    norm = normalize_output_path(output_path)
    parent = PurePosixPath(norm).parent
    if str(parent) in ("", "."):
        return ""
    return "../" * len(parent.parts)


def slugify_id(text: str) -> str:
    """Convert a symbol/section/command name into a deterministic HTML id."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "item"


def _strip_whitespace(text: str, limit: int = 320) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _extract_group_items(raw_group: dict[str, Any]) -> list[dict[str, Any]]:
    """Support doc_groups entries that carry items under 'items', 'pages', or 'docs'."""
    for key in ("items", "pages", "docs"):
        val = raw_group.get(key)
        if isinstance(val, list):
            return val
    return []


def _build_navigation(
    docs_list: list[dict[str, Any]],
    raw_doc_groups: list[dict[str, Any]],
    counts: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble the 7 canonical sidebar groups + ordered flat doc sequence for prev/next links."""
    docs_by_slug = {d["slug"]: d for d in docs_list}
    docs_by_out = {d["output_path"]: d for d in docs_list}

    # Build normalized groups from introspect.doc_groups (or derive from docs if absent).
    group_map: dict[str, dict[str, Any]] = {}
    ordered_group_ids: list[str] = []

    if raw_doc_groups:
        for grp in raw_doc_groups:
            gid = str(grp.get("id") or slugify_id(grp.get("title", "docs")))
            title = str(grp.get("title") or gid.replace("-", " ").title())
            entry = group_map.setdefault(gid, {"id": gid, "title": title, "items": []})
            if gid not in ordered_group_ids:
                ordered_group_ids.append(gid)
            for item in _extract_group_items(grp):
                if isinstance(item, str):
                    doc = docs_by_slug.get(item)
                    if doc:
                        entry["items"].append(
                            {
                                "title": doc["title"],
                                "url": doc["output_path"],
                                "slug": doc["slug"],
                                "badge": "",
                            }
                        )
                elif isinstance(item, dict):
                    raw_url = item.get("output_path") or item.get("url") or ""
                    slug = item.get("slug", "")
                    if not raw_url and slug in docs_by_slug:
                        raw_url = docs_by_slug[slug]["output_path"]
                    if not raw_url:
                        continue
                    url = normalize_output_path(str(raw_url))
                    doc = docs_by_out.get(url) or docs_by_slug.get(slug)
                    entry["items"].append(
                        {
                            "title": str(item.get("title") or (doc["title"] if doc else slug)),
                            "url": url,
                            "slug": slug or (doc["slug"] if doc else slugify_id(url)),
                            "badge": str(item.get("badge") or ""),
                        }
                    )

    # Ensure every doc in docs_list is placed in its group (no orphans).
    seen_urls: set[str] = {
        item["url"] for g in group_map.values() for item in g["items"]
    }
    for doc in docs_list:
        url = doc["output_path"]
        if url in seen_urls:
            continue
        gid = str(doc.get("group_id") or "guides")
        gtitle = str(doc.get("group_title") or gid.replace("-", " ").title())
        entry = group_map.setdefault(gid, {"id": gid, "title": gtitle, "items": []})
        if gid not in ordered_group_ids:
            ordered_group_ids.append(gid)
        entry["items"].append(
            {
                "title": doc["title"],
                "url": url,
                "slug": doc["slug"],
                "badge": "",
            }
        )
        seen_urls.add(url)

    def _find_group_id(candidates: tuple[str, ...], title_keywords: tuple[str, ...]) -> str | None:
        for cid in candidates:
            if cid in group_map:
                return cid
        for gid in ordered_group_ids:
            t = group_map[gid]["title"].lower()
            if any(k in t for k in title_keywords):
                return gid
        return None

    def _ensure_item(gid: str, default_title: str, item: dict[str, Any], *, prepend: bool = False) -> None:
        if gid not in group_map:
            group_map[gid] = {"id": gid, "title": default_title, "items": []}
            ordered_group_ids.append(gid)
        items = group_map[gid]["items"]
        if any(it["url"] == item["url"] for it in items):
            return
        if prepend:
            items.insert(0, item)
        else:
            items.append(item)

    # Inject bespoke top-level pages into their canonical IA groups (if not already present).
    gs_id = _find_group_id(("getting-started", "getting_started", "start"), ("getting started", "quickstart")) or "getting-started"
    _ensure_item(
        gs_id,
        "Getting Started",
        {"title": "Quickstart & Install", "url": "quickstart/index.html", "slug": "quickstart", "badge": "Start"},
        prepend=True,
    )

    ex_id = _find_group_id(("recipes-examples", "recipes", "examples"), ("recipe", "example")) or "recipes-examples"
    _ensure_item(
        ex_id,
        "Recipes & Examples",
        {
            "title": "Examples & Recipes Gallery",
            "url": "examples/index.html",
            "slug": "examples",
            "badge": str(counts.get("examples", "")) if counts.get("examples") else "",
        },
        prepend=True,
    )

    ref_id = _find_group_id(("reference", "auto-reference", "api-reference"), ("reference",)) or "reference"
    ref_items = [
        {"title": "Python API", "url": "reference/api/index.html", "slug": "reference-api", "badge": str(counts.get("api", ""))},
        {"title": "CLI Reference", "url": "reference/cli/index.html", "slug": "reference-cli", "badge": str(counts.get("cli", ""))},
        {"title": "TOML Model Schema", "url": "reference/schema/index.html", "slug": "reference-schema", "badge": str(counts.get("schema_sections", ""))},
        {"title": "HNDL Operators", "url": "reference/ops/index.html", "slug": "reference-ops", "badge": str(counts.get("ops", ""))},
    ]
    for r_item in ref_items:
        _ensure_item(ref_id, "Reference", r_item)

    nav_groups = [group_map[gid] for gid in ordered_group_ids if group_map[gid]["items"]]

    # Flatten for prev/next pagination across docs + quickstart.
    flat_reading: list[dict[str, Any]] = []
    seen_flat: set[str] = set()
    for grp in nav_groups:
        for item in grp["items"]:
            if item["url"] not in seen_flat:
                flat_reading.append(item)
                seen_flat.add(item["url"])

    return nav_groups, flat_reading


def _build_search_index(
    data: dict[str, Any],
    rendered_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact client-side search-index.json entries covering all site surfaces."""
    entries: list[dict[str, Any]] = []

    # 1. Core landing & quickstart pages
    entries.append(
        {
            "kind": "doc",
            "title": "Quickstart & Install",
            "url": "quickstart/index.html",
            "section": "Getting Started",
            "summary": "Install HyperGAN with CUDA 13.0 or CPU PyTorch, scaffold a project, validate TOML, train, monitor in the Web Observatory, and export.",
            "tags": ["install", "quickstart", "pypi", "cuda", "cpu", "scaffold", "train", "export"],
            "headings": [
                "Install",
                "60-Second Quickstart",
                "Live Run Management",
                "Sampling, Evaluation & Standalone Export",
                "Python API Quickstart",
            ],
            "text": "pip install hypergan[all] torch torchvision cu130 hypergan new validate train status view sample evaluate export",
        }
    )

    # 2. Rendered markdown docs (docs/*.md, GLOSSARY, CHANGELOG)
    for doc in rendered_docs:
        headings = [h["title"] for h in doc.get("toc", [])][:18]
        entries.append(
            {
                "kind": "doc",
                "title": doc["title"],
                "url": doc["output_path"],
                "section": doc.get("group_title") or "Documentation",
                "summary": _strip_whitespace(doc.get("description") or "", 180),
                "tags": [doc["slug"], doc.get("group_id", "")],
                "headings": headings,
                "text": _strip_whitespace(doc.get("plain_text") or "", 420),
            }
        )

    # 3. Python API symbols
    for grp in data.get("api", {}).get("groups", []):
        gtitle = grp.get("title", "Python API")
        for item in grp.get("items", []):
            name = item["name"]
            anchor = f"api-{slugify_id(name)}"
            aliases = list(item.get("aliases") or [])
            field_names = [f.get("name", "") for f in (item.get("fields") or [])]
            method_names = [m.get("name", "") for m in (item.get("methods") or [])]
            entries.append(
                {
                    "kind": "api",
                    "title": f"hg.{name}",
                    "url": f"reference/api/index.html#{anchor}",
                    "section": gtitle,
                    "summary": _strip_whitespace(item.get("summary") or item.get("signature") or "", 180),
                    "tags": [item.get("kind", "function"), name, *aliases],
                    "headings": [*field_names[:8], *method_names[:8]],
                    "text": _strip_whitespace(
                        f"{item.get('signature', '')} {item.get('doc', '')}", 260
                    ),
                }
            )

    # 4. CLI commands & nested subcommands
    def _add_cli_entry(cmd: dict[str, Any], gtitle: str) -> None:
        full_cmd = cmd.get("full_command") or f"hypergan {cmd.get('name', '')}"
        anchor = f"cli-{slugify_id(full_cmd)}"
        opt_flags = [o.get("flags", "") for o in (cmd.get("options") or [])]
        entries.append(
            {
                "kind": "cli",
                "title": full_cmd,
                "url": f"reference/cli/index.html#{anchor}",
                "section": gtitle,
                "summary": _strip_whitespace(cmd.get("summary") or cmd.get("description") or "", 180),
                "tags": [cmd.get("name", ""), *(cmd.get("aliases") or [])],
                "headings": opt_flags[:10],
                "text": _strip_whitespace(
                    f"{cmd.get('usage', '')} {cmd.get('description', '')} {' '.join(opt_flags)}",
                    260,
                ),
            }
        )
        for sub in cmd.get("subcommands") or []:
            _add_cli_entry(sub, gtitle)

    for grp in data.get("cli", {}).get("groups", []):
        gtitle = grp.get("title", "CLI")
        for cmd in grp.get("commands", []):
            _add_cli_entry(cmd, gtitle)

    # 5. TOML Schema top-level & sections
    schema_data = data.get("schema", {})
    for field in schema_data.get("top_level", []):
        fname = field["name"]
        entries.append(
            {
                "kind": "schema",
                "title": fname,
                "url": f"reference/schema/index.html#schema-top-{slugify_id(fname)}",
                "section": "Top-Level Fields",
                "summary": _strip_whitespace(field.get("description") or "", 180),
                "tags": ["toml", "hypergan-model/1", fname, str(field.get("type", ""))],
                "headings": [],
                "text": _strip_whitespace(
                    f"{fname} {field.get('type', '')} {field.get('default', '')} {field.get('description', '')}",
                    220,
                ),
            }
        )
    for sec in schema_data.get("sections", []):
        sec_id = f"schema-{slugify_id(sec.get('id') or sec.get('name', 'section'))}"
        field_names = [f.get("name", "") for f in sec.get("fields", [])]
        entries.append(
            {
                "kind": "schema",
                "title": sec.get("toml_header") or f"[{sec.get('name', '')}]",
                "url": f"reference/schema/index.html#{sec_id}",
                "section": "TOML Schema",
                "summary": _strip_whitespace(sec.get("description") or "", 180),
                "tags": ["toml", sec.get("name", ""), *field_names[:8]],
                "headings": field_names[:14],
                "text": _strip_whitespace(
                    f"{sec.get('description', '')} {' '.join(field_names)}", 260
                ),
            }
        )

    # 6. HNDL Operators
    for cat in data.get("ops", {}).get("categories", []):
        cat_name = cat.get("name", "HNDL")
        for op in cat.get("ops", []):
            op_name = op["name"]
            arg_names = [a.get("name", "") for a in (op.get("args") or [])]
            entries.append(
                {
                    "kind": "op",
                    "title": op_name,
                    "url": f"reference/ops/index.html#op-{slugify_id(op_name)}",
                    "section": f"HNDL · {cat_name}",
                    "summary": _strip_whitespace(op.get("summary") or op.get("description") or "", 180),
                    "tags": ["hndl", cat_name.lower(), op_name, *arg_names[:6]],
                    "headings": arg_names[:10],
                    "text": _strip_whitespace(
                        f"{op_name} {op.get('description', '')} {' '.join(arg_names)}", 240
                    ),
                }
            )

    # 7. Examples & Packaged Recipes
    examples_data = data.get("examples", {})
    for bucket_key in (
        "packaged_recipes",
        "toml_examples",
        "python_examples",
        "hndl_networks",
        "execution_profiles",
        "starter_templates",
    ):
        for ex in examples_data.get(bucket_key, []):
            ex_id = slugify_id(ex.get("id") or ex.get("name") or ex.get("title") or "example")
            entries.append(
                {
                    "kind": "example",
                    "title": ex.get("title") or ex.get("name") or ex_id,
                    "url": f"examples/index.html#example-{ex_id}",
                    "section": ex.get("category_label") or bucket_key.replace("_", " ").title(),
                    "summary": _strip_whitespace(ex.get("summary") or ex.get("description") or "", 180),
                    "tags": [*(ex.get("tags") or []), ex.get("category", bucket_key)],
                    "headings": [],
                    "text": _strip_whitespace(
                        f"{ex.get('path', '')} {ex.get('run_command', '')} {ex.get('summary', '')}",
                        240,
                    ),
                }
            )

    return entries


def _copy_static_assets(repo_root: Path, out_dir: Path, pygments_css_text: str) -> None:
    """Copy 255labs tokens/fonts/logos, docs/assets, site CSS/JS, pygments CSS, llms.txt, and .nojekyll."""
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Official 255labs web assets from src/hypergan/web_assets/
    web_assets_src = repo_root / "src" / "hypergan" / "web_assets"
    for filename in WEB_ASSET_FILES:
        src = web_assets_src / filename
        if not src.is_file():
            raise FileNotFoundError(f"Expected 255labs web asset not found: {src}")
        shutil.copy2(src, assets_dir / filename)

    # 2. Repository docs/assets/* (lockups, colorizer banner, logos, grid)
    docs_assets_src = repo_root / "docs" / "assets"
    if docs_assets_src.is_dir():
        for src in sorted(docs_assets_src.iterdir()):
            if src.is_file():
                shutil.copy2(src, assets_dir / src.name)

    # 3. Site static CSS & JS
    shutil.copy2(STATIC_DIR / "site.css", assets_dir / "site.css")
    shutil.copy2(STATIC_DIR / "site.js", assets_dir / "site.js")

    # 4. Generated Pygments CSS
    (assets_dir / "pygments.css").write_text(pygments_css_text, encoding="utf-8")

    # 5. llms.txt & .nojekyll at site root
    llms_src = repo_root / "llms.txt"
    if llms_src.is_file():
        shutil.copy2(llms_src, out_dir / "llms.txt")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


def _write_html(out_dir: Path, output_path: str, html: str) -> None:
    dest = out_dir / normalize_output_path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")


def _prepare_examples_gallery(
    examples_data: dict[str, Any],
    highlight_code: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten examples buckets into gallery cards with guaranteed syntax-highlighted HTML."""
    category_specs = [
        ("packaged_recipes", "packaged", "Packaged Recipes"),
        ("toml_examples", "toml", "TOML Configs"),
        ("python_examples", "python", "Python API"),
        ("hndl_networks", "hndl", "HNDL Networks"),
        ("execution_profiles", "execution", "Execution Profiles"),
        ("starter_templates", "templates", "Starter Templates"),
    ]
    items: list[dict[str, Any]] = []
    category_counts: list[dict[str, Any]] = []

    for bucket_key, cat_slug, cat_label in category_specs:
        bucket = examples_data.get(bucket_key) or []
        category_counts.append(
            {"id": cat_slug, "label": cat_label, "count": len(bucket)}
        )
        for raw in bucket:
            ex = dict(raw)
            ex["anchor_id"] = f"example-{slugify_id(ex.get('id') or ex.get('name') or ex.get('title') or 'item')}"
            ex["filter_category"] = cat_slug
            ex["category_label"] = ex.get("category_label") or cat_label
            ex["title"] = ex.get("title") or ex.get("name") or ex["anchor_id"]
            ex["summary"] = ex.get("summary") or ex.get("description") or ""
            ex["tags"] = list(ex.get("tags") or [])
            ex["run_command"] = ex.get("run_command") or ""
            ex["language"] = ex.get("language") or (
                "python" if cat_slug == "python" else "hndl" if cat_slug == "hndl" else "toml"
            )
            code = ex.get("code") or ""
            if code and not ex.get("code_html"):
                ex["code_html"] = highlight_code(code, ex["language"])
            items.append(ex)

    return items, category_counts


def build_site(
    repo_root: Path,
    out_dir: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Build the complete HyperGAN documentation site into ``out_dir``."""
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).resolve()

    introspect = _load_sibling_module("introspect")
    markdown_render = _load_sibling_module("markdown_render")
    verify_mod = _load_sibling_module("verify") if strict else None

    data = introspect.collect_site_data(repo_root)
    meta = dict(data.get("meta") or {})
    meta.setdefault("version", "2.1.1")
    meta.setdefault("commit", "master")
    meta.setdefault("site_url", "https://hypergan.github.io/HyperGAN/")
    meta.setdefault("repo_url", "https://github.com/HyperGAN/HyperGAN")
    meta.setdefault("ops_mode", data.get("ops", {}).get("mode", "fallback"))
    counts = dict(meta.get("counts") or {})
    counts.setdefault("api", data.get("api", {}).get("total", 0))
    counts.setdefault("cli", data.get("cli", {}).get("total", 0))
    counts.setdefault("schema_sections", data.get("schema", {}).get("total_sections", 0))
    counts.setdefault("schema_fields", data.get("schema", {}).get("total_fields", 0))
    counts.setdefault("ops", data.get("ops", {}).get("total", 0))
    counts.setdefault("docs", len(data.get("docs") or []))
    meta["counts"] = counts

    # Normalize doc entries & assemble complete route_map for Markdown relative link rewriting.
    docs_list: list[dict[str, Any]] = []
    route_map: dict[str, str] = {
        "README.md": "quickstart/index.html",
        "GLOSSARY.md": "glossary/index.html",
        "CHANGELOG.md": "changelog/index.html",
        "llms.txt": "llms.txt",
        "examples/README.md": "examples/index.html",
    }

    docs_assets_dir = repo_root / "docs" / "assets"
    if docs_assets_dir.is_dir():
        for asset in docs_assets_dir.iterdir():
            if asset.is_file():
                route_map[f"docs/assets/{asset.name}"] = f"assets/{asset.name}"

    for raw_doc in data.get("docs") or []:
        doc = dict(raw_doc)
        doc["output_path"] = normalize_output_path(doc.get("output_path") or f"docs/{doc['slug']}/index.html")
        src_path = str(doc.get("source_path") or "").replace("\\", "/").lstrip("./")
        doc["source_path"] = src_path
        if src_path:
            route_map[src_path] = doc["output_path"]
        docs_list.append(doc)

    # Render every Markdown doc using markdown_render.render_markdown().
    rendered_docs: list[dict[str, Any]] = []
    doc_slug_to_url: dict[str, str] = {}
    for doc in docs_list:
        rendered = markdown_render.render_markdown(
            doc.get("raw_markdown", ""),
            source_path=doc["source_path"],
            page_url_path=doc["output_path"],
            route_map=route_map,
            repo_url=meta["repo_url"],
            commit=meta.get("commit", "master"),
        )
        doc["html"] = rendered.get("html", "")
        doc["toc"] = rendered.get("toc") or []
        doc["plain_text"] = rendered.get("plain_text", "")
        if not doc.get("title") and rendered.get("title"):
            doc["title"] = rendered["title"]
        if not doc.get("description") and rendered.get("description"):
            doc["description"] = rendered["description"]
        doc_slug_to_url[doc["slug"]] = doc["output_path"]
        rendered_docs.append(doc)

    # Build sidebar navigation & reading order for prev/next pagination.
    nav_groups, reading_order = _build_navigation(
        rendered_docs,
        data.get("doc_groups") or [],
        counts,
    )
    reading_index = {item["url"]: idx for idx, item in enumerate(reading_order)}

    # Prepare clean output directory & copy static assets.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_static_assets(repo_root, out_dir, markdown_render.pygments_css())

    # Configure Jinja2 environment.
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["slugify"] = slugify_id

    common_ctx: dict[str, Any] = {
        "meta": meta,
        "nav_groups": nav_groups,
        "doc_slug_to_url": doc_slug_to_url,
        "has_llms_txt": (out_dir / "llms.txt").is_file(),
    }

    def _render_page(
        template_name: str,
        output_path: str,
        *,
        page_title: str,
        page_description: str,
        active_nav: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        norm_out = normalize_output_path(output_path)
        prefix = root_prefix_for(norm_out)
        ctx = {
            **common_ctx,
            "root_prefix": prefix,
            "current_url": norm_out,
            "page_title": page_title,
            "page_description": page_description,
            "active_nav": active_nav,
        }
        if extra:
            ctx.update(extra)
        html = env.get_template(template_name).render(**ctx)
        _write_html(out_dir, norm_out, html)

    # 1. Landing Page -> index.html
    landing_tabs = []
    for tab_id, spec in LANDING_SNIPPETS.items():
        landing_tabs.append(
            {
                "id": tab_id,
                "label": spec["label"],
                "lang": spec["lang"],
                "filename": spec["filename"],
                "code": spec["code"],
                "code_html": markdown_render.highlight_code(spec["code"], spec["lang"]),
            }
        )

    packaged_recipes = list((data.get("examples") or {}).get("packaged_recipes") or [])
    for rec in packaged_recipes:
        rec["anchor_id"] = f"example-{slugify_id(rec.get('id') or rec.get('title') or 'recipe')}"

    _render_page(
        "landing.html",
        "index.html",
        page_title=f"HyperGAN v{meta['version']} — Train generative models. Follow their progress. Export the result.",
        page_description=(
            "HyperGAN trains generative models from declarative hypergan-model/1 TOML files "
            "using HNDL networks, ParticleGAN unified_v11, a real-time Web Observatory, and standalone export."
        ),
        active_nav="home",
        extra={
            "layout_mode": "landing",
            "landing_tabs": landing_tabs,
            "packaged_recipes": packaged_recipes,
        },
    )

    # 2. Quickstart -> quickstart/index.html
    qs_blocks: dict[str, dict[str, str]] = {}
    for key, spec in QUICKSTART_SNIPPETS.items():
        code_text = spec["code"].replace("{version}", str(meta["version"]))
        qs_blocks[key] = {
            "lang": spec["lang"],
            "code": code_text,
            "code_html": markdown_render.highlight_code(code_text, spec["lang"]),
        }
    qs_toc = [
        {"level": 2, "id": "install", "title": "Install"},
        {"level": 2, "id": "sixty-second-quickstart", "title": "60-Second Quickstart"},
        {"level": 2, "id": "validate-and-train", "title": "Validate & Train"},
        {"level": 2, "id": "live-run-management", "title": "Live Run Management & Observatory"},
        {"level": 2, "id": "sampling-evaluation--standalone-export", "title": "Sampling, Evaluation & Export"},
        {"level": 2, "id": "which-run-rm-gc-stop-and-tag-act-on", "title": "Strict Run Targeting"},
        {"level": 2, "id": "python-api-quickstart", "title": "Python API Quickstart"},
    ]
    qs_idx = reading_index.get("quickstart/index.html")
    qs_next = (
        reading_order[qs_idx + 1]
        if qs_idx is not None and qs_idx + 1 < len(reading_order)
        else None
    )
    _render_page(
        "quickstart.html",
        "quickstart/index.html",
        page_title=f"Quickstart & Install — HyperGAN v{meta['version']}",
        page_description="Install HyperGAN 2.1.1 with CUDA 13.0 or CPU PyTorch, scaffold a project, train, monitor in the Web Observatory, and export standalone generators.",
        active_nav="quickstart",
        extra={
            "layout_mode": "docs",
            "qs_blocks": qs_blocks,
            "toc": qs_toc,
            "prev_page": None,
            "next_page": qs_next,
        },
    )

    # 3. All Documentation Pages (49 docs/*.md + GLOSSARY + CHANGELOG)
    for doc in rendered_docs:
        idx = reading_index.get(doc["output_path"])
        prev_page = reading_order[idx - 1] if idx is not None and idx > 0 else None
        next_page = (
            reading_order[idx + 1]
            if idx is not None and idx + 1 < len(reading_order)
            else None
        )
        _render_page(
            "doc.html",
            doc["output_path"],
            page_title=f"{doc['title']} — HyperGAN Docs",
            page_description=doc.get("description") or f"{doc['title']} documentation for HyperGAN.",
            active_nav="docs",
            extra={
                "layout_mode": "docs",
                "doc": doc,
                "toc": doc.get("toc") or [],
                "prev_page": prev_page,
                "next_page": next_page,
            },
        )

    # 4. Python API Reference -> reference/api/index.html
    api_data = data.get("api") or {"total": 0, "groups": []}
    kind_counts: dict[str, int] = {}
    for grp in api_data.get("groups", []):
        for item in grp.get("items", []):
            k = item.get("kind", "function")
            kind_counts[k] = kind_counts.get(k, 0) + 1
            item["anchor_id"] = f"api-{slugify_id(item['name'])}"
    _render_page(
        "api_index.html",
        "reference/api/index.html",
        page_title=f"Python API Reference ({api_data.get('total', 0)} Symbols) — HyperGAN",
        page_description="Complete auto-generated Python API reference for import hypergan as hg: functions, dataclasses, RunHandle, LoadedGenerator, and exceptions.",
        active_nav="api",
        extra={
            "layout_mode": "reference",
            "api": api_data,
            "kind_counts": kind_counts,
        },
    )

    # 5. CLI Reference -> reference/cli/index.html
    cli_data = data.get("cli") or {"total": 0, "global_flags": [], "groups": []}
    def _stamp_cli_ids(cmd: dict[str, Any]) -> None:
        full = cmd.get("full_command") or f"hypergan {cmd.get('name', '')}"
        cmd["anchor_id"] = f"cli-{slugify_id(full)}"
        for sub in cmd.get("subcommands") or []:
            _stamp_cli_ids(sub)
    for grp in cli_data.get("groups", []):
        for cmd in grp.get("commands", []):
            _stamp_cli_ids(cmd)
    _render_page(
        "cli_reference.html",
        "reference/cli/index.html",
        page_title=f"CLI Reference ({cli_data.get('total', 0)} Commands) — HyperGAN",
        page_description="Complete auto-generated CLI reference for hypergan: scaffolding, validation, training, run control, sampling, evaluation, export, queue, and Web Observatory.",
        active_nav="cli",
        extra={
            "layout_mode": "reference",
            "cli": cli_data,
        },
    )

    # 6. TOML Model Schema Reference -> reference/schema/index.html
    schema_data = data.get("schema") or {"total_sections": 0, "total_fields": 0, "top_level": [], "sections": []}
    for field in schema_data.get("top_level", []):
        field["anchor_id"] = f"schema-top-{slugify_id(field['name'])}"
    for sec in schema_data.get("sections", []):
        sec["anchor_id"] = f"schema-{slugify_id(sec.get('id') or sec.get('name', 'section'))}"
    _render_page(
        "schema_reference.html",
        "reference/schema/index.html",
        page_title=f"TOML Schema Reference (hypergan-model/1) — HyperGAN",
        page_description="Interactive hypergan-model/1 TOML schema reference covering every section, field, type, default, constraint, and allowed value.",
        active_nav="schema",
        extra={
            "layout_mode": "reference",
            "schema": schema_data,
        },
    )

    # 7. HNDL Operators Reference -> reference/ops/index.html
    ops_data = data.get("ops") or {"total": 0, "mode": "fallback", "categories": []}
    for cat in ops_data.get("categories", []):
        cat["anchor_id"] = f"ops-cat-{slugify_id(cat.get('id') or cat.get('name', 'cat'))}"
        for op in cat.get("ops", []):
            op["anchor_id"] = f"op-{slugify_id(op['name'])}"
    _render_page(
        "ops_reference.html",
        "reference/ops/index.html",
        page_title=f"HNDL Operators Reference ({ops_data.get('total', 0)} Ops) — HyperGAN",
        page_description="Complete filterable catalog of HNDL neural network operators with ports, categories, and argument signatures.",
        active_nav="ops",
        extra={
            "layout_mode": "reference",
            "ops": ops_data,
        },
    )

    # 8. Examples & Packaged Recipes Gallery -> examples/index.html
    gallery_items, gallery_categories = _prepare_examples_gallery(
        data.get("examples") or {},
        markdown_render.highlight_code,
    )
    _render_page(
        "examples.html",
        "examples/index.html",
        page_title="Examples & Packaged Recipes — HyperGAN",
        page_description="Filterable gallery of HyperGAN packaged wheel recipes, TOML model configs, Python API scripts, HNDL networks, execution profiles, and starter templates.",
        active_nav="examples",
        extra={
            "layout_mode": "reference",
            "gallery_items": gallery_items,
            "gallery_categories": gallery_categories,
        },
    )

    # 9. 404 Page -> 404.html
    _render_page(
        "404.html",
        "404.html",
        page_title="Page Not Found — HyperGAN",
        page_description="The requested HyperGAN documentation page could not be found.",
        active_nav="",
        extra={"layout_mode": "minimal"},
    )

    # 10. Search Index -> search-index.json
    search_entries = _build_search_index(data, rendered_docs)
    (out_dir / "search-index.json").write_text(
        json.dumps(search_entries, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    summary = {
        "out_dir": str(out_dir),
        "version": meta["version"],
        "ops_mode": meta["ops_mode"],
        "docs_count": len(rendered_docs),
        "api_count": api_data.get("total", 0),
        "cli_count": cli_data.get("total", 0),
        "schema_sections": schema_data.get("total_sections", 0),
        "schema_fields": schema_data.get("total_fields", 0),
        "ops_count": ops_data.get("total", 0),
        "examples_count": len(gallery_items),
        "search_entries": len(search_entries),
    }

    # 11. Strict post-build verification
    if strict and verify_mod is not None:
        errors = verify_mod.verify_site(out_dir, strict=True)
        if errors:
            formatted = "\n".join(f"  - {err}" for err in errors[:50])
            more = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
            raise RuntimeError(
                f"Site verification failed with {len(errors)} error(s):\n{formatted}{more}"
            )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="site/build.py",
        description="Build the HyperGAN 255labs-branded static documentation site.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("_site"),
        help="Output directory for the built static site (default: _site).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=SITE_DIR.parent,
        help="Path to the HyperGAN repository root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run site/verify.py strict HTML link, anchor, and asset checks after building.",
    )
    args = parser.parse_args(argv)

    summary = build_site(args.repo_root, args.out, strict=args.strict)
    print(
        f"Built HyperGAN v{summary['version']} site -> {summary['out_dir']} "
        f"(docs={summary['docs_count']}, api={summary['api_count']}, cli={summary['cli_count']}, "
        f"schema={summary['schema_sections']}/{summary['schema_fields']}, "
        f"ops={summary['ops_count']} [{summary['ops_mode']}], "
        f"examples={summary['examples_count']}, search={summary['search_entries']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
