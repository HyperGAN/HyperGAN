"""Unit, introspection, renderer, verifier, and E2E tests for the HyperGAN docs site."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = REPO_ROOT / "site"
SRC_DIR = REPO_ROOT / "src"

for _p in (str(SITE_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import introspect  # noqa: E402
import markdown_render  # noqa: E402
import verify  # noqa: E402


def _has_live_hndl() -> bool:
    try:
        import torch  # noqa: F401
        import hndl  # noqa: F401
        return True
    except Exception:
        return False


class IntrospectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = introspect.collect_site_data(REPO_ROOT)

    def test_top_level_contract_keys(self):
        expected_keys = {"meta", "api", "cli", "schema", "ops", "examples", "docs", "doc_groups"}
        self.assertTrue(expected_keys.issubset(self.data.keys()))
        meta = self.data["meta"]
        self.assertEqual(meta["site_url"], "https://hypergan.github.io/HyperGAN/")
        self.assertEqual(meta["repo_url"], "https://github.com/HyperGAN/HyperGAN")
        self.assertIn(meta["ops_mode"], ("live", "fallback"))
        counts = meta["counts"]
        for k in ("api", "cli", "schema_sections", "schema_fields", "ops", "examples", "docs"):
            self.assertGreater(counts[k], 0, f"expected positive meta.counts[{k!r}]")

    def test_api_introspects_all_129_public_symbols(self):
        import hypergan

        api = self.data["api"]
        expected_all = set(hypergan.__all__)
        self.assertEqual(api["total"], len(expected_all))
        self.assertEqual(api["total"], 129)

        seen = {}
        for grp in api["groups"]:
            self.assertTrue(grp["id"])
            self.assertTrue(grp["title"])
            for item in grp["items"]:
                seen[item["name"]] = item
                self.assertIn(item["kind"], ("function", "dataclass", "class", "exception", "module"))
                self.assertIsInstance(item["aliases"], list)
                self.assertIsInstance(item["fields"], list)
                self.assertIsInstance(item["methods"], list)
                self.assertTrue(item["summary"], f"missing summary for API symbol {item['name']}")
                self.assertTrue(item["source_file"], f"missing source_file for API symbol {item['name']}")
                self.assertGreater(item["source_line"], 0)
                self.assertIn("#L", item["source_url"])

        self.assertEqual(set(seen.keys()), expected_all)

        # Compatibility aliases attached to canonical entries, not duplicated
        self.assertIn("validate_config", seen["validate"]["aliases"])
        self.assertIn("runs", seen["list_runs"]["aliases"])
        self.assertIn("runs_ls", seen["list_runs"]["aliases"])
        self.assertIn("prune", seen["gc"]["aliases"])
        self.assertNotIn("validate_config", seen)

        # Callable metrics module handled cleanly
        metrics_entry = seen["metrics"]
        self.assertEqual(metrics_entry["kind"], "module")
        self.assertIn("run", metrics_entry["signature"])

        # Dataclass fields extracted
        model_entry = seen["Model"]
        self.assertEqual(model_entry["kind"], "dataclass")
        self.assertGreater(len(model_entry["fields"]), 0)

        # LoadedGenerator present with methods and signature
        lg = seen["LoadedGenerator"]
        self.assertEqual(lg["kind"], "class")
        method_names = {m["name"] for m in lg["methods"]}
        self.assertTrue({"sample", "forward", "interpolate", "sample_latents"}.issubset(method_names))

    def test_loaded_generator_ast_fallback(self):
        lg_ast = introspect._introspect_loaded_generator_ast(
            REPO_ROOT, introspect.REPO_URL, "master", []
        )
        self.assertEqual(lg_ast["name"], "LoadedGenerator")
        self.assertEqual(lg_ast["kind"], "class")
        self.assertEqual(lg_ast["source_file"], "src/hypergan/artifacts.py")
        self.assertGreater(lg_ast["source_line"], 100)
        self.assertIn("bundle", lg_ast["signature"])
        method_names = {m["name"] for m in lg_ast["methods"]}
        self.assertTrue({"sample", "forward", "interpolate", "sample_latents", "to"}.issubset(method_names))

    def test_cli_introspects_commands_subcommands_and_global_flags(self):
        cli = self.data["cli"]
        self.assertEqual(cli["total"], 52)
        self.assertEqual(len(cli["groups"]), 6)

        global_flag_strs = " ".join(f["flags"] for f in cli["global_flags"])
        self.assertIn("--json", global_flag_strs)
        self.assertIn("--quiet", global_flag_strs)

        top_by_name = {}
        nested_count = 0
        for grp in cli["groups"]:
            for cmd in grp["commands"]:
                top_by_name[cmd["name"]] = cmd
                nested_count += len(cmd["subcommands"])
                # Global flags should not be duplicated in per-command options
                for opt in cmd["options"]:
                    self.assertNotIn("--json", opt["flags"])
                    self.assertNotIn("--quiet", opt["flags"])
                    self.assertNotIn("--experimental", opt["flags"])

        self.assertEqual(len(top_by_name), 42)
        self.assertEqual(nested_count, 10)

        # Alias deduplication
        self.assertNotIn("eval", top_by_name)
        self.assertIn("eval", top_by_name["evaluate"]["aliases"])

        # Fallback descriptions supplied for serve/server-status/stop-server
        for name in ("serve", "server-status", "stop-server"):
            self.assertTrue(top_by_name[name]["summary"], f"missing summary for CLI command {name}")

        # Exact help match check (e.g. ps must not collide with ops)
        self.assertIn("run", top_by_name["ps"]["summary"].lower())
        self.assertIn("hndl", top_by_name["ops"]["summary"].lower())

    def test_schema_introspects_18_sections_and_111_fields(self):
        schema = self.data["schema"]
        self.assertEqual(schema["total_sections"], 18)
        self.assertEqual(schema["total_fields"], 111)
        self.assertEqual(len(schema["top_level"]), 5)
        self.assertEqual(len(schema["sections"]), 13)

        sec_map = {s["name"]: s for s in schema["sections"]}
        # Ensure all 3 field keys (`fields`, `properties`, `items["fields"]`) worked
        self.assertEqual(len(sec_map["data"]["fields"]), 15)
        self.assertEqual(len(sec_map["networks"]["fields"]), 13)
        self.assertEqual(len(sec_map["losses"]["fields"]), 13)
        self.assertEqual(sec_map["losses"]["toml_header"], "[[losses]]")
        self.assertEqual(sec_map["networks"]["toml_header"], "[networks.<name>]")

    def test_hndl_ops_fallback_mode(self):
        fallback_data = introspect.collect_site_data(REPO_ROOT, _force_ops_fallback=True)
        ops = fallback_data["ops"]
        self.assertEqual(ops["mode"], "fallback")
        self.assertEqual(len(ops["categories"]), 12)
        self.assertGreaterEqual(ops["total"], 85)

        all_ops = {op["name"]: op for cat in ops["categories"] for op in cat["ops"]}
        self.assertIn("diff_augment", all_ops)
        self.assertIn("conv", all_ops)
        self.assertIn("attention", all_ops)
        da = all_ops["diff_augment"]
        da_arg_names = {a["name"] for a in da["args"]}
        self.assertEqual(da_arg_names, {"transforms", "translation_ratio", "cutout_probability"})

    def test_hndl_ops_live_mode_when_available(self):
        if not _has_live_hndl():
            self.skipTest("torch/hndl not installed in this Python environment")
        ops = self.data["ops"]
        self.assertEqual(ops["mode"], "live")
        self.assertEqual(ops["total"], 85)
        self.assertEqual(len(ops["categories"]), 12)
        all_ops = {op["name"]: op for cat in ops["categories"] for op in cat["ops"]}
        self.assertIn("diff_augment", all_ops)
        self.assertTrue(all_ops["diff_augment"]["inputs"])
        self.assertTrue(all_ops["diff_augment"]["outputs"])

    def test_examples_and_all_49_docs_indexed(self):
        examples = self.data["examples"]
        self.assertEqual(len(examples["packaged_recipes"]), 5)
        self.assertEqual(len(examples["starter_templates"]), 8)
        self.assertGreaterEqual(len(examples["toml_examples"]), 10)
        self.assertEqual(len(examples["python_examples"]), 6)
        self.assertEqual(len(examples["hndl_networks"]), 22)
        self.assertEqual(len(examples["execution_profiles"]), 3)

        for cat_name, items in examples.items():
            for item in items:
                for key in (
                    "id",
                    "title",
                    "category",
                    "category_label",
                    "path",
                    "github_url",
                    "summary",
                    "tags",
                    "run_command",
                    "language",
                    "code",
                    "code_html",
                    "doc_slug",
                ):
                    self.assertIn(key, item, f"missing key {key!r} in example {cat_name}/{item.get('id')}")
        for tpl in examples["starter_templates"]:
            self.assertIn("name", tpl)
            self.assertIn("description", tpl)

        # Verify all docs/*.md + GLOSSARY.md + CHANGELOG.md are indexed with zero omissions
        expected_docs = {f"docs/{p.name}" for p in (REPO_ROOT / "docs").glob("*.md")}
        self.assertGreaterEqual(len(expected_docs), 50)
        self.assertIn("docs/colab.md", expected_docs)
        expected_all_docs = expected_docs | {"GLOSSARY.md", "CHANGELOG.md"}

        indexed_sources = {d["source_path"] for d in self.data["docs"]}
        self.assertEqual(indexed_sources, expected_all_docs)
        self.assertEqual(len(self.data["doc_groups"]), 7)


class MarkdownRenderTests(unittest.TestCase):
    def test_hndl_highlighting_and_copy_wrapper(self):
        code = "# Generator\nh = conv(64, kernel=3, padding=1)\nout = tanh(h)\n"
        html_out = markdown_render.highlight_code(code, "hndl")
        self.assertIn('data-lang="hndl"', html_out)
        self.assertIn("data-copy-code", html_out)
        self.assertIn(">HNDL<", html_out)
        self.assertIn('<span class="nf">conv</span>', html_out)
        self.assertIn('<span class="na">kernel</span>', html_out)

    def test_pygments_css_supports_dark_and_light(self):
        css = markdown_render.pygments_css()
        self.assertIn(":root", css)
        self.assertIn('[data-theme="light"]', css)
        self.assertIn("--syn-keyword", css)
        self.assertIn(".highlight .nf", css)

    def test_github_alerts_rendering(self):
        md_text = (
            "# Title\n\n"
            "> [!NOTE]\n"
            "> Remember to validate configs.\n\n"
            "> [!WARNING]\n"
            "> Destructive action ahead.\n"
        )
        res = markdown_render.render_markdown(
            md_text,
            source_path="docs/sample.md",
            page_url_path="docs/sample/index.html",
            route_map={},
            repo_url=introspect.REPO_URL,
            commit="master",
        )
        self.assertIn('class="callout callout-note"', res["html"])
        self.assertIn('<div class="callout-title">Note</div>', res["html"])
        self.assertIn('class="callout callout-warning"', res["html"])
        self.assertIn('<div class="callout-title">Warning</div>', res["html"])
        self.assertNotIn("[!NOTE]", res["html"])

    def test_link_rewriting_and_heading_anchors(self):
        route_map = {
            "docs/configuration.md": "docs/configuration/index.html",
            "GLOSSARY.md": "glossary/index.html",
            "docs/assets/hypergan-logo.png": "assets/hypergan-logo.png",
        }
        md_text = (
            "# Guide Title\n\n"
            "Intro paragraph for search and summary.\n\n"
            "## Sampling, Evaluation & Standalone Export\n\n"
            "See [Config](configuration.md#train), [Glossary](../GLOSSARY.md), "
            "[Source](../src/hypergan/api.py), [Examples Dir](../examples/), "
            "and [README Install](../README.md#install).\n\n"
            "![Logo](assets/hypergan-logo.png)\n\n"
            "### Details\n\n"
            "### Details\n"
        )
        res = markdown_render.render_markdown(
            md_text,
            source_path="docs/guide.md",
            page_url_path="docs/guide/index.html",
            route_map=route_map,
            repo_url=introspect.REPO_URL,
            commit="abc1234",
        )
        self.assertEqual(res["title"], "Guide Title")
        self.assertIn("Intro paragraph", res["description"])
        self.assertIn("Guide Title", res["plain_text"])

        toc_ids = [item["id"] for item in res["toc"]]
        self.assertEqual(
            toc_ids,
            ["sampling-evaluation--standalone-export", "details", "details-1"],
        )

        html_out = res["html"]
        self.assertIn('href="../configuration/index.html#train"', html_out)
        self.assertIn('href="../../glossary/index.html"', html_out)
        self.assertIn('href="https://github.com/HyperGAN/HyperGAN/blob/abc1234/src/hypergan/api.py"', html_out)
        self.assertIn('href="https://github.com/HyperGAN/HyperGAN/tree/abc1234/examples"', html_out)
        self.assertIn('href="https://github.com/HyperGAN/HyperGAN/blob/abc1234/README.md#install"', html_out)
        self.assertIn('src="../../assets/hypergan-logo.png"', html_out)

    def test_all_repo_docs_render_with_valid_internal_links(self):
        data = introspect.collect_site_data(REPO_ROOT, _force_ops_fallback=True)
        route_map = {d["source_path"]: d["output_path"] for d in data["docs"]}
        for asset in (REPO_ROOT / "docs" / "assets").glob("*"):
            if asset.is_file():
                route_map[f"docs/assets/{asset.name}"] = f"assets/{asset.name}"

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Copy assets so verify_site can resolve them
            assets_out = out / "assets"
            assets_out.mkdir(parents=True, exist_ok=True)
            for asset in (REPO_ROOT / "docs" / "assets").glob("*"):
                if asset.is_file():
                    (assets_out / asset.name).write_bytes(asset.read_bytes())

            for doc in data["docs"]:
                rendered = markdown_render.render_markdown(
                    doc["raw_markdown"],
                    source_path=doc["source_path"],
                    page_url_path=doc["output_path"],
                    route_map=route_map,
                    repo_url=data["meta"]["repo_url"],
                    commit=data["meta"]["commit"],
                )
                dest = out / doc["output_path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(f"<!doctype html><html><body>{rendered['html']}</body></html>", encoding="utf-8")

            # Verify every internal link and #fragment across all 51 rendered docs
            # (non-strict because synthetic dir lacks landing/reference pages)
            errors = verify.verify_site(out, strict=False)
            link_errors = [
                e for e in errors
                if "missing required site file" not in e and "missing search-index.json" not in e
            ]
            self.assertEqual(link_errors, [], f"Broken internal links/anchors in repo docs:\n" + "\n".join(link_errors))


class VerifySiteTests(unittest.TestCase):
    def _create_valid_site(self, root: Path) -> None:
        (root / ".nojekyll").write_text("", encoding="utf-8")
        (root / "llms.txt").write_text("# HyperGAN\n", encoding="utf-8")
        (root / "assets").mkdir(parents=True, exist_ok=True)
        (root / "assets" / "logo.png").write_bytes(b"png")

        search_entries = [
            {"kind": "doc", "title": "Config", "url": "docs/configuration/index.html"},
            {"kind": "api", "title": "train", "url": "reference/api/index.html#train"},
            {"kind": "cli", "title": "hypergan train", "url": "reference/cli/index.html#train"},
            {"kind": "schema", "title": "[train]", "url": "reference/schema/index.html#train"},
            {"kind": "op", "title": "conv", "url": "reference/ops/index.html#conv"},
            {"kind": "example", "title": "gaussian-grid", "url": "examples/index.html#gaussian-grid"},
        ]
        (root / "search-index.json").write_text(json.dumps(search_entries), encoding="utf-8")

        pages = {
            "index.html": '<html id="top"><body><a href="quickstart/index.html#install">Start</a><img src="assets/logo.png"></body></html>',
            "404.html": '<html><body><a href="index.html">Home</a></body></html>',
            "quickstart/index.html": '<html><body><h2 id="install">Install</h2><a href="../index.html">Home</a></body></html>',
            "examples/index.html": '<html><body><div id="gaussian-grid"></div></body></html>',
            "glossary/index.html": '<html><body><h1 id="glossary">Glossary</h1></body></html>',
            "changelog/index.html": '<html><body><h1 id="changelog">Changelog</h1></body></html>',
            "reference/api/index.html": '<html><body><div id="train"></div></body></html>',
            "reference/cli/index.html": '<html><body><div id="train"></div></body></html>',
            "reference/schema/index.html": '<html><body><div id="train"></div></body></html>',
            "reference/ops/index.html": '<html><body><div id="conv"></div></body></html>',
        }
        for i in range(49):
            pages[f"docs/page-{i}/index.html"] = '<html><body><a href="../../index.html">Home</a></body></html>'

        for rel, content in pages.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    def test_valid_site_passes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_valid_site(root)
            self.assertEqual(verify.verify_site(root, strict=True), [])
            self.assertEqual(verify.main([str(root)]), 0)

    def test_catches_broken_link_and_broken_anchor(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_valid_site(root)
            (root / "index.html").write_text(
                '<html><body>'
                '<a href="missing/page.html">Missing</a>'
                '<a href="quickstart/index.html#nonexistent-anchor">Bad Anchor</a>'
                '<img src="assets/missing.png">'
                '<a href="/absolute-path/">Non Relative</a>'
                '</body></html>',
                encoding="utf-8",
            )
            errors = verify.verify_site(root, strict=True)
            joined = "\n".join(errors)
            self.assertIn("missing/page.html", joined)
            self.assertIn("#nonexistent-anchor", joined)
            self.assertIn("assets/missing.png", joined)
            self.assertIn("non-relative root-anchored URL", joined)
            self.assertEqual(verify.main([str(root)]), 1)


class EndToEndSiteBuildTests(unittest.TestCase):
    def test_full_site_build_and_strict_verify(self):
        build_script = SITE_DIR / "build.py"
        if not build_script.is_file():
            self.skipTest("site/build.py not yet present in this branch (built by site-designer)")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "_site"
            proc = subprocess.run(
                [sys.executable, str(build_script), "--out", str(out_dir), "--strict", "--repo-root", str(REPO_ROOT)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"site/build.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            errors = verify.verify_site(out_dir, strict=True)
            self.assertEqual(errors, [], "verify_site reported errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
