"""Built-site HTML link, anchor, asset, and search-index verifier."""
from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


_CORE_REQUIRED_FILES = [
    "index.html",
    "404.html",
    ".nojekyll",
    "llms.txt",
    "search-index.json",
    "quickstart/index.html",
    "examples/index.html",
    "glossary/index.html",
    "changelog/index.html",
    "reference/api/index.html",
    "reference/cli/index.html",
    "reference/schema/index.html",
    "reference/ops/index.html",
]

_EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "javascript:",
    "data:",
)


class _PageLinkParser(HTMLParser):
    """Collect element IDs and internal href/src references from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.refs: list[tuple[str, str, str]] = []  # (tag, attr, value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        elem_id = attr_map.get("id")
        if elem_id:
            self.ids.add(elem_id)
        if tag == "a":
            anchor_name = attr_map.get("name")
            if anchor_name:
                self.ids.add(anchor_name)
            href = attr_map.get("href")
            if href is not None:
                self.refs.append((tag, "href", href))
        elif tag == "link":
            href = attr_map.get("href")
            if href is not None:
                self.refs.append((tag, "href", href))
        elif tag in ("img", "script", "source"):
            src = attr_map.get("src")
            if src is not None:
                self.refs.append((tag, "src", src))


def _resolve_target(out_dir: Path, current_html: Path, raw_url: str, strict: bool) -> tuple[Path | None, str, str | None]:
    """Resolve an internal URL to ``(target_file, fragment, error_msg)``."""
    stripped = raw_url.strip()
    if not stripped:
        return None, "", None
    if stripped.lower().startswith(_EXTERNAL_PREFIXES):
        return None, "", None

    parts = urlsplit(stripped)
    path_part = unquote(parts.path)
    fragment = unquote(parts.fragment)

    if not path_part:
        return current_html, fragment, None

    if path_part.startswith("/"):
        if strict:
            return None, fragment, f"non-relative root-anchored URL {raw_url!r} (must use relative root_prefix)"
        candidate = (out_dir / path_part.lstrip("/")).resolve()
    else:
        candidate = (current_html.parent / path_part).resolve()

    out_resolved = out_dir.resolve()
    try:
        candidate.relative_to(out_resolved)
    except ValueError:
        return None, fragment, f"link escapes site output root: {raw_url!r}"

    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif path_part.endswith("/") and not candidate.exists():
        candidate = candidate / "index.html"

    return candidate, fragment, None


def _verify_search_index(out_dir: Path) -> list[str]:
    errors: list[str] = []
    idx_path = out_dir / "search-index.json"
    if not idx_path.is_file():
        return ["missing search-index.json"]

    try:
        payload = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid search-index.json: {exc}"]

    if isinstance(payload, dict):
        entries = payload.get("items") or payload.get("entries") or payload.get("documents")
    else:
        entries = payload

    if not isinstance(entries, list) or not entries:
        return ["search-index.json contains no entries"]

    found_kinds: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        raw_kind = str(
            item.get("kind")
            or item.get("type")
            or item.get("category")
            or item.get("section")
            or ""
        ).strip().lower()
        if raw_kind:
            found_kinds.add(raw_kind)

    required_groups = {
        "docs": {"doc", "docs", "guide", "markdown", "page"},
        "api": {"api", "python-api", "symbol", "function", "dataclass", "class", "exception"},
        "cli": {"cli", "command", "subcommand"},
        "schema": {"schema", "toml", "config", "section", "field"},
        "ops": {"op", "ops", "hndl", "hndl op", "operator"},
        "examples": {"example", "examples", "recipe", "template"},
    }
    for label, accepted in required_groups.items():
        if not (found_kinds & accepted):
            errors.append(
                f"search-index.json missing entries for {label!r} (found kinds: {sorted(found_kinds)})"
            )
    return errors


def verify_site(out_dir: Path, *, strict: bool = True) -> list[str]:
    """Verify built HTML pages, links, anchors, assets, and search-index.json.

    Returns a list of human-readable error strings (empty list on pass).
    """
    out_root = Path(out_dir).resolve()
    errors: list[str] = []

    if not out_root.is_dir():
        return [f"output directory does not exist: {out_root}"]

    # 1. Core required files
    for rel in _CORE_REQUIRED_FILES:
        if not (out_root / rel).is_file():
            errors.append(f"missing required site file: {rel}")

    # 2. Verify docs page coverage
    docs_pages = list((out_root / "docs").glob("*/index.html")) if (out_root / "docs").is_dir() else []
    if strict and len(docs_pages) < 48:
        errors.append(f"expected at least 48 rendered docs/*/index.html pages, found {len(docs_pages)}")

    # 3. Verify search-index.json
    errors.extend(_verify_search_index(out_root))

    # 4. Parse every HTML file under out_root
    html_files = sorted(out_root.rglob("*.html"))
    if not html_files:
        errors.append(f"no HTML files found under {out_root}")
        return errors

    parsed_pages: dict[Path, _PageLinkParser] = {}
    for html_file in html_files:
        parser = _PageLinkParser()
        try:
            parser.feed(html_file.read_text(encoding="utf-8"))
            parser.close()
        except Exception as exc:
            rel_file = html_file.relative_to(out_root).as_posix()
            errors.append(f"{rel_file}: HTML parse error: {exc}")
            continue
        parsed_pages[html_file.resolve()] = parser

    # 5. Validate every internal link, asset src, and #fragment anchor
    for html_file, parser in parsed_pages.items():
        rel_src = html_file.relative_to(out_root).as_posix()
        for tag, attr, raw_val in parser.refs:
            target, fragment, resolve_err = _resolve_target(out_root, html_file, raw_val, strict=strict)
            if resolve_err:
                errors.append(f"{rel_src}: <{tag} {attr}={raw_val!r}>: {resolve_err}")
                continue
            if target is None:
                continue
            if not target.is_file():
                try:
                    rel_target = target.relative_to(out_root).as_posix()
                except ValueError:
                    rel_target = str(target)
                errors.append(
                    f"{rel_src}: broken <{tag} {attr}={raw_val!r}> -> missing {rel_target}"
                )
                continue
            if fragment and target.suffix.lower() == ".html":
                target_parser = parsed_pages.get(target.resolve())
                if target_parser is not None and fragment not in target_parser.ids:
                    rel_target = target.relative_to(out_root).as_posix()
                    errors.append(
                        f"{rel_src}: broken anchor <{tag} {attr}={raw_val!r}> -> #{fragment} not found in {rel_target}"
                    )

    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python site/verify.py <out_dir> [--no-strict]")
        return 0 if args else 2

    out_dir = Path(args[0])
    strict = "--no-strict" not in args[1:]
    errors = verify_site(out_dir, strict=strict)
    if errors:
        print(f"Site verification FAILED with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    html_count = len(list(out_dir.rglob("*.html"))) if out_dir.is_dir() else 0
    print(f"Site verification PASSED ({html_count} HTML files checked, 0 errors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
