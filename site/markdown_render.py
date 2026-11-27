"""Markdown & Pygments syntax-highlighting renderer for HyperGAN docs."""
from __future__ import annotations

import html
import posixpath
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import RegexLexer, bygroups, words
from pygments.lexers import (
    BashLexer,
    JsonLexer,
    PythonLexer,
    TextLexer,
    TOMLLexer,
    YamlLexer,
    get_lexer_by_name,
)
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)
from pygments.util import ClassNotFound


_ALERT_RE = re.compile(
    r"^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\s*\n|\s*$)",
    re.IGNORECASE,
)

_ALERT_LABELS = {
    "note": "Note",
    "tip": "Tip",
    "important": "Important",
    "warning": "Warning",
    "caution": "Caution",
}

# Known historical / renamed anchors in repository Markdown documents so
# cross-references resolve cleanly without modifying docs/ content.
_LEGACY_ANCHOR_ALIASES: dict[tuple[str, str], str] = {
    (
        "docs/configuration.md",
        "custom-metrics-and-explicit-snapshot-evaluation",
    ): "7-evaluations--scalar-metrics-evaluations--metrics",
    (
        "CHANGELOG.md",
        "Least-Squares-GAN",
    ): "08--least-squares-gan-and-api-examples",
}

_KNOWN_REPO_DIRS = {
    "docs",
    "docs/assets",
    "examples",
    "examples/api",
    "examples/api/config_first",
    "examples/archive",
    "examples/execution",
    "examples/networks",
    "frontend",
    "reports",
    "reports/archive",
    "scripts",
    "src",
    "tests",
    ".github",
    ".github/workflows",
}


class HndlLexer(RegexLexer):
    """Pygments lexer for HNDL (HyperGAN Network Description Language)."""

    name = "HNDL"
    aliases = ["hndl"]
    filenames = ["*.hndl"]
    mimetypes = ["text/x-hndl"]

    tokens = {
        "root": [
            (r"\s+", Text.Whitespace),
            (r"#[^\n]*", Comment.Single),
            (
                words(
                    (
                        "for",
                        "in",
                        "range",
                        "if",
                        "elif",
                        "else",
                        "and",
                        "or",
                        "not",
                        "return",
                        "let",
                        "def",
                    ),
                    prefix=r"\b",
                    suffix=r"\b",
                ),
                Keyword,
            ),
            (
                words(("True", "False", "None"), prefix=r"\b", suffix=r"\b"),
                Keyword.Constant,
            ),
            # Symbolic dimension literals like 'B', 'C', 'H', 'W', 'S', 'D'
            (r"'(?:B|C|H|W|S|D|N|T|L)'", Name.Constant),
            (r'"(?:B|C|H|W|S|D|N|T|L)"', Name.Constant),
            (r'"(?:\\.|[^"\\])*"', String.Double),
            (r"'(?:\\.|[^'\\])*'", String.Single),
            # Operator / function call
            (
                r"([a-zA-Z_][a-zA-Z0-9_]*)(\s*)(\()",
                bygroups(Name.Function, Text.Whitespace, Punctuation),
            ),
            # Keyword argument inside a call
            (
                r"([a-zA-Z_][a-zA-Z0-9_]*)(\s*)(=)(?!=)",
                bygroups(Name.Attribute, Text.Whitespace, Operator),
            ),
            # Numbers (scientific, float, int)
            (r"\b\d+(?:\.\d*)?[eE][+-]?\d+\b", Number.Float),
            (r"\b\d+\.\d+\b", Number.Float),
            (r"\b\d+\b", Number.Integer),
            (r"->|==|!=|<=|>=|[+\-*/%=<>]", Operator),
            (r"[()\[\]{},.:;]", Punctuation),
            (r"[a-zA-Z_][a-zA-Z0-9_]*", Name.Variable),
        ]
    }


_LANG_ALIASES: dict[str, tuple[str, str]] = {
    "hndl": ("hndl", "HNDL"),
    "toml": ("toml", "TOML"),
    "py": ("python", "Python"),
    "python": ("python", "Python"),
    "python3": ("python", "Python"),
    "sh": ("bash", "Shell"),
    "bash": ("bash", "Bash"),
    "shell": ("bash", "Shell"),
    "zsh": ("bash", "Shell"),
    "console": ("bash", "Shell"),
    "json": ("json", "JSON"),
    "yaml": ("yaml", "YAML"),
    "yml": ("yaml", "YAML"),
    "ebnf": ("text", "EBNF"),
    "md": ("markdown", "Markdown"),
    "markdown": ("markdown", "Markdown"),
    "html": ("html", "HTML"),
    "css": ("css", "CSS"),
    "js": ("javascript", "JavaScript"),
    "javascript": ("javascript", "JavaScript"),
    "ts": ("typescript", "TypeScript"),
    "typescript": ("typescript", "TypeScript"),
    "dockerfile": ("docker", "Dockerfile"),
    "docker": ("docker", "Dockerfile"),
    "diff": ("diff", "Diff"),
    "text": ("text", "Text"),
    "txt": ("text", "Text"),
    "plain": ("text", "Text"),
    "": ("text", "Text"),
}


def _resolve_lexer(lang: str):
    key = (lang or "").strip().lower()
    if key == "hndl":
        return HndlLexer(), "hndl", "HNDL"
    if key == "toml":
        return TOMLLexer(), "toml", "TOML"
    if key in ("py", "python", "python3"):
        return PythonLexer(), "python", "Python"
    if key in ("sh", "bash", "shell", "zsh", "console"):
        return BashLexer(), "bash", _LANG_ALIASES.get(key, ("bash", "Shell"))[1]
    if key == "json":
        return JsonLexer(), "json", "JSON"
    if key in ("yaml", "yml"):
        return YamlLexer(), "yaml", "YAML"
    if key in ("", "text", "txt", "plain", "ebnf"):
        label = _LANG_ALIASES.get(key, ("text", key.upper() or "Text"))[1]
        return TextLexer(), "text", label
    slug, label = _LANG_ALIASES.get(key, (re.sub(r"[^a-z0-9_-]", "", key) or "text", key.upper()))
    try:
        return get_lexer_by_name(slug, stripall=False), slug, label
    except ClassNotFound:
        return TextLexer(), slug, label


def highlight_code(code: str, lang: str = "") -> str:
    """Syntax-highlight a code snippet with Pygments + custom HNDL lexer.

    Returns a complete code-block wrapper with language pill and copy button hook
    (``data-copy-code``).
    """
    lexer, lang_slug, lang_label = _resolve_lexer(lang)
    formatter = HtmlFormatter(nowrap=True, classprefix="")
    # Preserve exact code content without injecting a trailing extra blank line
    normalized = code[:-1] if code.endswith("\n") else code
    highlighted = highlight(normalized, lexer, formatter)
    if highlighted.endswith("\n"):
        highlighted = highlighted[:-1]
    return (
        f'<div class="code-block" data-lang="{html.escape(lang_slug, quote=True)}">'
        f'<div class="code-block-header">'
        f'<span class="code-lang">{html.escape(lang_label)}</span>'
        f'<button type="button" class="code-copy-btn" data-copy-code aria-label="Copy code">Copy</button>'
        f"</div>"
        f'<pre class="highlight"><code class="language-{html.escape(lang_slug, quote=True)}">'
        f"{highlighted}"
        f"</code></pre>"
        f"</div>"
    )


def pygments_css() -> str:
    """Return CSS variable-driven Pygments token styles for Ink (dark) & Paper (light)."""
    return """/* Syntax highlighting tokens mapped to 255labs Ink (dark) and Paper (light) themes */
:root {
  --syn-bg: #151714;
  --syn-fg: #F1F2EA;
  --syn-comment: #8B9285;
  --syn-keyword: #D7BDF6;
  --syn-function: #AACBFA;
  --syn-string: #D7FF3F;
  --syn-number: #F7C97E;
  --syn-operator: #C8CEC2;
  --syn-attr: #9CE5C4;
  --syn-type: #AACBFA;
  --syn-constant: #F5B88E;
  --syn-error: #FF7A7A;
}

[data-theme="light"] {
  --syn-bg: #E7E8DF;
  --syn-fg: #101110;
  --syn-comment: #5B6155;
  --syn-keyword: #6D28D9;
  --syn-function: #1D4ED8;
  --syn-string: #2D5F0B;
  --syn-number: #B45309;
  --syn-operator: #35392F;
  --syn-attr: #0F766E;
  --syn-type: #1E40AF;
  --syn-constant: #9A3412;
  --syn-error: #B91C1C;
}

.highlight { color: var(--syn-fg); }
.highlight .c,
.highlight .ch,
.highlight .cm,
.highlight .cp,
.highlight .cpf,
.highlight .c1,
.highlight .cs { color: var(--syn-comment); font-style: italic; }
.highlight .k,
.highlight .kc,
.highlight .kd,
.highlight .kn,
.highlight .kp,
.highlight .kr,
.highlight .kt { color: var(--syn-keyword); font-weight: 500; }
.highlight .n,
.highlight .nv,
.highlight .vi,
.highlight .vg,
.highlight .vc { color: var(--syn-fg); }
.highlight .nf,
.highlight .fm,
.highlight .nb,
.highlight .bp { color: var(--syn-function); }
.highlight .nc,
.highlight .nn,
.highlight .ne { color: var(--syn-type); font-weight: 500; }
.highlight .na,
.highlight .nd,
.highlight .nt { color: var(--syn-attr); }
.highlight .no { color: var(--syn-constant); font-weight: 500; }
.highlight .s,
.highlight .sa,
.highlight .sb,
.highlight .sc,
.highlight .dl,
.highlight .sd,
.highlight .s2,
.highlight .se,
.highlight .sh,
.highlight .si,
.highlight .sx,
.highlight .sr,
.highlight .s1,
.highlight .ss { color: var(--syn-string); }
.highlight .m,
.highlight .mb,
.highlight .mf,
.highlight .mh,
.highlight .mi,
.highlight .mo,
.highlight .il { color: var(--syn-number); }
.highlight .o,
.highlight .ow,
.highlight .p { color: var(--syn-operator); }
.highlight .gd { color: var(--syn-error); }
.highlight .gi { color: var(--syn-string); }
.highlight .err { color: var(--syn-error); }
"""


def slugify_heading(text: str, seen: dict[str, int] | None = None) -> str:
    """Generate a GitHub-compatible heading anchor ID with duplicate numbering."""
    # Strip any inline HTML tags
    plain = re.sub(r"<[^>]+>", "", text)
    plain = html.unescape(plain).strip().lower()
    # Keep letters, digits, underscores, hyphens, and whitespace; drop punctuation
    cleaned = re.sub(r"[^\w\s-]", "", plain, flags=re.UNICODE)
    slug = re.sub(r"\s", "-", cleaned)
    if not slug:
        slug = "section"
    if seen is None:
        return slug
    count = seen.get(slug, 0)
    seen[slug] = count + 1
    if count == 0:
        return slug
    return f"{slug}-{count}"


def _token_plain_text(token: Token) -> str:
    """Extract readable plain text from an inline markdown-it token."""
    if not token.children:
        return token.content or ""
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif child.type == "image":
            parts.append(_token_plain_text(child))
    return "".join(parts)


def _page_dir_from_url_path(page_url_path: str) -> str:
    """Compute the POSIX directory of a built page URL relative to site root."""
    cleaned = page_url_path.strip().lstrip("/")
    if not cleaned:
        return "."
    if cleaned.endswith("/"):
        stripped = cleaned.rstrip("/")
        return stripped or "."
    if cleaned.endswith(".html"):
        parent = posixpath.dirname(cleaned)
        return parent or "."
    return cleaned


def _relative_site_href(target_site_path: str, page_dir: str) -> str:
    """Compute a clean relative href from ``page_dir`` to ``target_site_path``."""
    target = target_site_path.strip().lstrip("/")
    trailing_slash = target.endswith("/")
    norm_target = target.rstrip("/") if trailing_slash else target

    if not norm_target:
        # Root landing page
        if page_dir in ("", "."):
            return "./"
        rel = posixpath.relpath(".", page_dir)
        return rel.rstrip("/") + "/"

    rel = posixpath.relpath(norm_target, page_dir)
    if trailing_slash:
        if rel == ".":
            return "./"
        return rel.rstrip("/") + "/"
    return rel


_SAFE_DATA_IMAGE_RE = re.compile(
    r"^data:image/(?:png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+$",
    re.IGNORECASE,
)


def _rewrite_url(
    url: str,
    *,
    source_path: str,
    page_dir: str,
    route_map: dict[str, str],
    repo_url: str,
    commit: str,
    local_heading_ids: set[str],
    is_image: bool = False,
) -> str:
    raw = (url or "").strip()
    if not raw:
        return raw

    # Normalize away ASCII control characters and whitespace when checking URI schemes
    # so obfuscated payloads like "java\tscript:..." or " data:text/html,..." are caught.
    scheme_probe = re.sub(r"[\x00-\x20]+", "", raw).lower()
    if scheme_probe.startswith(("javascript:", "vbscript:")):
        return ""
    if scheme_probe.startswith("data:"):
        if is_image and _SAFE_DATA_IMAGE_RE.fullmatch(scheme_probe):
            return raw
        return ""
    if scheme_probe.startswith(("http://", "https://")):
        return raw
    if not is_image and scheme_probe.startswith(("mailto:", "tel:")):
        return raw

    probed_parts = urlsplit(scheme_probe)
    if probed_parts.scheme:
        return ""

    parts = urlsplit(raw)
    path_part = parts.path
    fragment = parts.fragment

    # Pure same-page anchor link
    if not path_part and fragment:
        aliased = _LEGACY_ANCHOR_ALIASES.get((source_path, fragment), fragment)
        if aliased not in local_heading_ids and aliased.lower() in local_heading_ids:
            aliased = aliased.lower()
        return f"#{aliased}"

    # Resolve relative to source_path's parent directory in the repository
    source_dir = posixpath.dirname(source_path.lstrip("/"))
    joined = posixpath.join(source_dir, path_part) if source_dir else path_part
    resolved = posixpath.normpath(joined).lstrip("/")
    if resolved == ".":
        resolved = ""

    # Remap known historical/renamed anchors
    if fragment:
        fragment = _LEGACY_ANCHOR_ALIASES.get((resolved, fragment), fragment)

    # README.md with a fragment points to GitHub README so anchors always resolve
    if resolved == "README.md" and fragment:
        return f"{repo_url.rstrip('/')}/blob/{commit}/README.md#{fragment}"

    # Direct hit in site route_map (docs/*.md, GLOSSARY.md, CHANGELOG.md, etc.)
    if resolved in route_map:
        target_site_path = route_map[resolved]
        # Self-link with fragment
        if resolved == source_path.lstrip("/") and fragment:
            if fragment in local_heading_ids:
                return f"#{fragment}"
            if fragment.lower() in local_heading_ids:
                return f"#{fragment.lower()}"
            return f"{repo_url.rstrip('/')}/blob/{commit}/{resolved}#{fragment}"
        rel_href = _relative_site_href(target_site_path, page_dir)
        return f"{rel_href}#{fragment}" if fragment else rel_href

    # Built assets copied from docs/assets/* -> assets/*
    if resolved.startswith("docs/assets/"):
        asset_site_path = "assets/" + resolved[len("docs/assets/") :]
        rel_href = _relative_site_href(asset_site_path, page_dir)
        return f"{rel_href}#{fragment}" if fragment else rel_href

    # Otherwise rewrite to canonical GitHub blob/tree URL
    is_dir = path_part.endswith("/") or resolved in _KNOWN_REPO_DIRS
    kind = "tree" if is_dir else "blob"
    gh_base = f"{repo_url.rstrip('/')}/{kind}/{commit}/{resolved}" if resolved else repo_url.rstrip("/")
    if parts.query:
        gh_base = f"{gh_base}?{parts.query}"
    if fragment:
        return f"{gh_base}#{fragment}"
    return gh_base


def _transform_github_alerts(tokens: list[Token]) -> list[Token]:
    """Convert GitHub-style alert blockquotes (> [!NOTE], etc.) into callout tokens."""
    out: list[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type != "blockquote_open":
            out.append(tok)
            i += 1
            continue

        # Find matching blockquote_close respecting nesting
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if tokens[j].type == "blockquote_open":
                depth += 1
            elif tokens[j].type == "blockquote_close":
                depth -= 1
            j += 1
        close_idx = j - 1

        # Inspect first paragraph inline token inside blockquote
        alert_kind: str | None = None
        if close_idx > i + 2 and tokens[i + 1].type == "paragraph_open" and tokens[i + 2].type == "inline":
            inline_tok = tokens[i + 2]
            match = _ALERT_RE.match(inline_tok.content or "")
            if match:
                alert_kind = match.group(1).lower()
                # Strip the [!KIND] marker from inline token and its children
                prefix_len = match.end()
                inline_tok.content = inline_tok.content[prefix_len:]
                if inline_tok.children:
                    new_children: list[Token] = []
                    stripped = False
                    skip_next_break = False
                    for ch in inline_tok.children:
                        if not stripped and ch.type == "text":
                            m2 = _ALERT_RE.match(ch.content)
                            if m2:
                                rem = ch.content[m2.end() :]
                                stripped = True
                                if not rem:
                                    skip_next_break = True
                                    continue
                                ch.content = rem
                                new_children.append(ch)
                                continue
                        if skip_next_break and ch.type in ("softbreak", "hardbreak"):
                            skip_next_break = False
                            continue
                        skip_next_break = False
                        new_children.append(ch)
                    inline_tok.children = new_children

        if alert_kind is None:
            out.append(tok)
            i += 1
            continue

        label = _ALERT_LABELS.get(alert_kind, alert_kind.title())
        open_html = Token("html_block", "", 0)
        open_html.content = (
            f'<div class="callout callout-{alert_kind}" role="note">'
            f'<div class="callout-title">{html.escape(label)}</div>'
            f'<div class="callout-body">\n'
        )
        close_html = Token("html_block", "", 0)
        close_html.content = "</div></div>\n"

        out.append(open_html)
        inner = tokens[i + 1 : close_idx]
        # Drop an empty leading paragraph if the marker was the entire first paragraph
        if (
            len(inner) >= 3
            and inner[0].type == "paragraph_open"
            and inner[1].type == "inline"
            and inner[2].type == "paragraph_close"
            and not (inner[1].content or "").strip()
            and not inner[1].children
        ):
            inner = inner[3:]
        out.extend(inner)
        out.append(close_html)
        i = close_idx + 1

    return out


def _strip_html_to_text(rendered_html: str) -> str:
    # Remove code-copy button labels so search text stays clean
    cleaned = re.sub(r"<button[^>]*>.*?</button>", " ", rendered_html, flags=re.S)
    cleaned = re.sub(r'<a class="heading-anchor"[^>]*>.*?</a>', " ", cleaned, flags=re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_description(tokens: list[Token]) -> str:
    """Return the first clean descriptive paragraph from parsed markdown tokens."""
    in_blockquote = 0
    in_list = 0
    for idx, tok in enumerate(tokens):
        if tok.type == "blockquote_open":
            in_blockquote += 1
        elif tok.type == "blockquote_close":
            in_blockquote = max(0, in_blockquote - 1)
        elif tok.type in ("bullet_list_open", "ordered_list_open"):
            in_list += 1
        elif tok.type in ("bullet_list_close", "ordered_list_close"):
            in_list = max(0, in_list - 1)
        elif tok.type == "paragraph_open" and in_blockquote == 0 and in_list == 0:
            if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
                text = re.sub(r"\s+", " ", _token_plain_text(tokens[idx + 1])).strip()
                if text and not text.startswith("[!"):
                    if len(text) > 220:
                        cut = text[:217].rsplit(" ", 1)[0]
                        return cut.rstrip(".,;:") + "..."
                    return text
    return ""


def render_markdown(
    text: str,
    *,
    source_path: str,
    page_url_path: str,
    route_map: dict[str, str],
    repo_url: str = "https://github.com/HyperGAN/HyperGAN",
    commit: str = "master",
) -> dict[str, Any]:
    """Render Markdown to HTML with HNDL highlighting, alerts, TOC, and rewritten links."""
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(
        ["table", "strikethrough"]
    )

    tokens = md.parse(text or "")
    tokens = _transform_github_alerts(tokens)

    # First pass: collect heading IDs, title, and TOC (h2/h3)
    seen_slugs: dict[str, int] = {}
    local_heading_ids: set[str] = set()
    toc: list[dict[str, Any]] = []
    title = ""

    for idx, tok in enumerate(tokens):
        if tok.type == "heading_open":
            level = int(tok.tag[1]) if tok.tag and len(tok.tag) == 2 else 2
            heading_text = ""
            if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
                heading_text = _token_plain_text(tokens[idx + 1]).strip()
            hid = slugify_heading(heading_text, seen_slugs)
            tok.attrSet("id", hid)
            local_heading_ids.add(hid)
            if level == 1 and not title and heading_text:
                title = heading_text
            if level in (2, 3) and heading_text:
                toc.append({"level": level, "id": hid, "title": heading_text})

    description = _extract_description(tokens)

    if not title:
        stem = PurePosixPath(source_path).stem
        title = "HyperGAN Documentation" if stem.lower() == "readme" else stem.replace("-", " ").title()

    # Second pass: rewrite links and images on inline children
    page_dir = _page_dir_from_url_path(page_url_path)

    def _rewrite_inline_tokens(tok_list: list[Token]) -> None:
        for tok in tok_list:
            if tok.type == "link_open":
                href = tok.attrGet("href")
                if href:
                    new_href = _rewrite_url(
                        str(href),
                        source_path=source_path,
                        page_dir=page_dir,
                        route_map=route_map,
                        repo_url=repo_url,
                        commit=commit,
                        local_heading_ids=local_heading_ids,
                    )
                    tok.attrSet("href", new_href)
                    if new_href.startswith(("http://", "https://")):
                        tok.attrSet("target", "_blank")
                        tok.attrSet("rel", "noopener noreferrer")
            elif tok.type == "image":
                src = tok.attrGet("src")
                if src:
                    new_src = _rewrite_url(
                        str(src),
                        source_path=source_path,
                        page_dir=page_dir,
                        route_map=route_map,
                        repo_url=repo_url,
                        commit=commit,
                        local_heading_ids=local_heading_ids,
                        is_image=True,
                    )
                    tok.attrSet("src", new_src)
                    tok.attrSet("loading", "lazy")
            if tok.children:
                _rewrite_inline_tokens(tok.children)

    _rewrite_inline_tokens(tokens)

    # Custom renderer rules for fences, heading anchor links, and responsive tables
    def _render_fence(tokens_arg, idx, options, env):
        tok = tokens_arg[idx]
        info = (tok.info or "").strip()
        lang = info.split()[0] if info else ""
        return highlight_code(tok.content, lang) + "\n"

    def _render_heading_close(tokens_arg, idx, options, env):
        tok = tokens_arg[idx]
        level = int(tok.tag[1]) if tok.tag and len(tok.tag) == 2 else 2
        anchor_html = ""
        if level in (2, 3, 4) and idx >= 2 and tokens_arg[idx - 2].type == "heading_open":
            hid = tokens_arg[idx - 2].attrGet("id") or ""
            if hid:
                anchor_html = (
                    f'<a class="heading-anchor" href="#{html.escape(str(hid), quote=True)}" '
                    f'aria-label="Permalink">#</a>'
                )
        return f"{anchor_html}</{tok.tag}>\n"

    def _render_table_open(tokens_arg, idx, options, env):
        return '<div class="table-wrap"><table>\n'

    def _render_table_close(tokens_arg, idx, options, env):
        return "</table></div>\n"

    md.renderer.rules["fence"] = _render_fence
    md.renderer.rules["heading_close"] = _render_heading_close
    md.renderer.rules["table_open"] = _render_table_open
    md.renderer.rules["table_close"] = _render_table_close

    rendered_html = md.renderer.render(tokens, md.options, {})
    plain_text = _strip_html_to_text(rendered_html)

    return {
        "html": rendered_html,
        "toc": toc,
        "title": title,
        "description": description,
        "plain_text": plain_text,
    }


def render_docstring_html(doc: str) -> str:
    """Render a Python docstring (supporting Markdown and reST ``code``) to clean HTML."""
    if not doc or not doc.strip():
        return ""
    cleaned = inspect_cleandoc(doc)
    # Normalize Sphinx/reST double-backtick inline literals to Markdown single backticks
    cleaned = re.sub(r"``([^`]+)``", r"`\1`", cleaned)
    md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable(
        ["table", "strikethrough"]
    )
    return md.render(cleaned)


def inspect_cleandoc(doc: str) -> str:
    import inspect

    return inspect.cleandoc(doc)
