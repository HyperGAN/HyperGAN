"""Build the GitBook tree from an exact HyperGAN commit."""

import argparse
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
from urllib.parse import quote, unquote, urlsplit


SOURCE = "https://github.com/HyperGAN/HyperGAN"
LINK = re.compile(r"(?P<prefix>\]\([ \t]*<?|^ {0,3}\[[^\]\n]+\]:[ \t]*<?)(?P<target>[^\s<>()]+)")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")


def git(*args):
    return subprocess.check_output(["git", *args])


def rewrite_links(text, page, revision, files, directories, preview_files):
    def replace(match):
        if any(start <= match.start() < end for start, end in code_spans):
            return match[0]
        href = urlsplit(match["target"])
        if href.scheme or href.netloc or not href.path or href.path.startswith("/"):
            return match[0]
        path = posixpath.normpath(posixpath.join(page.parent.as_posix(), unquote(href.path)))
        if path in preview_files:
            return match[0]
        if path in files:
            kind = "blob"
        elif path in directories:
            kind = "tree"
        else:
            raise ValueError(f"{page} links to an untracked path: {href.path}")
        link = f"{SOURCE}/{kind}/{revision}/{quote(path, safe='/')}"
        if href.query:
            link += "?" + href.query
        if href.fragment:
            link += "#" + href.fragment
        return match["prefix"] + link

    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
        else:
            code_spans = [segment.span() for segment in INLINE_CODE.finditer(line)]
            line = LINK.sub(replace, line)
        result.append(line)
    return "".join(result)


def summary(pages):
    sections = ["# Summary", "* [HyperGAN documentation](README.md)"]
    linked = {"README.md"}
    for line in pages["README.md"].splitlines():
        if line.startswith("## "):
            sections.append(line)
        if not line.startswith("- "):
            continue
        for match in LINK.finditer(line):
            target = urlsplit(match["target"]).path
            if target in pages and target not in linked:
                heading = next(line[2:] for line in pages[target].splitlines() if line.startswith("# "))
                sections.append(f"* [{heading}]({target})")
                linked.add(target)
    remaining = sorted(pages.keys() - linked)
    if remaining:
        sections.append("## Additional documentation")
        for target in remaining:
            heading = next(line[2:] for line in pages[target].splitlines() if line.startswith("# "))
            sections.append(f"* [{heading}]({target})")
    return "\n\n".join(sections) + "\n"


def build(output, revision="HEAD"):
    revision = git("rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    rows = git("ls-tree", "-rz", "--full-tree", revision).decode().rstrip("\0").split("\0")
    files = {}
    for row in rows:
        metadata, name = row.split("\t", 1)
        files[name] = metadata.split()[0]
    directories = {parent.as_posix() for name in files for parent in PurePosixPath(name).parents}
    preview_files = {name for name in files if name.startswith("docs/") and
                     (name.endswith(".md") or name.startswith("docs/assets/"))}

    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    assets, pages = {}, {}
    for name in sorted(preview_files):
        if files[name] not in {"100644", "100755"}:
            raise ValueError(f"Expected a regular documentation file: {name}")
        content = git("show", f"{revision}:{name}")
        local = name.removeprefix("docs/")
        if local in {"SUMMARY.md", "SOURCE_REVISION"}:
            raise ValueError(f"Documentation file collides with GitBook output: {name}")
        if name.endswith(".md"):
            pages[local] = rewrite_links(content.decode(), PurePosixPath(name), revision,
                                         files, directories, preview_files)
        else:
            assets[local] = content
    navigation = summary(pages)
    output.mkdir(parents=True)
    for name, content in {**assets, **{name: text.encode() for name, text in pages.items()}}.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (output / "SUMMARY.md").write_text(navigation, encoding="utf-8")
    (output / ".gitbook.yaml").write_text(
        "root: ./\nstructure:\n  readme: README.md\n  summary: SUMMARY.md\n", encoding="utf-8"
    )
    (output / "SOURCE_REVISION").write_text(revision + "\n", encoding="ascii")
    print(f"Built {len(pages)} pages and {len(assets)} assets from {revision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="HEAD")
    arguments = parser.parse_args()
    build(arguments.output, arguments.revision)
