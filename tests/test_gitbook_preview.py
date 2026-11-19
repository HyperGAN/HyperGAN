from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class GitBookPreviewTests(unittest.TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.origin = self.root / "origin.git"
        self.run_command("git", "init", "-q", "--bare", str(self.origin))
        self.run_command("git", "init", "-q", "-b", "master")
        self.run_command("git", "config", "user.name", "Fixture")
        self.run_command("git", "config", "user.email", "fixture@example.org")
        self.run_command("git", "config", "commit.gpgsign", "false")
        self.run_command("git", "remote", "add", "origin", str(self.origin))
        self.write("README.md", "# Project root\n")
        self.write("docs/README.md", "# Docs\n\n## Guide\n\n- [Setup](setup.md)\n")
        self.write("docs/setup.md", "# Setup\n[Root](../README.md#install)\n![Logo](assets/logo.png)\n"
                   "[Source tree](../examples/)\n[license]: ../LICENSE\n"
                   "`[example](../example.md)`\n```md\n[example](../example.md)\n```\n")
        self.write("docs/assets/logo.png", "fixture image bytes")
        self.write("examples/model.toml", "[model]\n")
        self.write("LICENSE", "fixture license\n")
        self.source = self.commit("Initial docs")

    def run_command(self, *args, check=True):
        return subprocess.run(args, cwd=self.repo, check=check, capture_output=True, text=True)

    def write(self, name, text):
        file = self.repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")

    def commit(self, name):
        self.run_command("git", "add", ".")
        self.run_command("git", "commit", "-qm", name)
        return self.run_command("git", "rev-parse", "HEAD").stdout.strip()

    def build(self, name, check=True):
        output = self.root / name
        result = self.run_command("python3", str(SCRIPTS / "build_gitbook_preview.py"), str(output), check=check)
        return output, result

    def publish(self, output, check=True):
        return self.run_command("bash", str(SCRIPTS / "publish_gitbook_preview.sh"), str(output), check=check)

    def remote_tip(self):
        row = self.run_command("git", "ls-remote", "origin", "refs/heads/gitbook-preview").stdout
        return row.split("\t")[0]

    def test_tree_uses_tracked_bytes_and_revision_pinned_links(self):
        self.write("docs/setup.md", "# Uncommitted replacement\n")
        book, _ = self.build("one")
        path = f"https://github.com/HyperGAN/HyperGAN"
        page = (book / "setup.md").read_text()
        self.assertIn(f"[Root]({path}/blob/{self.source}/README.md#install)", page)
        self.assertIn(f"[Source tree]({path}/tree/{self.source}/examples)", page)
        self.assertIn(f"[license]: {path}/blob/{self.source}/LICENSE", page)
        self.assertIn("![Logo](assets/logo.png)", page)
        self.assertIn("`[example](../example.md)`", page)
        self.assertIn("```md\n[example](../example.md)\n```", page)
        self.assertIn("## Guide\n\n* [Setup](setup.md)", (book / "SUMMARY.md").read_text())
        self.assertEqual((book / "assets/logo.png").read_bytes(), b"fixture image bytes")
        self.assertEqual((book / "SOURCE_REVISION").read_text().strip(), self.source)

    def test_missing_relative_link_does_not_produce_a_book(self):
        self.write("docs/setup.md", "# Setup\n[Missing](missing.md)\n")
        self.commit("Missing link")
        book, result = self.build("broken", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("links to an untracked path: missing.md", result.stderr)
        self.assertFalse(book.exists())

    def test_packaged_preview_advances_and_refuses_unowned_tip(self):
        previous_book, _ = self.build("previous")
        self.publish(previous_book)
        previous_tip = self.remote_tip()
        self.publish(previous_book)
        self.assertEqual(self.remote_tip(), previous_tip)

        self.write("docs/setup.md", "# Setup\nReviewed update.\n")
        current_source = self.commit("Reviewed docs update")
        current_book, _ = self.build("current")
        self.publish(current_book)
        current_tip = self.remote_tip()
        self.assertNotEqual(previous_tip, current_tip)
        self.assertEqual(self.run_command("git", "rev-parse", f"{current_tip}^").stdout.strip(), previous_tip)
        self.assertEqual(self.run_command("git", "show", f"{current_tip}:SOURCE_REVISION").stdout.strip(), current_source)
        self.assertEqual(self.run_command("git", "show", f"{current_tip}:setup.md").stdout,
                         "# Setup\nReviewed update.\n")
        self.run_command("git", "checkout", "-q", "--detach", self.source)
        self.publish(previous_book)
        self.assertEqual(self.remote_tip(), current_tip)

        tree = self.run_command("git", "rev-parse", f"{current_tip}^{{tree}}").stdout.strip()
        foreign = self.run_command("git", "commit-tree", tree, "-p", current_tip, "-m", "GitBook edit").stdout.strip()
        self.run_command("git", "push", "origin", f"{foreign}:refs/heads/gitbook-preview")
        result = self.publish(previous_book, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Review the existing GitBook branch", result.stderr)
        self.assertEqual(self.remote_tip(), foreign)


if __name__ == "__main__":
    unittest.main()
