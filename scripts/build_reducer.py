"""Rebuild the bundled core WASM with the pinned Rust toolchain and Cargo.lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="rebuild and compare without modifying bundled files")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "reducers/core"
ASSETS = ROOT / "src/hypergan/metrics_reducer/assets"
cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve()
env = dict(os.environ, RUSTFLAGS="-C link-arg=--max-memory=16777216 -C link-arg=-zstack-size=1048576 --remap-path-prefix=" + str(ROOT) + "=/hypergan --remap-path-prefix=" + str(cargo_home) + "=/cargo")
subprocess.run(["cargo", "build", "--locked", "--release", "--target", "wasm32-unknown-unknown"], cwd=CORE, env=env, check=True)
module = (CORE / "target/wasm32-unknown-unknown/release/hypergan_reducer.wasm").read_bytes()
ASSETS.mkdir(parents=True, exist_ok=True)
manifest = json.dumps({"abi": 1, "sha256": hashlib.sha256(module).hexdigest(), "bytes": len(module), "rust": "1.90.0", "max_memory_bytes": 16777216, "max_request_bytes": 262144, "max_batch_values": 1024, "reducers": ["mean/v1", "envelope/v1"]}, indent=2) + "\n"
if args.check:
    if (ASSETS / "reducer.wasm").read_bytes() != module or (ASSETS / "reducer.json").read_text() != manifest:
        raise SystemExit("Bundled reducer differs from pinned rebuild")
else:
    (ASSETS / "reducer.wasm").write_bytes(module)
    (ASSETS / "reducer.json").write_text(manifest)
print(f"Built {len(module)} bytes: {hashlib.sha256(module).hexdigest()}")
