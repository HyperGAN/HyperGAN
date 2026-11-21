"""Run one deterministic file shard of a pytest selection."""
import argparse
import hashlib

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int)
    parser.add_argument('count', type=int)
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.index < args.count or not args.pytest_args:
        parser.error('require 0 <= index < count and a pytest selection')

    class FileShard:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            selected, deselected = [], []
            for item in items:
                path = item.path.relative_to(config.rootpath).as_posix()
                digest = hashlib.sha256(path.encode()).digest()
                owner = int.from_bytes(digest, 'big') % args.count
                (selected if owner == args.index else deselected).append(item)
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

    return pytest.main(args.pytest_args, plugins=[FileShard()])


if __name__ == '__main__':
    raise SystemExit(main())
