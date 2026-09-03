"""Validate or exactly compare FAHM oracle fixture directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fixture(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "fahm-oracle-v1":
        raise ValueError(f"Unsupported schema: {manifest['schema_version']}")
    for item in manifest["deck_files"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["nbytes"]:
            raise ValueError(f"Wrong byte count for {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['path']}")
    for item in manifest["arrays"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["nbytes"]:
            raise ValueError(f"Wrong byte count for {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['path']}")
        values = np.fromfile(path, dtype=np.dtype(item["dtype"]))
        expected = int(np.prod(item["shape"], dtype=np.int64))
        if values.size != expected:
            raise ValueError(f"Wrong element count for {item['path']}")
        values.reshape(tuple(item["shape"]), order=item["order"])
    return manifest


def compare_fixtures(left: Path, right: Path) -> None:
    validate_fixture(left)
    validate_fixture(right)
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    if left_files != right_files:
        raise ValueError("Fixture file lists differ")
    differences = [path for path in left_files if (left / path).read_bytes() != (right / path).read_bytes()]
    if differences:
        raise ValueError("Fixture bytes differ: " + ", ".join(map(str, differences)))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("fixture", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        validate_fixture(args.fixture.resolve())
    else:
        compare_fixtures(args.left.resolve(), args.right.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
