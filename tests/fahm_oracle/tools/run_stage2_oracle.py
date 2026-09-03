"""Regenerate FAHM static/dynamic oracles twice and enforce the Stage 2 gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tests.fahm_oracle.tools.extract_fahm_static_oracle import extract
from tests.fahm_oracle.tools.verify_fahm_oracle import compare_fixtures


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _matlab_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _run_matlab_export(matlab: Path, workspace: Path, output: Path) -> None:
    exporter = workspace / "RSTCore-main" / "tests" / "fahm_oracle" / "matlab"
    command = (
        f"addpath('{_matlab_quote(exporter)}'); "
        f"export_fahm_minimal_oracle('{_matlab_quote(workspace / 'MRST')}',"
        f"'{_matlab_quote(output)}');"
    )
    subprocess.run([str(matlab), "-batch", command], check=True)


def run_gate(workspace: Path, matlab: Path) -> dict[str, object]:
    fixture = workspace / "RSTCore-main" / "tests" / "fixtures" / "fahm_oracle" / "v1"
    with tempfile.TemporaryDirectory(prefix="fahm_static_") as static_tmp:
        regenerated_static = Path(static_tmp)
        extract(workspace, regenerated_static)
        if _tree_bytes(regenerated_static) != _tree_bytes(fixture / "static"):
            raise ValueError("Committed FAHM static oracle is stale")

    with tempfile.TemporaryDirectory(prefix="fahm_dynamic_1_") as first_tmp, tempfile.TemporaryDirectory(
        prefix="fahm_dynamic_2_"
    ) as second_tmp:
        first = Path(first_tmp)
        second = Path(second_tmp)
        _run_matlab_export(matlab, workspace, first)
        _run_matlab_export(matlab, workspace, second)
        compare_fixtures(first, second)
        compare_fixtures(first, fixture / "golden")
        files = [path for path in first.rglob("*") if path.is_file()]
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        return {
            "arrays": len(manifest["arrays"]),
            "dynamic_exports": 2,
            "exact_dynamic_bytes": True,
            "exact_static_bytes": True,
            "files": len(files),
            "oracle_id": manifest["oracle_id"],
            "status": "PASS",
            "total_bytes": sum(path.stat().st_size for path in files),
        }


def main() -> int:
    default_workspace = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument(
        "--matlab",
        type=Path,
        default=Path(r"D:\Program Files\MATLAB\R2022b\bin\matlab.exe"),
    )
    args = parser.parse_args()
    report = run_gate(args.workspace_root.resolve(), args.matlab.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
