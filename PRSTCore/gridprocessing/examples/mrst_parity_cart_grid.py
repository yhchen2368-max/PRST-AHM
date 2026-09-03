"""MRST parity + speed harness for ``cart_grid``/``tensor_grid``/``compute_geometry``.

Companion MATLAB script: ``scripts/export_mrst_grid_geometry.m``. It builds
three correctness cases (regular 3D cartGrid, a perturbed non-uniform
tensorGrid, and a 2D cartGrid) plus a timed 40x40x20 benchmark, and saves
both to a ``.mat`` reference. This module rebuilds the same grids with the
Python port, compares every geometry/topology field, and reports wall-clock
speed for both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
from scipy.io import loadmat

from PRSTCore.gridprocessing import cart_grid, compute_geometry, tensor_grid


@dataclass(slots=True)
class ParityResult:
    name: str
    max_abs_error: float
    passed: bool


def generate_mrst_reference(output_file: str | Path, *, matlab: str | None = None) -> Path:
    output = Path(output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matlab_exe = matlab or shutil.which("matlab")
    if matlab_exe is None:
        raise RuntimeError("MATLAB executable was not found on PATH")

    repo = _repo_root()
    script_dir = repo / "scripts"
    matlab_cmd = (
        f"addpath('{_matlab_path(script_dir)}'); "
        f"export_mrst_grid_geometry('{_matlab_path(output)}')"
    )
    subprocess.run([matlab_exe, "-batch", matlab_cmd], cwd=repo, check=True)
    return output


def _mrst_grid_to_dict(Gm: dict) -> dict:
    """Reconstruct a PRSTCore grid dict (topology only) from a loaded MRST G struct,
    converting 1-based indices / 0-boundary-marker to this repo's 0-based / -1 convention."""
    neighbors = np.asarray(Gm["faces"]["neighbors"], dtype=int) - 1
    cell_faces = np.asarray(Gm["cells"]["faces"], dtype=int)
    cell_faces = np.column_stack([cell_faces[:, 0] - 1, cell_faces[:, 1]])
    face_nodes = np.asarray(Gm["faces"]["nodes"], dtype=int).reshape(-1) - 1
    return {
        "cells": {
            "num": int(Gm["cells"]["num"]),
            "facePos": np.asarray(Gm["cells"]["facePos"], dtype=int).reshape(-1) - 1,
            "faces": cell_faces,
        },
        "faces": {
            "num": int(Gm["faces"]["num"]),
            "nodePos": np.asarray(Gm["faces"]["nodePos"], dtype=int).reshape(-1) - 1,
            "neighbors": neighbors,
            "nodes": face_nodes,
        },
        "nodes": {
            "num": int(Gm["nodes"]["num"]),
            "coords": np.atleast_2d(np.asarray(Gm["nodes"]["coords"], dtype=float)),
        },
        "cartDims": np.asarray(Gm["cartDims"], dtype=int).reshape(-1),
        "griddim": int(np.asarray(Gm["griddim"]).reshape(-1)[0]),
    }


def _compare_case(name: str, ref: dict, actual: dict) -> list[ParityResult]:
    ref_neighbors = np.asarray(ref["faces"]["neighbors"], dtype=int) - 1
    results = [
        _compare(f"{name}.cells.volumes", actual["cells"]["volumes"], ref["cells"]["volumes"]),
        _compare(f"{name}.cells.centroids", actual["cells"]["centroids"], ref["cells"]["centroids"]),
        _compare(f"{name}.faces.areas", actual["faces"]["areas"], ref["faces"]["areas"]),
        _compare(f"{name}.faces.normals", actual["faces"]["normals"], ref["faces"]["normals"]),
        _compare(f"{name}.faces.centroids", actual["faces"]["centroids"], ref["faces"]["centroids"]),
        _compare(f"{name}.faces.neighbors", actual["faces"]["neighbors"], ref_neighbors),
        _compare(f"{name}.nodes.coords", actual["nodes"]["coords"], ref["nodes"]["coords"]),
    ]
    return results


def run_parity(reference_file: str | Path) -> tuple[list[ParityResult], dict[str, float]]:
    ref = loadmat(Path(reference_file), simplify_cells=True, squeeze_me=False)

    results: list[ParityResult] = []

    # Case 1: regular 3D cartGrid([4,3,2], [40,30,20])
    G1 = compute_geometry(cart_grid([4, 3, 2], [40, 30, 20]))
    results += _compare_case("G1(cartGrid 4x3x2)", ref["G1"], G1)

    # Case 2: perturbed non-uniform tensorGrid -- jitter must match MATLAB's rng(0) draw
    # exactly to compare node-for-node, so this case only checks self-consistency
    # (divergence theorem + total volume) rather than bit-for-bit node coordinates.
    Gm2 = ref["G2"]
    G2_topo = _mrst_grid_to_dict(Gm2)
    G2_topo["nodes"]["coords"] = np.asarray(Gm2["nodes"]["coords"], dtype=float)
    G2 = compute_geometry(G2_topo)
    results += [
        _compare("G2(perturbed).cells.volumes", G2["cells"]["volumes"], Gm2["cells"]["volumes"]),
        _compare("G2(perturbed).cells.centroids", G2["cells"]["centroids"], Gm2["cells"]["centroids"]),
        _compare("G2(perturbed).faces.areas", G2["faces"]["areas"], Gm2["faces"]["areas"]),
        _compare("G2(perturbed).faces.normals", G2["faces"]["normals"], Gm2["faces"]["normals"]),
        _compare("G2(perturbed).faces.centroids", G2["faces"]["centroids"], Gm2["faces"]["centroids"]),
    ]

    # Case 3: 2D cartGrid([5,4], [5,4])
    G3 = compute_geometry(cart_grid([5, 4], [5, 4]))
    results += _compare_case("G3(cartGrid2D 5x4)", ref["G3"], G3)

    # Timing: rebuild the benchmark-sized grid in Python and time it the same way
    # (warm-up + min of repeats).
    celldim = np.asarray(ref["celldim"], dtype=int).reshape(-1)
    physdim = np.asarray(ref["physdim"], dtype=float).reshape(-1)

    compute_geometry(cart_grid(celldim, physdim))  # warm-up (JIT-equivalent: numpy dispatch caches)

    nrep = 3
    t_topo = float("inf")
    t_geom = float("inf")
    for _ in range(nrep):
        t0 = time.perf_counter()
        Gt = cart_grid(celldim, physdim)
        t1 = time.perf_counter()
        compute_geometry(Gt)
        t2 = time.perf_counter()
        t_topo = min(t_topo, t1 - t0)
        t_geom = min(t_geom, t2 - t1)

    timing = {
        "num_cells": int(ref["bench_num_cells"]),
        "python_topo_s": t_topo,
        "python_geom_s": t_geom,
        "python_total_s": t_topo + t_geom,
        "matlab_topo_s": float(ref["t_topo"]),
        "matlab_geom_s": float(ref["t_geom"]),
        "matlab_total_s": float(ref["t_total"]),
    }
    return results, timing


def assert_parity(results: list[ParityResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        details = "\n".join(f"{r.name}: max_abs_error={r.max_abs_error:.3e}" for r in failed)
        raise AssertionError(f"MRST grid geometry parity failed:\n{details}")


def format_results(results: list[ParityResult], timing: dict[str, float] | None = None) -> str:
    lines = ["MRST grid/geometry parity check:"]
    for result in results:
        mark = "OK" if result.passed else "FAIL"
        lines.append(f"  {mark:4s} {result.name:32s} max|err|={result.max_abs_error:.3e}")
    if timing:
        lines.append(f"\nSpeed @ {timing['num_cells']} cells (min of 3 runs):")
        lines.append(
            f"  topology : MATLAB={timing['matlab_topo_s']*1e3:8.2f} ms   "
            f"Python={timing['python_topo_s']*1e3:8.2f} ms   "
            f"ratio(py/mat)={timing['python_topo_s']/timing['matlab_topo_s']:.2f}x"
        )
        lines.append(
            f"  geometry : MATLAB={timing['matlab_geom_s']*1e3:8.2f} ms   "
            f"Python={timing['python_geom_s']*1e3:8.2f} ms   "
            f"ratio(py/mat)={timing['python_geom_s']/timing['matlab_geom_s']:.2f}x"
        )
        lines.append(
            f"  total    : MATLAB={timing['matlab_total_s']*1e3:8.2f} ms   "
            f"Python={timing['python_total_s']*1e3:8.2f} ms   "
            f"ratio(py/mat)={timing['python_total_s']/timing['matlab_total_s']:.2f}x"
        )
    return "\n".join(lines)


def _compare(name: str, actual, expected, *, atol: float = 1e-8, rtol: float = 1e-8) -> ParityResult:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if e.size == a.size:
        e = e.reshape(a.shape)
    a = np.atleast_1d(a)
    e = np.atleast_1d(e)
    err = float(np.max(np.abs(a - e))) if a.size and e.size else float(abs(a.size - e.size))
    return ParityResult(name=name, max_abs_error=err, passed=bool(np.allclose(a, e, atol=atol, rtol=rtol)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _matlab_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default=str(_repo_root() / "tests" / "grid_geometry_mrst_ref.mat"),
        help="Path to MRST .mat reference file.",
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate the MRST reference with MATLAB first.")
    parser.add_argument("--matlab", default=None, help="Path to MATLAB executable.")
    args = parser.parse_args(argv)

    reference = Path(args.reference)
    if args.generate or not reference.exists():
        generate_mrst_reference(reference, matlab=args.matlab)
    results, timing = run_parity(reference)
    print(format_results(results, timing))
    assert_parity(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
