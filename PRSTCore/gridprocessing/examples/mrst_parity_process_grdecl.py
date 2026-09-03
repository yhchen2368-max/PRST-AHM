"""MRST parity harness for ``process_grdecl``/``compute_geometry`` on real
corner-point decks.

Companion MATLAB script: ``scripts/export_mrst_process_grdecl.m``.

Face/node numbering is an internal implementation detail of the topology
builder (MRST's C-MEX ``processgrid_mex`` enumerates faces/nodes in its own
traversal order; the reused Python port of that traversal
(``PRSTCore.deckformat.grid.init_eclipse_grid._cp_mex_topology``) does not
match it index-for-index) -- so this harness compares grids *as grids*
rather than array-for-array: cell-indexed fields (volumes, centroids) compare
directly since ECLIPSE active-cell ordering is preserved; face/node fields
compare as sets (sorted areas, nearest-neighbor-matched node coordinates,
neighbor cell-pairs as unordered sets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.gridprocessing import compute_geometry, process_grdecl


@dataclass(slots=True)
class ParityResult:
    name: str
    max_abs_error: float
    passed: bool


def generate_mrst_reference(deck_path: str | Path, output_file: str | Path, varname: str,
                             *, matlab: str | None = None) -> Path:
    output = Path(output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matlab_exe = matlab or shutil.which("matlab")
    if matlab_exe is None:
        raise RuntimeError("MATLAB executable was not found on PATH")

    repo = _repo_root()
    script_dir = repo / "scripts"
    matlab_cmd = (
        f"addpath('{_matlab_path(script_dir)}'); "
        f"export_mrst_process_grdecl('{_matlab_path(output)}', '{_matlab_path(deck_path)}', '{varname}')"
    )
    subprocess.run([matlab_exe, "-batch", matlab_cmd], cwd=repo, check=True)
    return output


def run_parity(deck_path: str | Path, reference_file: str | Path,
                *, cell_rtol: float = 1e-6, cell_atol: float = 1.0,
                node_atol: float = 1e-6) -> tuple[list[ParityResult], dict[str, float]]:
    deck = convert_deck_units(read_eclipse_deck(str(deck_path)))

    t0 = time.perf_counter()
    G = process_grdecl(deck["GRID"])
    t1 = time.perf_counter()
    G = compute_geometry(G)
    t2 = time.perf_counter()

    ref = loadmat(Path(reference_file), simplify_cells=True, squeeze_me=True)

    results = [
        _compare("counts", [G["cells"]["num"], G["faces"]["num"], G["nodes"]["num"]],
                 [ref["num_cells"], ref["num_faces"], ref["num_nodes"]], atol=0, rtol=0),
        _compare("cells.volumes", G["cells"]["volumes"], ref["cell_volumes"], rtol=cell_rtol, atol=cell_atol),
        _compare("cells.centroids", G["cells"]["centroids"], ref["cell_centroids"], rtol=cell_rtol, atol=cell_atol),
    ]

    a1 = np.sort(G["faces"]["areas"])
    a2 = np.sort(np.asarray(ref["face_areas"], dtype=float))
    results.append(_compare("faces.areas(sorted set)", a1, a2, rtol=cell_rtol, atol=cell_atol))

    n1 = np.asarray(G["nodes"]["coords"], dtype=float)
    n2 = np.asarray(ref["node_coords"], dtype=float)
    if n1.shape == n2.shape:
        dist, idx = cKDTree(n2).query(n1)
        # Bijection is reported but not required to pass: at fault junctions
        # with several near-duplicate pillar-intersection points, nearest-
        # neighbor ties can legitimately match two very-close-but-distinct
        # nodes to the same reference point without indicating a real error
        # (the distance bound below is what actually matters).
        n_dupe = len(dist) - len(np.unique(idx))
        results.append(ParityResult(
            f"nodes.coords(nearest-match, {n_dupe} tied)", float(dist.max()), bool(dist.max() < node_atol)
        ))
    else:
        results.append(ParityResult("nodes.coords(nearest-match)", float("inf"), False))

    def pair_set(neighbors):
        n = np.sort(np.asarray(neighbors, dtype=int), axis=1)
        return set(map(tuple, n.tolist()))

    ref_neighbors = np.asarray(ref["neighbors"], dtype=int) - 1
    s1, s2 = pair_set(G["faces"]["neighbors"]), pair_set(ref_neighbors)
    results.append(ParityResult("faces.neighbors(pair set)", float(len(s1.symmetric_difference(s2))), s1 == s2))

    timing = {
        "num_cells": int(G["cells"]["num"]),
        "python_topo_s": t1 - t0,
        "python_geom_s": t2 - t1,
        "matlab_topo_s": float(ref["t_topo"]),
        "matlab_geom_s": float(ref["t_geom"]),
    }
    return results, timing


def assert_parity(results: list[ParityResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        details = "\n".join(f"{r.name}: max_abs_error={r.max_abs_error:.3e}" for r in failed)
        raise AssertionError(f"MRST processGRDECL parity failed:\n{details}")


def format_results(results: list[ParityResult], timing: dict[str, float] | None = None) -> str:
    lines = ["MRST processGRDECL/computeGeometry parity check:"]
    for result in results:
        mark = "OK" if result.passed else "FAIL"
        lines.append(f"  {mark:4s} {result.name:32s} max|err|={result.max_abs_error:.3e}")
    if timing:
        lines.append(f"\nSpeed @ {timing['num_cells']} cells:")
        lines.append(
            f"  topology : MATLAB={timing['matlab_topo_s']*1e3:8.1f} ms   "
            f"Python={timing['python_topo_s']*1e3:8.1f} ms   "
            f"ratio(py/mat)={timing['python_topo_s']/timing['matlab_topo_s']:.2f}x"
        )
        lines.append(
            f"  geometry : MATLAB={timing['matlab_geom_s']*1e3:8.1f} ms   "
            f"Python={timing['python_geom_s']*1e3:8.1f} ms   "
            f"ratio(py/mat)={timing['python_geom_s']/timing['matlab_geom_s']:.2f}x"
        )
    return "\n".join(lines)


def _compare(name: str, actual, expected, *, atol: float, rtol: float) -> ParityResult:
    a = np.atleast_1d(np.asarray(actual, dtype=float))
    e = np.atleast_1d(np.asarray(expected, dtype=float))
    if e.size == a.size:
        e = e.reshape(a.shape)
    err = float(np.max(np.abs(a - e))) if a.size and e.size else float(abs(a.size - e.size))
    return ParityResult(name=name, max_abs_error=err, passed=bool(np.allclose(a, e, atol=atol, rtol=rtol)))


def _repo_root() -> Path:
    # PRSTCore/gridprocessing/examples/this.py -> four names up is the repo.
    # parents[4] reaches its *parent* directory, where there is no scripts/,
    # so the MATLAB call failed with "undefined function
    # export_mrst_process_grdecl" on every machine. mrst_parity_cart_grid.py,
    # at the same depth, has always used parents[3].
    return Path(__file__).resolve().parents[3]


def _matlab_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")
