"""MRST parity + speed harness for ``compute_trans``/``incomp_tpfa``.

Companion MATLAB script: ``scripts/export_mrst_incomp_tpfa.m``. It solves a
small 3D case (rate-controlled injector, bhp-controlled producer, a
Dirichlet-pressure boundary side) with MRST's own ``incompTPFA``, exports
the well/bc inputs it used (so this harness doesn't need a ported
``addWell``/``pside``/``makeRock`` first -- this test is scoped to
``incompTPFA``'s own linear algebra), and times a larger source-only case
for a speed comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.io import loadmat

from PRSTCore.gridprocessing import cart_grid, compute_geometry
from PRSTCore.solvers.incomp import compute_trans, incomp_tpfa


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
        f"export_mrst_incomp_tpfa('{_matlab_path(output)}')"
    )
    subprocess.run([matlab_exe, "-batch", matlab_cmd], cwd=repo, check=True)
    return output


def run_parity(reference_file: str | Path) -> tuple[list[ParityResult], dict[str, float]]:
    ref = loadmat(Path(reference_file), simplify_cells=True, squeeze_me=True)

    celldim = np.asarray(ref["celldim"], dtype=int).reshape(-1)
    physdim = np.asarray(ref["physdim"], dtype=float).reshape(-1)
    G = compute_geometry(cart_grid(celldim, physdim))
    rock = {"perm": np.asarray(ref["rock_perm"], dtype=float).reshape(-1)}
    T = compute_trans(G, rock)
    fluid = {"mu": 1e-3}

    well_cells = np.atleast_1d(ref["well_cells"])
    well_WI = np.atleast_1d(ref["well_WI"])
    well_type = np.atleast_1d(ref["well_type"])
    well_val = np.atleast_1d(ref["well_val"])
    wells = [
        {
            "cells": np.atleast_1d(well_cells[k]).astype(int).reshape(-1) - 1,
            "WI": np.atleast_1d(well_WI[k]).astype(float).reshape(-1),
            "type": str(well_type[k]),
            "val": float(well_val[k]),
        }
        for k in range(well_cells.size)
    ]

    bc_face = np.atleast_1d(ref["bc_face"]).astype(int).reshape(-1) - 1
    bc_type_raw = np.atleast_1d(ref["bc_type"])
    bc_type = [str(t) for t in bc_type_raw]
    bc_value = np.atleast_1d(ref["bc_value"]).astype(float).reshape(-1)
    bc = {"face": bc_face, "type": bc_type, "value": bc_value}

    state = incomp_tpfa(G, T, fluid, wells=wells, bc=bc)

    ref_neighbors = np.asarray(ref["neighbors"], dtype=int) - 1
    results = [
        _compare("pressure", state["pressure"], ref["pressure"]),
        _compare("flux", state["flux"], ref["flux"]),
        _compare("facePressure", state["facePressure"], ref["facePressure"]),
        _compare("neighbors(sanity)", G["faces"]["neighbors"], ref_neighbors),
        _compare(
            "well[0].flux(sum)",
            np.sum(state["wellSol"][0]["flux"]),
            np.atleast_1d(ref["well_flux"])[0] if np.ndim(ref["well_flux"]) else ref["well_flux"],
        ),
        _compare("well[0].pressure", state["wellSol"][0]["pressure"], np.atleast_1d(ref["well_pressure"])[0]),
        _compare("well[1].pressure", state["wellSol"][1]["pressure"], np.atleast_1d(ref["well_pressure"])[1]),
    ]

    # Timing: rebuild the source-only benchmark case and time it the same way.
    celldim_big = np.asarray(ref["celldim_big"], dtype=int).reshape(-1)
    Gbig = compute_geometry(cart_grid(celldim_big, celldim_big.astype(float)))
    rock_big = {"perm": np.full(Gbig["cells"]["num"], 1e-13)}
    src = {"cell": np.array([0, Gbig["cells"]["num"] - 1]), "rate": np.array([1e-3, -1e-3])}

    compute_trans(Gbig, rock_big)
    incomp_tpfa(Gbig, compute_trans(Gbig, rock_big), fluid, src=src)  # warm-up

    nrep = 3
    t_trans = float("inf")
    t_solve = float("inf")
    for _ in range(nrep):
        t0 = time.perf_counter()
        Tb = compute_trans(Gbig, rock_big)
        t1 = time.perf_counter()
        incomp_tpfa(Gbig, Tb, fluid, src=src)
        t2 = time.perf_counter()
        t_trans = min(t_trans, t1 - t0)
        t_solve = min(t_solve, t2 - t1)

    timing = {
        "num_cells": int(ref["bench_num_cells"]),
        "python_trans_s": t_trans,
        "python_solve_s": t_solve,
        "matlab_trans_s": float(ref["t_trans"]),
        "matlab_solve_s": float(ref["t_solve"]),
    }
    return results, timing


def assert_parity(results: list[ParityResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        details = "\n".join(f"{r.name}: max_abs_error={r.max_abs_error:.3e}" for r in failed)
        raise AssertionError(f"MRST incompTPFA parity failed:\n{details}")


def format_results(results: list[ParityResult], timing: dict[str, float] | None = None) -> str:
    lines = ["MRST incompTPFA parity check:"]
    for result in results:
        mark = "OK" if result.passed else "FAIL"
        lines.append(f"  {mark:4s} {result.name:24s} max|err|={result.max_abs_error:.3e}")
    if timing:
        lines.append(f"\nSpeed @ {timing['num_cells']} cells (min of 3 runs):")
        lines.append(
            f"  computeTrans : MATLAB={timing['matlab_trans_s']*1e3:8.2f} ms   "
            f"Python={timing['python_trans_s']*1e3:8.2f} ms   "
            f"ratio(py/mat)={timing['python_trans_s']/timing['matlab_trans_s']:.2f}x"
        )
        lines.append(
            f"  incompTPFA   : MATLAB={timing['matlab_solve_s']*1e3:8.2f} ms   "
            f"Python={timing['python_solve_s']*1e3:8.2f} ms   "
            f"ratio(py/mat)={timing['python_solve_s']/timing['matlab_solve_s']:.2f}x"
        )
    return "\n".join(lines)


def _compare(name: str, actual, expected, *, atol: float = 1e-6, rtol: float = 1e-6) -> ParityResult:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if e.size == a.size:
        e = e.reshape(a.shape)
    a = np.atleast_1d(a)
    e = np.atleast_1d(e)
    err = float(np.max(np.abs(a - e))) if a.size and e.size else float(abs(a.size - e.size))
    return ParityResult(name=name, max_abs_error=err, passed=bool(np.allclose(a, e, atol=atol, rtol=rtol)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _matlab_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default=str(_repo_root() / "tests" / "incomp_tpfa_mrst_ref.mat"),
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
