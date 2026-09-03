"""1D MRST parity example for PRSTCore flow diagnostics.

The companion MATLAB function ``scripts/flow_diagnostics_mrst_reference.m``
generates the MRST reference for the exact same state.  This module loads the
reference, rebuilds the Python/PRSTCore structures, runs diagnostics, and
compares all fields that the network/GPSNet path needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from scipy.io import loadmat

from PRSTCore.network_models import Network
from PRSTCore.visualization.diagnostics import (
    computeFandPhi,
    computeLorenz,
    computePressureAndDiagnostics,
    computeSweep,
    computeTOFandTracer,
    computeWellPairs,
)


@dataclass(slots=True)
class ParityResult:
    name: str
    max_abs_error: float
    passed: bool


def generate_mrst_reference(output_file: str | Path, *, matlab: str | None = None) -> Path:
    """Run MATLAB/MRST to generate the reference ``.mat`` file."""
    output = Path(output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matlab_exe = matlab or shutil.which("matlab")
    if matlab_exe is None:
        raise RuntimeError("MATLAB executable was not found on PATH")

    repo = _repo_root()
    script_dir = repo / "scripts"
    matlab_cmd = (
        f"addpath('{_matlab_path(script_dir)}'); "
        f"flow_diagnostics_mrst_reference('{_matlab_path(output)}')"
    )
    subprocess.run([matlab_exe, "-batch", matlab_cmd], cwd=repo, check=True)
    return output


def run_parity(reference_file: str | Path) -> list[ParityResult]:
    """Run the Python diagnostics and compare against a saved MRST reference."""
    ref = loadmat(Path(reference_file), simplify_cells=True, squeeze_me=True)
    G, rock, state, W, model = build_prstcore_case(ref)

    D = computeTOFandTracer(
        state,
        G,
        rock,
        wells=W,
        computeWellTOFs=True,
        firstArrival=False,
        model=model,
    )
    WP = computeWellPairs(state, G, rock, W, D)
    pressure_state, diagnostics = computePressureAndDiagnostics(
        model,
        wells=W,
        state=state,
        computeWellTOFs=True,
        firstArrival=False,
    )
    F, Phi = computeFandPhi(G["cells"]["volumes"] * rock["poro"], D.tof)
    Lorenz = computeLorenz(F, Phi)
    Ev, tD = computeSweep(F, Phi)

    problem = {
        "SimulatorSetup": {
            "model": model,
            "schedule": {"step": {"val": [1.0], "control": [1]}, "control": [{"W": W}]},
        },
        "OutputHandlers": {"states": [pressure_state]},
    }
    ntwrk = Network(W, G, type="fd_postprocessor", problem=problem, flow_filter=0.0)
    edge_T, edge_pv = ntwrk.get_edge_data()

    ref_D = ref["D"]
    ref_WP = ref["WP"]
    results = [
        _compare("D.inj", D.inj, _mrst_indices(ref_D["inj"])),
        _compare("D.prod", D.prod, _mrst_indices(ref_D["prod"])),
        _compare("D.tof", D.tof, ref_D["tof"]),
        _compare("D.itracer", D.itracer, ref_D["itracer"]),
        _compare("D.ptracer", D.ptracer, ref_D["ptracer"]),
        _compare("D.itof", D.itof, ref_D["itof"]),
        _compare("D.ptof", D.ptof, ref_D["ptof"]),
        _compare("WP.pairIx", WP.pairIx, _mrst_pair_ix(ref_WP["pairIx"])),
        _compare("WP.vols", WP.vols, ref_WP["vols"]),
        _compare("wellCommunication", diagnostics.wellCommunication, ref["wellCommunication"]),
        _compare("F", F, ref["F"]),
        _compare("Phi", Phi, ref["Phi"]),
        _compare("Lorenz", Lorenz, ref["Lorenz"]),
        _compare("Ev", Ev, ref["Ev"]),
        _compare("tD", tD, ref["tD"]),
        _compare("Network.num_edges", ntwrk.num_edges, 1),
        _compare("Network.T", edge_T, np.asarray([0.5])),
        _compare("Network.pv", edge_pv, np.asarray([3.0])),
    ]
    return results


def build_prstcore_case(ref: dict[str, Any]):
    """Convert the saved MRST structs into PRSTCore/Python dictionaries."""
    Gm = ref["G"]
    nc = int(Gm["cells"]["num"])
    neighbors = np.asarray(Gm["faces"]["neighbors"], dtype=int) - 1
    G = {
        "cells": {
            "num": nc,
            "centroids": np.asarray(Gm["cells"]["centroids"], dtype=float),
            "volumes": _as_vector(Gm["cells"]["volumes"]),
        },
        "faces": {"neighbors": neighbors},
    }
    rock = {
        "poro": _as_vector(ref["rock"]["poro"]),
        "perm": np.asarray(ref["rock"]["perm"], dtype=float).reshape((nc, -1)),
    }

    W = []
    well_count = _as_vector(ref["well_sign"]).size
    for i in range(well_count):
        W.append(
            {
                "cells": (_cell(ref["well_cells"], i).astype(int) - 1).tolist(),
                "name": str(np.atleast_1d(ref["well_names"])[i]),
                "sign": float(_as_vector(ref["well_sign"])[i]),
                "val": float(_as_vector(ref["well_val"])[i]),
                "status": bool(_as_vector(ref["well_status"])[i]),
                "refDepth": float(_as_vector(ref["well_refDepth"])[i]),
                "dZ": _cell(ref["well_dZ"], i).astype(float),
            }
        )

    state = {
        "pressure": _as_vector(ref["state"]["pressure"]),
        "flux": _as_vector(ref["state"]["flux"]),
        "wellSol": [],
    }
    for i in range(well_count):
        state["wellSol"].append(
            {
                "flux": _cell(ref["wellsol_flux"], i).astype(float),
                "pressure": float(_as_vector(ref["wellsol_pressure"])[i]),
                "bhp": float(_as_vector(ref["wellsol_bhp"])[i]),
                "status": bool(_as_vector(ref["well_status"])[i]),
                "sign": float(_as_vector(ref["well_sign"])[i]),
            }
        )

    internal = np.all(neighbors >= 0, axis=1)
    model = {
        "G": G,
        "rock": rock,
        "operators": {
            "N": neighbors[internal],
            "T": np.ones(np.count_nonzero(internal), dtype=float),
            "T_all": np.ones(np.count_nonzero(internal), dtype=float),
            "pv": G["cells"]["volumes"] * rock["poro"],
        },
    }
    return G, rock, state, W, model


def assert_parity(results: list[ParityResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        details = "\n".join(f"{r.name}: max_abs_error={r.max_abs_error:.3e}" for r in failed)
        raise AssertionError(f"MRST parity failed:\n{details}")


def format_results(results: list[ParityResult]) -> str:
    lines = ["MRST parity check:"]
    for result in results:
        mark = "OK" if result.passed else "FAIL"
        lines.append(f"  {mark:4s} {result.name:22s} max|err|={result.max_abs_error:.3e}")
    return "\n".join(lines)


def _compare(name: str, actual, expected, *, atol: float = 1e-10, rtol: float = 1e-10) -> ParityResult:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if e.size == a.size:
        e = e.reshape(a.shape)
    a = np.atleast_1d(a)
    e = np.atleast_1d(e)
    err = float(np.max(np.abs(a - e))) if a.size and e.size else float(abs(a.size - e.size))
    return ParityResult(name=name, max_abs_error=err, passed=bool(np.allclose(a, e, atol=atol, rtol=rtol)))


def _as_vector(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)


def _cell(values, index: int) -> np.ndarray:
    arr = np.atleast_1d(values)
    return np.atleast_1d(np.asarray(arr[index], dtype=float)).reshape(-1)


def _mrst_indices(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=int)).reshape(-1) - 1


def _mrst_pair_ix(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=int)).reshape((-1, 2)) - 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _matlab_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default=str(_repo_root() / "tests" / "flow_diagnostics_mrst_1d_ref.mat"),
        help="Path to MRST .mat reference file.",
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate the MRST reference with MATLAB first.")
    parser.add_argument("--matlab", default=None, help="Path to MATLAB executable.")
    args = parser.parse_args(argv)

    reference = Path(args.reference)
    if args.generate or not reference.exists():
        generate_mrst_reference(reference, matlab=args.matlab)
    results = run_parity(reference)
    print(format_results(results))
    assert_parity(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

