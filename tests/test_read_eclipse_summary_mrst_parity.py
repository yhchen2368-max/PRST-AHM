"""MRST parity + self-checks for read_eclipse_summary/convert_summary_to_well_sols
(PRSTCore.deckformat.resultinput.read_eclipse_summary), companion to
scripts/export_mrst_read_eclipse_summary.m. Uses the real SPE9 case's
ECLIPSE summary output (examples/SPE9/RESULTS/SPE9_CP.SMSPEC/.UNSMRY).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.deckformat.resultinput.read_eclipse_summary import (
    convert_summary_to_well_sols,
    read_eclipse_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPE9_PREFIX = REPO_ROOT / "examples" / "SPE9" / "RESULTS" / "SPE9_CP"


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_read_eclipse_summary_matches_mrst(tmp_path: Path):
    reference = tmp_path / "read_eclipse_summary_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_read_eclipse_summary('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    smry = read_eclipse_summary(SPE9_PREFIX)
    assert smry["data"].shape[1] == int(ref["nsteps_ref"])
    assert smry["data"].shape[0] == int(ref["nlist_ref"])

    time = smry["get"](":+:+:+:+", "TIME")
    assert np.allclose(time, np.atleast_1d(ref["time_ref"]))
    assert np.allclose(smry["get"]("PRODU2", "WOPR"), np.atleast_1d(ref["wopr_ref"]))
    assert np.allclose(smry["get"]("PRODU2", "WBHP"), np.atleast_1d(ref["wbhp_ref"]))
    assert np.allclose(smry["get"]("PRODU2", "WWCT"), np.atleast_1d(ref["wwct_ref"]))
    assert np.allclose(smry["get"]("INJE1", "WBHP"), np.atleast_1d(ref["injwbhp_ref"]))
    assert smry["get_unit"]("PRODU2", "WBHP") == str(ref["unit_wbhp"]).strip()

    well_sols, ws_time = convert_summary_to_well_sols(SPE9_PREFIX)
    ref_names = [str(n).strip() for n in np.atleast_1d(ref["names"])]
    names = [w["name"] for w in well_sols[0]]
    assert names == ref_names

    assert np.allclose(ws_time, np.atleast_1d(ref["wsTime"]).ravel())

    nt = len(well_sols)
    nw = len(names)
    for kt in range(nt):
        for kw in range(nw):
            w = well_sols[kt][kw]
            assert np.isclose(w["qOs"], ref["qOs"][kt, kw], atol=1e-10)
            assert np.isclose(w["qWs"], ref["qWs"][kt, kw], atol=1e-10)
            assert np.isclose(w["qGs"], ref["qGs"][kt, kw], atol=1e-8)
            assert np.isclose(w["bhp"], ref["bhp"][kt, kw], atol=1e-6)
            assert np.isclose(w["sign"], ref["sgn"][kt, kw], atol=1e-10)


def test_read_eclipse_summary_structural_on_real_spe9_output():
    """Structural check that runs without MATLAB: verifies well discovery,
    monotone time, and physically sane sign conventions on the real SPE9
    ECLIPSE summary output."""
    smry = read_eclipse_summary(SPE9_PREFIX)
    assert smry["data"].shape[0] == len(smry["KEYWORDS"]) == len(smry["WGNAMES"])
    assert smry["intehead_unit"] == 2  # FIELD units, per SPE9_CP.SMSPEC's INTEHEAD

    well_sols, time = convert_summary_to_well_sols(smry)
    assert time.size == len(well_sols)
    assert np.all(np.diff(time) >= 0)

    names = {w["name"] for w in well_sols[0]}
    assert "INJE1" in names
    assert any(n.startswith("PRODU") for n in names)

    last = {w["name"]: w for w in well_sols[-1]}
    # Injector INJE1 should be injecting water (qWs > 0) by MRST's rate
    # sign convention (positive = into the well/reservoir).
    assert last["INJE1"]["qWs"] > 0
    # A producing well should show a negative oil rate (out of the
    # reservoir) once flowing.
    producers = [w for n, w in last.items() if n.startswith("PRODU")]
    assert any(p["qOs"] < 0 for p in producers)
    # bhp values should be physically plausible reservoir pressures (Pa).
    assert all(w["bhp"] > 0 for w in last.values())
