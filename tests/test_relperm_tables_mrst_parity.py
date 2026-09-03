"""MRST parity + self-checks for SWFN/SGFN/SOF2/SOF3/Corey relperm readers
(PRSTCore.ad_props.relperm_tables), companion to scripts/export_mrst_relperm_tables.m.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_props.relperm_tables import assign_sgfn, assign_sof2, assign_sof3, assign_swfn, corey_relperm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_relperm_tables_match_mrst(tmp_path: Path):
    reference = tmp_path / "relperm_tables_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_relperm_tables('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    swfn = np.array([[0.2, 0.0, 5.0e4], [0.3, 0.02, 3.0e4], [0.5, 0.18, 1.0e4],
                      [0.7, 0.5, 3.0e3], [0.9, 0.85, 0], [1.0, 1.0, 0]])
    w = assign_swfn(swfn)
    assert np.allclose(w.krW(ref["sw_query"]), ref["krW_ref"], atol=1e-8)
    assert np.allclose(w.pcOW(ref["sw_query"]), ref["pcOW_ref"], atol=1e-6)
    assert np.allclose(w.points, ref["swfn_points"])

    sgfn = np.array([[0.0, 0.0, 0], [0.1, 0.0, 0], [0.3, 0.15, 500], [0.5, 0.4, 1200], [0.7, 0.8, 2500]])
    g = assign_sgfn(sgfn)
    assert np.allclose(g.krG(ref["sg_query"]), ref["krG_ref"], atol=1e-8)
    assert np.allclose(g.pcOG(ref["sg_query"]), ref["pcOG_ref"], atol=1e-6)
    assert np.allclose(g.points, ref["sgfn_points"])

    sof3 = np.array([[0.0, 0.0, 0.0], [0.2, 0.03, 0.02], [0.4, 0.2, 0.15], [0.6, 0.5, 0.45], [0.8, 1.0, 0.9]])
    o3 = assign_sof3(sof3)
    assert np.allclose(o3.krOW(ref["so_query"]), ref["krOW_sof3_ref"], atol=1e-8)
    assert np.allclose(o3.krOG(ref["so_query"]), ref["krOG_sof3_ref"], atol=1e-8)

    kr = corey_relperm(ref["s_query"], n=2.5, sr=0.15, sr_tot=0.35, kr_max=0.9)
    assert np.allclose(kr, ref["kr_corey_ref"], atol=1e-8)


def test_sof2_matches_the_same_interpolation_primitive_as_sof3():
    """SOF2's krO is a plain 1D linear interpolation of [So, krO] -- the
    same primitive validated against MRST for SWFN/SGFN/SOF3 above (SOF2's
    own MRST entry point uses an internal, undocumented region-mapping
    helper not exercised here; see export_mrst_relperm_tables.m)."""
    sof2 = np.array([[0.0, 0.0], [0.2, 0.05], [0.4, 0.25], [0.6, 0.55], [0.8, 1.0]])
    o2 = assign_sof2(sof2)
    so_q = np.linspace(0.0, 0.8, 25)
    assert np.allclose(o2.krO(so_q), np.interp(so_q, sof2[:, 0], sof2[:, 1]))


def test_corey_relperm_endpoints_and_monotonicity():
    s = np.linspace(0.0, 1.0, 50)
    kr = corey_relperm(s, n=2.0, sr=0.2, sr_tot=0.3, kr_max=0.8)
    assert np.isclose(kr[s <= 0.2].max(), 0.0)
    assert np.isclose(kr.max(), 0.8, atol=1e-8)
    assert np.all(np.diff(kr) >= -1e-12)  # monotone non-decreasing
