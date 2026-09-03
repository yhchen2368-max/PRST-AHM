"""MRST parity + self-checks for VFPPROD/VFPINJ table interpolation
(PRSTCore.ad_props.vfp_table), companion to scripts/export_mrst_vfp_table.m.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_props.vfp_table import assign_vfpinj, assign_vfpprod

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_vfp_table_matches_mrst(tmp_path: Path):
    reference = tmp_path / "vfp_table_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_vfp_table('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    prod = assign_vfpprod(ref["flo"], ref["thp"], ref["wfr"], ref["gfr"], ref["alq"], ref["Q"],
                           flowtype="OIL", wfrtype="WOR", gfrtype="GOR", ref_depth=1000)
    bhp = prod.evaluate_bhp(ref["flo_q"], ref["thp_q"], ref["wfr_q"], ref["gfr_q"], ref["alq_q"])
    assert np.allclose(bhp, np.atleast_1d(ref["bhp_prod_ref"]), atol=1e-8)

    prod1 = assign_vfpprod(ref["flo"], ref["thp"], ref["wfr"], ref["gfr"], [ref["alq"][0]], ref["Q1"][..., None],
                            flowtype="LIQ", wfrtype="WCT", gfrtype="GLR", ref_depth=1000)
    bhp1 = prod1.evaluate_bhp(ref["flo_q"], ref["thp_q"], ref["wfr_q"], ref["gfr_q"])
    assert np.allclose(bhp1, np.atleast_1d(ref["bhp_prod1_ref"]), atol=1e-8)

    inj = assign_vfpinj(ref["floi"], ref["thpi"], ref["BHP"], flowtype="WAT", ref_depth=900)
    bhp_inj = inj.evaluate_bhp(ref["flo_qi"], ref["thp_qi"])
    assert np.allclose(bhp_inj, np.atleast_1d(ref["bhp_inj_ref"]), atol=1e-8)


def test_vfpprod_multilinear_interp_and_extrap_matches_closed_form():
    """Self-consistency check independent of MRST: since the synthetic
    table is a separable affine function of its axes, exact multilinear
    interpolation (and linear extrapolation) must reproduce the closed
    form everywhere, inside and outside the table's grid."""
    flo = np.array([100.0, 200.0, 300.0])
    thp = np.array([10.0, 20.0, 30.0])
    wfr = np.array([0.1, 0.5])
    gfr = np.array([50.0, 150.0])
    alq = np.array([0.0, 10.0])

    def closed_form(f, t, w, g, a):
        return 100 + 0.01 * f + 0.5 * t + 2 * w + 3 * g + 0.1 * a

    FLO, THP, WFR, GFR, ALQ = np.meshgrid(flo, thp, wfr, gfr, alq, indexing="ij")
    Q = closed_form(FLO, THP, WFR, GFR, ALQ)

    table = assign_vfpprod(flo, thp, wfr, gfr, alq, Q)

    rng = np.random.default_rng(0)
    n = 200
    fq = rng.uniform(50, 350, n)
    tq = rng.uniform(0, 40, n)
    wq = rng.uniform(-0.2, 0.8, n)
    gq = rng.uniform(0, 200, n)
    aq = rng.uniform(-5, 15, n)

    bhp = table.evaluate_bhp(fq, tq, wq, gq, aq)
    assert np.allclose(bhp, closed_form(fq, tq, wq, gq, aq), atol=1e-8)


def test_vfpinj_evaluate_bhp_basic():
    flo = np.array([50.0, 150.0, 250.0])
    thp = np.array([5.0, 15.0, 25.0])
    FLO, THP = np.meshgrid(flo, thp, indexing="ij")
    BHP = 50 + 0.02 * FLO + 0.8 * THP
    table = assign_vfpinj(flo, thp, BHP, flowtype="WAT", ref_depth=900)

    bhp = table.evaluate_bhp([50.0, 100.0, 250.0], [5.0, 10.0, 25.0])
    expected = 50 + 0.02 * np.array([50.0, 100.0, 250.0]) + 0.8 * np.array([5.0, 10.0, 25.0])
    assert np.allclose(bhp, expected, atol=1e-10)
