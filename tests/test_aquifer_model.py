"""Tests for AquiferModel (AquiferModel.m + computeInitAquifer.m port) and
process_aquifer (processAquifer.m port).

process_aquifer is validated exactly against real MRST (the local
MSW.data deck, which has AQUANCON/AQUFETP) -- see
test_process_aquifer_mrst_parity below. AquiferModel's flux/init-pressure/
post-convergence-update logic has no reachable MRST reference within this
session's scope (it needs a full ReservoirModel to mock: getProps,
getPhaseNames, fluid.rhoWS, gravity), so it is validated by mathematical
self-consistency: the closed-form flux formula, the least-squares
optimality condition computeInitAquifer solves, and the mass-balance
identity update_after_convergence must satisfy exactly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.aquifer_model import AquiferModel
from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.params.process_aquifer import process_aquifer
from PRSTCore.gridprocessing.cart_grid import cart_grid
from PRSTCore.gridprocessing.compute_geometry import compute_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
MSW_DECK = REPO_ROOT / "mrst-2026a" / "modules" / "nwm" / "data" / "MSW.data"


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def _msw_grid_and_deck():
    deck = convert_deck_units(read_eclipse_deck(str(MSW_DECK)))
    cd_ = deck["GRID"]["cartDims"]
    G = cart_grid(cd_, [cd_[0] * 100.0, cd_[1] * 100.0, cd_[2] * 6.0])
    G["nodes"]["coords"][:, 2] += 1000.0
    G = compute_geometry(G)
    return deck, G


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_process_aquifer_matches_mrst(tmp_path: Path):
    reference = tmp_path / "process_aquifer_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_process_aquifer('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    deck, G = _msw_grid_and_deck()
    out = process_aquifer(deck, G)

    assert out["initval"]["pressures"][0] == pytest.approx(ref["initval"]["pressures"], rel=1e-10)
    assert out["initval"]["volumes"][0] == pytest.approx(ref["initval"]["volumes"], rel=1e-10)

    py_conn = out["aquifers"][:, out["aquind"]["conn"]].astype(int)
    ref_conn = ref["aquifers"][:, ref["aquind"]["conn"] - 1].astype(int) - 1
    assert set(py_conn.tolist()) == set(ref_conn.tolist())

    py_sorted = out["aquifers"][np.argsort(py_conn)]
    ref_sorted = ref["aquifers"][np.argsort(ref_conn)]
    for field in ("J", "C", "alpha", "depthconn", "depthaq", "pvttbl"):
        pi, ri = out["aquind"][field], ref["aquind"][field] - 1
        assert np.allclose(py_sorted[:, pi], ref_sorted[:, ri], rtol=1e-8, atol=1e-12), field


def test_process_aquifer_runs_on_real_deck_and_is_physically_sane():
    """Self-contained (no MATLAB) structural check."""
    deck, G = _msw_grid_and_deck()
    out = process_aquifer(deck, G)
    assert out["aquifers"].shape == (625, 8)
    conn = out["aquifers"][:, out["aquind"]["conn"]].astype(int)
    assert np.all(conn >= 0) and np.all(conn < G["cells"]["num"])
    assert np.isclose(np.sum(out["aquifers"][:, out["aquind"]["alpha"]]), 1.0)
    assert out["initval"]["pressures"][0] > 0
    assert out["initval"]["volumes"][0] > 0


def _synthetic_aquifer(n_aquifers=2, nconn_per=5, seed=0):
    rng = np.random.default_rng(seed)
    nconn = n_aquifers * nconn_per
    aquind = {"aquid": 0, "conn": 1, "pvttbl": 2, "J": 3, "C": 4,
              "alpha": 5, "depthconn": 6, "depthaq": 7}
    aquifers = np.zeros((nconn, 8))
    aquid = np.repeat(np.arange(1, n_aquifers + 1), nconn_per)
    aquifers[:, aquind["aquid"]] = aquid
    aquifers[:, aquind["conn"]] = np.arange(nconn)
    aquifers[:, aquind["J"]] = rng.uniform(1e-9, 1e-7, nconn)
    aquifers[:, aquind["C"]] = rng.uniform(1e-10, 1e-9, nconn)
    alpha = rng.uniform(0.5, 1.5, nconn)
    for a in range(1, n_aquifers + 1):
        m = aquid == a
        alpha[m] /= alpha[m].sum()
    aquifers[:, aquind["alpha"]] = alpha
    aquifers[:, aquind["depthconn"]] = rng.uniform(1000, 1050, nconn)
    aquifers[:, aquind["depthaq"]] = rng.uniform(1000, 1050, n_aquifers)[aquid - 1]
    aquiferprops = {"C": rng.uniform(1e-10, 1e-9, n_aquifers)}
    initval = {"pressures": rng.uniform(2.5e7, 3.5e7, n_aquifers),
               "volumes": rng.uniform(1e9, 1e11, n_aquifers)}
    return AquiferModel(aquifers, aquind, aquiferprops, initval), nconn


def test_compute_aquifer_fluxes_matches_closed_form_at_dt_zero():
    model, nconn = _synthetic_aquifer()
    rng = np.random.default_rng(1)
    p_aq = model.initval["pressures"]
    v_aq = model.initval["volumes"]
    pW = rng.uniform(2e7, 4e7, nconn)
    bW = rng.uniform(1.0, 1.05, nconn)
    rhoWS = 1000.0
    q = model.compute_aquifer_fluxes(p_aq=p_aq, v_aq=v_aq, pW_conn=pW, bW_conn=bW, rhoWS=rhoWS, dt=0.0)

    ix = model.aquind
    aquid2conn = model._aquid2conn()
    p_aq_conn = aquid2conn @ p_aq
    ix_alpha, ix_J = model.aquifers[:, ix["alpha"]], model.aquifers[:, ix["J"]]
    depthconn, depthaq = model.aquifers[:, ix["depthconn"]], model.aquifers[:, ix["depthaq"]]
    expected = ix_alpha * ix_J * (p_aq_conn - pW + bW * rhoWS * (9.80665 * (depthconn - depthaq)))
    assert np.allclose(q, expected, atol=1e-10)


def test_compute_init_aquifer_satisfies_least_squares_optimality():
    """The p_aq that computeInitAquifer returns must be the exact
    minimizer of ||q(p_aq)||^2 (q affine in p_aq), i.e. M^T @ q = 0 at the
    solution -- the normal-equations optimality condition, independent of
    any MRST reference."""
    model, nconn = _synthetic_aquifer(n_aquifers=3, nconn_per=8, seed=2)
    rng = np.random.default_rng(3)
    pW = rng.uniform(2e7, 4e7, nconn)
    bW = rng.uniform(1.0, 1.05, nconn)
    rhoWS = 1000.0
    init_vol = rng.uniform(1e9, 1e11, model.n_aquifers)

    result = model.compute_init_aquifer(pW_conn=pW, bW_conn=bW, rhoWS=rhoWS, initaqvolumes=init_vol)
    assert result["volume"] is not None
    assert np.array_equal(result["volume"], init_vol)

    # Optimality condition for min ||q(p_aq)||^2 with q = M @ p_aq + r is
    # M^T @ q(p_aq*) = 0 -- recover M the same way compute_init_aquifer
    # does (the Jacobian of q w.r.t. p_aq at any point, since q is affine).
    p_aq0_adi = SparseADI.variable(np.zeros(model.n_aquifers), model.n_aquifers, 0)
    q_adi = model.compute_aquifer_fluxes(p_aq=p_aq0_adi, v_aq=init_vol,
                                          pW_conn=pW, bW_conn=bW, rhoWS=rhoWS, dt=0.0)
    M = q_adi.jac.toarray()

    q = model.compute_aquifer_fluxes(p_aq=result["pressure"], v_aq=init_vol,
                                      pW_conn=pW, bW_conn=bW, rhoWS=rhoWS, dt=0.0)
    optimality_residual = M.T @ q
    assert np.max(np.abs(optimality_residual)) < 1e-6 * np.max(np.abs(M.T @ q_adi.val))

    # And it must actually be better (or equal) than the trivial p_aq=0 case.
    q0 = model.compute_aquifer_fluxes(p_aq=np.zeros(model.n_aquifers), v_aq=init_vol,
                                       pW_conn=pW, bW_conn=bW, rhoWS=rhoWS, dt=0.0)
    assert np.sum(q ** 2) <= np.sum(q0 ** 2) + 1e-6


def test_update_after_convergence_mass_balance_identity():
    model, nconn = _synthetic_aquifer(seed=4)
    aquifer_sol = model.init_state_aquifer()
    rng = np.random.default_rng(5)
    q = rng.uniform(-1.0, 1.0, nconn)
    dt = 86400.0

    updated = model.update_after_convergence(aquifer_sol, q, dt)
    aquid2conn = model._aquid2conn()
    Q = dt * (aquid2conn.T @ q)
    assert np.allclose(updated["volume"], aquifer_sol["volume"] - Q)
    C = np.asarray(model.aquiferprops["C"])
    expected_p = aquifer_sol["pressure"] - Q / (C * aquifer_sol["volume"])
    assert np.allclose(updated["pressure"], expected_p)


def test_add_aquifer_contribution_numeric_and_adi_agree_and_are_localized():
    model, nconn = _synthetic_aquifer(n_aquifers=1, nconn_per=4, seed=6)
    nc = 10
    conn = model.aquifers[:, model.aquind["conn"]].astype(int)
    q = np.random.default_rng(7).uniform(-1, 1, nconn)

    eq_val = np.arange(nc, dtype=float)
    eq_numeric = SparseADI.constant(eq_val, nc)
    out_numeric = model.add_aquifer_contribution(eq_numeric, q)

    expected = eq_val.copy()
    np.add.at(expected, conn, -q)
    assert np.allclose(out_numeric.val, expected)

    eq_adi = SparseADI.variable(eq_val, nc, 0)
    q_adi = SparseADI.constant(q, nc)
    out_adi = model.add_aquifer_contribution(eq_adi, q_adi)
    assert np.allclose(out_adi.val, expected)

    # Cells with no aquifer connection are untouched.
    untouched = np.setdiff1d(np.arange(nc), conn)
    assert np.allclose(out_numeric.val[untouched], eq_val[untouched])
