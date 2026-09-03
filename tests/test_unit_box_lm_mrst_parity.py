"""MRST parity + self-checks for unit_box_lm (PRSTCore.optimization.optim.unit_box_lm),
companion to scripts/export_mrst_unit_box_lm.m.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.optimization.optim.unit_box_lm import unit_box_lm

REPO_ROOT = Path(__file__).resolve().parents[1]

_A = np.array([
    [1.0, 0.5, -0.3, 0.2],
    [0.4, -1.0, 0.6, 0.1],
    [-0.2, 0.3, 1.2, -0.5],
    [0.6, 0.2, -0.4, 0.9],
    [-0.5, 0.7, 0.1, 0.3],
    [0.3, -0.2, 0.5, 1.1],
])
_B = np.array([0.8, -0.4, 0.6, 1.0, -0.2, 0.5])
_U0 = np.array([0.2, 0.3, 0.4, 0.5])

_A2 = np.array([[2.0, 0.1], [0.1, 2.0], [-1.0, 0.5]])
_B2 = np.array([5.0, -5.0, 0.0])
_U02 = np.array([0.5, 0.5])


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_unit_box_lm_matches_mrst(tmp_path: Path):
    reference = tmp_path / "unit_box_lm_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_unit_box_lm('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    v, u, history = unit_box_lm(_U0, lambda u: (_A @ u - _B, _A), verbose=False)
    assert np.isclose(v, ref["v"], atol=1e-8)
    assert np.allclose(u, ref["u"], atol=1e-8)
    assert np.allclose(np.array(history["u"]), ref["u_hist"], atol=1e-6)

    v2, u2, _ = unit_box_lm(_U02, lambda u: (_A2 @ u - _B2, _A2), verbose=False)
    assert np.isclose(v2, ref["v2"], atol=1e-8)
    assert np.allclose(u2, ref["u2"], atol=1e-8)


def test_unit_box_lm_recovers_unconstrained_least_squares_optimum():
    """Self-consistency check: for a linear residual with its unconstrained
    least-squares optimum strictly inside [0,1]^n, the box constraint never
    binds and unit_box_lm must converge to the exact normal-equations
    solution."""
    rng = np.random.default_rng(1)
    A = rng.uniform(-1, 1, (10, 3))
    u_star = np.array([0.3, 0.5, 0.4])
    b = A @ u_star

    v, u, history = unit_box_lm(np.array([0.5, 0.5, 0.5]), lambda u: (A @ u - b, A), verbose=False)
    # Gauss-Newton converges superlinearly on an exactly-linear residual, so
    # this legitimately stops via res_tol_abs (default 1e-5) well before
    # the gradient-norm tolerance -- both are valid stopping criteria.
    assert v < 1e-4
    assert np.allclose(u, u_star, atol=1e-2)
    assert len(history["val"]) <= 5


def test_unit_box_lm_respects_box_constraints():
    """The optimizer must never return an iterate outside [0,1]^n even
    when the unconstrained optimum lies outside the box."""
    A = np.array([[3.0, 0.0], [0.0, 3.0]])
    b = np.array([10.0, -10.0])  # unconstrained optimum at [10/3, -10/3]
    v, u, history = unit_box_lm(np.array([0.5, 0.5]), lambda u: (A @ u - b, A), verbose=False)
    assert np.all(u >= 0) and np.all(u <= 1)
    for uh in history["u"]:
        assert np.all(uh >= 0) and np.all(uh <= 1)
    assert np.allclose(u, [1.0, 0.0], atol=1e-8)
