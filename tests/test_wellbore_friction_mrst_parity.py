"""MRST parity test for well_bore_friction (wellBoreFriction.m port), the
segment friction pressure-drop model multi-segment wells (MSW) need.

Companion MATLAB script: scripts/export_mrst_wellbore_friction.m.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.wellbore_friction import well_bore_friction, well_bore_friction_adi

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def _generate_reference(output_file: Path, matlab: str | None = None) -> Path:
    matlab_exe = matlab or shutil.which("matlab")
    if matlab_exe is None:
        raise RuntimeError("MATLAB executable was not found on PATH")
    script_dir = REPO_ROOT / "scripts"
    cmd = (
        f"addpath('{_matlab_path(script_dir)}'); "
        f"export_mrst_wellbore_friction('{_matlab_path(output_file)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    return output_file


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_well_bore_friction_matches_mrst(tmp_path: Path):
    reference = tmp_path / "wellbore_friction_mrst_ref.mat"
    _generate_reference(reference)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    v, rho, mu = ref["v_massrate"], ref["rho"], ref["mu"]
    Do, L, rough = ref["Do"], ref["L"], ref["roughness"]

    dp = well_bore_friction(v, rho, mu, Do, L, rough, flowtype="massRate", assume_turbulent=False)
    assert np.allclose(dp, ref["dp_massrate"], rtol=1e-8, atol=1e-10)

    dp_t = well_bore_friction(v, rho, mu, Do, L, rough, flowtype="massRate", assume_turbulent=True)
    assert np.allclose(dp_t, ref["dp_massrate_turb"], rtol=1e-8, atol=1e-10)

    dp_vol = well_bore_friction(ref["v_vol"], rho, mu, Do, L, rough, flowtype="volumeRate")
    assert np.allclose(dp_vol, ref["dp_vol"], rtol=1e-4, atol=1e-10)

    dp_vel = well_bore_friction(ref["v_vel"], rho, mu, Do, L, rough, flowtype="velocity")
    assert np.allclose(dp_vel, ref["dp_vel"], rtol=1e-6, atol=1e-10)

    dp_ann = well_bore_friction(
        v, rho, mu, (float(ref["Di_scalar"]), float(ref["Do_scalar"])), L, rough, flowtype="massRate"
    )
    assert np.allclose(dp_ann, ref["dp_annulus"], rtol=1e-8, atol=1e-10)

    # ADI-differentiable counterpart: its .val must match MRST exactly too,
    # not just the plain-numpy well_bore_friction it reuses for the
    # non-ADI fallback path.
    n = v.size
    v_adi = SparseADI.variable(v, 2 * n, 0)
    rho_adi = SparseADI.variable(rho, 2 * n, n)
    dp_adi = well_bore_friction_adi(v_adi, rho_adi, mu, Do, L, rough, flowtype="massRate", assume_turbulent=False)
    assert np.allclose(dp_adi.val, ref["dp_massrate"], rtol=1e-8, atol=1e-10)


def test_well_bore_friction_adi_jacobian_matches_finite_difference():
    """No MRST reference differentiates this function (MRST relies on its
    own ADI class doing this automatically); validate well_bore_friction_adi's
    analytic Jacobian directly against central finite differences of the
    plain well_bore_friction, across laminar/transitional/turbulent
    regimes and both signs of flow."""
    rng = np.random.default_rng(2)
    n = 30
    v = rng.uniform(-3.0, 3.0, n)
    rho = rng.uniform(400.0, 1000.0, n)
    mu = rng.uniform(3e-4, 3e-3, n)
    D = rng.uniform(0.05, 0.25, n)
    L = rng.uniform(20.0, 200.0, n)
    rough = rng.uniform(1e-6, 2e-4, n)

    nvar = 2 * n
    v_adi = SparseADI.variable(v, nvar, 0)
    rho_adi = SparseADI.variable(rho, nvar, n)
    dp = well_bore_friction_adi(v_adi, rho_adi, mu, D, L, rough, flowtype="massRate")
    jac = dp.jac.toarray()

    eps_v = 1.0e-6 * np.maximum(np.abs(v), 1.0)
    eps_rho = 1.0e-6 * np.maximum(np.abs(rho), 1.0)
    for i in range(n):
        vp, vm = v.copy(), v.copy()
        vp[i] += eps_v[i]
        vm[i] -= eps_v[i]
        fd = (well_bore_friction(vp, rho, mu, D, L, rough, flowtype="massRate")
              - well_bore_friction(vm, rho, mu, D, L, rough, flowtype="massRate")) / (2 * eps_v[i])
        assert np.allclose(jac[:, i], fd, atol=1e-5, rtol=1e-2), f"dv column {i} mismatch"

        rp, rm = rho.copy(), rho.copy()
        rp[i] += eps_rho[i]
        rm[i] -= eps_rho[i]
        fd = (well_bore_friction(v, rp, mu, D, L, rough, flowtype="massRate")
              - well_bore_friction(v, rm, mu, D, L, rough, flowtype="massRate")) / (2 * eps_rho[i])
        assert np.allclose(jac[:, n + i], fd, atol=1e-5, rtol=1e-2), f"drho column {i} mismatch"
