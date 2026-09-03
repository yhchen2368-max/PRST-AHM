"""MRST parity + speed comparison for the full 90-report-step SPE9 run
(companion to test_spe1_mrst_parity.py; SPE9's first Newton step was
already validated bit-for-bit against MRST in earlier session work --
this covers the full multi-step run's well solutions and timing).

Companion MATLAB script: scripts/export_mrst_spe9_full.m.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

REPO_ROOT = Path(__file__).resolve().parents[1]
SPE9_DECK = REPO_ROOT / "examples" / "SPE9" / "SPE9_CP.DATA"


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_spe9_full_matches_mrst_and_is_competitive_in_speed(tmp_path: Path):
    reference = tmp_path / "spe9_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_spe9_full('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True, timeout=3600)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    state0, model, schedule, solver = init_eclipse_problem_ad(str(SPE9_DECK))
    t0 = time.perf_counter()
    well_sols, states = simulate_schedule_ad(state0, model, schedule, NonLinearSolver=solver)
    elapsed = time.perf_counter() - t0

    assert len(states) == 90
    names = [w["name"] for w in well_sols[0]]
    ref_names = [str(n).strip() for n in np.atleast_1d(ref["names"])]
    assert set(names) == set(ref_names)

    nt = len(well_sols)
    for field in ("qOs", "qWs", "bhp"):
        ref_q = np.atleast_2d(ref[field])
        for pi, name in enumerate(names):
            mi = ref_names.index(name)
            got = np.array([well_sols[kt][pi][field] for kt in range(nt)])
            want = ref_q[:, mi]
            atol = 2e6 if field == "bhp" else 5e-3
            assert np.allclose(got, want, rtol=2e-2, atol=atol), f"{field}/{name} mismatch"

    assert np.allclose(np.asarray(states[-1]["sW"]), ref["sw_final"].ravel(), atol=5e-2)
    assert np.allclose(np.asarray(states[-1]["pressure"]), ref["pressure_final"].ravel(),
                        rtol=5e-3, atol=1e5)

    print(f"\nSPE9 (90 steps, 9000 cells): MRST {float(ref['elapsed']):.2f}s vs "
          f"PRSTCore {elapsed:.2f}s (ratio {float(ref['elapsed']) / elapsed:.2f}x)")
    assert elapsed < 3.0 * float(ref["elapsed"])
