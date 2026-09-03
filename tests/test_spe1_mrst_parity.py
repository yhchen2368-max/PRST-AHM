"""MRST parity + speed comparison for the SPE1 (Odeh) benchmark: runs
PRSTCore's full three-phase black-oil AD pipeline
(init_eclipse_problem_ad + simulate_schedule_ad) on the same deck MRST's
own blackoilTutorialSPE1.m uses, and compares well rates/BHP/final state
against MRST, plus wall-clock time.

Companion MATLAB script: scripts/export_mrst_spe1.m.
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
SPE1_DECK = REPO_ROOT / "examples" / "SpE1" / "BENCH_SPE1.DATA"


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def _run_prst_spe1():
    state0, model, schedule, solver = init_eclipse_problem_ad(str(SPE1_DECK))
    t0 = time.perf_counter()
    well_sols, states = simulate_schedule_ad(state0, model, schedule, NonLinearSolver=solver)
    elapsed = time.perf_counter() - t0
    return well_sols, states, elapsed


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_spe1_matches_mrst_and_is_competitive_in_speed(tmp_path: Path):
    reference = tmp_path / "spe1_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_spe1('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    well_sols, states, elapsed = _run_prst_spe1()

    names = [w["name"] for w in well_sols[0]]
    ref_names = [str(n).strip() for n in np.atleast_1d(ref["names"])]
    assert set(names) == set(ref_names)

    nt = len(well_sols)
    for field in ("qOs", "qWs", "qGs", "bhp"):
        ref_q = np.atleast_2d(ref[field])
        for pi, name in enumerate(names):
            mi = ref_names.index(name)
            got = np.array([well_sols[kt][pi][field] for kt in range(nt)])
            want = ref_q[:, mi]
            # bhp differs by <1% relative at the physical scale (~5e7 Pa);
            # rates match to numerical noise except where the true value
            # is itself numerically ~0 (well not yet flowing that phase).
            atol = 1e6 if field == "bhp" else 1e-3
            assert np.allclose(got, want, rtol=1e-2, atol=atol), f"{field}/{name} mismatch"

    sw_final = np.asarray(states[-1]["sW"])
    sg_final = np.asarray(states[-1]["sG"])
    assert np.allclose(sw_final, ref["sw_final"].ravel(), atol=1e-4)
    assert np.allclose(sg_final, ref["sg_final"].ravel(), atol=2e-2)
    assert np.allclose(np.asarray(states[-1]["pressure"]), ref["pressure_final"].ravel(),
                        rtol=1e-3, atol=1e4)

    print(f"\nSPE1: MRST {float(ref['elapsed']):.2f}s vs PRSTCore {elapsed:.2f}s "
          f"(ratio {float(ref['elapsed']) / elapsed:.2f}x)")
    # Not a hard perf gate (machine-dependent) -- documents the comparison;
    # fails only if PRSTCore regresses to grossly (>3x) slower than MRST.
    assert elapsed < 3.0 * float(ref["elapsed"])


def test_spe1_runs_end_to_end_and_is_physically_sane():
    """Self-contained (no MATLAB) smoke test: the full 120-step SPE1
    schedule solves without error and produces physically sane rates."""
    well_sols, states, elapsed = _run_prst_spe1()
    assert len(states) == 120
    names = [w["name"] for w in well_sols[0]]
    assert "INJECTOR" in names and "PRODUCER" in names

    last = {w["name"]: w for w in well_sols[-1]}
    # Gas injector: injecting (qGs > 0). Producer: producing oil (qOs < 0).
    assert last["INJECTOR"]["qGs"] > 0
    assert last["PRODUCER"]["qOs"] < 0

    pressure = np.asarray(states[-1]["pressure"])
    assert np.all(np.isfinite(pressure)) and np.all(pressure > 0)
    sw = np.asarray(states[-1]["sW"])
    sg = np.asarray(states[-1]["sG"])
    assert np.all(sw >= -1e-8) and np.all(sg >= -1e-8) and np.all(sw + sg <= 1 + 1e-6)
