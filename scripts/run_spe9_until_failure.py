"""Run SPE9 with the deck-selected solver and stop at the first failure."""

from copy import deepcopy
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, ".")
sys.setrecursionlimit(10000)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


def residual_breakdown(model, old_state, state, dt, forces):
    problem, _ = model.get_equations(old_state, state, dt, forces)
    residual = np.asarray(problem["Residuals"], dtype=float).ravel()
    nc = len(old_state["pressure"])
    return {
        "max": float(np.max(np.abs(residual))),
        "water": float(np.max(np.abs(residual[:nc]))),
        "oil": float(np.max(np.abs(residual[nc : 2 * nc]))),
        "gas": float(np.max(np.abs(residual[2 * nc : 3 * nc]))),
    }


def main():
    state, model, schedule, solver = init_eclipse_problem_ad("examples/SPE9/SPE9_CP.DATA")
    print(
        "init cells={} steps={} solver={} linear={} nls=({}, min={}, cuts={})".format(
            len(state["pressure"]),
            len(schedule["step"]["val"]),
            type(solver).__name__,
            type(solver.LinearSolver).__name__,
            solver.maxIterations,
            solver.minIterations,
            solver.maxTimestepCuts,
        ),
        flush=True,
    )
    solver.verbose = True
    if hasattr(solver.LinearSolver, "verbose"):
        solver.LinearSolver.verbose = False
    solver.timeStepSelector.reset()

    previous_control = None
    elapsed_total = time.perf_counter()
    for step, dt in enumerate(schedule["step"]["val"], start=1):
        control_id = int(schedule["step"]["control"][step - 1])
        forces = model.getDrivingForces(schedule["control"][control_id])
        if control_id != previous_control:
            model, state = model.updateForChangedControls(state, forces)
            previous_control = control_id
        old_state = deepcopy(state)
        t0 = time.perf_counter()
        try:
            state, report, _ = solver.solveTimestep(
                old_state,
                float(dt),
                model,
                drivingForces=forces,
                initialGuess=deepcopy(state),
                controlId=control_id,
            )
        except Exception as exc:
            print(f"FAIL step={step} exception={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return 1
        elapsed = time.perf_counter() - t0
        ok = bool(report.get("Converged", False))
        try:
            res = residual_breakdown(model, old_state, state, float(dt), forces)
            res_text = (
                f"resmax={res['max']:.6e} "
                f"rw={res['water']:.6e} ro={res['oil']:.6e} rg={res['gas']:.6e}"
            )
        except Exception as exc:
            res_text = f"residual_check_failed={type(exc).__name__}:{exc}"
        print(
            "STEP={:03d}/{} days={:.9g} ok={} iterations={} ministeps={} "
            "cuts={} elapsed={:.2f}s total={:.2f}s {}".format(
                step,
                len(schedule["step"]["val"]),
                float(dt) / 86400.0,
                ok,
                int(report.get("Iterations", 0)),
                int(report.get("AcceptedMinisteps", 0)),
                int(report.get("MinistepCuttingCount", 0)),
                elapsed,
                time.perf_counter() - elapsed_total,
                res_text,
            ),
            flush=True,
        )
        if not ok:
            print(f"STOP first_nonconverged_step={step}", flush=True)
            return 2
    print("COMPLETE all SPE9 steps converged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
