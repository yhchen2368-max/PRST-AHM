"""Run the full EGG schedule with per-step nonlinear progress logging."""

from copy import deepcopy
from pathlib import Path
import csv
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


OUT = ROOT / "results" / "egg_full_run"
DECK = ROOT / "examples" / "EGG" / "Egg_Model_ECL.DATA"


def _well_array(wells, key):
    return np.asarray([float(w.get(key, np.nan)) for w in wells], dtype=float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state, model, schedule, solver = init_eclipse_problem_ad(str(DECK))
    solver.verbose = True
    solver.timeStepSelector.reset()
    if hasattr(solver.LinearSolver, "verbose"):
        solver.LinearSolver.verbose = False

    print(
        "EGG init cells={} steps={} solver={} linear={} nls=({}, min={}, cuts={})".format(
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

    rows = []
    previous_control = None
    sim_time = 0.0
    converged_all = True
    failure = ""

    for step, dt_value in enumerate(schedule["step"]["val"], start=1):
        control_id = int(schedule["step"]["control"][step - 1])
        forces = model.getDrivingForces(schedule["control"][control_id])
        if control_id != previous_control:
            model, state = model.updateForChangedControls(state, forces)
            previous_control = control_id

        old_state = deepcopy(state)
        step_started = time.perf_counter()
        try:
            state, report, ministates = solver.solveTimestep(
                old_state,
                float(dt_value),
                model,
                drivingForces=forces,
                initialGuess=deepcopy(state),
                controlId=control_id,
            )
        except Exception as exc:
            converged_all = False
            failure = f"{type(exc).__name__}: {exc}"
            rows.append({"step": step, "converged": False, "error": failure})
            print(f"EGG FAIL step={step}: {failure}", flush=True)
            break

        elapsed = time.perf_counter() - step_started
        ok = bool(report.get("Converged", False))
        converged_all = converged_all and ok
        sim_time += float(dt_value)
        wells = state.get("wellSol", [])
        bhp = _well_array(wells, "bhp")
        qws = _well_array(wells, "qWs")
        qos = _well_array(wells, "qOs")
        row = {
            "step": step,
            "time_days": sim_time / 86400.0,
            "dt_days": float(dt_value) / 86400.0,
            "converged": ok,
            "iterations": int(report.get("Iterations", 0)),
            "accepted_ministeps": int(report.get("AcceptedMinisteps", len(ministates))),
            "cut_count": int(report.get("MinistepCuttingCount", 0)),
            "elapsed_seconds": elapsed,
            "pressure_min": float(np.min(state["pressure"])),
            "pressure_max": float(np.max(state["pressure"])),
            "sW_min": float(np.min(state["sW"])),
            "sW_max": float(np.max(state["sW"])),
            "bhp_min": float(np.nanmin(bhp)) if bhp.size else np.nan,
            "bhp_max": float(np.nanmax(bhp)) if bhp.size else np.nan,
            "qWs_sum": float(np.nansum(qws)) if qws.size else np.nan,
            "qOs_sum": float(np.nansum(qos)) if qos.size else np.nan,
        }
        rows.append(row)
        print(
            "EGG STEP={:03d}/{} time_days={:.6g} ok={} iterations={} "
            "ministeps={} cuts={} elapsed={:.2f}s p=[{:.7g},{:.7g}] sw=[{:.6g},{:.6g}]".format(
                step,
                len(schedule["step"]["val"]),
                row["time_days"],
                ok,
                row["iterations"],
                row["accepted_ministeps"],
                row["cut_count"],
                elapsed,
                row["pressure_min"],
                row["pressure_max"],
                row["sW_min"],
                row["sW_max"],
            ),
            flush=True,
        )
        if not ok:
            failure = f"nonlinear solver did not converge at step {step}"
            break

        if step % 5 == 0 or step == len(schedule["step"]["val"]):
            _write_outputs(rows, converged_all, failure, started)

    _write_outputs(rows, converged_all, failure, started)
    print(
        "EGG COMPLETE converged={} steps={} elapsed={:.2f}s".format(
            converged_all,
            len(rows),
            time.perf_counter() - started,
        ),
        flush=True,
    )
    return 0 if converged_all else 1


def _write_outputs(rows, converged, failure, started):
    report_path = OUT / "report.json"
    csv_path = OUT / "steps.csv"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": "EGG",
                "deck": str(DECK),
                "converged": bool(converged),
                "failure": failure,
                "elapsed_seconds": time.perf_counter() - started,
                "steps_completed": len(rows),
                "rows": rows,
            },
            handle,
            indent=2,
        )
    if rows:
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
