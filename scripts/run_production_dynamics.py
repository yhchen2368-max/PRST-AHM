"""Run complete schedules for the bundled Eclipse decks and plot well dynamics."""
from __future__ import annotations

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

CASES = {
    "SPE1": ROOT / "examples/SpE1/BENCH_SPE1.DATA",
    "SPE9": ROOT / "examples/SPE9/SPE9_CP.DATA",
    "EGG": ROOT / "examples/EGG/Egg_Model_ECL.DATA",
    "NORNE": ROOT / "examples/Norne/Norne_simplified/NORNE_ATW2013.DATA",
}
OUT = ROOT / "results" / "production_dynamics"


def _well_value(well_sols, well_index, field):
    values = []
    for wells in well_sols:
        if well_index < len(wells):
            values.append(float(wells[well_index].get(field, np.nan)))
        else:
            values.append(np.nan)
    return values


def _write_svg_plot(path, times, series, title, ylabel):
    width, height = 1000, 600
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 55, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    finite = [v for values in series for v in values if np.isfinite(v)]
    if not finite:
        finite = [0.0, 1.0]
    xmin, xmax = min(times, default=0.0), max(times, default=1.0)
    ymin, ymax = min(finite), max(finite)
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0

    def point(x, y):
        px = margin_left + (x - xmin) / max(xmax - xmin, 1.0) * plot_w
        py = margin_top + (ymax - y) / (ymax - ymin) * plot_h
        return px, py

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="black"/>',
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="14">Time (days)</text>',
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height / 2:.1f})">{ylabel}</text>',
    ]
    for index, values in enumerate(series):
        coords = [point(x, y) for x, y in zip(times, values) if np.isfinite(y)]
        if len(coords) >= 2:
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            parts.append(f'<polyline fill="none" stroke="{palette[index % len(palette)]}" stroke-width="1.5" points="{polyline}"/>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def run_case(name: str, deck: Path):
    started = time.perf_counter()
    state, model, schedule, solver = init_eclipse_problem_ad(str(deck))
    solver.timeStepSelector.reset()
    previous_control = None
    times = []
    rows = []
    well_history = []
    simulation_time = 0.0
    converged = True
    failure = ""

    for step_index, dt_value in enumerate(schedule["step"]["val"]):
        control_id = int(schedule["step"]["control"][step_index])
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
            step_elapsed = time.perf_counter() - step_started
            ok = bool(report.get("Converged", False))
            converged = converged and ok
            simulation_time += float(dt_value)
            times.append(simulation_time / 86400.0)
            wells = state.get("wellSol", [])
            well_history.append(wells)
            rows.append({
                "step": step_index + 1,
                "time_days": times[-1],
                "dt_days": float(dt_value) / 86400.0,
                "converged": ok,
                "iterations": int(report.get("Iterations", 0)),
                "accepted_ministeps": int(report.get("AcceptedMinisteps", len(ministates))),
                "step_elapsed_seconds": step_elapsed,
                "pressure_min": float(np.min(state["pressure"])),
                "pressure_max": float(np.max(state["pressure"])),
                "well_count": len(wells),
            })
            print(
                f"{name} step={step_index + 1}/{len(schedule['step']['val'])} "
                f"converged={ok} iterations={rows[-1]['iterations']} "
                f"ministeps={rows[-1]['accepted_ministeps']} "
                f"elapsed={step_elapsed:.2f}s",
                flush=True,
            )
            if not ok:
                failure = f"nonlinear solver failed at step {step_index + 1}"
                break
        except Exception as exc:
            converged = False
            failure = f"{type(exc).__name__}: {exc}"
            rows.append({"step": step_index + 1, "converged": False, "error": failure})
            print(f"{name} FAIL step={step_index + 1}: {failure}", flush=True)
            break

    total_elapsed = time.perf_counter() - started
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "case": name,
            "deck": str(deck),
            "converged": converged,
            "failure": failure,
            "elapsed_seconds": total_elapsed,
            "report_steps_completed": len(rows),
            "rows": rows,
        }, handle, indent=2)
    if rows:
        with (case_dir / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)

        n_wells = max((len(wells) for wells in well_history), default=0)
        fields = [("qOs", "Oil rate"), ("qWs", "Water rate"), ("qGs", "Gas rate"), ("bhp", "BHP")]
        for field, label in fields:
            _write_svg_plot(
                case_dir / f"{field}.svg",
                times,
                [_well_value(well_history, wi, field) for wi in range(n_wells)],
                f"{name} production dynamics - {label}",
                label,
            )
    print(f"{name} COMPLETE converged={converged} steps={len(rows)} total={total_elapsed:.2f}s", flush=True)
    return {
        "case": name,
        "converged": converged,
        "failure": failure,
        "elapsed_seconds": total_elapsed,
        "steps": len(rows),
        "output": str(case_dir),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, deck in CASES.items():
        print(f"\n=== {name} ===", flush=True)
        summary.append(run_case(name, deck))
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
