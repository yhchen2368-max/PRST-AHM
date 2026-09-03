"""Export a full SPE1 nonlinear/Jacobian trace from the Python port.

The simulator path is the normal ``init_eclipse_problem_ad`` solver path.
This script only wraps equation assembly and linear solves to record what
the solver already computes.
"""

from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is required for the real case
    sp = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


DECK = ROOT / "examples" / "SpE1" / "BENCH_SPE1.DATA"
OUTPUT = ROOT / "spe1_python_full_trace.npz"


def _as_csr(A):
    if sp is not None and sp.issparse(A):
        return A.tocoo()
    if sp is None:
        raise RuntimeError("scipy.sparse is required to export Jacobians")
    return sp.coo_matrix(np.asarray(A, dtype=float))


def _field(report, name, default=np.nan):
    if isinstance(report, dict):
        return report.get(name, default)
    return default


def main():
    state, model, schedule, solver = init_eclipse_problem_ad(str(DECK))

    current = {"step": 0, "ministep": 0, "dt": np.nan}
    records = []
    pending_by_id = {}

    jac_rows = []
    jac_cols = []
    jac_vals = []
    jac_ptr = [0]
    res_vec = []
    res_ptr = [0]

    original_prepare = model.prepareTimestep
    original_get_equations = model.get_equations
    original_check = model.checkConvergence
    original_solve_linear = solver.LinearSolver.solveLinearProblem
    original_stabilize = solver.stabilizeNewtonIncrements

    def traced_prepare(state_arg, state0_arg, dt_arg, driving_forces):
        current["ministep"] += 1
        current["dt"] = float(dt_arg)
        return original_prepare(state_arg, state0_arg, dt_arg, driving_forces)

    def traced_get_equations(state0_arg, state_arg, dt_arg, driving_forces=None, **kwargs):
        t0 = time.perf_counter()
        problem, assembled_state = original_get_equations(
            state0_arg, state_arg, dt_arg, driving_forces, **kwargs
        )
        assembly_time = time.perf_counter() - t0
        if "Jacobian" in problem and "Residuals" in problem:
            J = _as_csr(problem["Jacobian"])
            r = np.asarray(problem["Residuals"], dtype=float).ravel()
            jac_rows.append(J.row.astype(np.int64) + 1)
            jac_cols.append(J.col.astype(np.int64) + 1)
            jac_vals.append(J.data.astype(float))
            jac_ptr.append(jac_ptr[-1] + int(J.nnz))
            res_vec.append(r.astype(float))
            res_ptr.append(res_ptr[-1] + int(r.size))
            rec = {
                "step": int(current["step"]),
                "ministep": int(current["ministep"]),
                "iteration": int(kwargs.get("iteration", -1)),
                "dt": float(dt_arg),
                "assembly_time": float(assembly_time),
                "jac_shape": tuple(int(x) for x in J.shape),
                "res_size": int(r.size),
                "residual_values": None,
                "residual_converged": None,
                "linear_iterations": np.nan,
                "linear_residual": np.nan,
                "linear_time": np.nan,
                "linear_solution_time": np.nan,
                "linear_converged": False,
                "relaxation": np.nan,
                "solved": False,
            }
            records.append(rec)
            pending_by_id[id(problem)] = len(records) - 1
        return problem, assembled_state

    def traced_check(problem):
        converged, values, names = original_check(problem)
        ix = pending_by_id.get(id(problem))
        if ix is not None:
            records[ix]["residual_values"] = np.asarray(values, dtype=float).ravel().copy()
            records[ix]["residual_converged"] = np.asarray(converged, dtype=bool).ravel().copy()
            records[ix]["residual_names"] = np.asarray(names, dtype=object)
        return converged, values, names

    def traced_solve_linear(problem, model_arg=None):
        out = original_solve_linear(problem, model_arg)
        report = out[2] if len(out) > 2 else {}
        ix = pending_by_id.get(id(problem))
        if ix is not None:
            records[ix]["solved"] = True
            records[ix]["linear_iterations"] = float(_field(report, "Iterations", np.nan))
            records[ix]["linear_residual"] = float(_field(report, "Residual", np.nan))
            records[ix]["linear_time"] = float(_field(report, "SolverTime", np.nan))
            records[ix]["linear_solution_time"] = float(_field(report, "LinearSolutionTime", np.nan))
            records[ix]["linear_converged"] = bool(_field(report, "Converged", False))
        return out

    def traced_stabilize(dx):
        out = original_stabilize(dx)
        dx_out, report = out
        if records:
            records[-1]["relaxation"] = float(_field(report, "relaxationParameter", np.nan))
        return dx_out, report

    model.prepareTimestep = traced_prepare
    model.get_equations = traced_get_equations
    model.checkConvergence = traced_check
    solver.LinearSolver.solveLinearProblem = traced_solve_linear
    solver.stabilizeNewtonIncrements = traced_stabilize

    solver.timeStepSelector.reset()
    previous_control = None
    step_iterations = []
    step_ministeps = []
    step_times = []

    started = time.perf_counter()
    for step, dt in enumerate(schedule["step"]["val"], start=1):
        current["step"] = step
        current["ministep"] = 0
        control_id = int(schedule["step"]["control"][step - 1])
        forces = model.getDrivingForces(schedule["control"][control_id])
        if control_id != previous_control:
            model, state = model.updateForChangedControls(state, forces)
            previous_control = control_id

        state0 = deepcopy(state)
        state, report, _ = solver.solveTimestep(
            state0,
            float(dt),
            model,
            drivingForces=forces,
            initialGuess=deepcopy(state),
            controlId=control_id,
        )
        if not report.get("Converged", False):
            raise RuntimeError(f"SPE1 did not converge at report step {step}")
        step_iterations.append(int(report["Iterations"]))
        step_ministeps.append(int(report["AcceptedMinisteps"]))
        step_times.append(float(report.get("SimulationTime", np.nan)))
        print(
            "STEP={:03d} records={} iterations={} ministeps={} elapsed={:.2f}s".format(
                step,
                len(records),
                report["Iterations"],
                report["AcceptedMinisteps"],
                time.perf_counter() - started,
            ),
            flush=True,
        )

    max_res_values = max(
        len(r["residual_values"]) if r.get("residual_values") is not None else 0
        for r in records
    )
    residual_values = np.full((len(records), max_res_values), np.nan, dtype=float)
    residual_converged = np.zeros((len(records), max_res_values), dtype=bool)
    for i, rec in enumerate(records):
        values = rec.get("residual_values")
        flags = rec.get("residual_converged")
        if values is not None:
            residual_values[i, : len(values)] = values
        if flags is not None:
            residual_converged[i, : len(flags)] = flags

    np.savez(
        OUTPUT,
        trace_step=np.asarray([r["step"] for r in records], dtype=np.int32),
        trace_ministep=np.asarray([r["ministep"] for r in records], dtype=np.int32),
        trace_iteration=np.asarray([r["iteration"] for r in records], dtype=np.int32),
        trace_dt=np.asarray([r["dt"] for r in records], dtype=float),
        trace_assembly_time=np.asarray([r["assembly_time"] for r in records], dtype=float),
        trace_linear_iterations=np.asarray([r["linear_iterations"] for r in records], dtype=float),
        trace_linear_residual=np.asarray([r["linear_residual"] for r in records], dtype=float),
        trace_linear_time=np.asarray([r["linear_time"] for r in records], dtype=float),
        trace_linear_solution_time=np.asarray([r["linear_solution_time"] for r in records], dtype=float),
        trace_linear_converged=np.asarray([r["linear_converged"] for r in records], dtype=bool),
        trace_relaxation=np.asarray([r["relaxation"] for r in records], dtype=float),
        trace_solved=np.asarray([r["solved"] for r in records], dtype=bool),
        trace_residual_values=residual_values,
        trace_residual_converged=residual_converged,
        trace_jac_rows=np.concatenate(jac_rows) if jac_rows else np.array([], dtype=np.int64),
        trace_jac_cols=np.concatenate(jac_cols) if jac_cols else np.array([], dtype=np.int64),
        trace_jac_vals=np.concatenate(jac_vals) if jac_vals else np.array([], dtype=float),
        trace_jac_ptr=np.asarray(jac_ptr, dtype=np.int64),
        trace_residual_vector=np.concatenate(res_vec) if res_vec else np.array([], dtype=float),
        trace_residual_ptr=np.asarray(res_ptr, dtype=np.int64),
        trace_jac_shape=np.asarray([r["jac_shape"] for r in records], dtype=np.int32),
        trace_residual_size=np.asarray([r["res_size"] for r in records], dtype=np.int32),
        step_iterations=np.asarray(step_iterations, dtype=np.int32),
        step_ministeps=np.asarray(step_ministeps, dtype=np.int32),
        step_times=np.asarray(step_times, dtype=float),
    )
    print(f"Wrote {len(records)} nonlinear assemblies to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
