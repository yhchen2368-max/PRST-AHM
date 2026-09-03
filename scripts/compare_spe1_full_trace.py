"""Compare SPE1 full-state and nonlinear/Jacobian traces against MRST."""

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
MRST_FULL = ROOT / "spe1_mrst_full.mat"
PY_FULL = ROOT / "spe1_python_full_parity.npz"
MRST_TRACE = ROOT / "spe1_mrst_full_trace.mat"
PY_TRACE = ROOT / "spe1_python_full_trace.npz"


def _flat(a):
    return np.asarray(a).reshape(-1)


def _mat_vec(m, name):
    return _flat(m[name]).astype(float)


def _mat_bool(m, name):
    return _flat(m[name]).astype(bool)


def _sparse_from(raw, idx):
    ptr = raw["ptr"]
    rows = raw["rows"][ptr[idx] : ptr[idx + 1]].astype(np.int64) - 1
    cols = raw["cols"][ptr[idx] : ptr[idx + 1]].astype(np.int64) - 1
    vals = raw["vals"][ptr[idx] : ptr[idx + 1]].astype(float)
    shape = tuple(raw["shape"][idx].astype(int))
    return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


def _residual_from(raw, idx):
    ptr = raw["res_ptr"]
    return raw["res"][ptr[idx] : ptr[idx + 1]].astype(float)


def compare_final_states():
    mrst = loadmat(MRST_FULL, squeeze_me=True, struct_as_record=False)
    py = np.load(PY_FULL, allow_pickle=True)
    rows = np.asarray(py["rows"], dtype=float)
    max_cols = rows[:, 1:11].max(axis=0)
    py_iterations = rows[:, -2].astype(int)
    py_ministeps = rows[:, -1].astype(int)
    mrst_iterations = np.asarray(mrst["iterations"], dtype=int).reshape(-1)
    mrst_ministeps = np.asarray(mrst["ministeps"], dtype=int).reshape(-1)
    print("Final-state report-step comparison")
    print(f"  report steps: {rows.shape[0]}")
    print(f"  iterations: Python={py_iterations.sum()} MRST={mrst_iterations.sum()} exact={np.array_equal(py_iterations, mrst_iterations)}")
    print(f"  ministeps:  Python={py_ministeps.sum()} MRST={mrst_ministeps.sum()} exact={np.array_equal(py_ministeps, mrst_ministeps)}")
    print(
        "  max abs [p, sw, sg, rs, bhp, qgs] = "
        f"[{max_cols[0]:.6e}, {max_cols[2]:.6e}, {max_cols[4]:.6e}, "
        f"{max_cols[6]:.6e}, {max_cols[8]:.6e}, {max_cols[9]:.6e}]"
    )


def compare_trace():
    if not MRST_TRACE.is_file():
        raise FileNotFoundError(f"Missing MRST trace: {MRST_TRACE}")
    if not PY_TRACE.is_file():
        raise FileNotFoundError(f"Missing Python trace: {PY_TRACE}")

    mrst = loadmat(MRST_TRACE, squeeze_me=True, struct_as_record=False)
    py = np.load(PY_TRACE, allow_pickle=True)

    mrst_axes = {
        "step": _flat(mrst["trace_step"]).astype(int),
        "ministep": _flat(mrst["trace_ministep"]).astype(int),
        "iteration": _flat(mrst["trace_iteration"]).astype(int),
        "dt": _flat(mrst["trace_dt"]).astype(float),
        "solved": _flat(mrst["trace_solved"]).astype(bool),
    }
    py_axes = {
        "step": py["trace_step"].astype(int),
        "ministep": py["trace_ministep"].astype(int),
        "iteration": py["trace_iteration"].astype(int),
        "dt": py["trace_dt"].astype(float),
        "solved": py["trace_solved"].astype(bool),
    }

    print("Nonlinear trace comparison")
    print(f"  records: Python={len(py_axes['step'])} MRST={len(mrst_axes['step'])}")
    for name in ("step", "ministep", "iteration", "solved"):
        print(f"  {name}: exact={np.array_equal(py_axes[name], mrst_axes[name])}")
    print(f"  dt max abs: {np.max(np.abs(py_axes['dt'] - mrst_axes['dt'])):.6e}")

    py_res_values = py["trace_residual_values"].astype(float)
    mrst_res_values = np.vstack([np.asarray(x, dtype=float).reshape(-1) for x in np.ravel(mrst["trace_residual_values"])])
    py_res_conv = py["trace_residual_converged"].astype(bool)
    mrst_res_conv = np.vstack([np.asarray(x, dtype=bool).reshape(-1) for x in np.ravel(mrst["trace_residual_converged"])])
    print(f"  convergence values max abs: {np.nanmax(np.abs(py_res_values - mrst_res_values)):.6e}")
    print(f"  convergence flags exact: {np.array_equal(py_res_conv, mrst_res_conv)}")

    py_raw = {
        "rows": py["trace_jac_rows"],
        "cols": py["trace_jac_cols"],
        "vals": py["trace_jac_vals"],
        "ptr": py["trace_jac_ptr"].astype(np.int64),
        "shape": py["trace_jac_shape"],
        "res": py["trace_residual_vector"],
        "res_ptr": py["trace_residual_ptr"].astype(np.int64),
    }
    mrst_raw = {
        "rows": _flat(mrst["trace_jac_rows"]).astype(np.int64),
        "cols": _flat(mrst["trace_jac_cols"]).astype(np.int64),
        "vals": _flat(mrst["trace_jac_vals"]).astype(float),
        "ptr": _flat(mrst["jac_ptr"]).astype(np.int64),
        "shape": np.asarray(mrst["trace_jac_shape"], dtype=int),
        "res": _flat(mrst["trace_residual_vector"]).astype(float),
        "res_ptr": _flat(mrst["residual_ptr"]).astype(np.int64),
    }

    max_j = 0.0
    norm_j = 0.0
    max_r = 0.0
    norm_r = 0.0
    worst_j = -1
    worst_r = -1
    nnz_exact = True
    for i in range(len(py_axes["step"])):
        A_py = _sparse_from(py_raw, i)
        A_mrst = _sparse_from(mrst_raw, i)
        dA = (A_py - A_mrst).tocoo()
        local_max_j = float(np.max(np.abs(dA.data))) if dA.nnz else 0.0
        local_norm_j = float(sp.linalg.norm(dA)) if dA.nnz else 0.0
        if local_max_j > max_j:
            max_j = local_max_j
            worst_j = i
        norm_j = max(norm_j, local_norm_j)
        nnz_exact = nnz_exact and A_py.nnz == A_mrst.nnz

        r_py = _residual_from(py_raw, i)
        r_mrst = _residual_from(mrst_raw, i)
        dr = r_py - r_mrst
        local_max_r = float(np.max(np.abs(dr))) if dr.size else 0.0
        local_norm_r = float(np.linalg.norm(dr)) if dr.size else 0.0
        if local_max_r > max_r:
            max_r = local_max_r
            worst_r = i
        norm_r = max(norm_r, local_norm_r)

    print(f"  Jacobian nnz per record exact: {nnz_exact}")
    print(f"  Jacobian diff max={max_j:.6e} worst_record={worst_j + 1} norm_max={norm_j:.6e}")
    print(f"  full residual diff max={max_r:.6e} worst_record={worst_r + 1} norm_max={norm_r:.6e}")
    print(
        "  linear time totals: "
        f"Python={np.nansum(py['trace_linear_time']):.6f}s "
        f"MRST={np.nansum(_mat_vec(mrst, 'trace_linear_time')):.6f}s"
    )
    print(
        "  report-step wall totals: "
        f"Python={np.nansum(py['step_times']):.6f}s "
        f"MRST={np.nansum(_mat_vec(mrst, 'step_times')):.6f}s"
    )


def main():
    compare_final_states()
    compare_trace()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
