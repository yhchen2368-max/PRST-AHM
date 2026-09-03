"""Inspect the singular Jacobian of a model: find zero rows/columns.

Loads the deck, builds the first Newton system and reports which unknown
blocks (pressure / saturations / well equations) are structurally
singular, so we can tell a well-assembly bug from a cell problem.
"""
import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad)

deck = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(ROOT, "examples", "spe3", "SPE3CASE1.DATA")
print("deck:", deck)
state0, model, schedule, solver = init_eclipse_problem_ad(
    deck, RemoveZeroPoreVolume=True)
nc = len(state0["pressure"])
print("nc=%d  phases(o/w/g)=%s/%s/%s" % (nc, model.oil, model.water,
                                         model.gas))

# wells in the first control
control = int(schedule["step"]["control"][0])
ctrl = schedule["control"][control]
print("control %d: %d wells" % (control, len(ctrl.get("W", []))))
for w in ctrl.get("W", []):
    name = w.get("name")
    cells = w.get("cells") or []
    wi = w.get("WI") or []
    print("  well %-8s status=%s type=%s phase=%s nconn=%d WI=%s"
          % (name, w.get("status"), w.get("type"), w.get("phase"),
             len(cells), wi[:3]))

# build the first Newton system
forces = model.getDrivingForces(ctrl)
model, st = model.updateForChangedControls(state0, forces)
dt = float(schedule["step"]["val"][0])

# the same call stepFunction uses to assemble the linearized problem
problem, st2 = model.get_equations(
    state0, deepcopy(st), dt, forces, iteration=1)
state = st2

A = None
if hasattr(problem, "getLinearSystem"):
    A, b = problem.getLinearSystem()
elif isinstance(problem, dict) and "Jacobian" in problem:
    A = problem["Jacobian"]
    b = -np.asarray(problem["Residuals"], dtype=float)

if A is None:
    print("no linear system available (problem type=%s)" % type(problem).__name__)
    sys.exit(0)

A = A.tocsr() if hasattr(A, "tocsr") else np.asarray(A, dtype=float)
n = A.shape[0]
print("Jacobian shape: %s  nnz=%d" % (A.shape, A.nnz))

rows = np.asarray(A.sum(axis=1)).ravel()
cols = np.asarray(A.sum(axis=0)).ravel()
zr = np.where(np.abs(rows) < 1e-30)[0]
zc = np.where(np.abs(cols) < 1e-30)[0]
print("zero rows: %d  zero cols: %d" % (len(zr), len(zc)))

# dense rank / null space of the full Jacobian (only for small systems)
Ad = A.toarray()
if n <= 1500:
    rank = np.linalg.matrix_rank(Ad, tol=1e-8)
    print("dense rank: %d of %d  (rank deficient by %d)"
          % (rank, n, n - rank))
    s_all = np.linalg.svd(Ad, compute_uv=False)
    print("smallest singular values: %s"
          % [float(x) for x in s_all[-6:]])
    for tol in (1e-8, 1e-10, 1e-12, 1e-14):
        print("  rank(tol=%.0e) = %d" % (tol, np.sum(s_all > tol)))
    if n - np.sum(s_all > 1e-10) > 0:
        u, s, vh = np.linalg.svd(Ad)
        rank10 = np.sum(s > 1e-10)
        for k in range(min(3, n - rank10)):
            v = vh[rank10 + k]
            big = np.argsort(np.abs(v))[::-1][:12]
            print("  nullvec %d: |v|=%.3e  (idx,val): %s"
                  % (k, np.linalg.norm(v),
                     [(int(i), round(float(v[i]), 4)) for i in big]))
        ul = u[:, rank10:]
        for k in range(min(2, n - rank10)):
            v = ul[:, k]
            big = np.argsort(np.abs(v))[::-1][:12]
            print("  left-nullvec %d: rows (idx,val): %s"
                  % (k, [(int(i), round(float(v[i]), 4)) for i in big]))
else:
    from scipy.sparse.linalg import svds
    try:
        _, s, _ = svds(A, k=4, which="SM", maxiter=3000)
        print("smallest singular values (svds): %s" % [float(x) for x in s])
    except Exception as exc:
        print("svds failed: %s" % exc)

# residual
res = np.asarray(b, dtype=float).ravel() if b is not None else None
if res is not None:
    print("residual norm: %.3e  max|res|: %.3e"
          % (np.linalg.norm(res), np.abs(res).max()))
    print("near-zero residual equations:", np.where(
        np.abs(res) < 1e-10 * (np.abs(res).max() + 1e-30))[0][:30].tolist())

# well unknowns: the last few columns.  Which columns have any coupling to
# the well rows, and are the well equations present?
ncell = nc
print("unknown count n=%d, cell dof block=%d, well block=%d:%d"
      % (n, 3 * ncell, 3 * ncell, n))
Acsr = A.tocsr()
# reservoir-only block rank
Ad_res = A[:3 * ncell, :3 * ncell].toarray()
r_res = np.linalg.matrix_rank(Ad_res, tol=1e-8)
print("reservoir-only block rank: %d of %d (deficient by %d)"
      % (r_res, 3 * ncell, 3 * ncell - r_res))
if 3 * ncell - r_res > 0:
    u, s, vh = np.linalg.svd(Ad_res)
    for k in range(min(3, 3 * ncell - r_res)):
        v = vh[r_res + k]
        big = np.argsort(np.abs(v))[::-1][:10]
        print("  res nullvec %d: entries (idx -> cell,var) %s"
              % (k, [(int(i), int(i) % nc, "p/sW/x"[int(i) // nc])
                     for i in big]))
    ul = u[:, r_res:]
    for k in range(min(3, 3 * ncell - r_res)):
        v = ul[:, k]
        big = np.argsort(np.abs(v))[::-1][:10]
        print("  res left-nullvec %d: rows (idx -> cell,eq) %s"
              % (k, [(int(i), int(i) % nc, "W/O/G"[int(i) // nc])
                     for i in big]))
# full-precision well block
well_blk = Ad[3 * ncell:, 3 * ncell:]
print("well block (8x8), full precision:")
for i in range(8):
    row = well_blk[i]
    nz = np.where(np.abs(row) > 0)[0]
    print("  row %d: %s" % (972 + i,
          [(int(j), float(row[j])) for j in nz]))
# and the reservoir coupling of the bhp columns at full precision
print("bhp column couplings (full precision):")
for j in (978, 979):
    c = Acsr.getcol(j).toarray().ravel()
    nz = np.where(np.abs(c) > 0)[0]
    print("  col %d: %s" % (j, [(int(i), float(c[i])) for i in nz]))

# map unknown index ranges: MRST/PRSTCore order is [pressure, sw, sg, ...] per
# cell, then well/facility unknowns at the end.
nvar = 0
if hasattr(model, "getPrimaryVariableNames"):
    names = model.getPrimaryVariableNames()
    print("primary variables:", names)
    nvar = len(names)
elif hasattr(model, "names"):
    names = model.names if isinstance(model.names, (list, tuple)) else None
    print("model names:", names)

if zr.size or zc.size:
    # show a sample of zero rows with their index ranges
    print("first 20 zero rows:", zr[:20].tolist())
    print("first 20 zero cols:", zc[:20].tolist())
    if nvar:
        print("per-cell dof =", nvar)
        print("cell count implied by n =", n, " nvar =", nvar,
              " -> ncell =", n / nvar)
