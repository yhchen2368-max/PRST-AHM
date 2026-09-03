"""Levenberg-Marquardt optimization on the unit box [0,1]^n.

1:1 Python translation of MRST's ``unitBoxLM.m``
(mrst-2026a/autodiff/optimization/optim/unitBoxLM.m). Companion to
:mod:`PRSTCore.optimization.optim.optimize_bound_constrained` (the BFGS
sibling, ``unitBoxBFGS.m``'s port): this one expects a residual-vector
objective ``v, J = f(u)`` (Gauss-Newton/LM formulation) rather than a
scalar-objective-plus-gradient interface.

Scope: the ``lsqTol > 0`` branch (MATLAB's ``lsqminnorm``, a truncated-SVD
least-squares solve for rank-deficient damped systems) is approximated here
via ``numpy.linalg.lstsq`` with ``rcond=lsq_tol`` rather than reproducing
``lsqminnorm``'s exact algorithm; this only affects callers who explicitly
opt into that path (default ``lsq_tol=0`` uses the same direct
``numpy.linalg.solve`` as MATLAB's ``mldivide`` on the well-conditioned
damped normal equations). Plotting (``plotEvolution``) is not ported.

Note: this is a distinct, separately-namespaced implementation from the
simplified ``unit_box_lm`` helper already defined in
``PRSTCore.optimization`` (used by ``eggCoarseModelAdjointCalibration.py``
and ``tests/test_cgnet_upscaling.py``, which depend on that helper's
different call signature -- ``residual_func(p) -> (r, J, extra)``, no
history/box active-set/convergence-criteria support). That helper is left
untouched to avoid breaking its existing callers; use this module directly
(``PRSTCore.optimization.optim.unit_box_lm.unit_box_lm``) for the faithful,
MRST-parity-validated port.
"""

from __future__ import annotations

import numpy as _np


def unit_box_lm(u0, f, *, lambda_init=0.01, lambda_increase=8.0, lambda_decrease=5.0,
                 radius_increase=2.0, radius_decrease=4.0, lambda_max=1e6, lambda_min=1e-6,
                 ratio_thresholds=(0.25, 0.75), scaled_damping=False, update_strategy="simple",
                 grad_tol=1e-6, update_tol=1e-6, res_tol_abs=1e-5, res_tol_rel=0.0,
                 res_change_tol_rel=-_np.inf, max_it=20, max_fun_evals=None, lsq_tol=0.0,
                 verbose=True):
    """Port of ``unitBoxLM.m``.

    Parameters
    ----------
    u0 : (n,) array
        Initial guess, ``0 <= u0 <= 1``.
    f : callable
        ``v, J = f(u)`` returning an ``(m,)`` residual vector ``v`` and an
        ``(m, n)`` Jacobian ``J`` such that ``J.T @ v`` is the gradient of
        ``sum(v**2)`` w.r.t. ``u``.

    Returns
    -------
    v : float
        Best objective value (``sum(v**2)``).
    u : (n,) ndarray
        Control vector corresponding to ``v``.
    history : dict
        Per-iteration ``val``, ``u``, ``pg`` (projected-gradient norm),
        ``lambda``, ``rho``, ``nIt``, ``du``, truncated to the iterations
        actually taken.
    """
    if max_fun_evals is None:
        max_fun_evals = 2 * max_it

    if update_strategy != "TR":
        good_fac, bad_fac = lambda_decrease, lambda_increase
    else:
        good_fac, bad_fac = radius_increase, radius_decrease

    u0 = _np.asarray(u0, dtype=float).ravel()
    n = u0.size
    cap = lambda x: _np.clip(x, 0.0, 1.0)

    u = u0.copy()
    lam = lambda_init
    du = _np.zeros(n)
    radius = _np.nan

    nbuf = max_it + 2
    h = {
        "val": _np.full(nbuf, _np.nan), "u": [None] * nbuf, "lambda": _np.full(nbuf, _np.nan),
        "rho": _np.full(nbuf, _np.nan), "nIt": _np.zeros(nbuf, dtype=int), "pg": _np.full(nbuf, _np.nan),
        "du": _np.full(nbuf, _np.nan),
    }

    it = 0
    accept = True
    isFree = _np.ones(n, dtype=bool)
    Jr = gr = JJr = Dr = None

    def converged(it_):
        if it_ <= 0:
            return False
        dv = _np.inf
        if it_ > 1:
            dv = abs((h["val"][it_] - h["val"][it_ - 1]) / h["val"][it_])
        flags = [
            it_ >= max_it + 1,
            h["nIt"][:it_ + 1].sum() >= max_fun_evals,
            h["pg"][it_] < grad_tol,
            h["du"][it_] < update_tol,
            h["val"][it_] < res_tol_abs,
            h["val"][it_] / h["val"][1] < res_tol_rel,
            dv < res_change_tol_rel,
        ]
        if any(flags) and verbose:
            reasons = [
                f"Reached maximal number of iterations ({it_})",
                f"Reached maximal number of function evaluations ({h['nIt'][:it_ + 1].sum()})",
                f"Norm of projected gradient below tolerance ({h['pg'][it_]:.2e} < {grad_tol:.2e})",
                f"Norm of update below tolerance ({h['du'][it_]:.2e} < {update_tol:.2e})",
                f"Absolute mismatch below tolerance ({h['val'][it_]:.2e} < {res_tol_abs:.2e})",
                f"Relative mismatch below tolerance ({h['val'][it_] / h['val'][1]:.2e} < {res_tol_rel:.2e})",
                f"Relative mismatch change below tolerance ({dv:.2e} < {res_change_tol_rel:.2e})",
            ]
            idx = next(i for i, fl in enumerate(flags) if fl)
            print(f"Optimization finished: {reasons[idx]}")
        return any(flags)

    def lsq_solve(A, b):
        if lsq_tol > _np.finfo(float).eps:
            sol, *_r = _np.linalg.lstsq(A, b, rcond=lsq_tol)
            return sol
        return _np.linalg.solve(A, b)

    def compute_update(JJ, D, g, lam_, r_):
        if update_strategy != "TR":
            lam_ = max(lambda_min, min(lambda_max, lam_))
            du_ = -lsq_solve(JJ + lam_ * D, g)
            return du_, lam_, r_
        if not _np.isfinite(r_):
            du_ = -lsq_solve(JJ + lam_ * D, g)
            return du_, lam_, float(_np.linalg.norm(du_))
        it2 = 0
        ndu = _np.inf
        lam0 = lam_
        du_ = -lsq_solve(JJ + lam_ * D, g)
        while abs(ndu - r_) > 0.1 * r_ and it2 < 20:
            it2 += 1
            du_ = -lsq_solve(JJ + lam_ * D, g)
            dut = -lsq_solve(JJ + lam_ * D, du_)
            ndu = float(_np.linalg.norm(du_))
            ndut = float(du_ @ dut) / ndu
            lam_ = lam_ + (1 - ndu / r_) * ndu / ndut
            lam_ = max(lambda_min, min(lambda_max, lam_))
        if it2 == 20:
            lam_ = lam0 * lambda_increase
            du_ = -lsq_solve(JJ + lam_ * D, g)
            r_ = float(_np.linalg.norm(du_))
        else:
            r_ = float(_np.linalg.norm(du_))
        return du_, lam_, r_

    while not converged(it):
        it += 1
        uNew = cap(u + du)
        resNew, JNew = f(uNew)
        resNew = _np.asarray(resNew, dtype=float).ravel()
        JNew = _np.asarray(JNew, dtype=float)
        val = float(_np.sum(resNew ** 2))

        h["val"][it], h["lambda"][it] = val, lam
        h["u"][it], h["nIt"][it] = uNew, h["nIt"][it] + 1

        if it > 1:
            denom = float(dur @ (lam * Dr @ dur - gr))
            rho = -(val - h["val"][it - 1]) / denom
            h["rho"][it] = rho
            accept = rho > 0
            if rho < ratio_thresholds[0]:
                lam = lam * bad_fac
                radius = radius / bad_fac
            elif rho > ratio_thresholds[1]:
                lam = lam / good_fac
                radius = radius * good_fac

        if it > 1 and not accept:
            it -= 1
        else:
            u, r, J = uNew, resNew, JNew
            g = J.T @ r
            pg = float(_np.linalg.norm(u - cap(u - g)))
            h["pg"][it] = pg
            if pg < grad_tol or it >= max_it + 1:
                continue
            isFree = ~((u == 0) & (g > 0)) & ~((u == 1) & (g < 0))
            Jr = J[:, isFree]
            gr = g[isFree]
            JJr = Jr.T @ Jr
            if scaled_damping:
                dr = _np.diag(JJr).copy()
                mval = 1e-3 * dr.max()
                dr[dr < mval] = mval
                Dr = _np.diag(dr)
            else:
                Dr = _np.eye(int(isFree.sum()))

        dur, lam, radius = compute_update(JJr, Dr, gr, lam, radius)
        du = _np.zeros(n)
        du[isFree] = dur
        h["du"][it] = float(_np.linalg.norm(du))
        if verbose:
            print(f"It: {it - (0 if accept else 1):2d} | val: {h['val'][it]:.3e} | "
                  f"its: {h['nIt'][it]:3d} | lambda: {h['lambda'][it]:.3e} | pgrad: {h['pg'][it]:.3e}")

    if it == 1:
        pass  # matches MRST's "finished after first function evaluation" warning
    elif h["val"][it] > h["val"][it - 1]:
        it -= 1

    v = h["val"][it]
    u_out = h["u"][it]
    history = {k: (_np.asarray(vv[1:it + 1]) if k != "u" else vv[1:it + 1]) for k, vv in h.items()}
    return v, u_out, history
