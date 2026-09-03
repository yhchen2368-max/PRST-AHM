"""Python port of MRST's explicit two-phase transport solver
(mrst-2026a/solvers/incomp/transport/{explicitTransport,private/twophaseUpwFE,
private/initTransport}.m), no-gravity path.

Solves the Buckley-Leverett transport equation ``s_t + f(s)_x = q`` with a
first-order upwind discretization in space and forward Euler in time, given
the face fluxes produced by :func:`PRSTCore.solvers.incomp.incomp_tpfa`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(slots=True)
class TwoPhaseFluid:
    """Minimal two-phase fluid model: constant viscosities + a relperm
    callable ``relperm(sw) -> (krw, kro)``."""
    mu: tuple[float, float]
    relperm: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    rhoWS: float | None = None  # surface density, water (kg/m^3); informational only here
    rhoOS: float | None = None  # surface density, oil (kg/m^3); informational only here


def corey_fluid(mu_w: float, mu_o: float, *, nw: float = 2.0, no: float = 2.0,
                 krw_max: float = 1.0, kro_max: float = 1.0,
                 swc: float = 0.0, sor: float = 0.0) -> TwoPhaseFluid:
    """Port of MRST ``ad-props/simple``-style Corey relative permeability,
    as a convenience fluid for ``explicit_transport``/``implicit_transport``."""
    span = max(1.0 - swc - sor, 1e-12)

    def relperm(sw: np.ndarray):
        sw = np.clip(np.asarray(sw, dtype=float), swc, 1.0 - sor)
        se = (sw - swc) / span
        krw = krw_max * se**nw
        kro = kro_max * (1.0 - se) ** no
        return krw, kro

    return TwoPhaseFluid(mu=(mu_w, mu_o), relperm=relperm)


def linear_fluid(mu_w: float, mu_o: float) -> TwoPhaseFluid:
    """Linear (kr=s) relperm -- gives f_w(s)=s under equal viscosities, i.e.
    pure linear advection with a known-exact solution. Useful for testing."""
    def relperm(sw: np.ndarray):
        sw = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
        return sw, 1.0 - sw

    return TwoPhaseFluid(mu=(mu_w, mu_o), relperm=relperm)


def total_mobility(fluid: TwoPhaseFluid, s: np.ndarray) -> np.ndarray:
    """Total mobility ``krw(s)/mu_w + kro(s)/mu_o``, for the ``mob=``
    argument of :func:`PRSTCore.solvers.incomp.incomp_tpfa.incomp_tpfa`."""
    mu_w, mu_o = fluid.mu
    krw, kro = fluid.relperm(s)
    return krw / mu_w + kro / mu_o


def _boundary_inflow_per_cell(G: dict, flux: np.ndarray, nc: int) -> np.ndarray:
    neighbors = G["faces"]["neighbors"]
    boundary = (neighbors[:, 0] < 0) | (neighbors[:, 1] < 0)
    bidx = np.nonzero(boundary)[0]
    if bidx.size == 0:
        return np.zeros(nc)
    boundary_on_side0 = neighbors[bidx, 0] < 0
    c = np.where(boundary_on_side0, neighbors[bidx, 1], neighbors[bidx, 0])
    sgn = np.where(boundary_on_side0, 1.0, -1.0)
    return np.bincount(c, weights=sgn * flux[bidx], minlength=nc)


def _source_term(G: dict, state: dict, wells) -> np.ndarray:
    """Net rate *into* each cell (m^3/s, positive = injection) from boundary
    faces (whatever bc/src incomp_tpfa already baked into state['flux']) and
    wells. Equivalent to MRST's computeTransportSourceTerm, derived directly
    from mass conservation rather than re-parsing bc/src structures."""
    nc = G["cells"]["num"]
    q = _boundary_inflow_per_cell(G, np.asarray(state["flux"], dtype=float), nc)
    if wells:
        for w, wsol in zip(wells, state["wellSol"]):
            wc = np.asarray(w["cells"], dtype=int)
            q += np.bincount(wc, weights=np.atleast_1d(wsol["flux"]), minlength=nc)
    return q


def _upwind_edges(G: dict, flux: np.ndarray, nc: int):
    """Port of ``initTransport.m``'s no-gravity ``matrices_nograv``, as raw
    (dst, src, weight) triples: ``weight`` is the positive flux magnitude
    flowing from cell ``src`` into cell ``dst`` -- built directly from face
    fluxes rather than MRST's half-face detour, since our face-level flux
    already carries a clean sign convention. Also returns each cell's total
    outflow (sum of weights where it is the source)."""
    neighbors = G["faces"]["neighbors"]
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)
    a = neighbors[internal, 0]
    b = neighbors[internal, 1]
    v = flux[internal]

    # v > 0: flow a->b (inflow into b, magnitude v). v < 0: flow b->a (inflow into a, magnitude -v).
    dst = np.where(v >= 0, b, a)
    src = np.where(v >= 0, a, b)
    w = np.abs(v)

    outflow_per_cell = np.bincount(src, weights=w, minlength=nc)
    return dst, src, w, outflow_per_cell


def _fractional_flow_and_deriv(fluid: TwoPhaseFluid, s: np.ndarray, *, eps: float = 1e-6):
    def f_w(sat: np.ndarray) -> np.ndarray:
        mu_w, mu_o = fluid.mu
        krw, kro = fluid.relperm(sat)
        mob_w, mob_o = krw / mu_w, kro / mu_o
        return mob_w / (mob_w + mob_o)

    f = f_w(s)
    s_p = np.clip(s + eps, 0.0, 1.0)
    s_m = np.clip(s - eps, 0.0, 1.0)
    denom = s_p - s_m
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    df = (f_w(s_p) - f_w(s_m)) / denom
    return f, df


def explicit_transport(G: dict, state: dict, rock: dict, fluid: TwoPhaseFluid, tf: float, *,
                        wells=None, dt: float | None = None, sat_warn: float = 1e-8) -> dict:
    """Port of MRST ``explicitTransport.m`` / ``twophaseUpwFE.m`` (no gravity).

    ``state`` must contain ``'flux'``/``'wellSol'`` from a prior
    :func:`incomp_tpfa` call, and ``'s'`` (water saturation per cell). Returns
    a new state dict with updated ``'s'``.
    """
    nc = G["cells"]["num"]
    pv = np.asarray(G["cells"]["volumes"], dtype=float) * np.asarray(rock["poro"], dtype=float)
    q = _source_term(G, state, wells)
    flux = np.asarray(state["flux"], dtype=float)
    dst, src, w, outflow = _upwind_edges(G, flux, nc)

    s = np.array(state["s"], dtype=float, copy=True)
    step_dt = tf if dt is None else dt
    if step_dt <= 0 or tf <= 0:
        raise ValueError("dt and tf must be positive")

    t = 0.0
    while t < tf:
        h = min(step_dt, tf - t)
        mu_w, mu_o = fluid.mu
        krw, kro = fluid.relperm(s)
        mob_w, mob_o = krw / mu_w, kro / mu_o
        f_w = mob_w / (mob_w + mob_o)

        # dz[i] = (upwind-weighted inflow into i) - (i's own outflow, weighted by i's own f_w):
        # net water accumulation rate. Conservation form: pv*ds/dt = dz + source, so a
        # positive net inflow *increases* saturation.
        dz = np.bincount(dst, weights=w * f_w[src], minlength=nc) - outflow * f_w
        s = s + (h / pv) * (dz + np.maximum(q, 0.0) + np.minimum(q, 0.0) * f_w)

        if np.any(s > 1 + sat_warn) or np.any(s < -sat_warn):
            pass  # matches MRST's non-fatal saturation warning; values are clamped below
        s = np.clip(s, 0.0, 1.0)

        t += h

    return {**state, "s": s}


def implicit_transport(G: dict, state: dict, rock: dict, fluid: TwoPhaseFluid, tf: float, *,
                        wells=None, dt: float | None = None, tol: float = 1e-6,
                        maxiter: int = 25) -> dict:
    """Port of MRST ``implicitTransport.m`` / ``twophaseJacobian.m`` (no gravity):
    backward-Euler upwind transport, solved per step with Newton-Raphson using
    an analytic (finite-difference-in-``f_w``) sparse Jacobian.

    Unconditionally stable (no CFL restriction), at the cost of a nonlinear
    solve per step -- the standard trade-off against :func:`explicit_transport`.
    Unlike MRST's ``implicitTransport``, time-step chopping on non-convergence
    and line search are not implemented; a fixed (or externally sub-stepped)
    ``dt`` is used and a `RuntimeError` is raised if Newton fails to converge.
    """
    nc = G["cells"]["num"]
    pv = np.asarray(G["cells"]["volumes"], dtype=float) * np.asarray(rock["poro"], dtype=float)
    q = _source_term(G, state, wells)
    flux = np.asarray(state["flux"], dtype=float)
    dst, src, w, outflow = _upwind_edges(G, flux, nc)

    q_pos = np.maximum(q, 0.0)
    q_neg = np.minimum(q, 0.0)
    rows_diag = np.arange(nc)

    def residual(s: np.ndarray, s0: np.ndarray, h: float, f: np.ndarray) -> np.ndarray:
        dz = np.bincount(dst, weights=w * f[src], minlength=nc) - outflow * f
        return s - s0 - (h / pv) * (dz + q_pos + q_neg * f)

    def jacobian(h: float, f: np.ndarray, df: np.ndarray):
        diag = 1.0 + (h / pv) * (outflow - q_neg) * df
        off_vals = -(h / pv[dst]) * w * df[src]
        rows = np.concatenate([rows_diag, dst])
        cols = np.concatenate([rows_diag, src])
        vals = np.concatenate([diag, off_vals])
        return sp.coo_matrix((vals, (rows, cols)), shape=(nc, nc)).tocsr()

    s = np.array(state["s"], dtype=float, copy=True)
    step_dt = tf if dt is None else dt
    if step_dt <= 0 or tf <= 0:
        raise ValueError("dt and tf must be positive")

    t = 0.0
    while t < tf:
        h = min(step_dt, tf - t)
        s0 = s.copy()
        s_iter = s0.copy()
        converged = False
        for _ in range(maxiter):
            f, df = _fractional_flow_and_deriv(fluid, s_iter)
            R = residual(s_iter, s0, h, f)
            if np.linalg.norm(R, ord=np.inf) < tol:
                converged = True
                break
            J = jacobian(h, f, df)
            ds = spla.spsolve(J, -R)
            s_iter = np.clip(s_iter + ds, 0.0, 1.0)
        if not converged:
            f, _ = _fractional_flow_and_deriv(fluid, s_iter)
            final_res = np.linalg.norm(residual(s_iter, s0, h, f), ord=np.inf)
            raise RuntimeError(
                f"implicit_transport: Newton failed to converge in {maxiter} iterations "
                f"(residual={final_res:.3e}, tol={tol:.3e})"
            )
        s = s_iter
        t += h

    return {**state, "s": s}
