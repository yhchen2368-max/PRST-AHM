"""SR1 trust-region optimization (unconstrained).

1:1 Python translation of MRST optimizeSR1.m
"""

import numpy as np


def compute_step(g, B, delta):
    """Compute the quasi-Newton step with trust-region constraint.

    Simplified dogleg / 2D subspace approach.
    """
    g = np.asarray(g, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    n = len(g)

    # Newton step
    try:
        pB = np.linalg.solve(B, -g)
    except np.linalg.LinAlgError:
        pB = -g

    if np.linalg.norm(pB) <= delta:
        s = pB
        Bstar = B
    else:
        # Cauchy point
        gBg = g.dot(B @ g)
        if gBg <= 0:
            tau = 1.0
        else:
            tau = min(g.dot(g) / gBg, delta / np.linalg.norm(g))
        pU = -tau * g

        if np.linalg.norm(pU) >= delta:
            s = delta * pU / np.linalg.norm(pU)
            Bstar = B
        else:
            # Dogleg: convex combination
            try:
                a = pB - pU
                b = pU
                aa = a.dot(a)
                bb = b.dot(b)
                ab = a.dot(b)
                disc = ab * ab - aa * (bb - delta * delta)
                if disc < 0:
                    beta = 1.0
                else:
                    beta = (-ab + np.sqrt(disc)) / aa
                    beta = max(0.0, min(1.0, beta))
                s = pU + beta * (pB - pU)
            except Exception:
                s = delta * (-g) / np.linalg.norm(g)
            Bstar = B

    return s, Bstar


def optimize_sr1(u0, f, B_init=None, B_scale=1.0, delta=1.0,
                 eta=1e-3, epsilon=1e-8, funval_tol=1e-8,
                 r=1e-3, max_it=100, backup_file=None,
                 plot_evolution=False, rat_lim=0.75, delta_fac=0.8):
    """SR1 trust-region optimization (minimization).

    Parameters
    ----------
    u0 : ndarray
        Initial guess.
    f : callable
        Objective function returning (value, gradient).
    B_init : ndarray, optional
        Initial quasi-Hessian.
    B_scale : float
        Scale for initial quasi-Hessian if B_init not provided.
    delta : float
        Initial trust-region radius.
    eta : float
        Acceptance threshold for step.
    epsilon : float
        Gradient norm convergence tolerance.
    funval_tol : float
        Function change tolerance.
    r : float
        SR1 update threshold.
    max_it : int
        Maximum iterations.

    Returns
    -------
    v : float
        Optimal value.
    u : ndarray
        Optimal point.
    history : list of dict
        Iteration history.
    status : int
        1 for convergence, -1 for non-convergence.
    """
    u = np.asarray(u0, dtype=float).ravel()
    n = len(u)

    if B_init is not None:
        B = np.asarray(B_init, dtype=float)
    else:
        B = B_scale * np.eye(n)

    v0, g0 = f(u)
    g0 = np.asarray(g0, dtype=float).ravel()

    history = [{"val": v0, "u": u.copy(), "gnorm": np.linalg.norm(g0)}]
    status = 1

    if np.linalg.norm(g0) <= epsilon:
        return v0, u, history, status

    for it in range(max_it):
        s, Bstar = compute_step(g0, B, delta)

        u_new = u + s
        v, g = f(u_new)
        g = np.asarray(g, dtype=float).ravel()

        dg = g - g0
        act_red = v0 - v
        pre_red = -(g0.dot(s) + 0.5 * s.dot(Bstar @ s))

        ratio = act_red / max(abs(pre_red), 1e-15)

        if ratio > eta:
            u = u_new
            v0, g0 = v, g
            history.append({"val": v, "u": u.copy(), "gnorm": np.linalg.norm(g, np.inf)})

            if abs(act_red) < funval_tol:
                break
        elif abs(act_red) < funval_tol:
            status = 2
            break

        # Trust-region update
        if ratio > rat_lim:
            if np.linalg.norm(s) > delta * delta_fac:
                delta *= 2
        elif ratio < 0.1:
            if np.linalg.norm(s) < delta:
                delta = np.linalg.norm(s) / 2
            else:
                delta /= 2

        # SR1 Hessian update
        tmp = dg - B @ s
        if abs(s.dot(tmp)) > r * np.linalg.norm(s) * np.linalg.norm(tmp):
            B = B + np.outer(tmp, tmp) / tmp.dot(s)

        if np.linalg.norm(g0) <= epsilon:
            break
    else:
        status = -1

    return v0, u, history, status
