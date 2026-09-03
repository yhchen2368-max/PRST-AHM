import numpy as np


def negate_point(p):
    """Negate value and derivative of a point struct (for argmaxCubic)."""
    return {"a": p["a"], "v": -p["v"], "dv": -p["dv"]}


def argmax_cubic(p1, p2):
    """Find max of cubic polynomial through points p1, p2.

    1:1 Python translation of MRST argmaxCubic.m

    Parameters
    ----------
    p1, p2 : dict
        Points with keys 'a' (step), 'v' (value), 'dv' (directional derivative).

    Returns
    -------
    x_opt : float
        Optimal step length.
    poly : ndarray
        Polynomial coefficients [c3, c2, c1, c0].
    """
    shift = p1["a"]
    a1 = 0.0
    a2 = p2["a"] - shift
    v1 = p1["v"]
    dv1 = p1["dv"]
    v2 = p2["v"]
    dv2 = p2["dv"]

    poly = np.zeros(4)
    poly[2] = dv1
    poly[3] = v1

    # Solve for cubic coefficients: [a^3, a^2] @ [c3; c2] = [rhs1; rhs2]
    A = np.array([[a2**3, a2**2], [3 * a2**2, 2 * a2]])
    rhs = np.array([v2 - dv1 * a2 - v1, dv2 - dv1])
    try:
        coeffs = np.linalg.solve(A, rhs)
        poly[0] = coeffs[0]
        poly[1] = coeffs[1]
    except np.linalg.LinAlgError:
        poly[0] = 0.0
        poly[1] = 0.0

    # Derivative roots: 3*c3*x^2 + 2*c2*x + c1 = 0
    c3, c2, c1 = poly[0], poly[1], poly[2]
    if abs(c3) < 1e-15:
        if abs(c2) < 1e-15:
            xe = -np.inf
        elif c2 < 0:
            xe = -c1 / (2 * c2)
        else:
            xe = -np.inf
    else:
        disc = 4 * c2**2 - 12 * c3 * c1
        if disc < 0:
            xe = np.inf
        else:
            root1 = (-2 * c2 + np.sqrt(disc)) / (6 * c3)
            root2 = (-2 * c2 - np.sqrt(disc)) / (6 * c3)
            v1_root = np.polyval(poly, root1)
            v2_root = np.polyval(poly, root2)
            xe = root1 if v1_root > v2_root else root2
            if xe < a1:
                xe = -np.inf

    x_opt = xe + shift
    return x_opt, poly


def assign_point(a, v, dv):
    """Create a point dict for line search."""
    return {"a": float(a), "v": float(v), "dv": float(dv)}


def line_search(u0, v0, g0, d, f, opt):
    """Line search based on Wolfe conditions.

    1:1 Python translation of MRST lineSearch.m

    Parameters
    ----------
    u0 : ndarray
        Current control vector.
    v0 : float
        Current objective value.
    g0 : ndarray
        Current gradient.
    d : ndarray
        Search direction (must satisfy d'*g0 <= 0).
    f : callable
        Objective function returning (value, gradient).
    opt : dict
        Options with keys: wolfe1, wolfe2, safeguardFac, stepIncreaseTol,
        lineSearchMaxIt, maxStep.

    Returns
    -------
    u : ndarray
    v : float
    g : ndarray
    info : dict
    """
    c1 = opt.get("wolfe1", 1e-4)
    c2 = opt.get("wolfe2", 0.9)
    sgf = opt.get("safeguardFac", 1e-5)
    inc_tol = opt.get("stepIncreaseTol", 10)
    max_it = opt.get("lineSearchMaxIt", 5)
    a_max = opt.get("maxStep", np.inf)

    assert d.dot(g0) <= 0, "Search direction must be decreasing"

    p0 = assign_point(0, v0, d.dot(g0))

    def w1(p):
        return p["v"] <= p0["v"] + c1 * p["a"] * p0["dv"]

    def w2(p):
        return abs(p["dv"]) <= c2 * abs(p0["dv"])

    p1 = p0
    p2 = assign_point(a_max, np.inf, np.inf)
    a = 1.0
    done = False
    it = 0
    flag = 1
    vals = np.full(max_it, np.nan)

    while not done and it < max_it:
        it += 1
        u = u0 + a * d
        v, g = f(u)
        g = np.asarray(g, dtype=float).ravel()
        vals[it - 1] = v
        p = assign_point(a, v, d.dot(g))

        if w1(p) and w2(p):
            done = True
            flag = 1
        elif abs(a_max - p["a"]) < np.sqrt(np.finfo(float).eps) and p["v"] < p0["v"] and p["dv"] < 0:
            done = True
            flag = -1
        else:
            if p["a"] > p2["a"]:
                p1, p2 = p2, p
            elif p["dv"] >= 0:
                p2 = p
            elif p1["v"] <= p2["v"]:
                p2 = p
            else:
                p1 = p

            if p1["v"] > p2["v"] and p1["dv"] >= p2["dv"]:
                a = np.inf
            else:
                np1 = negate_point(p1)
                np2 = negate_point(p2)
                a, _ = argmax_cubic(np1, np2)

            if a > p2["a"]:
                a = max(a, (1 + sgf) * p2["a"])
                a = min(a, min(inc_tol * p2["a"], a_max))
            elif a > p1["a"]:
                a = max(a, p1["a"] + sgf * (p2["a"] - p1["a"]))
                a = min(a, p2["a"] - sgf * (p2["a"] - p1["a"]))
            else:
                a = (p1["a"] + p2["a"]) / 2

    if not done:
        flag = -2
        if p1["v"] < p2["v"]:
            u = u0 + p1["a"] * d
            v, g = f(u)

    # objVals is the trace of objective values tried, in the order tried.
    # Callers use objVals[0] as the value at the *first* trial step, which
    # is what the quadratic-model ratio rho is measured against.
    info = {"flag": flag, "step": a, "nits": it, "objVals": vals[:it]}
    return u, v, g, info
