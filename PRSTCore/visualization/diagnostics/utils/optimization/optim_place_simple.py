"""MRST ``optimPlaceSimple.m`` counterpart."""

from scipy.optimize import minimize


def optim_place_simple(p, problem, **kwargs):
    fun = problem.get("objective") if isinstance(problem, dict) else None
    if fun is None:
        return p
    res = minimize(fun, p, method=kwargs.get("method", "L-BFGS-B"))
    return res.x


optimPlaceSimple = optim_place_simple

