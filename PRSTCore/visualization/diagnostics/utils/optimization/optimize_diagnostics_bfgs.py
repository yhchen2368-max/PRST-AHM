"""MRST ``optimizeDiagnosticsBFGS.m`` counterpart."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimize_diagnostics_bfgs(G, W, fluid, pv, T, s, state, boxLims, objective, *, maxiter=50, **kwargs):
    del G, fluid, pv, T, s, kwargs
    box = np.asarray(boxLims, dtype=float)
    x0 = np.mean(box, axis=1)

    def fun(x):
        if hasattr(objective, "compute"):
            Wnew = [dict(w) for w in W]
            for k, value in enumerate(x[: len(Wnew)]):
                Wnew[k]["val"] = float(value)
            return float(objective.compute(Wnew, state=state, computeGradient=False)["value"])
        return float(objective(x))

    res = minimize(fun, x0, method="L-BFGS-B", bounds=[tuple(b) for b in box], options={"maxiter": maxiter})
    info = {"result": res, "objective": res.fun}
    return [state], [], info


optimizeDiagnosticsBFGS = optimize_diagnostics_bfgs

