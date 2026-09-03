"""MRST trajectory ``perturbWell.m`` counterpart."""

import numpy as np


def perturb_well(model, state, W, getResiduals, posControl, **kwargs):
    """Finite-difference perturbation of well-position controls."""
    del model
    eps = float(kwargs.get("epsilon", 1e-6))
    base = getResiduals(state, W)
    params = getattr(posControl, "parameters", {})
    perturb = np.asarray(params.get("perturbation", []), dtype=float) if isinstance(params, dict) else np.asarray(getattr(params, "perturbation", []), dtype=float)
    out = []
    for k in range(len(perturb)):
        Wp = [dict(w) for w in W]
        if hasattr(posControl, "update"):
            Wp = posControl.update(Wp, k, eps)
        res = getResiduals(state, Wp)
        out.append((np.asarray(res, dtype=float) - np.asarray(base, dtype=float)) / eps)
    return out


perturbWell = perturb_well

