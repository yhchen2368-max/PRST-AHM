"""MRST ``evalObjectiveDiagnostics.m`` counterpart."""

from .controls import control2well


def eval_objective_diagnostics(u, obj, state, system, G, fluid, pv, T, W, scaling=None, **kwargs):
    """Update controls, evaluate diagnostics objective, return MRST-like tuple."""
    del system, G, fluid, pv, T
    Wnew = control2well(u, W, scaling=scaling) if u is not None else W
    result = obj.compute(Wnew, state=state, **kwargs) if hasattr(obj, "compute") else {"value": obj(Wnew), "gradient": None, "D": None}
    return result.get("value"), result.get("gradient"), Wnew, result.get("state", state), result.get("D")


evalObjectiveDiagnostics = eval_objective_diagnostics

