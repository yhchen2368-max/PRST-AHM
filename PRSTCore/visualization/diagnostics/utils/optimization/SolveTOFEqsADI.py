"""Compatibility wrapper for MRST ``SolveTOFEqsADI.m``."""

from __future__ import annotations

from PRSTCore.visualization.diagnostics import computeTOFandTracer


def SolveTOFEqsADI(eqs, state, W, computeTracer=True, linsolve=None):
    """Solve TOF equations from a Python diagnostics model-like object.

    ``eqs`` may be a dictionary containing ``G`` and ``rock``.  ADI equation
    objects are not currently reproduced in PRSTCore's diagnostics layer.
    """
    if not isinstance(eqs, dict) or "G" not in eqs or "rock" not in eqs:
        raise NotImplementedError("SolveTOFEqsADI requires a dict with G and rock in PRSTCore.")
    return computeTOFandTracer(state, eqs["G"], eqs["rock"], wells=W, computeWellTOFs=bool(computeTracer), solver=linsolve, model=eqs.get("model"))

