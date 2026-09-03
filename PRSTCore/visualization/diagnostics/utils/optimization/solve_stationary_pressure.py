"""MRST ``solveStationaryPressure.m`` diagnostics-facing counterpart."""

from __future__ import annotations

from PRSTCore.visualization.diagnostics.preprocessorGUI.utils import computePressureAndDiagnostics


def solve_stationary_pressure(G, state, operators, W, fluid=None, pv=None, T=None, **kwargs):
    """Solve a stationary pressure proxy and optionally diagnostics.

    The full MRST function is ADI-heavy.  This wrapper uses PRSTCore's
    diagnostics pressure helper and returns ``(state, D)`` when diagnostics
    are requested.
    """
    del fluid, pv, T
    model = {"G": G, "rock": kwargs.pop("rock", {"poro": operators.get("pv") if isinstance(operators, dict) and "pv" in operators else None}), "operators": operators}
    if model["rock"].get("poro") is None:
        import numpy as np

        model["rock"] = {"poro": np.ones(G["cells"]["num"])}
    out_state, diagnostics = computePressureAndDiagnostics(model, wells=W, state=state, **kwargs)
    return out_state, diagnostics.D


solveStationaryPressure = solve_stationary_pressure

