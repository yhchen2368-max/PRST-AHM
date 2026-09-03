"""Python counterpart for MRST ``DiagnosticsNPV.m``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics_objective import DiagnosticsObjective


@dataclass
class DiagnosticsNPV(DiagnosticsObjective):
    timeHorizon: float = 10.0
    timeUnit: str = "year"
    ro: float = 50.0
    rwi: float = -5.0
    rwp: float = -3.0
    rgi: float = 0.0
    rgp: float = 0.0
    discount: float = 0.1
    lengthCost: float = 0.0
    scaleProduction: float = 1.0
    scaleBreakthrough: float = 1.0
    pRef: float = 200e5

    def evaluate(self, state, D, W):
        """Evaluate a lightweight diagnostics NPV proxy from well rates."""
        del D
        value = 0.0
        for ws, w in zip(state.get("wellSol", []), W, strict=False):
            sign = float(ws.get("sign", w.get("sign", 0.0)))
            q = float(np.sum(np.asarray(ws.get("flux", [w.get("val", 0.0)]), dtype=float)))
            if sign > 0:
                value += self.rwi * abs(q)
            else:
                value += self.ro * abs(q) + self.rwp * abs(q) * 0.0
        return float(value)


def makeCompatibleForObjective(dm):
    if isinstance(dm, dict):
        dm = dict(dm)
        dm.update({"computeForward": False, "computeBackward": True, "tracerWells": "none", "computeWellTOFs": False})
    else:
        dm.computeForward = False
        dm.computeBackward = True
        dm.tracerWells = "none"
        dm.computeWellTOFs = False
    return dm

