"""Python counterpart for MRST ``DiagnosticsObjective.m``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiagnosticsObjective:
    model: Any
    wellControl: bool = True

    def compute(self, W, *, state0=None, state=None, D=None, computeGradient=True, outputState=True, outputDiagnostics=True):
        result = {"value": None, "gradient": None, "state": None, "D": None}
        if state is None and hasattr(self.model, "solvePressure"):
            state = self.model.solvePressure(W, state0=state0)
        if D is None and hasattr(self.model, "solveDiagnostics"):
            state, D = self.model.solveDiagnostics(state)
        if outputState:
            result["state"] = state
        if outputDiagnostics:
            result["D"] = D
        if computeGradient:
            value = self.evaluate(state, D, W)
            if isinstance(value, tuple):
                result["value"] = value[0]
                result["gradient"] = value[1:]
            else:
                result["value"] = value
        else:
            result["value"] = self.evaluate(state, D, W)
        return result

    def evaluate(self, state, D, W):
        raise NotImplementedError("'DiagnosticsObjective' does not contain an objective that can be evaluated")

    def getEquationPartials(self, system, state, forces):
        return {"well": None, "position": None}


def assembleGradient(L, dFdu, dJdu, W, isWellControl=True):
    del L, dFdu, W, isWellControl
    return dJdu

