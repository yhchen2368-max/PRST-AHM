"""MRST ``getObjectiveDiagnostics.m`` counterpart."""

from .diagnostics_npv import DiagnosticsNPV


def get_objective_diagnostics(G, rock, type, **kwargs):
    """Create a diagnostics objective descriptor."""
    model = kwargs.pop("model", {"G": G, "rock": rock})
    if str(type).lower() in {"npv", "diagnosticsnpv"}:
        return DiagnosticsNPV(model=model, **kwargs)
    return {"G": G, "rock": rock, "type": type, **kwargs}


getObjectiveDiagnostics = get_objective_diagnostics

