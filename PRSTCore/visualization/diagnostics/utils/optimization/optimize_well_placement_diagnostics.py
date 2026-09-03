"""MRST ``optimizeWellPlacementDiagnostics.m`` counterpart."""


def optimize_well_placement_diagnostics(W, *args, **kwargs):
    return W, [W], {"args": args, "kwargs": kwargs}


optimizeWellPlacementDiagnostics = optimize_well_placement_diagnostics

