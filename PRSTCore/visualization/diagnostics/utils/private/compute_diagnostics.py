"""MRST private ``computeDiagnostics.m`` counterpart."""

from PRSTCore.visualization.diagnostics import computeTOFandTracer, computeWellPairs


def compute_diagnostics(G, data, maxTOF=None, ix=None, precomp=None):
    if precomp is not None:
        return precomp
    states = data.get("states", [])
    rock = data.get("rock", {"poro": data.get("PORV")})
    wells = data.get("W", data.get("wells", []))
    indices = range(len(states)) if ix is None else ix
    diagnostics = []
    for i in indices:
        W = wells[i] if wells and isinstance(wells[0], list) else wells
        D = computeTOFandTracer(states[i], G, rock, wells=W, maxTOF=maxTOF, computeWellTOFs=True)
        WP = computeWellPairs(states[i], G, rock, W, D)
        diagnostics.append({"D": D, "WP": WP})
    out = dict(data)
    out["diagnostics"] = diagnostics
    return out


computeDiagnostics = compute_diagnostics

