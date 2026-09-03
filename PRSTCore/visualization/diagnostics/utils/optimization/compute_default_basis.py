"""MRST ``computeDefaultBasis.m`` counterpart."""

from __future__ import annotations

from PRSTCore.visualization.diagnostics import computeTOFandTracer


def compute_default_basis(basis, G, state, system, W, fluid, pv, T, **kwargs):
    """Populate a simple diagnostics basis dictionary."""
    del system, fluid, pv, T
    out = {} if basis is None else dict(basis)
    rock = kwargs.get("rock", {"poro": kwargs.get("poro", None)})
    if rock.get("poro") is None:
        import numpy as np

        rock = {"poro": np.ones(G["cells"]["num"])}
    out["D"] = computeTOFandTracer(state, G, rock, wells=W, **{k: v for k, v in kwargs.items() if k != "rock"})
    return out


computeDefaultBasis = compute_default_basis

