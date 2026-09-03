"""MRST ``validateStateForDiagnostics.m`` counterpart."""

from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import get_field, set_field


def validate_state_for_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    """Ensure total face and well fluxes exist for diagnostics.

    MRST collapses multi-component flux arrays by summing over columns.  This
    helper follows the same rule for ``state['flux']``, ``wellSol[*]['flux']``,
    and falls back to ``wellSol[*]['cqs']`` when needed.
    """
    out = dict(state)
    flux = get_field(out, "flux", None)
    if flux is None:
        raise ValueError("Reservoir state must provide total Darcy flux")
    flux = np.asarray(flux, dtype=float)
    if flux.ndim > 1:
        flux = np.sum(flux, axis=1)
    out["flux"] = flux.ravel()

    well_sols = list(get_field(out, "wellSol", []) or [])
    for idx, ws in enumerate(well_sols):
        wsd = dict(ws)
        wflux = get_field(wsd, "flux", None)
        if wflux is not None:
            arr = np.asarray(wflux, dtype=float)
            if arr.ndim > 1:
                arr = np.sum(arr, axis=1)
            wsd["flux"] = arr.ravel()
        elif get_field(wsd, "cqs", None) is not None:
            wsd["flux"] = np.sum(np.asarray(get_field(wsd, "cqs"), dtype=float), axis=1).ravel()
        well_sols[idx] = wsd
    set_field(out, "wellSol", well_sols)
    return out


validateStateForDiagnostics = validate_state_for_diagnostics

