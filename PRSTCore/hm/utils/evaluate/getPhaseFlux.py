"""Port of MRST ``getPhaseFlux.m`` (mrst-2026a/hm/utils/evaluate).

Per-perforation phase flux at reservoir conditions, expanded back to the
full perforation list.

The facility only carries the *active* wells, so the computed values are
scattered into a full-length vector and inactive wells' perforations stay
at zero -- a shut well contributes nothing rather than shifting the
indices of the wells after it.
"""

import numpy as _np


def getPhaseFlux(model, state):
    """Return one flux array per phase, over every perforation."""
    map_ = model.getProp(state, 'FacilityWellMapping')
    phaseq = model.getProp(state, 'PhaseFlux')
    phaseb = model.ReservoirModel.getProp(state, 'ShrinkageFactors')
    if not isinstance(phaseb, (list, tuple)):
        phaseb = [phaseb]
    cells = _np.asarray(_get(map_, 'cells'), dtype=int).ravel()
    phaseb = [b[cells] for b in phaseb]

    nph = model.getNumberOfPhases()
    wellSol = state['wellSol']
    nwt = len(wellSol)
    num_cells = _np.asarray(
        [_np.atleast_1d(_np.asarray(w['cells'])).ravel().size for w in wellSol],
        dtype=int)
    nwc = int(num_cells.sum())

    active_well = _np.zeros(nwt, dtype=bool)
    active_well[_np.asarray(_get(map_, 'active'), dtype=int).ravel()] = True
    active = _np.repeat(active_well, num_cells)

    flux = []
    for i in range(nph):
        v = phaseq[i] * phaseb[i]
        if hasattr(v, 'val'):
            from PRSTCore.ad_core.adi import SparseADI
            tmp = SparseADI.scatter(_np.flatnonzero(active), v, nwc)
        else:
            tmp = _np.zeros(nwc)
            tmp[active] = _np.asarray(v, dtype=float).ravel()
        flux.append(tmp)
    return flux


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
