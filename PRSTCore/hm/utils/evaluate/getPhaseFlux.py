"""Port of MRST ``getPhaseFlux.m`` (mrst-2026a/hm/utils/evaluate).

Per-perforation phase flux at reservoir conditions, expanded back to the
full perforation list.

The facility only carries the *active* wells, so the computed values are
scattered into a full-length vector and inactive wells' perforations stay
at zero -- a shut well contributes nothing rather than shifting the
indices of the wells after it.
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import is_ad as _is_ad


def getPhaseFlux(model, state):
    """Return one flux array per phase, over every perforation."""
    map_ = model.getProp(state, 'FacilityWellMapping')
    phaseq = model.getProp(state, 'PhaseFlux')
    phaseb = model.ReservoirModel.getProp(state, 'ShrinkageFactors')
    if not isinstance(phaseb, (list, tuple)):
        phaseb = [phaseb]
    cells = _np.asarray(_get(map_, 'cells'), dtype=int).ravel()

    nph = model.getNumberOfPhases()
    if len(phaseq) != nph or len(phaseb) != nph:
        raise ValueError(
            'PhaseFlux and ShrinkageFactors must contain %d phases' % nph)
    phaseb = [_take(b, cells) for b in phaseb]

    wellSol = state['wellSol']
    nwt = len(wellSol)
    num_cells = _np.asarray(
        [_np.atleast_1d(_np.asarray(w['cells'])).ravel().size for w in wellSol],
        dtype=int)
    nwc = int(num_cells.sum())

    active_index = _np.asarray(_get(map_, 'active'), dtype=int).ravel()
    if (_np.any(active_index < 0) or _np.any(active_index >= nwt) or
            _np.unique(active_index).size != active_index.size):
        raise ValueError('FacilityWellMapping.active is invalid')
    active_well = _np.zeros(nwt, dtype=bool)
    active_well[active_index] = True
    active = _np.repeat(active_well, num_cells)
    active_perforations = int(_np.count_nonzero(active))
    if cells.size != active_perforations:
        raise ValueError(
            'FacilityWellMapping.cells has width %d; expected %d active '
            'perforations' % (cells.size, active_perforations))

    flux = []
    for i in range(nph):
        q_width = (phaseq[i].val.size if _is_ad(phaseq[i])
                   else _np.asarray(phaseq[i]).size)
        b_width = (phaseb[i].val.size if _is_ad(phaseb[i])
                   else _np.asarray(phaseb[i]).size)
        if q_width != active_perforations:
            raise ValueError(
                'PhaseFlux[%d] has width %d; expected %d; padding/trimming '
                'is forbidden' % (i, q_width, active_perforations))
        if b_width != active_perforations:
            raise ValueError(
                'ShrinkageFactors[%d] has width %d; expected %d'
                % (i, b_width, active_perforations))
        v = phaseq[i] * phaseb[i]
        width = v.val.size if _is_ad(v) else _np.asarray(v).size
        if width != active_perforations:
            raise ValueError(
                'PhaseFlux[%d] has width %d; expected %d; padding/trimming '
                'is forbidden' % (i, width, active_perforations))
        if _is_ad(v):
            tmp = _SparseADI.scatter(_np.flatnonzero(active), v, nwc)
        else:
            tmp = _np.zeros(nwc)
            tmp[active] = _np.asarray(v, dtype=float).ravel()
        flux.append(tmp)
    return flux


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _take(value, indices):
    if _is_ad(value):
        return value[indices]
    array = _np.asarray(value, dtype=float).ravel()
    if indices.size and int(indices.max()) >= array.size:
        raise IndexError('FacilityWellMapping.cells is outside shrinkage data')
    return array[indices]
