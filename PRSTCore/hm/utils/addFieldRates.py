"""Port of MRST ``addFieldRates.m`` (mrst-2026a/hm/utils).

Appends two synthetic field-level entries -- ``'producers'`` and
``'injectors'`` -- to every well solution, carrying the summed phase rates
and the mean bottom-hole pressure over the wells of that sign.

Any entry already carrying one of those names is dropped first, so calling
this twice does not accumulate duplicates.
"""

import copy as _copy

import numpy as _np


def addFieldRates(wellSols):
    """Add field aggregates to each timestep's well solution list."""
    return [_addGroupRates(_addGroupRates(ws, 'producers'), 'injectors')
            for ws in wellSols]


def _addGroupRates(wellSol, name):
    """Port of the local ``addGroupRates``.

    ``sign == -1`` selects producers, ``+1`` injectors; only wells that are
    both open (``status``) and of that sign contribute.
    """
    wellSol = [w for w in wellSol
               if str(w.get('name', '')).lower() != name.lower()]
    if not wellSol:
        return wellSol

    flag = -1 if name.lower() == 'producers' else 1
    picked = [w for w in wellSol
              if bool(w.get('status', True)) and int(w.get('sign', 0)) == flag]

    qWt = qOt = qGt = bhp = 0.0
    if picked:
        qWt = float(_np.sum([_scalar(w.get('qWs', 0.0)) for w in picked]))
        qOt = float(_np.sum([_scalar(w.get('qOs', 0.0)) for w in picked]))
        qGt = float(_np.sum([_scalar(w.get('qGs', 0.0)) for w in picked]))
        bhp = float(_np.mean([_scalar(w.get('bhp', 0.0)) for w in picked]))

    # MATLAB seeds the new entry from the last well so it inherits every
    # other field of the struct array.
    tmp = _copy.deepcopy(wellSol[-1])
    tmp.update({'name': name, 'qWs': qWt, 'qOs': qOt, 'qGs': qGt,
                'bhp': bhp, 'status': bool(picked), 'sign': flag})
    return wellSol + [tmp]


def _scalar(v):
    arr = _np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
    return float(_np.sum(arr)) if arr.size else 0.0
