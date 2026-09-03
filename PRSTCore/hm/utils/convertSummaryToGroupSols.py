"""Port of MRST ``convertSummaryToGroupSols.m`` (mrst-2026a/hm/utils).

The group-level counterpart of ``convertSummaryToWellSols``: builds a
per-report-step group-solution history from an ECLIPSE summary file, using
the ``G*`` summary vectors.

Rate reconstruction follows the MATLAB's fallbacks:

* oil from GOPR;
* gas from GGPR, or -- failing that -- from the gas/oil ratio GGOR times
  the oil rate;
* water from GWPR, or from the water cut GWCT as ``wcut*qOs/(1-wcut)``;
* injection (GWIR, GGIR) is *added* to the corresponding production rate,
  so a group's entry is its net rate.

Production is reported negative (rates are negated on the way in),
injection positive, so ``sign`` reads as MRST's well-sign convention.
"""

import numpy as _np

from PRSTCore.deckformat.resultinput.read_eclipse_summary import (
    _get_units, read_eclipse_summary)

_TIME_FIELD = ':+:+:+:+'


def convertSummaryToGroupSols(fn, unit=None, groupname=None, time=None):
    """Return ``(groupSols, time)``.

    ``groupSols`` is one list of group dicts per report step.
    """
    smry = fn if isinstance(fn, dict) else read_eclipse_summary(fn)
    u = _extract_units(smry, unit)

    qOs, qWs, qGs, gns, t = _extract_quantities(smry, u)
    groupSols = _assign_groupsols(qOs, qWs, qGs, gns)

    if groupname:
        wanted = [groupname] if isinstance(groupname, str) else list(groupname)
        keep = [i for i, n in enumerate(gns) if n in wanted]
        groupSols = [[gs[i] for i in keep] for gs in groupSols]

    if time is not None and t is not None and len(t):
        wanted = _np.atleast_1d(_np.asarray(time, dtype=float)).ravel()
        keep = [i for i, tv in enumerate(t) if tv in wanted]
        t = _np.asarray([t[i] for i in keep], dtype=float)
        groupSols = [groupSols[i] for i in keep]

    return groupSols, t


def _extract_units(smry, unit):
    """Port of ``extract_units``: explicit name, else INTEHEAD, else metric."""
    if isinstance(unit, str):
        return _get_units(unit)
    intehead = smry.get('intehead_unit') if isinstance(smry, dict) else None
    if intehead is not None:
        return _get_units({1: 'metric', 2: 'field', 3: 'lab'}.get(int(intehead),
                                                                  'metric'))
    if unit:
        return unit
    import warnings
    warnings.warn('No unit given, assuming metric', RuntimeWarning)
    return _get_units('metric')


def _extract_quantities(smry, u):
    """Port of ``extract_quantities`` for the group vectors."""
    gns = _get_group_names(smry)
    t = smry['get'](_TIME_FIELD, 'TIME')
    time = (_np.asarray(t, dtype=float).ravel() * u['t']
            if t is not None else _np.zeros(0))

    ng = len(gns)
    nt = int(time.size) if time.size else int(smry['data'].shape[1])
    qOs = _np.zeros((nt, ng))
    qWs = _np.zeros((nt, ng))
    qGs = _np.zeros((nt, ng))

    for k, gn in enumerate(gns):
        akw = set(smry['get_keywords'](gn))

        if 'GOPR' in akw:
            qOs[:, k] = -_col(smry, gn, 'GOPR', nt) * u['ql']

        if 'GGPR' in akw:
            qGs[:, k] = -_col(smry, gn, 'GGPR', nt) * u['qg']
        elif 'GGOR' in akw:
            # Gas/oil ratio times the oil rate.
            qGs[:, k] = qOs[:, k] * _col(smry, gn, 'GGOR', nt)

        if 'GWPR' in akw:
            qWs[:, k] = -_col(smry, gn, 'GWPR', nt) * u['ql']
        elif 'GWCT' in akw:
            wcut = _col(smry, gn, 'GWCT', nt)
            with _np.errstate(divide='ignore', invalid='ignore'):
                qWs[:, k] = wcut * qOs[:, k] / (1.0 - wcut)
            qWs[~_np.isfinite(qWs[:, k]), k] = 0.0

        # Injection adds to the (negative) production rate.
        if 'GWIR' in akw:
            qWs[:, k] = qWs[:, k] + _col(smry, gn, 'GWIR', nt) * u['ql']
        if 'GGIR' in akw:
            qGs[:, k] = qGs[:, k] + _col(smry, gn, 'GGIR', nt) * u['qg']

    return qOs, qWs, qGs, gns, time


def _assign_groupsols(qOs, qWs, qGs, gns):
    """Port of ``assign_groupsols``."""
    nt = qOs.shape[0]
    out = []
    for kt in range(nt):
        step = []
        for kg, name in enumerate(gns):
            total = qWs[kt, kg] + qOs[kt, kg] + qGs[kt, kg]
            step.append({
                'name': name,
                'qOs': float(qOs[kt, kg]),
                'qWs': float(qWs[kt, kg]),
                'qGs': float(qGs[kt, kg]),
                'sign': float(_np.sign(total)),
                'status': bool(abs(total) > 0),
                'x': None, 'y': None, 'z': None,
            })
        out.append(step)
    return out


def _get_group_names(smry):
    """Port of ``get_group_names``: every name appearing under a G* vector."""
    names = set()
    for kw in smry['KEYWORDS']:
        if str(kw).startswith('G'):
            names.update(smry['get_names'](kw))
    names.discard(_TIME_FIELD)
    return sorted(names)


def _col(smry, name, keyword, nt):
    """One summary vector as a length-``nt`` float array."""
    values = smry['get'](name, keyword)
    if values is None:
        return _np.zeros(nt)
    values = _np.asarray(values, dtype=float).ravel()
    if values.size >= nt:
        return values[:nt]
    out = _np.zeros(nt)
    out[:values.size] = values
    return out
