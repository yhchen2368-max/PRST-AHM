"""Port of MRST ``perturbWellPosition.m`` (mrst-2026a/hm/utils).

Finite-difference sensitivity of the residual equations with respect to a
well's trajectory control points -- the gradient a well-placement
optimisation needs, where the trajectory enters through the *set of
perforated cells* and so cannot be differentiated analytically.

Each control point is displaced by ``+pert`` (and, for the default
two-sided quotient, ``-pert``); the well is re-perforated, the residuals
re-evaluated, and the difference divided by ``norm(pert)`` times one or
two.

Only cells perforated by either the plus or the minus trajectory can have
changed, so the cell equations are differenced on that union alone.
"""

import copy as _copy

import numpy as _np
import scipy.sparse as _sp


def perturbWellPosition(model, state, W, getResiduals, posControl,
                        approxType='twoSided'):
    """Return ``dFdU``: one sparse column block per equation."""
    if approxType not in ('twoSided', 'oneSided'):
        raise ValueError("approxType must be 'twoSided' or 'oneSided'")

    r = getResiduals(state, W)
    nc = int(_np.asarray(state['pressure']).size)
    params = posControl['parameters'] if isinstance(posControl, dict) \
        else posControl.parameters
    pntIx = _np.atleast_1d(_np.asarray(_get(params, 'pointIx'), dtype=int)).ravel()
    v = _np.atleast_2d(_np.asarray(_get(params, 'perturbation'), dtype=float))
    nu = pntIx.size
    neq = len(r['equations'])
    dFdU = [[None] * nu for _ in range(neq)]

    names = [w['name'] for w in W]
    target = _get(posControl, 'w')['name'] if isinstance(_get(posControl, 'w'), dict) \
        else _get(posControl, 'w').name
    matches = [i for i, n in enumerate(names) if n == target]
    assert len(matches) == 1, 'posControl must name exactly one well'
    wno = matches[0]

    w0 = _copy.deepcopy(W[wno])
    ws0 = _copy.deepcopy(state['wellSol'][wno])
    points0 = _np.array(_get(posControl, 'controlPoints'), dtype=float, copy=True)

    r_m = r
    wcIx_m = _np.zeros(nc, dtype=bool)
    if approxType == 'oneSided':
        wcIx_m = _cell_mask(W[wno], nc)

    for uno in range(nu):
        pert = v[uno, :]

        wcIx_p, r_p = _evaluate(model, state, W, wno, w0, ws0, posControl,
                                points0, pntIx[uno], pert, getResiduals, nc)
        if approxType == 'twoSided':
            wcIx_m, r_m = _evaluate(model, state, W, wno, w0, ws0, posControl,
                                    points0, pntIx[uno], -pert, getResiduals, nc)

        delta = (1.0 if approxType == 'oneSided' else 2.0) * _np.linalg.norm(pert)
        for eqno, eq_type in enumerate(r['types']):
            if eq_type == 'cell':
                # Only the perforated cells can have moved.
                cix = _np.flatnonzero(wcIx_p | wcIx_m)
                tmp = ((_np.asarray(r_p['equations'][eqno]).ravel()[cix]
                        - _np.asarray(r_m['equations'][eqno]).ravel()[cix]) / delta)
                dFdU[eqno][uno] = _sp.csc_matrix(
                    (tmp, (cix, _np.zeros(cix.size, dtype=int))), shape=(nc, 1))
            else:
                r_p_tmp = _checkStatus(r_p['equations'][eqno], W, wno)
                r_m_tmp = _checkStatus(r_m['equations'][eqno], W, wno)
                dFdU[eqno][uno] = ((r_p_tmp - r_m_tmp) / delta).reshape(-1, 1)

    _set(posControl, 'controlPoints', points0)
    return [_hstack(cols) for cols in dFdU]


def _evaluate(model, state, W, wno, w0, ws0, posControl, points0, point,
              pert, getResiduals, nc):
    """Displace one control point, re-perforate, and re-evaluate."""
    # MRST resolves this from visualization/diagnostics/utils/trajectory,
    # not from hm; PRSTCore already mirrors it at the same path.
    from PRSTCore.visualization.diagnostics.utils.trajectory import (
        updateWellTrajectory as _traj)
    updateWellTrajectory = _traj.updateWellTrajectory

    points = _np.array(points0, dtype=float, copy=True)
    points[point, :] = points0[point, :] + pert
    _set(posControl, 'controlPoints', points)

    trajectory = _get(posControl, 'getTrajectory')()
    w_p, ws_p = updateWellTrajectory(model, w0, ws0, trajectory)

    if not w_p.get('status', True):
        import warnings
        warnings.warn('Well-placement outside grid, this is currently not handled',
                      RuntimeWarning)
        mask = _np.zeros(nc, dtype=bool)
    else:
        mask = _cell_mask(w_p, nc)

    W[wno].update(w_p)
    state['wellSol'][wno].update(ws_p)
    return mask, getResiduals(state, W)


def _cell_mask(well, nc):
    mask = _np.zeros(nc, dtype=bool)
    cells = _np.atleast_1d(_np.asarray(well.get('cells', []), dtype=int)).ravel()
    if cells.size:
        mask[cells] = True
    return mask


def _checkStatus(r, W, wno):
    """Port of ``checkStatus``: pad a shut well's missing entry with zero."""
    r = _np.atleast_1d(_np.asarray(r, dtype=float)).ravel()
    if r.size == len(W):
        return r
    assert r.size + 1 == len(W), "Check what's going on here!"
    out = _np.zeros(len(W), dtype=float)
    keep = _np.ones(len(W), dtype=bool)
    keep[wno] = False
    out[keep] = r
    return out


def _hstack(columns):
    columns = [c for c in columns if c is not None]
    if not columns:
        return None
    if any(_sp.issparse(c) for c in columns):
        return _sp.hstack([_sp.csc_matrix(c) for c in columns], format='csc')
    return _np.hstack([_np.asarray(c).reshape(-1, 1) for c in columns])


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)
