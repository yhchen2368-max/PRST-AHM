"""Port of MRST ``checkParameterConsistency.m``
(mrst-2026a/hm/utils/optimizer).

Saturation endpoints are not independent: connate water must sit below
critical water, the mobile ranges must not overlap, and the residual
saturations must leave room for each other. Tuning them freely produces
unphysical relative-permeability curves.

This resolves those requirements two ways:

* where only *one* of a pair is being tuned, the other's tabulated value
  is known, so the constraint collapses to a **box limit** on the tuned
  parameter and is enforced by tightening it directly;
* where *both* are tuned, neither bound is known in advance, so the
  requirement becomes a **linear inequality** ``A*p <= b`` for the
  optimiser to honour.

The pair requirements (``P1 <= P2`` style, in scaled form)::

    Swl  <= Swcr        Sgl  <= Sgcr
    Swl  + Sgu  <= 1    Sgl  + Swu  <= 1
    Sowcr + Swcr <= 1

and the triple::

    Sogcr + Sgcr + Swl <= 1
"""

import warnings as _warnings

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.hm.utils.getRelpermScalingPoints import (as_dict,
                                                       getRelpermScalingPoints)

_EPS = _np.finfo(float).eps

# Pairwise requirements: (P1, P2), their signs, and the right-hand side.
_PAIRS = (
    (('Swl', 'Swcr'), (1, -1), 0.0),
    (('Sgl', 'Sgcr'), (1, -1), 0.0),
    (('Swl', 'Sgu'), (1, 1), 1.0),
    (('Sgl', 'Swu'), (1, 1), 1.0),
    (('Sowcr', 'Swcr'), (1, 1), 1.0 - _EPS),
)

# The three-way requirement.
_TRIPLE = (('Sogcr', 'Sgcr', 'Swl'), (1, 1, 1), 1.0 - _EPS)


def checkParameterConsistency(params, model):
    """Return ``(params, constraints)``.

    ``params`` comes back with any box limits tightened; ``constraints``
    is ``{'A', 'b'}`` or ``None``.
    """
    nParams = [int(_get(p, 'nParam')) for p in params]
    pos = _np.concatenate([[0], _np.cumsum(nParams)]).astype(int)
    names = [str(_get(p, 'name')) for p in params]
    scaling = as_dict(getRelpermScalingPoints(model))

    A_rows, b_rows, record = [], [], []
    n = int(sum(nParams))

    for keys, sgn, rhs in _PAIRS:
        ind = [_find(names, k) for k in keys]
        if all(i is None for i in ind):
            continue
        limits, subset = _getBoxLimits(params, ind)

        # One tuned, one tabulated -> tighten the tuned one's box.
        for k in (0, 1):
            other = 1 - k
            if ind[k] is None:
                continue
            sub, ia = _setdiff_stable(subset[k], subset[other])
            if sub.size == 0:
                continue
            cons = _getConstraints(scaling, keys[other], sub)
            if cons is None:
                continue
            cons = rhs - sgn[other] * cons
            flag = 'u' if sgn[k] > 0 else 'l'
            limits[k], done = _enforceBoxLimits(limits[k], cons, ia, flag)
            if done:
                _set(params[ind[k]], 'boxLims', limits[k])
                record.append(names[ind[k]])

        # Both tuned -> a linear inequality.
        if ind[0] is not None and ind[1] is not None:
            sub, ia, ib = _intersect_stable(subset[0], subset[1])
            if sub.size:
                Ai = _makeInequalityMatrix(pos, limits, ind, [ia, ib],
                                           sub.size, n)
                A_rows.append(sgn[0] * Ai[0] + sgn[1] * Ai[1])
                bi = (sgn[0] * limits[0][ia, 0] + sgn[1] * limits[1][ib, 0])
                b_rows.append(rhs - bi)

    keys, sgn, rhs = _TRIPLE
    ind = [_find(names, k) for k in keys]
    if not all(i is None for i in ind):
        limits, subset = _getBoxLimits(params, ind)
        for k in range(3):
            others = [j for j in range(3) if j != k]
            if ind[k] is None:
                continue
            union = _np.union1d(subset[others[0]], subset[others[1]])
            sub, ia = _setdiff_stable(subset[k], union)
            if sub.size == 0:
                continue
            cons = [_getConstraints(scaling, keys[j], sub) for j in others]
            # The MATLAB substitutes zero for a missing second constraint
            # on the first two rows only.
            if k != 2 and cons[1] is None:
                cons[1] = 0.0
            if cons[0] is None or cons[1] is None:
                continue
            value = rhs - (sgn[others[0]] * cons[0] + sgn[others[1]] * cons[1])
            flag = 'u' if sgn[k] > 0 else 'l'
            limits[k], done = _enforceBoxLimits(limits[k], value, ia, flag)
            if done:
                _set(params[ind[k]], 'boxLims', limits[k])
                record.append(names[ind[k]])

        # Two tuned, one tabulated.
        for f1, f2, f3 in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            if ind[f1] is None or ind[f2] is None:
                continue
            sub, ia, ib = _intersect_stable(subset[f1], subset[f2])
            if sub.size == 0:
                continue
            sub2, s = _setdiff_stable(sub, subset[f3])
            if sub2.size == 0:
                continue
            cons = _getConstraints(scaling, keys[f3], sub2)
            if cons is None:
                cons = 0.0
            Ai = _makeInequalityMatrix(pos, limits, [ind[f1], ind[f2]],
                                       [ia[s], ib[s]], sub2.size, n)
            A_rows.append(sgn[f1] * Ai[0] + sgn[f2] * Ai[1])
            bi = (sgn[f1] * limits[f1][ia[s], 0]
                  + sgn[f2] * limits[f2][ib[s], 0] + sgn[f3] * cons)
            b_rows.append(rhs - bi)

        # All three tuned.
        if all(i is not None for i in ind):
            sub, ia, ib = _intersect_stable(subset[0], subset[1])
            if sub.size:
                sub2, s, ic = _intersect_stable(sub, subset[2])
                if sub2.size:
                    ia, ib = ia[s], ib[s]
                    Ai = _makeInequalityMatrix(pos, limits, ind, [ia, ib, ic],
                                               sub2.size, n)
                    A_rows.append(sgn[0] * Ai[0] + sgn[1] * Ai[1]
                                  + sgn[2] * Ai[2])
                    bi = (sgn[0] * limits[0][ia, 0] + sgn[1] * limits[1][ib, 0]
                          + sgn[2] * limits[2][ic, 0])
                    b_rows.append(rhs - bi)

    if record:
        _warnings.warn(
            'The box-limits of the following parameters are adjusted '
            'according to consistency requirements: %s' % ', '.join(record),
            RuntimeWarning)

    if not A_rows:
        return params, None
    return params, {'A': _sp.vstack(A_rows, format='csr'),
                    'b': _np.concatenate([_np.atleast_1d(b) for b in b_rows])}


def _makeInequalityMatrix(pos, limits, ind, sub, m, n):
    """Port of ``makeInequalityMatrix``.

    Each row scales the parameter's unit-box coordinate by its box width,
    since the constraint is stated in physical units but the optimiser
    works in ``[0, 1]``.
    """
    out = []
    for i, index in enumerate(ind):
        if index is None:
            out.append(_sp.csr_matrix((m, n)))
            continue
        J = _np.arange(pos[index], pos[index + 1])
        V = _np.diff(limits[i], axis=1).ravel()
        if sub[i] is not None and _np.size(sub[i]):
            J = J[_np.asarray(sub[i], dtype=int)]
            V = V[_np.asarray(sub[i], dtype=int)]
        rows = _np.arange(m)
        out.append(_sp.csr_matrix((V[:m], (rows, J[:m])), shape=(m, n)))
    return out


def _getBoxLimits(params, ind):
    """Port of ``getBoxLimits``: per-entry limits and the tuned subset."""
    limits, subset = [], []
    for index in ind:
        if index is None:
            limits.append(_np.zeros((0, 2)))
            subset.append(_np.zeros(0, dtype=int))
            continue
        p = params[index]
        lim = _np.atleast_2d(_np.asarray(_get(p, 'boxLims'), dtype=float))
        if lim.shape[0] == 1:
            lim = _np.tile(lim, (int(_get(p, 'nParam')), 1))
        limits.append(lim)
        sub = _get(p, 'subset')
        if sub is None or isinstance(sub, str):
            sub = _np.arange(int(_get(p, 'nParam')))
        subset.append(_np.atleast_1d(_np.asarray(sub, dtype=int)).ravel())
    return limits, subset


def _getConstraints(scaling, name, subset):
    """The tabulated value of ``name`` for the given entries, or None."""
    for key, values in scaling.items():
        if key.lower() == str(name).lower():
            values = _np.asarray(values, dtype=float).ravel()
            idx = _np.asarray(subset, dtype=int)
            idx = idx[idx < values.size]
            return values[idx]
    return None


def _enforceBoxLimits(limits, value, ia, flag):
    """Port of ``enforceBoxLimits``: tighten one side of the box."""
    limits = _np.array(limits, dtype=float, copy=True)
    ix = _np.zeros(limits.shape[0], dtype=bool)
    ix[_np.asarray(ia, dtype=int)] = True
    value = _np.broadcast_to(_np.atleast_1d(_np.asarray(value, dtype=float)),
                             (limits.shape[0],)) \
        if _np.size(value) == 1 else _np.asarray(value, dtype=float)

    full = _np.zeros(limits.shape[0])
    full[_np.asarray(ia, dtype=int)] = value[:_np.size(ia)] \
        if _np.size(value) == _np.size(ia) else value[ix]

    if str(flag).lower() == 'u':
        ix = (limits[:, 1] > full) & ix
        if _np.any(ix):
            limits[ix, 1] = full[ix]
            return limits, True
    else:
        ix = (limits[:, 0] < full) & ix
        if _np.any(ix):
            limits[ix, 0] = full[ix]
            return limits, True
    return limits, False


def _find(names, target):
    lowered = str(target).lower()
    for i, n in enumerate(names):
        if str(n).lower() == lowered:
            return i
    return None


def _setdiff_stable(a, b):
    """``setdiff(a, b, 'stable')`` with the positions kept."""
    a = _np.atleast_1d(_np.asarray(a)).ravel()
    b = set(_np.atleast_1d(_np.asarray(b)).ravel().tolist())
    keep = [i for i, v in enumerate(a) if v not in b]
    return a[keep], _np.asarray(keep, dtype=int)


def _intersect_stable(a, b):
    """``intersect(a, b, 'stable')`` with both index vectors."""
    a = _np.atleast_1d(_np.asarray(a)).ravel()
    b = _np.atleast_1d(_np.asarray(b)).ravel()
    lookup = {v: i for i, v in enumerate(b.tolist())}
    values, ia, ib = [], [], []
    for i, v in enumerate(a.tolist()):
        if v in lookup and v not in values:
            values.append(v)
            ia.append(i)
            ib.append(lookup[v])
    return (_np.asarray(values), _np.asarray(ia, dtype=int),
            _np.asarray(ib, dtype=int))


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)
