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

import copy as _copy
import itertools as _itertools
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
    """Owned parameters and A*u<=b, in the exact FAHM pair/triple order.

    FAHM-FIX-027..030: signed fixed endpoints, subset indexing, triple
    column selection and constraints expressed in the FINAL tightened box.
    """
    params = _copy.deepcopy(params)
    nParams = [int(_get(p, 'nParam')) for p in params]
    pos = _np.r_[0, _np.cumsum(nParams)].astype(int)
    names = [str(_get(p, 'name')) for p in params]
    scaling = as_dict(getRelpermScalingPoints(model))
    templates, record = [], []
    for keys, signs, rhs in (*_PAIRS, _TRIPLE):
        ind = [_find(names, key) for key in keys]
        limits, subsets = _getBoxLimits(params, ind)
        count = len(keys)
        for k in range(count):
            if ind[k] is None:
                continue
            others = [j for j in range(count) if j != k]
            union = _np.unique(_np.concatenate([subsets[j] for j in others]))
            sub, ia = _setdiff_stable(subsets[k], union)
            if not sub.size:
                continue
            known = [_getConstraints(scaling, keys[j], sub) for j in others]
            if count == 3 and k != 2 and known[1] is None:
                known[1] = 0.0
            if any(v is None for v in known):
                continue
            value = (rhs - sum(signs[j] * v for j, v in zip(others, known))) / signs[k]
            limits[k], done = _enforceBoxLimits(limits[k], value, ia, 'u' if signs[k] > 0 else 'l')
            if done:
                _set(params[ind[k]], 'boxLims', limits[k])
                record.append(names[ind[k]])
        # Pair rows, then triple rows; stable cell order within each group.
        for size in range(2, count + 1):
            for chosen in _itertools.combinations(range(count), size):
                if any(ind[k] is None for k in chosen):
                    continue
                cells = subsets[chosen[0]]
                for k in chosen[1:]:
                    cells, _, _ = _intersect_stable(cells, subsets[k])
                others = [j for j in range(count) if j not in chosen]
                for j in others:
                    cells, _ = _setdiff_stable(cells, subsets[j])
                if not cells.size:
                    continue
                selected = [[int(_np.flatnonzero(subsets[k] == c)[0]) for c in cells] for k in chosen]
                fixed = _np.zeros(cells.size)
                for j in others:
                    value = _getConstraints(scaling, keys[j], cells)
                    if value is not None:
                        fixed += signs[j] * value
                templates.append(([ind[k] for k in chosen], selected,
                                  [signs[k] for k in chosen], rhs, fixed))
    if record:
        _warnings.warn('Box-limits adjusted according to consistency requirements: ' +
                       ', '.join(record), RuntimeWarning)
    A_rows, b_rows = [], []
    for ind, selected, signs, rhs, fixed in templates:
        limits, _ = _getBoxLimits(params, ind)
        m = len(selected[0])
        matrices = _makeInequalityMatrix(pos, limits, ind, selected, m, int(pos[-1]))
        A_rows.append(sum(s * a for s, a in zip(signs, matrices)))
        lower = sum(s * lim[sel, 0] for s, lim, sel in zip(signs, limits, selected))
        b_rows.append(rhs - (lower + fixed))
    if not A_rows:
        return params, None
    return params, {'A': _sp.vstack(A_rows, format='csr'), 'b': _np.concatenate(b_rows)}


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
        if V.size != m or J.size != m:
            raise ValueError('Constraint row/column sizes differ; no pad/trim is allowed')
        out.append(_sp.csr_matrix((V, (rows, J)), shape=(m, n)))
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
            if _np.any(idx < 0) or _np.any(idx >= values.size):
                raise ValueError('Constraint subset is outside active-cell values')
            return values[idx]
    return None


def _enforceBoxLimits(limits, value, ia, flag):
    limits = _np.array(limits, dtype=float, copy=True)
    ia = _np.asarray(ia, dtype=int)
    value = _np.broadcast_to(_np.asarray(value), (ia.size,))
    col = 1 if flag == 'u' else 0
    changed = limits[ia, col] > value if flag == 'u' else limits[ia, col] < value
    limits[ia[changed], col] = value[changed]
    return limits, bool(_np.any(changed))


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
