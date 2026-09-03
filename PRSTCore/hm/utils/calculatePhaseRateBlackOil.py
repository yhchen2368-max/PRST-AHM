"""Port of MRST ``calculatePhaseRateBlackOil.m`` (mrst-2026a/hm/utils).

Per-perforation phase rates for a black-oil well, replacing the mobility
of *injecting* perforations by a total mobility split along the injection
composition -- so an injector delivers its declared mixture rather than
the reservoir's mobility ratio.

The composition is the well's ``compi``, except where cross-flow occurs
(an injecting perforation on a producer, or vice versa): there it is
re-derived from the fluid actually entering the wellbore, with the
dissolved-gas / vaporised-oil cross terms folded in.
"""

import numpy as _np


def calculatePhaseRateBlackOil(Tdp, mobw, bw, q_s, rs, rv, map_, allowCrossFlow,
                               model):
    """Return the per-phase perforation rates ``q_ph``.

    ``Tdp`` is the transmissibility-weighted drawdown (positive injecting),
    ``mobw`` a list of per-phase perforation mobilities, ``bw`` an
    ``(nperf, nph)`` shrinkage-factor array, ``q_s`` an ``(nwell, nph)``
    surface-rate array, and ``map_`` carries ``perf2well``, ``isInjector``
    and ``W``.
    """
    perf2well = _np.asarray(map_['perf2well'], dtype=int).ravel()
    is_injector_well = _np.asarray(map_['isInjector'], dtype=bool).ravel()
    isInjector = is_injector_well[perf2well]
    W = map_['W']

    vTdp = _value(Tdp)
    injection = vTdp > 0
    production = (~injection) & (vTdp != 0)
    crossflow = ((injection & ~isInjector) | (production & isInjector))
    crossflow = crossflow & bool(allowCrossFlow)

    w, o, g = _phase_index(model)
    nph = len(mobw)
    bw = _np.atleast_2d(_value(bw))
    rs = _np.zeros(perf2well.size) if rs is None else _np.asarray(_value(rs)).ravel()
    rv = _np.zeros(perf2well.size) if rv is None else _np.asarray(_value(rv)).ravel()
    q_s = _np.atleast_2d(_np.asarray(q_s, dtype=float))

    mobw = list(mobw)
    if _np.any(injection):
        alpha = _np.atleast_2d(_np.asarray(
            [_np.asarray(well['compi'], dtype=float).ravel() for well in W]))

        if _np.any(crossflow):
            # q_wb: the mobility-weighted flux each perforation carries.
            q_wb = _np.column_stack([_value(m) for m in mobw]) * vTdp[:, None]
            for i in range(len(W)):
                ix = perf2well == i
                if not _np.any(crossflow[ix]):
                    continue
                flux = q_wb[ix, :] * bw[ix, :]
                flux_in = -_np.minimum(flux, 0.0)
                fw, fo, fg = flux_in[:, w], flux_in[:, o], flux_in[:, g]
                if _np.any(rv != 0):
                    fo = fo + rv[ix] * flux_in[:, g]
                if _np.any(rs != 0):
                    fg = fg + rs[ix] * flux_in[:, o]
                sum_in = _np.array([fw.sum(), fo.sum(), fg.sum()])
                if is_injector_well[i]:
                    sum_in = sum_in + q_s[i, :]
                if not _np.all(sum_in == 0):
                    alpha[i, :] = sum_in / sum_in.sum()

        alpha = alpha[perf2well, :]

        mobt = _np.zeros(int(_np.count_nonzero(injection)))
        for i in range(nph):
            mobt = mobt + _value(mobw[i])[injection]

        Fw = alpha[:, w] / bw[:, w]
        Fo = alpha[:, o] / bw[:, o]
        Fg = alpha[:, g] / bw[:, g]
        if not _np.all(rv == 0):
            ix = Fo != 0
            Fo[ix] = Fo[ix] - rv[ix] * alpha[ix, g] / bw[ix, o]
        if not _np.all(rs == 0):
            ix = Fg != 0
            Fg[ix] = Fg[ix] - rs[ix] * alpha[ix, o] / bw[ix, g]

        F = _np.column_stack([Fw, Fo, Fg])
        denom = F[injection, :].sum(axis=1)
        for i in range(nph):
            values = _np.array(_value(mobw[i]), dtype=float, copy=True)
            values[injection] = mobt * (F[injection, i] / denom)
            mobw[i] = values

    return [mobw[i] * Tdp for i in range(nph)]


def _phase_index(model):
    """``model.ReservoirModel.getPhaseIndex('W', 'O', 'G')``."""
    reservoir = getattr(model, 'ReservoirModel', model)
    getter = getattr(reservoir, 'getPhaseIndex', None)
    if getter is not None:
        return tuple(getter('W', 'O', 'G'))
    return 0, 1, 2


def _value(x):
    return x.val if hasattr(x, 'val') else _np.asarray(x, dtype=float)
