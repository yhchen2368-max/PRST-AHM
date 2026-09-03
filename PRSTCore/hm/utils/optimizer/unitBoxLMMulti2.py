"""Port of MRST ``unitBoxLMMulti2.m`` (mrst-2026a/hm/utils/optimizer).

Identical to :mod:`unitBoxLMMulti` except for how the multi-case gradient
is formed.

``unitBoxLMMulti`` accumulates each case's own contribution and averages::

    g = (1/nc) * sum_c J_c' * r(:,c)

``unitBoxLMMulti2`` averages the *residuals* across cases first, then
applies the averaged Jacobian once::

    r = mean(r, 2);  Jr = (1/nc) * sum_c J_c;  g = Jr' * r

The two agree when every case has the same Jacobian, and differ otherwise:
the first weights each case by its own sensitivity, the second by the
ensemble-mean sensitivity. The MATLAB keeps both as separate files, so
this port does too rather than adding a flag to one of them.
"""

import numpy as _np

from .unitBoxLMMulti import (_cap, _computeUpdate, _converged, _handleOutput,
                             _initializeHistory, _printInfo)


def unitBoxLMMulti2(f, u0, lambdaInit=1.0, lambdaMax=1e10, lambdaMin=1e-6,
                    lambdaIncrease=5.0, lambdaDecrease=2.0,
                    radiusIncrease=2.0, radiusDecrease=4.0,
                    ratioThresholds=(0.25, 0.75), scaledDamping=False,
                    updateStrategy='simple', gradTol=1e-6, updateTol=1e-6,
                    resTolAbs=1e-5, resTolRel=0.0, resChangeTolRel=-_np.inf,
                    maxIt=20, maxFunEvals=None, lsqTol=0.0, verbose=True,
                    plotEvolution=False):
    """Return ``(v, u, history)`` -- see :mod:`unitBoxLMMulti`."""
    if maxFunEvals is None:
        maxFunEvals = 2 * maxIt
    opt = dict(lambdaMax=lambdaMax, lambdaMin=lambdaMin, lsqTol=lsqTol,
               lambdaIncrease=lambdaIncrease, updateStrategy=updateStrategy)

    if updateStrategy != 'TR':
        goodFac, badFac = lambdaDecrease, lambdaIncrease
    else:
        goodFac, badFac = radiusIncrease, radiusDecrease

    u = _np.asarray(u0, dtype=float).ravel().copy()
    lam = float(lambdaInit)
    du = _np.zeros_like(u)
    h = _initializeHistory(maxIt)

    it = 0
    accept = True
    radius = _np.nan
    stalled = 0
    isFree = _np.ones(u.size, dtype=bool)
    JJr = Dr = gr = dur = None

    while not _converged(it, h, maxIt, maxFunEvals, gradTol, updateTol,
                         resTolAbs, resTolRel, resChangeTolRel, verbose):
        it += 1
        uNew = _cap(u + du)
        resNew, JNew = f(uNew)
        resNew = _np.atleast_2d(_np.asarray(resNew, dtype=float))
        if resNew.ndim == 1:
            resNew = resNew.reshape(-1, 1)
        nc = len(JNew)
        val = float(_np.sum(resNew ** 2))

        h['val'][it] = val
        h['lambda'][it] = lam
        h['u'][it] = uNew.copy()
        h['nIt'][it] += 1

        if it > 1:
            predicted = float(dur @ (lam * (Dr @ dur) - gr))
            rho = -(val - h['val'][it - 1]) / predicted if predicted else -1.0
            h['rho'][it] = rho
            accept = rho > 0
            if rho < ratioThresholds[0]:
                lam *= badFac
                radius = radius / badFac
            elif rho > ratioThresholds[1]:
                lam /= goodFac
                radius = radius * goodFac

        if it > 1 and not accept:
            it -= 1
            stalled += 1
            if stalled > 20:
                break
        else:
            stalled = 0
            u, r, J = uNew, resNew, JNew

            # The one difference from unitBoxLMMulti: average first.
            r_mean = _np.mean(r, axis=1)
            Jr_full = 0.0
            for c in range(nc):
                Jr_full = Jr_full + _np.asarray(J[c])
            Jr_full = Jr_full / nc
            g = Jr_full.T @ r_mean

            pg = float(_np.linalg.norm(u - _cap(u - g)))
            h['pg'][it] = pg
            if pg < gradTol or it >= maxIt + 1:
                continue

            isFree = ~((u == 0) & (g > 0)) & ~((u == 1) & (g < 0))
            Jr = Jr_full[:, isFree]
            gr = g[isFree]
            JJr = Jr.T @ Jr
            if scaledDamping:
                dr = _np.diag(JJr).copy()
                mval = 1e-3 * dr.max() if dr.size else 0.0
                dr[dr < mval] = mval
                Dr = _np.diag(dr)
            else:
                Dr = _np.eye(int(_np.count_nonzero(isFree)))

        dur, lam, radius = _computeUpdate(JJr, Dr, gr, lam, radius, opt)
        du = _np.zeros_like(u)
        du[isFree] = dur
        h['du'][it] = float(_np.linalg.norm(du))
        if verbose:
            _printInfo(h, it if accept else it + 1)

    return _handleOutput(it, h)
