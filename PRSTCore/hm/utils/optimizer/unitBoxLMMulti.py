"""Port of MRST ``unitBoxLMMulti.m`` (mrst-2026a/hm/utils/optimizer).

Levenberg-Marquardt least squares on the unit box, over several cases
sharing one parameter vector.

The residual function returns ``(res, J)`` with ``res`` an
``(nresidual, ncase)`` array and ``J`` one Jacobian per case; the
objective is ``sum(res.^2)`` over every case and residual.

Box handling is projection, not a barrier: after each step the parameters
are clipped to ``[0, 1]``, and a component pinned at a bound whose
gradient points further out is *frozen* for that iteration (``isFree``),
so the reduced system only solves for components that can actually move.

Two damping strategies, as in the MATLAB:

``'simple'``  the classic LM damping ``(J'J + lam*D) du = -g``, with
              ``lam`` scaled by the gain ratio;
``'TR'``      a trust region: ``lam`` is iterated so that ``|du|`` matches
              the radius, using the standard Hebden/More correction.
"""

import numpy as _np
import scipy.linalg as _la


def unitBoxLMMulti(f, u0, lambdaInit=1.0, lambdaMax=1e10, lambdaMin=1e-6,
                   lambdaIncrease=5.0, lambdaDecrease=2.0,
                   radiusIncrease=2.0, radiusDecrease=4.0,
                   ratioThresholds=(0.25, 0.75), scaledDamping=False,
                   updateStrategy='simple', gradTol=1e-6, updateTol=1e-6,
                   resTolAbs=1e-5, resTolRel=0.0, resChangeTolRel=-_np.inf,
                   maxIt=20, maxFunEvals=None, lsqTol=0.0, verbose=True,
                   plotEvolution=False):
    """Return ``(v, u, history)``: best objective, its parameters, and the
    per-iteration record."""
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
            # Gain ratio: actual reduction over the reduction the damped
            # model predicted.
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
            # Reject: retry from the same point with heavier damping.
            it -= 1
            stalled += 1
            if stalled > 20:
                break
        else:
            stalled = 0
            u, r, J = uNew, resNew, JNew

            g = 0.0
            Jr_full = 0.0
            for c in range(nc):
                g = g + _np.asarray(J[c]).T @ r[:, c]
                Jr_full = Jr_full + _np.asarray(J[c])
            g = g / nc
            Jr_full = Jr_full / nc

            pg = float(_np.linalg.norm(u - _cap(u - g)))
            h['pg'][it] = pg
            if pg < gradTol or it >= maxIt + 1:
                continue

            # Freeze components pinned at a bound whose gradient pushes
            # them further out; they cannot move, so exclude them.
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


def _cap(x):
    return _np.clip(x, 0.0, 1.0)


def _computeUpdate(JJ, D, g, lam, radius, opt):
    """Port of ``computeUpdate``: the damped normal equations."""
    def lsqSolve(A, b):
        if opt['lsqTol'] > _np.finfo(float).eps:
            return _la.lstsq(A, b, cond=opt['lsqTol'])[0]
        try:
            return _np.linalg.solve(A, b)
        except _np.linalg.LinAlgError:
            return _la.lstsq(A, b)[0]

    if opt['updateStrategy'] != 'TR':
        lam = max(opt['lambdaMin'], min(opt['lambdaMax'], lam))
        return -lsqSolve(JJ + lam * D, g), lam, radius

    if not _np.isfinite(radius):
        du = -lsqSolve(JJ + lam * D, g)
        return du, lam, float(_np.linalg.norm(du))

    # Hebden/More: iterate lam until |du| matches the trust radius.
    it = 0
    ndu = _np.inf
    lam0 = lam
    du = None
    while abs(ndu - radius) > 0.1 * radius and it < 20:
        it += 1
        du = -lsqSolve(JJ + lam * D, g)
        dut = -lsqSolve(JJ + lam * D, du)
        ndu = float(_np.linalg.norm(du))
        ndut = float(du @ dut) / ndu if ndu else 1.0
        lam = lam + (1 - ndu / radius) * ndu / ndut if ndut else lam
        lam = max(opt['lambdaMin'], min(opt['lambdaMax'], lam))
    if it == 20:
        lam = lam0 * opt['lambdaIncrease']
        du = -lsqSolve(JJ + lam * D, g)
        radius = float(_np.linalg.norm(du))
    return du, lam, radius


def _converged(it, h, maxIt, maxFunEvals, gradTol, updateTol, resTolAbs,
               resTolRel, resChangeTolRel, verbose):
    """Port of ``converged``: seven independent stopping criteria."""
    if it <= 0:
        return False
    dv = _np.inf
    if it > 1 and h['val'][it]:
        dv = abs((h['val'][it] - h['val'][it - 1]) / h['val'][it])
    flags = [
        it >= maxIt + 1,
        _np.nansum(h['nIt']) >= maxFunEvals,
        h['pg'][it] < gradTol,
        h['du'][it] < updateTol,
        h['val'][it] < resTolAbs,
        (h['val'][it] / h['val'][1]) < resTolRel if h['val'][1] else False,
        dv < resChangeTolRel,
    ]
    flags = [bool(x) if x == x else False for x in flags]
    if any(flags) and verbose:
        reasons = [
            'Reached maximal number of iterations (%d)' % it,
            'Reached maximal number of function evaluations (%d)'
            % int(_np.nansum(h['nIt'])),
            'Norm of projected gradient below tolerance (%7.2e < %7.2e)'
            % (h['pg'][it], gradTol),
            'Norm of update below tolerance (%7.2e < %7.2e)'
            % (h['du'][it], updateTol),
            'Absolute mismatch below tolerance (%7.2e < %7.2e)'
            % (h['val'][it], resTolAbs),
            'Relative mismatch below tolerance (%7.2e < %7.2e)'
            % (h['val'][it] / h['val'][1] if h['val'][1] else _np.nan,
               resTolRel),
            'Relative mismatch change is below tolerance (%7.2e < %7.2e)'
            % (dv, resChangeTolRel),
        ]
        line = '-' * 65
        print(line)
        print('| Optimization finished:')
        print('| %s' % reasons[flags.index(True)])
        print(line)
    return any(flags)


def _handleOutput(it, h):
    """Port of ``handleOutput``: return the best iterate, trimmed history."""
    import warnings
    if it == 1:
        warnings.warn('Optimization finished after first function evaluation',
                      RuntimeWarning)
    elif it > 1 and h['val'][it] > h['val'][it - 1]:
        # The final step made things worse; report the previous one.
        it -= 1
    v = h['val'][it]
    u = h['u'][it]
    trimmed = {k: (values[:it + 1] if not isinstance(values, list)
                   else values[:it + 1])
               for k, values in h.items()}
    return v, u, trimmed


def _initializeHistory(maxIt):
    n = maxIt + 2
    return {'val': _np.full(n, _np.nan), 'u': [None] * n,
            'lambda': _np.full(n, _np.nan), 'rho': _np.full(n, _np.nan),
            'nIt': _np.zeros(n), 'pg': _np.full(n, _np.nan),
            'du': _np.full(n, _np.nan)}


def _printInfo(h, it):
    print('It: %3d | val: %10.4e | lambda: %8.2e | pg: %8.2e | du: %8.2e'
          % (it, h['val'][it] if it < h['val'].size else _np.nan,
             h['lambda'][it] if it < h['lambda'].size else _np.nan,
             h['pg'][it] if it < h['pg'].size else _np.nan,
             h['du'][it] if it < h['du'].size else _np.nan))
