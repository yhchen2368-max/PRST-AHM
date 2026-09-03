"""Port of MRST ``L_BFGS_B.m`` (mrst-2026a/hm/utils/optimizer).

The bound-constrained L-BFGS-B of Byrd, Lu, Nocedal & Zhu (1995), without
the general linear inequalities that :mod:`optimizeLinIneqConstrained`
adds. The two share the Cauchy-point/subspace-minimisation core; what is
different here is that a variable may be bounded on one side only, or not
at all, tracked per variable in ``nbd``::

    nbd[i] = 0  unbounded
             1  lower bound only
             2  both bounds
             3  upper bound only

That distinction runs through three places -- which variables the
projected gradient clips (:func:`projectedGrad`), which get breakpoints
in the Cauchy walk, and how far a step may go (:func:`findMaxStep`) --
so this is a sibling of the linear-inequality version rather than a
special case of it, and MRST keeps it as its own file.

**Unreachable MATLAB defect.** ``getCauchyPoint`` opens with

    if norm(pg, Inf) <= 0
        if iprint >= 0
            fprintf('Subgnorm = 0.  GCP = X.')

where ``iprint`` is never defined, so MATLAB would raise there. The
branch cannot be reached in practice: ``getSearchDirection`` returns
early whenever ``norm(pg, inf) <= gradTol`` and ``gradTol`` is positive.
This port keeps the early return and drops the print, since printing was
the only thing the broken line was for.
"""

import numpy as _np

from PRSTCore.optimization.optim.limited_memory_hessian import \
    LimitedMemoryHessian
from PRSTCore.optimization.optim.line_search import line_search

from .optimizeLinIneqConstrained import (gatherInfo, hpsolb,
                                         incorporateTrustRegion, printInfo,
                                         updateTrustRegion, _fNegative,
                                         _fScale)

_EPS = _np.finfo(float).eps
_SQEPS = _np.sqrt(_EPS)

DEFAULTS = dict(
    maximize=False, lb=0.0, ub=1.0, stepInit=_np.nan, maxInitialUpdate=0.05,
    gradTol=1e-3, objChangeTol=5e-4, objChangeTolRel=-_np.inf, maxIt=25,
    lineSearchMaxIt=5, wolfe1=1e-4, wolfe2=0.9, safeguardFac=1e-5,
    stepIncreaseTol=10, lbfgsNum=10, lbfgsStrategy='dynamic',
    lbfgsRequireWolfe=False, useTrustRegion=False, trustRegionInit=_np.nan,
    radiusIncrease=2.0, radiusDecrease=0.25, ratioThresholds=(0.25, 0.75),
    outputHessian=False, history=None, saveHistory=None, verbose=True,
)


def L_BFGS_B(u0, f, **kwargs):
    """Return ``(v, u, history)``. ``f(u)`` returns ``(value, gradient)``."""
    opt = dict(DEFAULTS)
    opt.update(kwargs)
    objSign = -1 if opt['maximize'] else 1

    lb_opt = _np.atleast_1d(_np.asarray(opt['lb'], dtype=float))
    ub_opt = _np.atleast_1d(_np.asarray(opt['ub'], dtype=float))
    objScale = bool(_np.any(lb_opt != 0) or _np.any(ub_opt != 1))
    f_user = f
    if objScale:
        f = lambda u: _fScale(u, f_user, lb_opt, ub_opt)
        u0 = (_np.asarray(u0, dtype=float) - lb_opt) / (ub_opt - lb_opt)

    u0 = _np.asarray(u0, dtype=float).ravel().copy()
    n = u0.size
    lb, ub = _np.zeros(n), _np.ones(n)
    opt['boundsType'] = boundsType(u0, lb, ub)
    iwhere = _np.zeros(n, dtype=int)

    history = opt['history']
    it = 0 if history is None else len(history['val'])
    maxIt = opt['maxIt'] + it

    f_origin = f
    f = lambda u: _fNegative(u, f_origin, opt)

    if _np.any(lb > u0) or _np.any(u0 > ub):
        import warnings
        warnings.warn('Initial guess was not within bounds, projecting to '
                      'feasible domain.', RuntimeWarning)
        u0 = _np.maximum(lb, _np.minimum(ub, u0))

    v0, g0 = f(u0)
    g0 = _np.asarray(g0, dtype=float).ravel()

    step = opt['stepInit']
    if not _np.isfinite(step) or step <= 0:
        gmax = float(_np.max(_np.abs(g0)))
        step = opt['maxInitialUpdate'] / gmax if gmax > 0 else 1.0
    rTrust = opt['trustRegionInit']
    if opt['useTrustRegion'] and not _np.isfinite(rTrust):
        rTrust = opt['maxInitialUpdate']

    Hi = LimitedMemoryHessian(init_scale=step, m=opt['lbfgsNum'],
                              init_strategy=opt['lbfgsStrategy'])
    HiPrev = Hi

    if history is None:
        history = gatherInfo(None, objSign * v0, u0, _np.linalg.norm(g0),
                             _np.nan, _np.nan, _np.nan, Hi, _np.nan, rTrust,
                             opt['outputHessian'])
    if opt['verbose']:
        printInfo(history, it)

    u, v, g, rho, pg = u0, v0, g0, _np.nan, g0
    success = False
    while not success:
        it += 1
        if not opt['useTrustRegion']:
            lbcur, ubcur = lb, ub
        else:
            lbcur, ubcur = incorporateTrustRegion(u0, rTrust, lb, ub)

        d, Hi, pg, maxStep, dObjEst, iwhere = getSearchDirection(
            u0, g0, Hi, HiPrev, lbcur, ubcur, iwhere, opt)

        if not (_np.linalg.norm(pg, _np.inf) < opt['gradTol']) and d is not None:
            lsopt = dict(opt)
            lsopt['maxStep'] = maxStep
            u, v, g, lsinfo = line_search(u0, v0, g0, d, f, lsopt)
            g = _np.asarray(g, dtype=float).ravel()
            dObjTrue = float(lsinfo['objVals'][0]) - v0
            rho = dObjTrue / dObjEst if dObjEst else _np.nan
            if opt['useTrustRegion']:
                rTrust = updateTrustRegion(rTrust, rho,
                                           _np.linalg.norm(d, _np.inf),
                                           lsinfo['step'], opt)
            du, dg = u - u0, g - g0
            doUpdate = (du @ dg > _SQEPS * _np.linalg.norm(du)
                        * _np.linalg.norm(dg))
            if opt['lbfgsRequireWolfe']:
                doUpdate = doUpdate and lsinfo['flag'] > 0
            if doUpdate:
                dg = _np.where(_np.isfinite(dg), dg, 0.0)
                HiPrev = Hi
                Hi = Hi.update(du, dg)
            elif opt['verbose']:
                print('Hessian not updated during iteration %d.' % it)
            history = gatherInfo(history, objSign * v, u,
                                 _np.linalg.norm(pg, _np.inf), lsinfo['step'],
                                 lsinfo['nits'], lsinfo['flag'], Hi, rho,
                                 rTrust, opt['outputHessian'])
        else:
            if it == 1:
                u, v, rho = u0, v0, _np.nan
            history = gatherInfo(history, objSign * v, u,
                                 _np.linalg.norm(pg, _np.inf), 0, 0, 0, Hi,
                                 rho, rTrust, opt['outputHessian'])

        success = (it >= maxIt
                   or _np.linalg.norm(pg, _np.inf) < opt['gradTol']
                   or abs(v - v0) < opt['objChangeTol']
                   or (v != 0 and abs((v - v0) / v) < opt['objChangeTolRel']))
        u0, v0, g0 = u, v, g

        if opt['verbose']:
            printInfo(history, it)
        if opt['saveHistory']:
            _np.savez(opt['saveHistory'],
                      val=_np.asarray(history['val'], dtype=float),
                      u=_np.asarray(history['u'], dtype=float))

    if objScale:
        u = u * (ub_opt - lb_opt) + lb_opt
    return objSign * v, u, history


def boundsType(x, l, u):
    """Port of ``boundsType``: which sides each variable is bounded on.

    The MATLAB decides this once for the whole vector from whether ``l``
    and ``u`` were supplied at all, so every entry gets the same value.
    """
    n = _np.size(x)
    has_l = l is not None and _np.size(l) > 0
    has_u = u is not None and _np.size(u) > 0
    if has_l and has_u:
        return _np.full(n, 2, dtype=int)
    if has_l:
        return _np.full(n, 1, dtype=int)
    if has_u:
        return _np.full(n, 3, dtype=int)
    return _np.zeros(n, dtype=int)


def projectedGrad(x, l, u, nbd, g):
    """Port of ``projectedGrad``.

    **Defect reproduced.** The two branches are transposed relative to
    the reference algorithm, which makes this a no-op inside the box.
    L-BFGS-B's own ``projgr`` clips against the bound the step is heading
    *towards*::

        if g <  0 and has upper bound:  g = max(x - u, g)
        if g >= 0 and has lower bound:  g = min(x - l, g)

    MRST pairs each sign with the other bound instead -- ``min(x-l, g)``
    when ``g < 0`` and ``max(x-u, g)`` when ``g > 0``. For any feasible
    ``x`` we have ``x - l >= 0 > g`` in the first case, so ``min`` always
    returns ``g``; and ``x - u <= 0 < g`` in the second, so ``max`` does
    too. The gradient therefore comes back unchanged and a variable
    pinned at a bound is never recognised as converged there.

    The effect is a stopping criterion that is harder to satisfy than
    intended, not a wrong answer: ``getCauchyPoint`` still fixes bounded
    variables correctly, so the iterates stay right. Reproduced rather
    than corrected, since fixing it would change where MRST stops.
    """
    pg = _np.asarray(g, dtype=float).copy()
    nbd = _np.asarray(nbd, dtype=int)
    ix1 = (nbd <= 2) & (nbd != 0) & (g < 0)
    ix2 = (nbd >= 2) & (nbd != 0) & (g > 0)
    pg[ix1] = _np.minimum(x[ix1] - l[ix1], g[ix1])
    pg[ix2] = _np.maximum(x[ix2] - u[ix2], g[ix2])
    return pg


def findMaxStep(x, d, l, u):
    """Port of ``findMaxStep``: how far along ``d`` the box allows."""
    d = _np.asarray(d, dtype=float)
    with _np.errstate(divide='ignore', invalid='ignore'):
        sl = (l - x) / d
        su = (u - x) / d
    s = _np.maximum(sl, su)
    s[d == 0] = _np.inf
    s[~_np.isfinite(s)] = _np.inf
    return float(_np.min(s)) if s.size else _np.inf


def getSearchDirection(x, g, Hi, HiPrev, l, u, iwhere, opt):
    """Port of ``getSearchDirection``: the Cauchy point, then the
    subspace minimiser, retrying with an older Hessian if the direction
    does not decrease."""
    n = x.size
    nbd = opt['boundsType']
    d = maxStep = dObj = None
    pg = g

    for nTrial in (1, 2, 3):
        if nTrial == 2:
            Hi = HiPrev
        elif nTrial == 3:
            Hi = Hi.reset()

        pg = projectedGrad(x, l, u, nbd, g)
        if _np.linalg.norm(pg, _np.inf) <= opt['gradTol']:
            return None, Hi, pg, 0.0, 0.0, iwhere

        S, Y = Hi.active_pairs()
        if Y is None or S is None or _np.size(Y) == 0 or _np.size(S) == 0:
            th = 1.0
            W = _np.zeros((n, 1))
            M = _np.zeros((1, 1))
        else:
            y, s = Y[:, -1], S[:, -1]
            th = float(y @ y) / float(y @ s)
            W = _np.hstack([Y, th * S])
            A = S.T @ Y
            L = _np.tril(A, -1)
            D = _np.diag(_np.diag(A))
            M = _np.linalg.inv(_np.block([[-D, L.T], [L, th * (S.T @ S)]]))

        xc, c, iwhere = getCauchyPoint(x, g, pg, l, u, nbd, th, W, M, iwhere)
        xbar = subspaceMin(xc, c, x, g, l, u, nbd, th, W, M)
        d = xbar - x
        # B = th*I - W*M*W', Equation (3.2).
        dObj = float(g @ d + 0.5 * d @ (th * d - (W @ M) @ (W.T @ d)))

        maxStep = findMaxStep(x, d, l, u)
        if maxStep < 1 - _SQEPS and opt.get('verbose', True):
            print('Problematic search direction, maximum step: %f < 1'
                  % maxStep)

        if float(d @ g) <= 0:
            break

        if opt.get('verbose', True):
            what = 'Non-inceasing search direction'
            if nTrial == 1:
                print('%s, trying previous Hessian approximation.' % what)
            elif nTrial == 2:
                print('%s, trying to reset Hessian to identity.' % what)
            else:
                print('Exiting: %s.' % what)
        if nTrial == 3:
            d, maxStep = None, None

    return d, Hi, pg, maxStep, dObj, iwhere


def getCauchyPoint(x, g, pg, l, u, nbd, theta, W, M, iwhere):
    """Port of ``getCauchyPoint`` -- Algorithm CP, pages 8-9, with the
    per-variable bound types.

    See the module docstring for the unreachable ``iprint`` defect this
    replaces with a plain early return.
    """
    if _np.linalg.norm(pg, _np.inf) <= 0:
        return x.copy(), _np.zeros(W.shape[1]), iwhere

    n = x.size
    bnded = True
    nfree = n
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    iorder = _np.zeros(n, dtype=int)

    d = -g.copy()
    t = _np.zeros(n)
    for i in range(n):
        if iwhere[i] != 3 and iwhere[i] != -1:
            tl = l[i] - x[i] if nbd[i] <= 2 else 0.0
            tu = u[i] - x[i] if nbd[i] >= 2 else 0.0
            xlower = nbd[i] <= 2 and tl >= 0
            xupper = nbd[i] >= 2 and tu <= 0
            iwhere[i] = 0
            if xlower:
                if d[i] <= 0:
                    iwhere[i] = 1
            elif xupper:
                if d[i] >= 0:
                    iwhere[i] = 2
            else:
                if abs(d[i]) <= 0:
                    iwhere[i] = -3
        if iwhere[i] != 0 and iwhere[i] != -1:
            d[i] = 0.0
        else:
            if nbd[i] <= 2 and nbd[i] != 0 and d[i] < 0:
                iorder[nbreak] = i
                t[nbreak] = (l[i] - x[i]) / d[i]
                if nbreak == 0 or t[nbreak] < bkmin:
                    bkmin, ibkmin = t[nbreak], nbreak
                nbreak += 1
            elif nbd[i] >= 2 and d[i] > 0:
                iorder[nbreak] = i
                t[nbreak] = (u[i] - x[i]) / d[i]
                if nbreak == 0 or t[nbreak] < bkmin:
                    bkmin, ibkmin = t[nbreak], nbreak
                nbreak += 1
            else:
                nfree -= 1
                iorder[nfree] = i
                if abs(d[i]) > 0:
                    bnded = False

    xc = x.copy()
    cvec = _np.zeros(W.shape[1])
    if nbreak == 0 and nfree == n:
        return xc, cvec, iwhere

    p = W.T @ d
    f1 = -float(d @ d)
    f2 = -theta * f1 - float(p @ (M @ p))
    f2_org = f2
    dtm = -f1 / f2 if f2 else 0.0
    tsum = 0.0

    nleft = nbreak
    if nbreak > 0:
        tj = 0.0
        ibp = -1
        for iteration in range(1, n + 1):
            tj0 = tj
            if iteration == 1:
                tj = bkmin
                ibp = iorder[ibkmin]
            if iteration == 2 and ibkmin != nbreak - 1:
                t[ibkmin] = t[nbreak - 1]
                iorder[ibkmin] = iorder[nbreak - 1]
            if iteration > 1:
                t, iorder = hpsolb(nleft, t, iorder, iteration - 3)
                tj = t[nleft - 1]
                ibp = iorder[nleft - 1]

            dt = tj - tj0
            if dtm < dt:
                break

            tsum += dt
            nleft -= 1
            dibp = d[ibp]
            d[ibp] = 0.0
            if dibp > 0:
                zibp = u[ibp] - x[ibp]
                xc[ibp] = u[ibp]
                iwhere[ibp] = 2
            else:
                zibp = l[ibp] - x[ibp]
                xc[ibp] = l[ibp]
                iwhere[ibp] = 1
            if nleft == 0 and nbreak == n:
                dtm = dt
                break

            cvec = cvec + dt * p
            wbp = W[ibp, :]
            p = p - dibp * wbp
            f1 = (f1 + dt * f2 + dibp ** 2 - theta * dibp * zibp
                  + dibp * float(wbp @ (M @ cvec)))
            f2 = (f2 - theta * dibp ** 2 + 2.0 * dibp * float(wbp @ (M @ p))
                  - dibp ** 2 * float(wbp @ (M @ wbp)))
            f2 = max(_EPS * f2_org, f2)

            if nleft > 0:
                dtm = -f1 / f2 if f2 else 0.0
            elif bnded:
                dtm = 0.0
                break
            else:
                dtm = -f1 / f2 if f2 else 0.0
                break

    if not (nleft == 0 and nbreak == n):
        dtm = max(dtm, 0.0)
        tsum += dtm
        xc = xc + tsum * d

    return xc, cvec + dtm * p, iwhere


def subspaceMin(xc, c, x, g, l, u, nbd, theta, W, M):
    """Port of ``subspaceMin``.

    Identical to the linear-inequality version's, which takes no ``nbd``;
    MRST threads the argument through without reading it. Delegating
    keeps the two from drifting apart.
    """
    from .optimizeLinIneqConstrained import subspaceMin as _subspaceMin
    return _subspaceMin(xc, c, x, g, l, u, theta, W, M)
