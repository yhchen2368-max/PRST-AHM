"""Port of MRST ``optimizeLinIneqConstrained.m``
(mrst-2026a/hm/utils/optimizer).

L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995) extended with general linear
inequality constraints ``A*u <= b``.

The bound-constrained part is the published algorithm: build the
generalised Cauchy point by walking the piecewise-linear path of steepest
descent through the box's breakpoints (:func:`getCauchyPoint`, Algorithm
CP), then minimise the quadratic model over the variables still free
(:func:`subspaceMin`, Direct Primal Method).

The extension is what follows. The Cauchy/subspace step ``d0`` only knows
about the box, so it is projected onto the nullspace of whichever general
constraints are active (:func:`projQ`), and then the segment from ``x``
to ``x + d`` is walked forward: each time it runs into a new constraint,
that constraint joins the active set, ``Q`` grows by one direction
(:func:`expandQ`), and the remainder is re-projected. The result is a
direction that stays feasible for the whole step rather than only near
``x``.

This module holds the algorithm; :mod:`optimizeBoundConstrainedForFAHM`
is MRST's FAHM-specific variant of the same routine and imports these
helpers rather than repeating them.
"""

import numpy as _np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla

from PRSTCore.optimization.optim.limited_memory_hessian import \
    LimitedMemoryHessian
from PRSTCore.optimization.optim.line_search import line_search

_EPS = _np.finfo(float).eps
_SQEPS = _np.sqrt(_EPS)

DEFAULTS = dict(
    maximize=False, lb=0.0, ub=1.0, stepInit=_np.nan, maxInitialUpdate=0.05,
    gradTol=1e-3, objChangeTol=5e-4, objChangeTolRel=-_np.inf, maxIt=25,
    lineSearchMaxIt=5, wolfe1=1e-4, wolfe2=0.9, safeguardFac=1e-5,
    stepIncreaseTol=10, lbfgsNum=10, lbfgsStrategy='dynamic',
    lbfgsRequireWolfe=False, useTrustRegion=False, trustRegionInit=_np.nan,
    radiusIncrease=2.0, radiusDecrease=0.25, ratioThresholds=(0.25, 0.75),
    linIneq=None, linEq=None, outputHessian=False, history=None,
    verbose=True,
)


def optimizeLinIneqConstrained(u0, f, **kwargs):
    """Return ``(v, u, history)``.

    ``f(u)`` must return ``(value, gradient)``. ``u0`` must be feasible.
    """
    opt = dict(DEFAULTS)
    opt.update(kwargs)
    return _run(u0, f, opt)


# --------------------------------------------------------- driver loop --

def _run(u0, f, opt, per_iteration=None, extra=None, on_iteration=None):
    """The shared main loop.

    ``f(u)`` returns ``(value, gradient)`` in the caller's own sign
    convention; the loop negates it when maximising.

    ``per_iteration(it)``, when given, returns the objective to use for
    iteration ``it`` instead of ``f`` -- this is what lets the FAHM
    variant hand each evaluation its own case directory. ``extra`` is
    stashed in every history entry, and ``on_iteration(history)`` is
    called once per iteration for checkpointing.
    """
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
    iwhere = _np.zeros(n, dtype=int)

    history = opt['history']
    it = 0 if history is None else len(history['val'])
    maxIt = opt['maxIt'] + it
    it0 = it

    f_origin = f

    def objective(iteration):
        base = f_origin if per_iteration is None else per_iteration(iteration)
        return lambda u: _fNegative(u, base, opt)

    f = objective(it)
    c = getConstraints(u0, opt)

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
                             opt['outputHessian'], extra)
    if opt['verbose']:
        printInfo(history, it)

    u, v, g, rho, pg = u0, v0, g0, _np.nan, g0
    success = False
    while not success:
        it += 1
        f = objective(it)

        if not opt['useTrustRegion']:
            lbcur, ubcur = lb, ub
        else:
            lbcur, ubcur = incorporateTrustRegion(u0, rTrust, lb, ub)

        d, Hi, pg, maxStep, dObjEst, iwhere = getSearchDirection(
            u0, g0, Hi, HiPrev, lbcur, ubcur, iwhere, c, opt)

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
                                 rTrust, opt['outputHessian'], extra)
        else:
            if it == 1:
                u, v, rho = u0, v0, _np.nan
            history = gatherInfo(history, objSign * v, u,
                                 _np.linalg.norm(pg, _np.inf), 0, 0, 0, Hi,
                                 rho, rTrust, opt['outputHessian'], extra)

        if on_iteration is not None:
            on_iteration(history)

        success = (it >= maxIt
                   or _np.linalg.norm(pg, _np.inf) < opt['gradTol']
                   or abs(v - v0) < opt['objChangeTol']
                   or (v != 0 and abs((v - v0) / v) < opt['objChangeTolRel']))
        u0, v0, g0 = u, v, g

        if opt['verbose']:
            printInfo(history, it)

    if objScale:
        u = u * (ub_opt - lb_opt) + lb_opt
    return objSign * v, u, history


# --------------------------------------------------------- direction ----

def getSearchDirection(x, g, Hi, HiPrev, l, u, iwhere, c, opt):
    """Minimise the quadratic model subject to the box and the active
    linear constraints.

    Retries with the previous Hessian and then with the identity if the
    direction it produces is not usefully decreasing -- the same three
    trials the MATLAB makes.
    """
    n = x.size
    d = maxStep = dObj = None

    for nTrial in (1, 2, 3):
        if nTrial == 2:
            Hi = HiPrev
        elif nTrial == 3:
            Hi = Hi.reset()

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

        xc, cp, iwhere = getCauchyPoint(x, g, l, u, th, W, M, iwhere)
        xbar = subspaceMin(xc, cp, x, g, l, u, th, W, M)
        d0 = xbar - x

        Q = c['e']['Q']
        d = projQ(d0, Q)

        # Grow the active set until it stops growing, re-projecting each
        # time a new constraint joins it.
        isActive = _np.zeros(c['i']['A'].shape[0], dtype=bool)
        na, na_prev = 0, -1
        while na > na_prev:
            _, active_cur = classifyConstraints(c['i']['A'], c['i']['b'], x, d)
            isActive = isActive | active_cur
            na_prev = na
            na = int(_np.count_nonzero(isActive))
            if na > na_prev:
                X = _sp.hstack([_sp.csr_matrix(c['i']['A'])[isActive, :].T,
                                _sp.csr_matrix(c['e']['A']).T]).tocsc()
                if opt['linIneq'] is not None or opt['linEq'] is not None:
                    Q = _leadingLeftSingularVectors(X)
                else:
                    Q = _np.abs(X.toarray())
                d = projQ(d0, Q)

        pg = -projQ(g, Q)
        if _np.linalg.norm(pg, _np.inf) <= _SQEPS * _np.linalg.norm(g, _np.inf):
            return None, Hi, pg, None, None, iwhere

        # Walk the segment, adding each constraint it reaches.
        dr, gr = d, g
        becomesActive = isActive.copy()
        d = _np.zeros(n)
        done = False
        while not done:
            if _np.linalg.norm(dr) > _SQEPS:
                sgn, _ = classifyConstraints(c['i']['A'], c['i']['b'], x + d, dr)
                ix, s = findNextConstraint(c['i']['A'], c['i']['b'], x + d, dr,
                                           (sgn <= 0) | becomesActive)
            else:
                ix, s = None, 0.0
            if ix is not None and s <= 1 + _SQEPS:
                becomesActive[ix] = True
                d = d + s * dr
                gr = (1 - s) * gr
                Q = expandQ(Q, _np.asarray(
                    _sp.csr_matrix(c['i']['A'])[ix, :].todense()).ravel())
                dr = -projQ(gr, Q, Hi)
            else:
                d = d + dr
                done = True

        # B = th*I - W*M*W', Equation (3.2).
        dObj = float(g @ d + 0.5 * d @ (th * d - (W @ M) @ (W.T @ d)))

        sgn, _ = classifyConstraints(c['i']['A'], c['i']['b'], x, d)
        _, maxStep = findNextConstraint(c['i']['A'], c['i']['b'], x, d,
                                        sgn <= 0)
        if maxStep < 1 - _SQEPS and opt['verbose']:
            print('Problematic search direction, maximum step: %f < 1'
                  % maxStep)
        if maxStep < 1:
            d, maxStep = maxStep * d, 1.0

        isDecreasing = float(d @ g) <= 0
        isZero = (_np.linalg.norm(d, _np.inf)
                  <= _SQEPS * _np.linalg.norm(Hi.dot(g), _np.inf))
        if isDecreasing and not isZero:
            break

        if opt['verbose']:
            what = 'Small norm of search direction' if not isZero \
                else 'Non-inceasing search direction'
            if nTrial == 1:
                print('%s, trying previous Hessian approximation.' % what)
            elif nTrial == 2:
                print('%s, trying to reset Hessian to identity.' % what)
            else:
                print('Exiting: %s.' % what)
        if nTrial == 3:
            d, maxStep = None, None

    return d, Hi, pg, maxStep, dObj, iwhere


def getCauchyPoint(x, g, l, u, theta, W, M, iwhere):
    """Port of ``getCauchyPoint`` -- Algorithm CP, pages 8-9.

    Walks the projected steepest-descent path through the box's
    breakpoints, fixing one variable at each breakpoint passed, and stops
    in the interval that contains the model's minimiser.

    ``iwhere`` records each variable's bound status: 0 or -3 free with
    bounds, 1 fixed at the lower bound, 2 fixed at the upper, 3 always
    fixed, -1 always free.
    """
    n = x.size
    bnded = True
    nfree = n            # MATLAB's n+1, as a 0-based exclusive index
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    iorder = _np.zeros(n, dtype=int)

    d = -g.copy()
    t = _np.zeros(n)
    for i in range(n):
        if iwhere[i] != 3 and iwhere[i] != -1:
            tl = l[i] - x[i]
            tu = u[i] - x[i]
            iwhere[i] = 0
            if tl >= 0:
                if d[i] <= 0:
                    iwhere[i] = 1
            elif tu <= 0:
                if d[i] >= 0:
                    iwhere[i] = 2
            else:
                if abs(d[i]) <= 0:
                    iwhere[i] = -3
        if iwhere[i] != 0 and iwhere[i] != -1:
            d[i] = 0.0
        else:
            if d[i] < 0:
                iorder[nbreak] = i
                t[nbreak] = (l[i] - x[i]) / d[i]
                if nbreak == 0 or t[nbreak] < bkmin:
                    bkmin, ibkmin = t[nbreak], nbreak
                nbreak += 1
            elif d[i] > 0:
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
            if iteration == 2:
                if ibkmin != nbreak - 1:
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

    cvec = cvec + dtm * p
    return xc, cvec, iwhere


def hpsolb(n, t, iorder, iheap):
    """Port of ``hpsolb``: pop the least breakpoint off a heap.

    ``iheap == 0`` first heapifies ``t[:n]``; then the smallest element is
    moved to position ``n-1`` and the rest restored to a heap.
    """
    t = _np.asarray(t, dtype=float)
    iorder = _np.asarray(iorder, dtype=int)

    if iheap == 0:
        for k in range(1, n):
            ddum = t[k]
            indxin = iorder[k]
            i = k
            while i > 0:
                j = (i + 1) // 2 - 1
                if ddum < t[j]:
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else:
                    break
            t[i] = ddum
            iorder[i] = indxin

    if n > 1:
        i = 0
        out = t[0]
        indxou = iorder[0]
        ddum = t[n - 1]
        indxin = iorder[n - 1]
        while True:
            j = 2 * i + 1
            if j <= n - 2:
                if t[j + 1] < t[j]:
                    j += 1
                if t[j] < ddum:
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else:
                    break
            else:
                break
        t[i] = ddum
        iorder[i] = indxin
        t[n - 1] = out
        iorder[n - 1] = indxou

    return t, iorder


def subspaceMin(xc, c, x, g, l, u, theta, W, M):
    """Port of ``subspaceMin`` -- Direct Primal Method, page 12.

    Minimises the model over the variables the Cauchy point left free,
    then projects back into the box. If projecting turns the direction
    into an ascent direction, backtracks to the first bound instead.
    """
    xc = _np.asarray(xc, dtype=float).copy()
    freeVars = _np.flatnonzero((xc != l) & (xc != u))
    if freeVars.size == 0:
        return xc

    n = xc.size
    Z = _np.eye(n)[:, freeVars]
    WtZ = W.T @ Z
    N = _np.eye(W.shape[1]) - (1.0 / theta) * M @ (WtZ @ WtZ.T)

    rc = Z.T @ (g + theta * (xc - x) - W @ (M @ c))
    v = WtZ @ rc
    v = _np.linalg.solve(N, M @ v)
    # Z'*W is (W'*Z)' == WtZ.T, so this is nfree-long, matching rc.
    d = -(1.0 / theta) * rc - (1.0 / theta ** 2) * (WtZ.T @ v)

    nsub = freeVars.size
    iword = 0
    xp = xc.copy()
    for i in range(nsub):
        k = freeVars[i]
        xk = max(l[k], xc[k] + d[i])
        xc[k] = min(u[k], xk)
        if xc[k] == l[k] or xc[k] == u[k]:
            iword = 1
    if iword == 0:
        return xc

    if float((xc - x) @ g) > 0:
        xc = xp
        print(' Positive dir derivative in projection. '
              ' Using the backtracking step ')
    else:
        return xc

    alpha = 1.0
    ibd = 0
    for i in range(nsub):
        k = freeVars[i]
        dk = d[i]
        temp1 = alpha
        if dk < 0:
            temp2 = l[k] - xc[k]
            if temp2 >= 0:
                temp1 = 0.0
            elif dk * alpha < temp2:
                temp1 = temp2 / dk
        if dk > 0:
            temp2 = u[k] - xc[k]
            if temp2 <= 0:
                temp1 = 0.0
            elif dk * alpha > temp2:
                temp1 = temp2 / dk
        if temp1 < alpha:
            alpha = temp1
            ibd = i

    if alpha < 1:
        dk = d[ibd]
        k = freeVars[ibd]
        if dk > 0:
            xc[k] = u[k]
            d[ibd] = 0.0
        if dk < 0:
            xc[k] = l[k]
            d[ibd] = 0.0

    for i in range(nsub):
        xc[freeVars[i]] += alpha * d[i]
    return xc


# -------------------------------------------------------- constraints ----

def projQ(v, Q, H=None):
    """Port of ``projQ``: project ``H*v`` onto the nullspace of ``Q``.

    Applied twice for numerical stability, as the MATLAB does. A
    :class:`LimitedMemoryHessian` instead absorbs ``Q`` as its nullspace.
    """
    v = _np.asarray(v, dtype=float).ravel()
    if Q is None or _np.size(Q) == 0:
        return v if H is None else _apply(H, v)
    Q = _np.asarray(Q.todense() if _sp.issparse(Q) else Q, dtype=float)
    if isinstance(H, LimitedMemoryHessian):
        return H.set_nullspace(Q).dot(v)
    tmp = _apply(H, v - Q @ (Q.T @ v))
    return tmp - Q @ (Q.T @ tmp)


def getConstraints(u, opt):
    """Port of ``getConstraints``: the box plus any supplied linear
    constraints, each scaled by its own norm."""
    nu = _np.size(u)
    A = _sp.vstack([-_sp.eye(nu), _sp.eye(nu)], format='csr')
    b = _np.concatenate([_np.zeros(nu), _np.ones(nu)])

    if opt.get('linIneq') is not None:
        Ai = _sp.csr_matrix(opt['linIneq']['A'])
        sc = _matrixNorm(Ai)
        A = _sp.vstack([A, Ai / sc], format='csr')
        b = _np.concatenate([b, _np.asarray(opt['linIneq']['b'],
                                            dtype=float).ravel() / sc])

    if opt.get('linEq') is not None:
        Ae = _sp.csr_matrix(opt['linEq']['A'])
        sc = _matrixNorm(Ae)
        Ae = Ae / sc
        be = _np.asarray(opt['linEq']['b'], dtype=float).ravel() / sc
        Q = _leadingLeftSingularVectors(Ae.T.tocsc())
        e = {'A': Ae, 'b': be, 'Q': Q}
    else:
        e = {'A': _sp.csr_matrix((0, nu)), 'b': _np.zeros(0),
             'Q': _np.zeros((nu, 0))}

    return {'i': {'A': A, 'b': b}, 'e': e}


def classifyConstraints(A, b, u, v):
    """Port of ``classifyConstraints``.

    ``sgn`` is -1 heading into the feasible side, 0 parallel, +1 heading
    out; ``act`` marks constraints already at their bound and heading out.
    """
    A = _sp.csr_matrix(A)
    sgn = A @ _np.asarray(v, dtype=float).ravel()
    sgn[_np.abs(sgn) < _SQEPS] = 0.0
    sgn = _np.sign(sgn)
    act = (A @ _np.asarray(u, dtype=float).ravel() - b > -_SQEPS) & (sgn > 0)
    return sgn, act


def findNextConstraint(A, b, u, d, ac):
    """Port of ``findNextConstraint``: the nearest constraint the ray
    ``u + s*d`` reaches, skipping those already active."""
    A = _sp.csr_matrix(A)
    with _np.errstate(divide='ignore', invalid='ignore'):
        s = (b - A @ _np.asarray(u, dtype=float).ravel()) \
            / (A @ _np.asarray(d, dtype=float).ravel())
    s[_np.asarray(ac, dtype=bool)] = _np.inf
    s[~_np.isfinite(s)] = _np.inf
    s[s < _EPS] = _np.inf
    ix = int(_np.argmin(s))
    smin = float(s[ix])
    return (None, smin) if not _np.isfinite(smin) else (ix, smin)


def expandQ(Q, v):
    """Port of ``expandQ``: append a newly active constraint's normal,
    orthogonalised against those already there."""
    Q = _np.asarray(Q.todense() if _sp.issparse(Q) else Q, dtype=float)
    v = _np.asarray(v, dtype=float).ravel()
    n0 = _np.linalg.norm(v)
    if Q.size:
        v = v - Q @ (Q.T @ v)
    if n0 > 0 and _np.linalg.norm(v) / n0 > _SQEPS:
        col = (v / _np.linalg.norm(v)).reshape(-1, 1)
        return col if Q.size == 0 else _np.hstack([Q, col])
    print('Newly active constraint is linear combination of other active '
          'constraints ??!!')
    return Q


# ------------------------------------------------------------ support ----

def incorporateTrustRegion(u, rTrust, lb, ub):
    """Port of ``incorporateTrustRegion``: the inf-norm trust region as
    tightened bounds."""
    return _np.maximum(lb, u - rTrust), _np.minimum(ub, u + rTrust)


def updateTrustRegion(r, rho, update, step, opt):
    """Port of ``updateTrustRegion``: grow or shrink on the model fit."""
    lo, hi = opt['ratioThresholds']
    if rho < lo:
        return opt['radiusDecrease'] * step * r
    if rho > hi and r < update * step * (1 + _SQEPS):
        return opt['radiusIncrease'] * step * r
    return r


def gatherInfo(hst, val, u, pg, alpha, lsit, lsfl, hess, rho, r, outputH,
               params=None):
    """Port of ``gatherInfo``: append one iteration to the history."""
    if not outputH:
        hess = None
    entry = dict(val=val, u=_np.asarray(u, dtype=float).copy(), pg=pg,
                 alpha=alpha, lsit=lsit, lsfl=lsfl, hess=hess, rho=rho, r=r,
                 params=params)
    if hst is None:
        return {k: [v] for k, v in entry.items()}
    for k, v in entry.items():
        hst[k].append(v)
    return hst


def printInfo(history, it):
    """Port of ``printInfo``."""
    print('It: %2d | val: %4.3e | ls-its: %3d | pgrad: %4.3e | rho: %4.3e'
          % (it, history['val'][-1], _int(history['lsit'][-1]),
             history['pg'][-1], history['rho'][-1]))


def _fNegative(u, f, opt):
    """Port of ``fNegative``: negate when maximising."""
    v, g = f(u)
    return (-v, -_np.asarray(g, dtype=float)) if opt['maximize'] else (v, g)


def _fScale(u, f, lb, ub):
    """Port of ``fScale``: evaluate in physical units, return a scaled
    gradient."""
    v, g = f(u * (ub - lb) + lb)
    return v, _np.asarray(g, dtype=float) * (ub - lb)


def _matrixNorm(A):
    """MATLAB's ``norm(A)`` -- the largest singular value."""
    A = A.toarray() if _sp.issparse(A) else _np.asarray(A, dtype=float)
    return float(_np.linalg.norm(A, 2))


def _leadingLeftSingularVectors(X):
    """The left singular vectors of ``X`` whose singular values are not
    numerically zero -- MATLAB's ``svds`` followed by the ``s > sqrt(eps)*
    s(1)`` filter."""
    dense = X.toarray() if _sp.issparse(X) else _np.asarray(X, dtype=float)
    if dense.size == 0:
        return _np.zeros((dense.shape[0], 0))
    Q, s, _ = _np.linalg.svd(dense, full_matrices=False)
    if s.size == 0:
        return _np.zeros((dense.shape[0], 0))
    return Q[:, s > _SQEPS * s[0]]


def _apply(H, v):
    if H is None:
        return v
    if isinstance(H, LimitedMemoryHessian):
        return H.dot(v)
    return H * v if _np.isscalar(H) else H @ v


def _int(x):
    return int(x) if _np.isfinite(x) else 0
