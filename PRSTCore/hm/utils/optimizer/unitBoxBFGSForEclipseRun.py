"""Port of MRST ``unitBoxBFGSForEclipseRun.m``
(mrst-2026a/hm/utils/optimizer).

Projected quasi-Newton on the unit box with general linear constraints,
driving an external simulator: each iteration gets a copy of the base
case directory to run in.

Where :mod:`optimizeLinIneqConstrained` builds its step from the
L-BFGS-B Cauchy point, this one takes the plain quasi-Newton direction
``-H*g`` and projects it onto the nullspace of the active constraints --
first for the gradient, then again for the direction, since projecting
can activate constraints the gradient did not touch. The constraint
bookkeeping (:func:`projQ`, :func:`classifyConstraints`,
:func:`findNextConstraint`, :func:`expandQ`, :func:`getConstraints`) is
shared with that module rather than duplicated.

It also has something the L-BFGS-B variants lack: :func:`checkFeasible`,
which repairs a point that violates its constraints instead of failing.
MRST's own comment notes this should be a QP solve and is iterative
projection only because no QP solver is at hand -- so it is meant for
mild violations, and says so when it gives up.

Two other differences from its siblings: the Hessian may be dense rather
than limited-memory (``limitedMemory=False``), and the BFGS update can be
switched off entirely (``useBFGS=False``), leaving steepest descent.
"""

import os as _os
import shutil as _shutil
import warnings as _warnings

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.optimization.optim.limited_memory_hessian import \
    LimitedMemoryHessian
from PRSTCore.optimization.optim.line_search import line_search

from .optimizeLinIneqConstrained import (_leadingLeftSingularVectors,
                                         classifyConstraints, expandQ,
                                         findNextConstraint, gatherInfo,
                                         getConstraints, printInfo, projQ)

_EPS = _np.finfo(float).eps
_SQEPS = _np.sqrt(_EPS)

DEFAULTS = dict(
    maximize=True, stepInit=_np.nan, maxInitialUpdate=0.05, gradTol=1e-3,
    objChangeTol=5e-4, objChangeTolRel=-_np.inf, maxIt=25, lineSearchMaxIt=5,
    wolfe1=1e-4, wolfe2=0.9, safeguardFac=1e-5, stepIncreaseTol=10,
    useBFGS=True, limitedMemory=True, lbfgsNum=5, lbfgsStrategy='dynamic',
    linEq=None, enforceFeasible=True, linIneq=None, outputHessian=False,
    history=None, params=None, verbose=True,
)


def unitBoxBFGSForEclipseRun(u0, f, dir, **kwargs):
    """Return ``(v, u, history)``.

    ``f(u, case_dir)`` returns ``(value, gradient)``. ``dir`` maps
    ``'base'`` to the case to copy and ``'work'`` to where the copies go.
    """
    opt = dict(DEFAULTS)
    opt.update(kwargs)
    objSign = -1 if opt['maximize'] else 1

    base = dir['base'] if isinstance(dir, dict) else getattr(dir, 'base')
    work = dir['work'] if isinstance(dir, dict) else getattr(dir, 'work')

    u0 = _np.asarray(u0, dtype=float).ravel().copy()
    c = getConstraints(u0, opt)
    f_origin = f

    history = opt['history']
    it = 0 if history is None else len(history['val'])
    maxIt = opt['maxIt'] + it

    u0, consOK, _ = checkFeasible(u0, c, opt['enforceFeasible'],
                                  'Initial guess')
    assert consOK, 'Infeasible initial guess'

    def objective(iteration):
        case_dir = _copyBase(base, work, iteration)

        def wrapped(u):
            v, g = f_origin(u, case_dir)
            return (-v, -_np.asarray(g, dtype=float)) if opt['maximize'] \
                else (v, g)
        return wrapped

    fk = objective(it)
    v0, g0 = fk(u0)
    g0 = _np.asarray(g0, dtype=float).ravel()

    step = opt['stepInit']
    if not _np.isfinite(step) or step <= 0:
        gmax = float(_np.max(_np.abs(g0)))
        step = opt['maxInitialUpdate'] / gmax if gmax > 0 else 1.0

    if not opt['limitedMemory']:
        Hi = step * _np.eye(u0.size)
    else:
        Hi = LimitedMemoryHessian(init_scale=step, m=opt['lbfgsNum'],
                                  init_strategy=opt['lbfgsStrategy'])
    HiPrev = Hi

    history = gatherInfo(history, objSign * v0, u0, _np.linalg.norm(g0),
                         _np.nan, _np.nan, _np.nan, Hi, _np.nan, _np.nan,
                         opt['outputHessian'], opt['params'])
    if opt['verbose']:
        printInfo(history, it)

    u, v, g = u0, v0, g0
    success = False
    while not success:
        it += 1
        fk = objective(it)

        d, Hi, pg, maxStep = getSearchDirection(u0, g0, Hi, HiPrev, c, opt)

        if not (_np.linalg.norm(pg, _np.inf) < opt['gradTol']) and d is not None:
            fixedU, flag, fixed = checkFeasible(u0 + d, c,
                                                opt['enforceFeasible'])
            if not flag and fixed:
                d = fixedU - u0

            lsopt = dict(opt)
            lsopt['maxStep'] = maxStep
            u, v, g, lsinfo = line_search(u0, v0, g0, d, fk, lsopt)
            g = _np.asarray(g, dtype=float).ravel()

            if opt['useBFGS']:
                du, dg = u - u0, g - g0
                dg = _np.where(_np.isfinite(dg), dg, 0.0)
                if du @ dg > _SQEPS * _np.linalg.norm(du) * _np.linalg.norm(dg):
                    HiPrev = Hi
                    if isinstance(Hi, LimitedMemoryHessian):
                        Hi = Hi.update(du, dg)
                    else:
                        # Dense inverse-BFGS: V'HV + r du du'.
                        r = 1.0 / float(du @ dg)
                        V = _np.eye(u.size) - r * _np.outer(dg, du)
                        Hi = V.T @ Hi @ V + r * _np.outer(du, du)
                elif opt['verbose']:
                    print('Hessian not updated during iteration %d.' % it)

            history = gatherInfo(history, objSign * v, u,
                                 _np.linalg.norm(pg, _np.inf), lsinfo['step'],
                                 lsinfo['nits'], lsinfo['flag'], Hi, _np.nan,
                                 _np.nan, opt['outputHessian'], opt['params'])
        else:
            history = gatherInfo(history, objSign * v, u,
                                 _np.linalg.norm(pg, _np.inf), 0, 0, 0, Hi,
                                 _np.nan, _np.nan, opt['outputHessian'],
                                 opt['params'])

        _save(work, history)

        success = (it >= maxIt
                   or _np.linalg.norm(pg, _np.inf) < opt['gradTol']
                   or abs(v - v0) < opt['objChangeTol']
                   or (v != 0 and abs((v - v0) / v) < opt['objChangeTolRel']))
        u0, v0, g0 = u, v, g

        if opt['verbose']:
            printInfo(history, it)

    return objSign * v, u, history


def getSearchDirection(u0, g0, Hi, HiPrev, c, opt):
    """Port of ``getSearchDirection``.

    Projects the gradient and the quasi-Newton direction separately (the
    ``kd`` loop): the direction can activate constraints the gradient
    misses, so each needs its own active set. Then walks the segment,
    adding constraints as it reaches them.
    """
    d = maxStep = None
    pg = g0

    for k in (1, 2, 3):
        if k == 2:
            Hi = HiPrev
        elif k == 3:
            Hi = Hi.reset() if isinstance(Hi, LimitedMemoryHessian) \
                else _np.eye(u0.size)

        Q = c['e']['Q']
        pg = -projQ(g0, Q)
        d = -projQ(g0, Q, Hi)

        isActive = _np.zeros(c['i']['A'].shape[0], dtype=bool)
        for kd in (1, 2):
            na, na_prev = 0, -1
            while na > na_prev:
                probe = pg if kd == 1 else d
                _, active_cur = classifyConstraints(c['i']['A'], c['i']['b'],
                                                    u0, probe)
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
                    if kd == 1:
                        pg = -projQ(g0, Q)
                    else:
                        d = -projQ(g0, Q, Hi)

        if _np.linalg.norm(pg, _np.inf) <= _SQEPS * _np.linalg.norm(g0, _np.inf):
            return None, Hi, pg, None

        dr, gr = d, g0
        becomesActive = isActive.copy()
        d = _np.zeros(u0.size)
        done = False
        while not done:
            if _np.linalg.norm(dr) > _SQEPS:
                sgn, _ = classifyConstraints(c['i']['A'], c['i']['b'],
                                             u0 + d, dr)
                ix, s = findNextConstraint(c['i']['A'], c['i']['b'], u0 + d,
                                           dr, (sgn <= 0) | becomesActive)
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

        sgn, _ = classifyConstraints(c['i']['A'], c['i']['b'], u0, d)
        _, maxStep = findNextConstraint(c['i']['A'], c['i']['b'], u0, d,
                                        sgn <= 0)
        if maxStep < 0.95 and opt['verbose']:
            _warnings.warn('Problematic constraint handling, relative step '
                           'length: %6.5f' % maxStep, RuntimeWarning)
        if maxStep < 1:
            d, maxStep = maxStep * d, 1.0

        isDecreasing = float(d @ g0) <= 0
        Hg = Hi.dot(g0) if isinstance(Hi, LimitedMemoryHessian) else Hi @ g0
        isZero = _np.linalg.norm(d, _np.inf) <= _SQEPS * _np.linalg.norm(
            Hg, _np.inf)
        if isDecreasing and not isZero:
            break

        if opt['verbose']:
            what = 'Small norm of search direction' if not isZero \
                else 'Non-inceasing search direction'
            if k == 1:
                print('%s, trying previous Hessian approximation.' % what)
            elif k == 2:
                print('%s, trying to reset Hessian to identity.' % what)
            else:
                print('Exiting: %s.' % what)
        if k == 3:
            d, maxStep = None, None

    return d, Hi, pg, maxStep


def checkFeasible(u, c, enforce=False, nm='Vector u'):
    """Port of ``checkFeasible``. Returns ``(u, flag, fixed)``.

    Equality violations are removed in one shot by the minimum-norm
    correction. Inequality violations are walked one at a time, each
    projected out along its own normal within the space left by the
    equalities and the constraints already handled. MRST's own comment
    calls this a stand-in for a QP solve, intended for mild violations.
    """
    u = _np.asarray(u, dtype=float).ravel().copy()
    hasEC = _np.size(c['e']['A']) > 0 and c['e']['A'].shape[0] > 0
    hasIC = _np.size(c['i']['A']) > 0 and c['i']['A'].shape[0] > 0
    ecOK, icOK = True, True

    if hasEC:
        Ae = c['e']['A'].toarray() if _sp.issparse(c['e']['A']) \
            else _np.asarray(c['e']['A'], dtype=float)
        be = _np.asarray(c['e']['b'], dtype=float).ravel()
        if _np.any(_np.abs(Ae @ u - be) > _SQEPS):
            u = u + Ae.T @ _np.linalg.solve(Ae @ Ae.T, be - Ae @ u)
            ecOK = False              # fixed now, but still worth warning

    flag = ecOK
    fixed = False
    maxIt = 100
    Ai = _sp.csr_matrix(c['i']['A']) if hasIC else None
    bi = _np.asarray(c['i']['b'], dtype=float).ravel() if hasIC else None

    it = 0
    for it in range(1, maxIt + 1):
        if hasIC:
            icOK = not _np.any(Ai @ u - bi > _SQEPS)
            flag = flag and icOK
        if not enforce:
            break
        if icOK:
            fixed = True
            break

        Q = _np.zeros((u.size, 0))
        if hasEC:
            Q = _np.asarray(c['e']['Q'], dtype=float)

        def proj(v):
            return v if Q.size == 0 else v - Q @ (Q.T @ v)

        done = False
        cnt = 0
        icIx = _np.zeros(0, dtype=int)
        while not done:
            if cnt == 0:
                icIx = _np.flatnonzero(Ai @ u - bi > _SQEPS)
                icIx = _np.roll(icIx, it)
            if icIx.size == 0 or cnt == icIx.size:
                done = True
                continue
            ix = int(icIx[cnt])
            a = _np.asarray(Ai[ix, :].todense()).ravel()
            b = float(bi[ix])
            pa = proj(a)
            if _np.linalg.norm(pa) < _SQEPS * _np.linalg.norm(a):
                cnt += 1
                continue
            cnt = 0
            u = u + pa * ((b - a @ u) / (a @ pa))
            if Q.shape[1] < Q.shape[0] - 1:
                Q = expandQ(Q, pa)
            else:
                done = True

    if it == maxIt:
        _warnings.warn('Failed attempt to fix feasibility of %s, continuing '
                       'anyway ...' % nm, RuntimeWarning)
    elif not flag:
        if not enforce:
            _warnings.warn('%s is not feasible within tollerance. Consider '
                           "running with option 'enforceFeasible'=True" % nm,
                           RuntimeWarning)
        else:
            _warnings.warn('%s was not feasible, fixed feasibility in %d '
                           'iteration(s)' % (nm, it - 1), RuntimeWarning)
    return u, flag, fixed


def _copyBase(base, work, iteration):
    """Give iteration ``iteration`` its own copy of the base case."""
    case_dir = _os.path.join(work, 'case%d' % iteration)
    if _os.path.isdir(case_dir):
        _shutil.rmtree(case_dir, ignore_errors=True)
    if base and _os.path.isdir(base):
        _shutil.copytree(base, case_dir)
    else:
        _os.makedirs(case_dir, exist_ok=True)
    return case_dir


def _save(work, history):
    """Checkpoint, as the MATLAB saves history.mat each iteration."""
    try:
        _os.makedirs(work, exist_ok=True)
        _np.savez(_os.path.join(work, 'history.npz'),
                  val=_np.asarray(history['val'], dtype=float),
                  u=_np.asarray(history['u'], dtype=float))
    except Exception:
        # Losing the checkpoint must not abort an expensive run.
        pass
