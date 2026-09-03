"""Port of MRST ``optimWellPlacementSimple.m`` (mrst-2026a/hm/utils).

A plain projected-gradient *ascent* on well trajectory controls: step along
the gradient, project the step back onto each well's trajectory
constraints, clip it to ``maxRelative`` per component and to the unit box,
and halve it until the objective improves.

Maximisation, not minimisation: the line search accepts on ``v_test > v``.

Trajectory plotting is omitted -- the MATLAB draws into a figure axis with
``plotGrid``/``plot3``; there is nothing to compute there.
"""

import numpy as _np


def optimWellPlacementSimple(u0, f, W, stepInit=1.0, maxSteps=20,
                             maxRelative=0.05, maxLineSearchIts=10,
                             plotTrajectories=False, plotAx=None,
                             verbose=True):
    """Return the optimised control vector ``u``.

    ``f(u)`` returns ``(value, gradient)``; ``W`` is the well list, each
    well carrying a ``posControl`` with ``nPoints``, ``parameters.nParam``
    and ``getProjectedUpdate``.
    """
    v, g = f(u0)
    u = _np.array(u0, dtype=float, copy=True)
    cix = _np.arange(u.size)
    if verbose:
        print('Initial objective value: %e' % v)

    step = float(stepInit)
    pc = [w['posControl'] if isinstance(w, dict) else w.posControl for w in W]
    np_points = _np.asarray([_get(p, 'nPoints') for p in pc], dtype=int)
    total = int(np_points.sum()) * 3

    for k in range(int(maxSteps)):
        if verbose:
            print('Outer iteration: %d' % (k + 1))
        ok = False
        lits = 0
        v_test, g_test, du_cur = v, g, _np.zeros_like(u)

        while (not ok) and lits < int(maxLineSearchIts):
            du_cur = step * _np.asarray(g, dtype=float)

            # The projection runs in the full (all control points) space.
            u_tmp = _np.zeros(total)
            du_tmp = _np.zeros(total)
            u_tmp[cix] = u
            du_tmp[cix] = du_cur
            du_cur = (2.0 ** (-lits)) * _getProjectedUpdate(pc, u_tmp, du_tmp)
            du_cur = du_cur[cix]

            # Clip each component, then clip the point to the unit box.
            du_c = _np.sign(du_cur) * _np.minimum(maxRelative, _np.abs(du_cur))
            u_c = _np.maximum(0.0, _np.minimum(1.0, u + du_c))
            du_cur = u_c - u

            v_test, g_test = f(u + du_cur)
            if verbose:
                print('Computed objective value: %e' % v_test)
            if v_test > v:
                ok = True
            lits += 1

        if lits == int(maxLineSearchIts) and not ok:
            if verbose:
                print('Reached maximal number of line search iterations, exiting.')
            break

        v, g = v_test, g_test
        gnorm = _np.linalg.norm(g)
        step = max(0.01, _np.linalg.norm(du_cur) / gnorm) if gnorm else 0.01
        u = u + du_cur

    return u


def _getProjectedUpdate(pc, u, du):
    """Port of the local ``getProjectedUpdate``.

    The unit-box clip is applied first, then each well's own trajectory
    projection overwrites its own slice.
    """
    u_tmp = _np.maximum(0.0, _np.minimum(1.0, u + du))
    dup = u_tmp - u
    ix = 0
    for control in pc:
        n_param = int(_get(_get(control, 'parameters'), 'nParam'))
        sl = slice(ix, ix + n_param)
        dup[sl] = _get(control, 'getProjectedUpdate')(u[sl], du[sl], True)
        ix += n_param
    return dup


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
