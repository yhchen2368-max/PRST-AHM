"""Port of MRST ``distmesh_2d_nwm``: the modified DistMesh 2D mesh generator
(Per-Olof Persson, mod. John Burkardt) using signed distance functions."""

import numpy as np

from .._core import delaunayn


def _matlab_colon(a, b, c):
    """MATLAB ``a:b:c`` (includes the endpoint when it is on the grid)."""
    if b == 0:
        return np.array([], dtype=float)
    n = int(np.floor((c - a) / b + 1e-10)) + 1
    vals = a + b * np.arange(n)
    return vals[vals <= c + 1e-9 * abs(b)]


def distmesh_2d_nwm(fd, fh, h0, box, iteration_max, pfix, plotMesh=False):
    """2D mesh generator using distance functions.

    Parameters
    ----------
    fd : callable
        Signed distance function ``d(x, y)`` (negative inside).
    fh : callable
        Scaled edge length function ``h(x, y)``.
    h0 : float
        The initial edge length.
    box : ndarray, ``2 x 2``
        The bounding box ``[xmin, ymin; xmax, ymax]``.
    iteration_max : int
        The maximum number of iterations.
    pfix : ndarray
        The fixed node positions.
    plotMesh : bool, optional
        Whether to plot the current mesh.

    Returns
    -------
    p : ndarray
        The node positions.
    t : ndarray
        The triangle indices.
    """
    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    iteration = 0
    triangulation_count = 0

    box = np.asarray(box, dtype=float)
    pfix = np.asarray(pfix, dtype=float)

    # 1. Create the initial point distribution by generating a rectangular
    # mesh in the bounding box.
    x = _matlab_colon(box[0, 0], h0, box[1, 0])
    y = _matlab_colon(box[0, 1], h0 * np.sqrt(3) / 2, box[1, 1])
    X, Y = np.meshgrid(x, y)
    # Shift the even rows of the mesh to create a 'perfect' mesh of
    # equilateral triangles.
    X[1::2, :] = X[1::2, :] + h0 / 2
    p = np.column_stack([X.ravel(order='F'), Y.ravel(order='F')])

    # 2. Remove mesh points that are outside the region, then satisfy the
    # density constraint.
    p = p[fd(p) < geps]

    r0 = 1.0 / fh(p) ** 2
    keep = np.random.rand(p.shape[0]) < r0 / np.max(r0)
    p = np.vstack([pfix, p[keep]])

    # Keep unique points, always keeping the fixed points at the beginning.
    q, i, j = np.unique(p, axis=0, return_index=True, return_inverse=True)
    k = np.unique(i)
    assert np.all(k[:pfix.shape[0]] == np.arange(pfix.shape[0]))
    p = p[k]

    N = p.shape[0]

    if iteration_max <= 0:
        t = delaunayn(p)
        triangulation_count += 1
        return p, t

    pold = np.inf
    Ftot = np.zeros((0, 2))
    d = np.array([])

    while iteration < iteration_max:
        iteration += 1

        if iteration % 100 == 0:
            with np.errstate(invalid='ignore'):
                m = (np.max(np.sqrt(np.sum((deltat * Ftot[d < -geps]) ** 2, axis=1)) / h0)
                     if Ftot.shape[0] else np.nan)
            print(f'      {iteration} iterations, {triangulation_count} triangulations, tol = {m}')

        # 3. Retriangulation by the Delaunay algorithm.
        if ttol < np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0):
            N = p.shape[0]
            pold = p
            t = delaunayn(p)
            triangulation_count += 1
            pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3
            t = t[fd(pmid) < -geps]

            # 4. Describe each bar by a unique pair of nodes.
            bars = np.vstack([t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]])
            bars = np.unique(np.sort(bars, axis=1), axis=0)

            # 5. Graphical output of the current mesh.
            if plotMesh:
                print('   (plotMesh is not implemented in the Python port)')

        # 6. Move mesh points based on bar lengths L and forces F.
        barvec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        hbars = fh((p[bars[:, 0]] + p[bars[:, 1]]) / 2)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0)
        Fvec = (F / L)[:, None] * barvec
        # MRST's ``Ftot = full(sparse(bars(:,[1,1,2,2]), ..., [Fvec,-Fvec], N, 2))``
        # is a scatter-add over possibly-repeated bar-endpoint indices.
        # ``np.add.at`` gives the same result but forgoes NumPy's buffered
        # ufunc loop to stay correct under duplicate indices, which makes it
        # dramatically slower than a bincount-based scatter-add for the
        # thousands of bars/iterations this generator runs -- exactly the
        # gap between it and MATLAB's compiled ``sparse`` accumulation.
        idx = np.concatenate([bars[:, 0], bars[:, 1]])
        vals = np.concatenate([Fvec, -Fvec])
        Ftot = np.column_stack([
            np.bincount(idx, weights=vals[:, 0], minlength=N),
            np.bincount(idx, weights=vals[:, 1], minlength=N),
        ])
        Ftot[:pfix.shape[0]] = 0
        p = p + deltat * Ftot

        # 7. Bring outside points back to the boundary.
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            dgradx = (fd(np.column_stack([p[ix, 0] + deps, p[ix, 1]])) - d[ix]) / deps
            dgrady = (fd(np.column_stack([p[ix, 0], p[ix, 1] + deps])) - d[ix]) / deps
            p[ix] = p[ix] - np.column_stack([d[ix] * dgradx, d[ix] * dgrady])

        # 8. Termination criterion: all interior nodes move less than dptol.
        with np.errstate(invalid='ignore'):
            if np.max(np.sqrt(np.sum((deltat * Ftot[d < -geps]) ** 2, axis=1)) / h0) < dptol:
                break

    return p, t
