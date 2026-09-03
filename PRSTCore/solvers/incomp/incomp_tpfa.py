"""Python port of MRST's ``incompTPFA.m`` (mrst-2026a/solvers/incomp).

Solves the incompressible pressure/flux equations with a two-point flux
approximation on a fully unstructured grid, given half-transmissibilities
from :func:`PRSTCore.solvers.incomp.compute_trans.compute_trans`.

Scope (a faithful port of the linear-algebra core, not every option):
  - wells (rate- and bhp-controlled), Dirichlet/Neumann boundary conditions
    (``bc``), and explicit source terms (``src``) are supported, matching
    MRST's default code path.
  - gravity and capillary pressure are NOT modelled (equivalent to calling
    MRST with ``gravity off`` and no ``pc``), and the ``'reduce'``/``'bcp'``/
    ``'use_trans'`` options are not implemented. These are the parts of
    ``incompTPFA.m`` exercised by the vast majority of tutorial/diagnostics
    workflows; gravity is a natural P2 follow-up once cell centroids-based
    hydrostatic well pressure drop is ported too.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def incomp_tpfa(G: dict, T: np.ndarray, fluid: dict, *, wells=None, bc=None, src=None,
                 mob=None, matrix_output: bool = False) -> dict:
    """Port of MRST ``incompTPFA.m`` (no gravity / capillary pressure).

    Parameters
    ----------
    G, T : grid (from ``PRSTCore.gridprocessing``) and half-transmissibilities
        (from ``compute_trans``).
    fluid : dict with ``'mu'`` (single-phase viscosity, Pa.s). Ignored if
        ``mob`` is given directly.
    wells : list of dicts, each ``{'cells', 'WI', 'type': 'rate'|'bhp', 'val'}``
        (0-based perforation cell indices; WI = Peaceman well index per
        perforation).
    bc : dict or None: ``{'face', 'type', 'value'}`` with ``type`` entries
        ``'pressure'`` (Dirichlet, Pa) or ``'flux'`` (Neumann, m^3/s *into*
        the domain), 0-based face indices.
    src : dict or None: ``{'cell', 'rate'}``, m^3/s (positive = injection),
        0-based cell indices.
    mob : optional per-cell total mobility (nc,), overrides ``fluid['mu']``.

    Returns
    -------
    state : dict with ``'pressure'`` (nc,), ``'facePressure'`` (nif,),
        ``'flux'`` (nif,), ``'wellSol'`` (list of ``{'flux', 'pressure'}``
        per well), and (if ``matrix_output``) ``'A'``/``'rhs'``.
    """
    nc = G["cells"]["num"]
    nif = G["faces"]["num"]
    face_pos = G["cells"]["facePos"]
    cell_faces = np.asarray(G["cells"]["faces"])
    cf = cell_faces[:, 0]
    cellNo = np.repeat(np.arange(nc), np.diff(face_pos))
    neighbors = np.asarray(G["faces"]["neighbors"])

    totmob = np.full(nc, 1.0 / fluid["mu"], dtype=float) if mob is None else np.asarray(mob, dtype=float)

    Th = T * totmob[cellNo]
    ft = 1.0 / np.bincount(cf, weights=1.0 / Th, minlength=nif)

    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)

    # --- boundary conditions -------------------------------------------------
    dF = np.zeros(nif, dtype=bool)
    dC = np.zeros(nif)
    neumann_value = np.zeros(nif)  # m^3/s into the domain, only meaningful where set
    has_neumann = np.zeros(nif, dtype=bool)

    if bc is not None:
        bc_faces = np.asarray(bc["face"], dtype=int)
        bc_types = list(bc["type"])
        bc_values = np.asarray(bc["value"], dtype=float)
        for f, kind, val in zip(bc_faces, bc_types, bc_values):
            if kind == "pressure":
                dF[f] = True
                dC[f] = val
            elif kind == "flux":
                neumann_value[f] += val
                has_neumann[f] = True
            else:
                raise ValueError(f"Unsupported bc type {kind!r}")

    nw = 0 if wells is None else len(wells)
    n = nc + nw

    d = np.zeros(nc)
    rhs = np.zeros(n)

    dF_hf = dF[cf]
    if np.any(dF_hf):
        d += np.bincount(cellNo[dF_hf], weights=Th[dF_hf], minlength=nc)
        rhs[:nc] += np.bincount(cellNo[dF_hf], weights=Th[dF_hf] * dC[cf[dF_hf]], minlength=nc)

    n0, n1 = neighbors[internal, 0], neighbors[internal, 1]
    ft_int = ft[internal]
    d += np.bincount(n0, weights=ft_int, minlength=nc) + np.bincount(n1, weights=ft_int, minlength=nc)

    if np.any(has_neumann):
        nf = np.nonzero(has_neumann)[0]
        # `neumann_value` is defined as the volumetric rate *into the cell* that
        # owns this boundary face, so it enters the mass balance unchanged --
        # no geometric sign factor needed here (unlike the `flux` array below,
        # which follows the neighbors[:,0]->neighbors[:,1] directional convention).
        c = np.where(neighbors[nf, 0] >= 0, neighbors[nf, 0], neighbors[nf, 1])
        rhs[:nc] += np.bincount(c, weights=neumann_value[nf], minlength=nc)

    if src is not None:
        src_cells = np.asarray(src["cell"], dtype=int)
        src_rate = np.asarray(src["rate"], dtype=float)
        rhs[:nc] += np.bincount(src_cells, weights=src_rate, minlength=nc)

    # --- assemble internal-face part of the matrix ---------------------------
    rows = [n0, n1, np.arange(nc)]
    cols = [n1, n0, np.arange(nc)]
    vals = [-ft_int, -ft_int, d]

    # --- wells -----------------------------------------------------------
    well_sol_meta = []
    if nw:
        c_rows, c_cols, c_vals = [], [], []
        D_diag = np.zeros(nw)
        for k, w in enumerate(wells):
            wc = np.asarray(w["cells"], dtype=int)
            WI = np.asarray(w["WI"], dtype=float)
            wi = WI * totmob[wc]
            widx = nc + k

            rows.append(wc); cols.append(wc); vals.append(wi)  # well contributes to cell diagonal

            if w["type"] == "bhp":
                ww = float(np.max(wi)) if wi.size else 0.0
                rhs[widx] += ww * w["val"]
                rhs[wc] += wi * w["val"]
                D_diag[k] = ww
                # C row is identically zero for bhp wells (matches MRST).
            elif w["type"] == "rate":
                rhs[widx] += w["val"]
                c_rows.append(np.full(wc.size, widx)); c_cols.append(wc); c_vals.append(-wi)
                c_rows.append(wc); c_cols.append(np.full(wc.size, widx)); c_vals.append(-wi)
                D_diag[k] = float(np.sum(wi))
            else:
                raise ValueError(f"Unsupported well type {w['type']!r}")

            well_sol_meta.append((wc, WI, wi, widx))

        if c_rows:
            rows.append(np.concatenate(c_rows))
            cols.append(np.concatenate(c_cols))
            vals.append(np.concatenate(c_vals))
        rows.append(np.arange(nc, n)); cols.append(np.arange(nc, n)); vals.append(D_diag)

    I = np.concatenate(rows).astype(int)
    J = np.concatenate(cols).astype(int)
    V = np.concatenate(vals)
    A = sp.coo_matrix((V, (I, J)), shape=(n, n)).tocsr()

    has_bhp_well = wells is not None and any(w["type"] == "bhp" for w in wells)
    if not np.any(dF) and not has_bhp_well:
        # Pure-Neumann system: pin the null space, matching MRST's anchor fix.
        A = A.tolil()
        if A[0, 0] > 0:
            A[0, 0] = 2 * A[0, 0]
        else:
            j = int(np.argmax(A.diagonal()))
            A[j, j] = 2 * A[j, j]
        A = A.tocsr()

    p = spla.spsolve(A, rhs)

    # --- face pressures & fluxes ---------------------------------------------
    p_cell = p[:nc]
    fpress = np.bincount(cf, weights=Th * p_cell[cellNo], minlength=nif) / np.bincount(cf, weights=Th, minlength=nif)

    flux = np.zeros(nif)
    flux[internal] = -ft_int * (p_cell[n1] - p_cell[n0])

    boundary = ~internal
    bidx = np.nonzero(boundary)[0]
    if bidx.size:
        # Matches MRST's `sgn = 2*(G.faces.neighbors(~i,2)==0)-1`: +1 when the
        # *second* neighbor column holds the boundary marker.
        boundary_on_side1 = neighbors[bidx, 1] < 0
        c = np.where(boundary_on_side1, neighbors[bidx, 0], neighbors[bidx, 1])
        sgn = np.where(boundary_on_side1, 1.0, -1.0)
        fpress[bidx] = p_cell[c]  # default: no-flow -> face pressure == adjacent cell pressure
        fpress[dF] = dC[dF]
        flux[bidx] = -sgn * ft[bidx] * (fpress[bidx] - p_cell[c])

        nb = has_neumann[bidx]
        if np.any(nb):
            f_nb = bidx[nb]
            c_nb = c[nb]
            sgn_nb = sgn[nb]
            # Flip relative to the RHS's cell-local sign: `flux` follows the
            # neighbors[:,0]->neighbors[:,1] directional convention, so inflow
            # into a side-0 cell (boundary_on_side1, sgn=+1) is *negative* flux.
            flux[f_nb] = -sgn_nb * neumann_value[f_nb]
            # Invert the general relation flux = -sgn*ft*(fpress-p_c) for fpress.
            fpress[f_nb] = p_cell[c_nb] - flux[f_nb] / (sgn_nb * ft[f_nb])

    well_sol = []
    for wc, WI, wi, widx in well_sol_meta:
        well_sol.append({
            "flux": wi * (p[widx] - p_cell[wc]),
            "pressure": float(p[widx]),
        })

    state = {"pressure": p_cell, "facePressure": fpress, "flux": flux, "wellSol": well_sol}
    if matrix_output:
        state["A"] = A
        state["rhs"] = rhs
    return state
