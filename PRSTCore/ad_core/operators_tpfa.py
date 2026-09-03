"""General-grid TPFA operators: ports of MRST's ``getFaceTransmissibility.m``
and ``setupOperatorsTPFA.m``.

Unlike :func:`PRSTCore.ad_core.operators.setup_operators`, which derives
connections from a logical Cartesian index map and therefore only accepts
the grid dicts produced by ``init_eclipse_grid``, everything here works
from the general ``grid_structure`` topology
(``cells.faces``/``cells.facePos``, ``faces.neighbors``/``centroids``/
``normals``, ``cells.centroids``) that ``compute_geometry`` produces.  That
is the representation MRST itself computes transmissibilities from, so the
same code covers Cartesian, tensor, corner-point and fully unstructured
grids alike.

The half-transmissibility kernel is *not* reimplemented here: it reuses
:func:`PRSTCore.solvers.incomp.compute_trans.compute_trans`, the existing
port of ``computeTrans.m`` that is already checked against MRST reference
data by ``tests/test_incomp_tpfa_mrst_parity.py``.

Index conventions follow the rest of :mod:`PRSTCore.gridprocessing`:
cells/faces are 0-based and ``faces.neighbors`` uses ``-1`` for "no cell"
where MRST uses ``0``.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from PRSTCore.solvers.incomp.compute_trans import compute_trans


def get_face_transmissibility(G, rock, fix_negative=True):
    """Port of MRST ``getFaceTransmissibility``: the one-sided (half-face)
    transmissibilities of ``computeTrans`` reduced harmonically onto faces.
    """
    hT = compute_trans(G, rock, fix_negative=fix_negative)
    cf_face = np.asarray(G['cells']['faces'], dtype=np.int64)[:, 0]
    nf = int(G['faces']['num'])
    # apply_harmonic_reduction: 1 ./ accumarray(cf, 1./hT, [nf, 1]).  A face
    # with no contribution accumulates 0, whose reciprocal MATLAB reports as
    # Inf; such a face cannot be internal and is dropped by the
    # internal-only selection below, so guard the division rather than warn.
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.bincount(cf_face, weights=1.0 / hT, minlength=nf)
        T = 1.0 / inv
    T[~np.isfinite(T)] = 0.0
    return T


def pore_volume(G, rock):
    """Port of MRST ``poreVolume``: ``poro .* G.cells.volumes .* ntg``."""
    # setupOperatorsTPFA.pore_volume first checks the simulator-derived
    # ``G.cells.PORV`` field.  FAHM creates that field from INIT precisely
    # so deck/geometry reconstruction cannot perturb accumulation volumes.
    porv = G.get('cells', {}).get('PORV') if isinstance(G, dict) else None
    if porv is not None:
        values = np.asarray(porv, dtype=float).ravel()
        if values.size == int(G['cells']['num']):
            return values.copy()
    volumes = G['cells'].get('volumes')
    if volumes is None:
        volumes = G.get('cell_volumes')
    if volumes is None:
        raise ValueError('Grid has no cell volumes; run compute_geometry first')
    pv = np.asarray(volumes, dtype=float).ravel()
    rock = rock if isinstance(rock, dict) else {}
    poro = rock.get('poro')
    if poro is not None:
        pv = pv * np.asarray(poro, dtype=float).ravel()
    ntg = rock.get('ntg')
    if ntg is not None:
        ntg = np.asarray(ntg, dtype=float).ravel()
        pv = pv * (float(ntg[0]) if ntg.size == 1 else ntg)
    return pv


def _neighbor_subset_index(N, N_sub):
    """Port of MRST-0's ``getNeighborSubsetIndex``.

    Marks the grid faces whose (unordered) cell pair appears in
    ``N_sub``.  ECLIPSE's INIT file lists the connections it actually
    solved, which may be fewer than the grid's internal faces -- pinched
    layers and inactive cells drop out -- so the operators have to be
    built on that subset rather than on every internal face.
    """
    N = np.asarray(N, dtype=np.int64).reshape(-1, 2)
    N_sub = np.asarray(N_sub, dtype=np.int64).reshape(-1, 2)
    scale = int(max(N.max(), N_sub.max())) + 2

    a = np.sort(N, axis=1)
    b = np.sort(N_sub, axis=1)
    keys = a[:, 0] * scale + a[:, 1]
    wanted = b[:, 0] * scale + b[:, 1]
    return np.isin(keys, wanted)


def _mrst_corner_point_internal_order(N, G):
    """Permutation used by MRST ``processGRDECL`` for Cartesian faces.

    MRST stores I faces with I varying fastest, J faces with J varying
    fastest inside each I column, and K faces with K varying fastest inside
    each (I,J) column.  PRST's geometry is equivalent but creates faces in
    a different traversal order.  A consistent face permutation is
    mathematically neutral, yet FAHM's saved operators and parameter
    derivatives are ordered arrays, so normalize the internal subset here.
    """
    N = np.asarray(N, dtype=np.int64).reshape(-1, 2)
    if not N.size or not isinstance(G, dict):
        return np.arange(N.shape[0])
    dims = np.asarray(G.get('cartDims', []), dtype=int).ravel()
    index_map = np.asarray(G.get('cells', {}).get('indexMap', []),
                           dtype=np.int64).ravel()
    if dims.size != 3 or index_map.size < int(G['cells']['num']):
        return np.arange(N.shape[0])
    c0 = np.minimum(index_map[N[:, 0]], index_map[N[:, 1]])
    c1 = np.maximum(index_map[N[:, 0]], index_map[N[:, 1]])
    delta = c1 - c0
    nx, ny = int(dims[0]), int(dims[1])
    plane = nx * ny
    i = c0 % nx
    j = (c0 // nx) % ny
    k = c0 // plane

    groups = []
    for mask, keys in (
            (delta == 1, (i, j, k)),          # primary: k, j, i
            (delta == nx, (j, i, k)),         # primary: k, i, j
            (delta == plane, (k, i, j))):     # primary: j, i, k
        ix = np.flatnonzero(mask)
        if ix.size:
            groups.append(ix[np.lexsort(tuple(key[ix] for key in keys))])
    regular = (delta == 1) | (delta == nx) | (delta == plane)
    groups.append(np.flatnonzero(~regular))
    return np.concatenate(groups) if groups else np.arange(N.shape[0])


def setup_operators_tpfa(G, rock=None, neighbors=None, trans=None, porv=None):
    """Port of MRST ``setupOperatorsTPFA``.

    Returns the same operator set MRST builds: ``T``, ``T_all``, ``N``,
    ``internalConn``, ``pv``, the divergence matrix ``C`` with ``Grad``/
    ``Div``/``AccDiv``, the face-average ``M``/``faceAvg``, and
    ``faceUpstr``.  ``Grad``/``Div``/``faceAvg``/``faceUpstr`` accept both
    plain arrays and :class:`~PRSTCore.ad_core.adi.SparseADI` values.
    """
    cells = G['cells']
    nc = int(cells['num'])

    # MRST dispatches on ``trans``, not on ``neighbors``: supplying only
    # the cell pairs still takes the grid-based branch, where they select
    # a *subset* of the grid's own faces.  That is MRST-0's
    # ``% edited by zhang`` change -- 2026a asserts the neighbours are
    # empty here instead -- and it is the call HistoryMatching makes,
    # handing over the cell pairs ECLIPSE's INIT file reports while
    # letting the transmissibility come from the grid.
    N_all = np.asarray(G['faces']['neighbors'], dtype=np.int64).reshape(-1, 2)

    if trans is None:
        # grid_based_trans
        T_all = (np.asarray(G['faces']['TRANS'], dtype=float).ravel()
                 if 'TRANS' in G['faces'] else get_face_transmissibility(G, rock))
        if neighbors is None:
            internal = np.all(N_all >= 0, axis=1)
        else:
            internal = _neighbor_subset_index(N_all, neighbors)
        N = N_all[internal, :]
        T = T_all[internal]
        # Python's topology builder and MRST ``processGRDECL`` traverse
        # Cartesian faces differently.  Normalize every grid-derived
        # operator, not only the INIT-neighbour subset: apart from matching
        # FAHM's saved array order, this keeps ``neighbors=base.N`` an
        # identity operation as it is in setupOperatorsTPFA.
        order = _mrst_corner_point_internal_order(N, G)
        N, T = N[order], T[order]
    else:
        # user_provided_trans
        supplied = np.asarray(trans, dtype=float).ravel()
        if neighbors is None:
            internal = np.all(N_all >= 0, axis=1)
            N = N_all[internal, :]
            n_if = int(np.count_nonzero(internal))
        else:
            N_in = np.asarray(neighbors, dtype=np.int64).reshape(-1, 2)
            internal = np.all(N_in >= 0, axis=1)
            n_if = int(np.count_nonzero(internal))
            # explicit_cell_pairs: if the supplied pairs number exactly the
            # grid's internal faces, index into the grid instead.
            internal_grid = np.all(N_all >= 0, axis=1)
            if int(np.count_nonzero(internal_grid)) == n_if:
                internal = internal_grid
            N = N_in[np.all(N_in >= 0, axis=1), :]

        if supplied.size == n_if:
            # Internal-interface transmissibilities only.  MRST-0 sizes
            # T_all by the *grid's* face count, not by ``intInx``.
            T = supplied
            T_all = np.zeros(N_all.shape[0], dtype=float)
            T_all[internal] = supplied
        else:
            if supplied.size != internal.size:
                raise ValueError('Transmissibility vector matches neither the '
                                 'internal nor the full interface count')
            T_all = supplied
            T = supplied[internal]

    if np.any(T < 0.0):
        warnings.warn('Negative transmissibilities in %d interfaces.'
                      % int(np.count_nonzero(T < 0.0)), RuntimeWarning)

    # setupOperatorsTPFA adds G.nnc after constructing the grid-face
    # operators, rejecting inactive and duplicate pairs.  These rows are
    # not represented in G.faces, so they must extend T_all/internalConn.
    nnc = G.get('nnc') if isinstance(G, dict) else None
    if isinstance(nnc, dict) and nnc.get('cells') is not None:
        nnc_cells = np.asarray(nnc['cells'], dtype=np.int64).reshape(-1, 2)
        nnc_trans = np.asarray(nnc.get('trans', []), dtype=float).ravel()
        if nnc_cells.shape[0] != nnc_trans.size:
            raise ValueError('G.nnc cells/trans length mismatch')
        existing = {tuple(sorted(pair)) for pair in np.asarray(N, dtype=int)}
        keep = np.asarray([
            np.all(pair >= 0) and tuple(sorted(pair)) not in existing
            for pair in nnc_cells
        ], dtype=bool)
        if np.any(keep):
            N = np.vstack([N, nnc_cells[keep]])
            T = np.concatenate([T, nnc_trans[keep]])
            T_all = np.concatenate([T_all, nnc_trans[keep]])
            internal = np.concatenate([
                np.asarray(internal, dtype=bool).ravel(),
                np.ones(int(np.count_nonzero(keep)), dtype=bool),
            ])

    pv = (pore_volume(G, rock) if porv is None
          else np.asarray(porv, dtype=float).ravel())
    if pv.size != nc:
        raise ValueError('Dimension mismatch between grid and supplied '
                         'pore-volumes.')

    nf = N.shape[0]
    # C - (transpose) divergence matrix, exactly setupOperatorsTPFA's
    # sparse([(1:nf)'; (1:nf)'], N, ones(nf,1)*[1,-1], nf, nc).
    if nf:
        rows = np.concatenate([np.arange(nf), np.arange(nf)])
        cols = np.concatenate([N[:, 0], N[:, 1]])
        C = sp.csr_matrix((np.concatenate([np.ones(nf), -np.ones(nf)]),
                           (rows, cols)), shape=(nf, nc))
        M = sp.csr_matrix((np.full(2 * nf, 0.5), (rows, cols)), shape=(nf, nc))
    else:
        C = sp.csr_matrix((0, nc))
        M = sp.csr_matrix((0, nc))
    negC = (-C).tocsr()
    Ct = C.T.tocsr()

    def _apply(matrix, x):
        # SparseADI carries its own Jacobian through a linear map; plain
        # arrays just multiply.
        return x.linear_map(matrix) if hasattr(x, 'linear_map') else matrix @ x

    ops = {
        'T': T, 'T_all': T_all, 'N': N, 'internalConn': internal, 'pv': pv,
        'C': C, 'M': M,
        'Grad': lambda x: _apply(negC, x),
        'faceAvg': lambda x: _apply(M, x),
    }
    if nf == 0:
        ops['Div'] = lambda x: np.zeros(nc)
        ops['AccDiv'] = lambda acc, flux: acc
    else:
        ops['Div'] = lambda x: _apply(Ct, x)
        ops['AccDiv'] = lambda acc, flux: acc + _apply(Ct, flux)

    def face_upstr(flag, x):
        """Port of ``faceUpstr``: a true flag selects N(:,1), else N(:,2)."""
        flag = np.asarray(flag, dtype=bool).ravel()
        if flag.size == 1:
            flag = np.repeat(flag, nf)
        if flag.size != nf:
            raise ValueError('One logical upstream flag must be supplied per '
                             'interface.')
        up = np.where(flag, N[:, 0], N[:, 1])
        if hasattr(x, 'linear_map'):
            S = sp.csr_matrix((np.ones(nf), (np.arange(nf), up)), shape=(nf, nc))
            return x.linear_map(S)
        return np.asarray(x)[up]

    ops['faceUpstr'] = face_upstr
    return ops
