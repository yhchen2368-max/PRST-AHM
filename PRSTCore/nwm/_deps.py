"""Dependency wiring for the ``nwm`` models.

Provides lazy accessors for the PRSTCore functions used by the NWM model
classes, and raises informative ``NotImplementedError`` for the heavy MRST
core / AD-solver routines (deck-schedule conversion, aquifers, multi-segment
well conversion, corner-point geometry, ...) that are not (yet) ported to
Python.
"""

from __future__ import annotations

import numpy as np


def _import(modpath, attr, what):
    import importlib
    try:
        mod = importlib.import_module(modpath)
        return getattr(mod, attr)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f'{what} is not yet available in the Python PRSTCore port '
            f'(tried {modpath}.{attr}).') from exc


def initDeckADIFluid(deck):
    """Port of MRST ``initDeckADIFluid`` (PRSTCore)."""
    return _import('PRSTCore.ad_core.initialization.init_deck_adi_fluid',
                   'init_deck_adi_fluid',
                   'initDeckADIFluid (AD fluid from deck)')(deck)


def initEclipseRock(deck):
    """Port of MRST ``initEclipseRock`` (PRSTCore)."""
    return _import('PRSTCore.deckformat.params.rock.init_eclipse_rock',
                   'init_eclipse_rock',
                   'initEclipseRock (rock from deck)')(deck)


def computeTrans(G, rock, **kwargs):
    """Port of MRST ``computeTrans`` (PRSTCore, ``K_system='xyz'`` path)."""
    fn = _import('PRSTCore.solvers.incomp.compute_trans', 'compute_trans',
                 'computeTrans (half transmissibility)')
    # The PRSTCore port computes the global-xyz TPFA half transmissibility;
    # the MRST ``'K_system', 'loc_xyz'`` + ``cellCenters/cellFaceCenters``
    # path of the corner-point branch is approximated by it.
    return fn(G, rock)


def GenericBlackOilModel(*args, **kwargs):
    """Port of MRST ``GenericBlackOilModel`` (PRSTCore)."""
    return _import('PRSTCore.ad_core.models.generic_black_oil_model',
                   'GenericBlackOilModel',
                   'GenericBlackOilModel')(*args, **kwargs)


def ThreePhaseBlackOilModel(*args, **kwargs):
    """Port of MRST ``ThreePhaseBlackOilModel`` (PRSTCore uses the same
    generic three-phase black-oil class)."""
    return _import('PRSTCore.ad_core.models.generic_black_oil_model',
                   'GenericBlackOilModel',
                   'ThreePhaseBlackOilModel')(*args, **kwargs)


def setupOperatorsTPFA(G, rock, neighbors=None, trans=None):
    """Minimal port of MRST ``setupOperatorsTPFA`` for the NWM hybrid grid:
    returns an operators dict carrying the user-supplied neighbour lists and
    transmissibilities (``N``/``T``, plus ``N_all``/``T_all`` set by the
    caller)."""
    return {'N': np.asarray(neighbors, dtype=np.int64) if neighbors is not None else None,
            'T': np.asarray(trans, dtype=float) if trans is not None else None}


def compressRock(rock, cells):
    """Port of MRST ``compressRock``: select the rock data of the given
    (0-based) cells."""
    cells = np.asarray(cells, dtype=np.int64).ravel()
    out = {}
    for fld, val in rock.items():
        val = np.asarray(val)
        if val.ndim == 0:
            out[fld] = val
        else:
            out[fld] = val[cells]
    return out


def computeCpGeometry(*args, **kwargs):
    """MRST ``computeCpGeometry`` (corner-point geometry) - not yet ported."""
    raise NotImplementedError(
        'computeCpGeometry (corner-point cell/face centres) is not yet '
        'available in the Python PRSTCore port.')


def convertDeckScheduleToMRST(model, deck, **kwargs):
    """Port of MRST ``convertDeckScheduleToMRST`` (deck SCHEDULE -> MRST
    schedule struct, with well I/J/K completions resolved to grid cells).

    Reuses ``init_eclipse_problem_ad._convert_deck_schedule_to_mrst``, the
    same deck-schedule conversion already exercised by the top-level
    ``init_eclipse_problem_ad`` pipeline for plain (non-NWM) decks;
    ``model.G``/``model.rock`` supply the grid/rock the well completions
    resolve against, exactly as ``initEclipseProblemAD`` passes its own
    ``G``/``rock`` to ``convertDeckScheduleToMRST`` in MRST.
    """
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        _convert_deck_schedule_to_mrst,
    )
    return _convert_deck_schedule_to_mrst(model, deck, G=model.G, rock=model.rock)


def addWell(W, G, rock, cells, name='W', refDepth=0.0, **kwargs):
    """Minimal port of MRST ``addWell``, covering only the ``dZ`` field: the
    per-completion depth offset from the well's BHP reference depth, used by
    :meth:`NearWellboreModel.getSimSchedule` to combine the HW's own
    completions with the deck's regular wells. Mirrors ``addWell.m``'s
    ``getDeltaZ`` (projection of cell centroids onto the gravity direction,
    which for PRSTCore's ``[0, 0, g]`` convention is just the z/depth
    coordinate) rather than the full well-object construction, which is all
    the NWM call site (``_deps.addWell({}, G, rockTmp, wc, ...)``) uses.
    """
    cells = np.asarray(cells, dtype=np.int64).ravel()
    centroids = np.asarray(G['cells']['centroids'], dtype=float)
    z = centroids[cells, 2]
    dZ = z - float(refDepth)
    out = dict(W)
    out.update({'name': name, 'cells': cells, 'refDepth': float(refDepth), 'dZ': dZ})
    return out


def initStateDeck(model, deck):
    """Port of MRST ``initStateDeck`` (EQUIL/direct-assignment equilibrium
    initialisation). Reuses ``init_eclipse_problem_ad._init_state_deck``,
    which implements the same EQUIL/direct-assignment branches and is
    grid-topology agnostic (only needs ``model.G['cells']['centroids']`` and
    ``indexMap``), so it applies unchanged to the NWM hybrid grid."""
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import _init_state_deck
    return _init_state_deck(model, deck)


def processAquifer(deck, G):
    """Port of MRST ``processAquifer`` (Fetkovich AQUANCON/AQUFETP
    aquifer connections from the deck)."""
    from PRSTCore.deckformat.params.process_aquifer import process_aquifer
    return process_aquifer(deck, G)


def AquiferModel(model, aquifers, aquind, aquiferprops, initval):
    """Port of MRST ``AquiferModel``. The Python physics class does not
    itself need the reservoir ``model`` reference (its coupling into the
    water mass balance is wired by the caller, per the class's own
    docstring), so it is accepted here -- matching the NWM call site's
    positional arguments, which mirror MRST's own constructor call -- but
    not forwarded."""
    from PRSTCore.ad_core.models.aquifer_model import AquiferModel as _AquiferModel
    return _AquiferModel(aquifers, aquind, aquiferprops, initval)


def makeSingleWellpath(pW):
    """Lightweight stand-in for MRST (wellpaths module) ``makeSingleWellpath``.

    Returns a simple dict with the discrete well-path points (the wellpaths
    module itself is not yet ported to Python).
    """
    return {'points': np.asarray(pW, dtype=float)}


def findWellPathCells(G, wph):
    """Lightweight stand-in for MRST (wellpaths module) ``findWellPathCells``:
    for each well-path point, return the grid cell whose centroid is nearest
    (this is exact for Cartesian grids and a good approximation elsewhere)."""
    pts = np.asarray(wph['points'], dtype=float)
    centroids = np.asarray(G['cells']['centroids'], dtype=float)
    # For each well point, the nearest cell (based on squared distance)
    out = []
    for p in pts:
        d = np.sum((centroids - p) ** 2, axis=1)
        out.append(int(np.argmin(d)))
    return np.unique(np.array(out, dtype=np.int64))


def convert2MSWell(w, cell2node=None, connDZ=None, nodeDepth=None, topo=None,
                    segLength=None, segRoughness=None, segFlowModel=None,
                    segType=None, segDiam=None, G=None, vol=None):
    """Port of MRST ``convert2MSWell``: derive/attach well nodes and
    segments (``w['nodes']``, ``w['segments']``) to a standard well dict,
    turning it into a multi-segment well.  All of the ``opt.*`` geometric
    inputs MRST can alternatively *compute* (from ``w['cells']``/``w['r']``
    and a grid ``G``, when the caller leaves them empty) are supported here
    too, matching ``convert2MSWell.m``'s fallback branches -- but NWM's
    ``MultiSegWellNWM.getSimSchedule`` always supplies them directly from
    its own wellbore-grid node/segment construction, so those branches are
    exercised only if a future caller omits them.
    """
    w = dict(w)
    cells = np.asarray(w['cells'], dtype=np.int64).ravel()
    nperf = cells.size

    if cell2node is None:
        c2n = np.eye(nperf)
    else:
        import scipy.sparse as _sp
        if _sp.issparse(cell2node):
            c2n = cell2node.toarray().astype(float)
        else:
            c2n = np.asarray(cell2node, dtype=float)
            if c2n.ndim == 1:
                # Perf-to-node given as a per-perforation node index
                # (1-based, MRST convention) -> expand to a dense
                # (nnode, nperf) matrix.
                idx = c2n.astype(np.int64)
                nn = int(idx.max()) if idx.size else 0
                dense = np.zeros((nn, nperf))
                dense[idx - 1, np.arange(nperf)] = 1.0
                c2n = dense
    w['cell2node'] = c2n

    cell_count = np.maximum(c2n.sum(axis=1), 1.0)
    nn = c2n.shape[0]

    if nodeDepth is None:
        depth = float(w['refDepth']) + (c2n @ np.asarray(w['dZ'], dtype=float)) / cell_count
    else:
        depth = np.asarray(nodeDepth, dtype=float).ravel()
    node_vol = np.ones(nn) if vol is None else np.asarray(vol, dtype=float).ravel()
    nodes = {'depth': depth, 'vol': node_vol, 'dist': np.full(nn, np.nan)}

    w['connDZ'] = (np.zeros_like(np.asarray(w['dZ'], dtype=float)) if connDZ is None
                   else np.asarray(connDZ, dtype=float).ravel())

    if topo is None:
        topo = np.column_stack([np.arange(nn - 1), np.arange(1, nn)])
    topo = np.asarray(topo, dtype=np.int64)
    ns = topo.shape[0]

    if segLength is None:
        assert G is not None, 'segLength requires G when not supplied directly'
        cc = np.asarray(G['cells']['centroids'], dtype=float)[cells]
        cn = (c2n @ cc) / cell_count[:, None]
        length = np.linalg.norm(cn[topo[:, 1]] - cn[topo[:, 0]], axis=1)
    else:
        length = np.asarray(segLength, dtype=float).ravel()

    if segDiam is None:
        rc = np.asarray(w['r'], dtype=float).ravel()
        if rc.size == 1:
            rc = np.full(nperf, rc[0])
        rn = (c2n @ rc) / cell_count
        diam = rn[topo[:, 1]] + rn[topo[:, 0]]
    else:
        diam = np.asarray(segDiam, dtype=float).ravel()

    delta_distance = np.zeros(ns)
    if segType is not None and len(segType) > 0:
        seg_type = np.asarray(segType)
        out_from_tube = np.flatnonzero(seg_type == 3)
        for seg_no in out_from_tube:
            d, n = 0.0, 0
            for node in topo[seg_no]:
                local = np.flatnonzero(np.any(topo == node, axis=1))
                for lseg in local:
                    if seg_type[lseg] == 1:
                        d += length[lseg]
                        n += 1
            assert n > 0
            delta_distance[seg_no] = d / n

    segments = {
        'length': length, 'diam': diam, 'topo': topo,
        'roughness': np.zeros(ns), 'flowModel': np.zeros(ns),
        'deltaDistance': delta_distance,
    }
    if segRoughness is not None and len(segRoughness) > 0:
        segments['roughness'] = np.asarray(segRoughness, dtype=float).ravel()
    if segFlowModel is not None and len(segFlowModel) > 0:
        segments['flowModel'] = np.asarray(segFlowModel).ravel()

    w['nodes'] = nodes
    w['segments'] = segments
    w['isMS'] = True
    return w


def wellBoreFriction(v, rho, mu, D, L, roughness, flowtype='massRate', **kwargs):
    """Port of MRST ``wellBoreFriction`` (``flowtype`` positional here to
    match the NWM call site ``_deps.wellBoreFriction(v, rho, mu, D, L,
    roughness, 'massRate')``, mirroring MRST's own positional call)."""
    from PRSTCore.ad_core.models.wellbore_friction import well_bore_friction
    return well_bore_friction(v, rho, mu, D, L, roughness, flowtype=flowtype, **kwargs)


def combineMSwithRegularWells(W_regular, W_ms):
    """Port of MRST ``combineMSwithRegularWells``: concatenate the regular
    (standard) wells with the multi-segment well(s) into one well list,
    tagging each with ``isMS``. MRST's field-set unification (padding
    missing struct-array fields with ``[]``) has no Python analogue -- a
    list of dicts does not require uniform keys across elements."""
    W_regular = list(W_regular) if isinstance(W_regular, (list, tuple)) else [W_regular]
    W_ms = list(W_ms) if isinstance(W_ms, (list, tuple)) else [W_ms]
    out = []
    for w in W_regular:
        w = dict(w)
        w['isMS'] = False
        out.append(w)
    for w in W_ms:
        w = dict(w)
        w['isMS'] = True
        out.append(w)
    return out
