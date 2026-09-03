"""Port of MRST ``assembleGrids``: assemble multiple grids without merging
common faces and without handling boundary intersections."""

import numpy as np


def _handleFaceDir(Gs):
    for i in range(len(Gs)):
        if Gs[i]['cells']['faces'].shape[1] < 2:
            col = np.full((Gs[i]['cells']['faces'].shape[0], 1), np.nan)
            Gs[i]['cells']['faces'] = np.hstack([Gs[i]['cells']['faces'], col])
    return Gs


def assembleGrids(Gs):
    """Assemble multiple grids into a combined grid ``G`` (does not merge
    common faces and does not handle boundary intersections).

    The input grids are deep-copied: they are not mutated by the assembly
    (unlike the MATLAB original, which appends a NaN direction column to
    ``cells.faces`` in place)."""
    from copy import deepcopy
    Gs = [deepcopy(g) for g in Gs]

    # G.cells
    nc = np.array([g['cells']['num'] for g in Gs], dtype=np.int64)
    G = {}
    G['cells'] = {}
    G['cells']['num'] = int(nc.sum())

    ncf = np.concatenate([np.diff(g['cells']['facePos']) for g in Gs])
    G['cells']['facePos'] = np.concatenate([[0], np.cumsum(ncf)])

    nf = np.array([g['faces']['num'] for g in Gs], dtype=np.int64)
    nf_cumsum = np.concatenate([[0], np.cumsum(nf)])
    cf = [g['cells']['faces'][:, 0] + nf_cumsum[i] for i, g in enumerate(Gs)]
    Gs = _handleFaceDir(Gs)
    dire = [g['cells']['faces'][:, 1] for g in Gs]
    G['cells']['faces'] = np.column_stack([np.concatenate(cf), np.concatenate(dire)])

    geocell = ('volumes', 'centroids', 'layers')
    for gname in geocell:
        try:
            G['cells'][gname] = np.concatenate([g['cells'][gname] for g in Gs])
        except (KeyError, ValueError):
            pass

    grdID_c = np.concatenate([np.full(int(n), i + 1, dtype=np.int64)
                              for i, n in enumerate(nc)])
    G['cells']['grdID'] = grdID_c

    # G.faces
    G['faces'] = {}
    G['faces']['num'] = int(nf.sum())

    nfn = np.concatenate([np.diff(g['faces']['nodePos']) for g in Gs])
    G['faces']['nodePos'] = np.concatenate([[0], np.cumsum(nfn)])

    nn = np.array([g['nodes']['num'] for g in Gs], dtype=np.int64)
    nn_cumsum = np.concatenate([[0], np.cumsum(nn)])
    fn = [g['faces']['nodes'] + nn_cumsum[i] for i, g in enumerate(Gs)]
    G['faces']['nodes'] = np.concatenate(fn)

    nc_cumsum = np.concatenate([[0], np.cumsum(nc)])
    neighbors = [g['faces']['neighbors'].copy() for g in Gs]
    for i in range(len(neighbors)):
        idx = neighbors[i] >= 0
        neighbors[i][idx] = neighbors[i][idx] + nc_cumsum[i]
    G['faces']['neighbors'] = np.vstack(neighbors)

    geoface = ('areas', 'normals', 'centroids', 'surfaces')
    for gname in geoface:
        try:
            G['faces'][gname] = np.concatenate([g['faces'][gname] for g in Gs])
        except (KeyError, ValueError):
            pass

    grdID_f = np.concatenate([np.full(int(n), i + 1, dtype=np.int64)
                              for i, n in enumerate(nf)])
    G['faces']['grdID'] = grdID_f

    # G.nodes
    G['nodes'] = {}
    G['nodes']['num'] = int(nn.sum())
    G['nodes']['coords'] = np.vstack([g['nodes']['coords'] for g in Gs])

    grdID_n = np.concatenate([np.full(int(n), i + 1, dtype=np.int64)
                              for i, n in enumerate(nn)])
    G['nodes']['grdID'] = grdID_n

    # G.griddim / G.subGrids / G.type
    griddim = np.unique([g['griddim'] for g in Gs])
    assert len(griddim) == 1, 'The grid dimensions are not consistent'
    G['griddim'] = int(griddim[0])
    G['subGrids'] = Gs
    G['type'] = ['assembleGrids']
    return G
