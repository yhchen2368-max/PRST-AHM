"""FAHM ModelParameter.m local-coordinate PERM -> half-face -> T chain.

Face indices and active cells are zero-based; cardinal half-face tags remain
MATLAB's 1..6 (x-, x+, y-, y+, z-, z+). No connectivity is regenerated.
"""
import re
import numpy as np
import scipy.sparse as sp


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def half_cells(G):
    return np.repeat(np.arange(G['cells']['num']), np.diff(G['cells']['facePos']))


def directional_trans(model, column, direction):
    G, rock = field(model, 'G'), field(model, 'rock')
    cf = np.asarray(G['cells']['faces'], dtype=int)
    cn = half_cells(G)
    tags = (cf[:, 1] + 1) // 2 - 1
    cp = G['cells'].get('cpgeometry', {})
    dim = np.asarray(G['nodes']['coords']).shape[1] if 'nodes' in G else 0
    has_cp = (np.shape(cp.get('centroids')) == (G['cells']['num'], dim)
              and np.shape(cp.get('facecentroids')) == (len(cf), dim))
    centers = cp['centroids'] if has_cp else G['cells']['centroids']
    faces = cp['facecentroids'] if has_cp else np.asarray(G['faces']['centroids'])[cf[:, 0]]
    C = faces - np.asarray(centers)[cn]
    sign = 2 * (cn == np.asarray(G['faces']['neighbors'])[cf[:, 0], 0]) - 1
    N = np.asarray(G['faces']['normals'])[cf[:, 0]] * sign[:, None]
    values = np.asarray(column.val if hasattr(column, 'val') else column).ravel(order='F')
    half = np.where(tags == direction, values[cn], 0) * np.sum(C * N, axis=1) / np.sum(C * C, axis=1)
    half = np.abs(half)
    if 'ntg' in rock and cf.shape[1] == 2:
        ntg = np.asarray(rock['ntg']).ravel()[cn].copy()
        ntg[tags == 2] = 1
        half *= ntg
    if hasattr(column, 'val'):
        from PRSTCore.ad_core.adi import ad_select
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = column / values
        ratio = ad_select(np.isnan(ratio.val), 0.0, ratio)
        return ratio[cn] * half
    return half


def _direction(name):
    name = re.sub(r'\W', '_', name.lower())
    return {'x_': 1, 'i_': 1, 'x': 2, 'i': 2,
            'y_': 3, 'j_': 3, 'y': 4, 'j': 4,
            'z_': 5, 'k_': 5, 'z': 6, 'k': 6}[name]


def cartesian_multipliers(G, multipliers):
    cf = np.asarray(G['cells']['faces'], dtype=int)
    cn = half_cells(G)
    result = np.ones(G['faces']['num'])
    for name, raw in multipliers.items():
        values = np.asarray(raw, dtype=float).ravel(order='F')
        if values.size == int(np.prod(G['cartDims'])) and values.size != G['cells']['num']:
            values = values[np.asarray(G['cells']['indexMap'])]
        if values.size != G['cells']['num']:
            raise ValueError('Cartesian multiplier must contain active or full-grid values')
        values = np.where(np.isfinite(values), values, 1.0)
        mask = cf[:, 1] == _direction(name)
        np.multiply.at(result, cf[mask, 0], values[cn[mask]])
    return result[cf[:, 0]]


def fault_multipliers(G, grid):
    """processFaults: select actual cell-face tags, union per named fault."""
    grid = {k.lower(): v for k, v in grid.items()}
    spec, mult = grid.get('faults', []), grid.get('multflt', [])
    result = np.ones(G['faces']['num'])
    if not len(spec) or not len(mult):
        return result
    cf = np.asarray(G['cells']['faces'], dtype=int)
    cn = half_cells(G)
    ijk = np.column_stack(np.unravel_index(G['cells']['indexMap'], G['cartDims'], order='F')) + 1
    factors = {row[0]: float(row[1]) for row in mult}
    for name in sorted({row[0] for row in spec}):
        value = factors.get(name, np.nan)
        if not np.isfinite(value):
            continue
        selected = np.zeros(G['faces']['num'], dtype=bool)
        for row in spec:
            if row[0] != name:
                continue
            lo = np.asarray(row[1:7:2], dtype=int)
            hi = np.asarray(row[2:7:2], dtype=int)
            cells = np.all((ijk >= lo) & (ijk <= hi), axis=1)
            direction = _direction(str(row[7]).replace('+', ''))
            mask = cells[cn] & (cf[:, 1] == direction)
            selected[cf[mask, 0]] = True
        result[selected] *= value
    return result


def assemble_trans(model, columns):
    G, rock, ops = field(model, 'G'), field(model, 'rock'), field(model, 'operators')
    half = sum(directional_trans(model, col, k) for k, col in enumerate(columns))
    if rock.get('multipliers'):
        half = half * cartesian_multipliers(G, rock['multipliers'])
    cf = np.asarray(G['cells']['faces'])[:, 0].astype(int)
    M = sp.csr_matrix((np.ones(cf.size), (cf, np.arange(cf.size))), shape=(G['faces']['num'], cf.size))
    internal = np.asarray(ops['internalConn'])
    if internal.dtype != bool:
        raise ValueError('operators.internalConn must be a logical face/connection mask')
    extra = None
    if 'nnc' in G:
        nc = np.asarray(G['nnc']['cells'], dtype=int)
        neighbors = {tuple(sorted(row)) for row in G['faces']['neighbors']}
        use = np.array([np.all(row >= 0) and tuple(sorted(row)) not in neighbors for row in nc])
        extra = np.asarray(G['nnc']['trans']).ravel()[use]
        if extra.size:
            internal = internal[:-extra.size]
    if internal.size != G['faces']['num']:
        raise ValueError('internalConn size differs from faces plus retained NNC')
    face_ids = np.asarray(ops.get('internalFaceIndices', np.flatnonzero(internal)), dtype=int)
    if not np.array_equal(np.sort(face_ids), np.flatnonzero(internal)):
        raise ValueError('internalFaceIndices must be a permutation of internalConn')
    with np.errstate(divide='ignore', invalid='ignore'):
        if hasattr(half, 'val'):
            from PRSTCore.ad_core.adi import ad_select
            closed = np.asarray(M @ (half.val == 0)) > 0
            # Zero, fixed permeability closes a face. Avoid inf*0 in its
            # sparse derivative; selected positive-cell parameters cannot
            # reopen that fixed half-face. MRST's sparse AD returns zero.
            rec = 1.0 / ad_select(half.val == 0, 1.0, half)
        else:
            rec = 1.0 / half
        mapped = rec.linear_map(M[face_ids]) if hasattr(rec, 'val') else M[face_ids] @ rec
        trans = 1.0 / mapped
        if hasattr(half, 'val'):
            trans = ad_select(closed[face_ids], 0.0, trans)
    deck = field(model, 'inputdata')
    if deck and 'GRID' in deck:
        trans = trans * fault_multipliers(G, deck['GRID'])[face_ids]
    if extra is not None:
        if hasattr(trans, 'val'):
            from PRSTCore.ad_core.adi import SparseADI
            trans = SparseADI.concat([trans, SparseADI.constant(extra, trans.nvar)])
        else:
            trans = np.concatenate([trans, extra])
    return trans
