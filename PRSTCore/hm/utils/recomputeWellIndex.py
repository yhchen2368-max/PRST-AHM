"""Port of MRST ``recomputeWellIndex.m`` (mrst-2026a/hm/utils).

Recomputes every schedule well's productivity index from the current
rock -- the step a history match needs after perturbing permeability --
and then re-applies the deck's WPIMULT multipliers on top.

A perforation is recomputed only where the deck defaulted it:
``WI <= 0 & Kh <= 0 & cstatus``.
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI

from .evaluate.computeWellIndexADI import computeWellIndexADI


def recomputeWellIndex(model, schedule):
    """Return ``schedule`` with each control's ``W[*]['WI']`` refreshed."""
    G = model.G
    dims = _cell_dims(G)

    controls = schedule.get('control', [])
    if not controls:
        return schedule

    first = controls[0].get('W', [])
    n_cells = [len(_np.atleast_1d(_np.asarray(w['cells'])).ravel()) for w in first]
    bounds = _np.concatenate([[0], _np.cumsum(n_cells)]).astype(int)
    index = [_np.arange(bounds[i], bounds[i + 1]) for i in range(len(n_cells))]

    for ctrl in controls:
        W = ctrl.get('W')
        if not W:
            continue
        if not any('defaulted' in w for w in W):
            continue

        r = _cat([w['r'] for w in W])
        direction = _np.concatenate(
            [_np.atleast_1d(_np.asarray(w['dir'])).ravel() for w in W])
        cells = _cat([w['cells'] for w in W]).astype(int)
        cstatus = _cat([w['cstatus'] for w in W]).astype(bool)
        WI = _cat([w['WI'] for w in W])
        Kh = _cat([w['defaulted']['Kh'] for w in W])
        Sk = _cat([w['defaulted']['Skin'] for w in W])
        WI_def = _cat([w['defaulted']['WI'] for w in W])

        compWI = (WI_def <= 0) & (Kh <= 0) & cstatus
        if not _np.any(compWI):
            continue

        WI_comp = computeWellIndexADI(G, model.rock, r, cells, Dir=direction,
                                      Skin=Sk, cellDims=dims, Kh=Kh,
                                      Subset=compWI)
        if isinstance(WI_comp, _SparseADI):
            WI = _SparseADI.constant(_np.asarray(WI, dtype=float), WI_comp.nvar)
            WI = WI + _SparseADI.scatter(_np.flatnonzero(compWI),
                                         WI_comp - WI[_np.flatnonzero(compWI)],
                                         WI.val.size)
            WI = WI * cstatus.astype(float)
        else:
            WI = _np.asarray(WI, dtype=float).copy()
            WI[compWI] = _np.asarray(WI_comp, dtype=float).ravel()
            WI = WI * cstatus.astype(float)
        for i, w in enumerate(W):
            w['WI'] = WI[index[i]]

    _apply_deck_wpimult(model, schedule)
    return schedule


def _apply_deck_wpimult(model, schedule):
    """Re-apply the deck's WPIMULT records over the recomputed indices."""
    deck = getattr(model, 'inputdata', None)
    if not isinstance(deck, dict) or 'SCHEDULE' not in deck:
        return
    ectrls = deck['SCHEDULE'].get('control', [])
    if not isinstance(ectrls, (list, tuple)) or not ectrls:
        return
    if not any(c.get('WPIMULT') for c in ectrls if isinstance(c, dict)):
        return

    IJK = _grid_logical_indices(model.G)
    W_prev = schedule['control'][0].get('W', [])
    WI_prev = [_np.asarray(w['WI'], dtype=float).copy() for w in W_prev]
    WI_raw = [v.copy() for v in WI_prev]

    for ctrl_idx, ctrl in enumerate(schedule['control']):
        wpi = ectrls[ctrl_idx].get('WPIMULT') if ctrl_idx < len(ectrls) else None
        wpi = _dedupe_wpimult(wpi)
        W = ctrl.get('W', [])
        W, WI_raw, WI_prev = _apply_wpimult(W, IJK, wpi, WI_raw, WI_prev)
        ctrl['W'] = W


def _dedupe_wpimult(wpi):
    """Port of the I/J/K == -1 split: keep only the last blanket record
    per (well, I, J, K)."""
    if not wpi:
        return []
    rows = [list(r) for r in wpi if r]
    blanket, specific = [], []
    for row in rows:
        if len(row) >= 5 and row[2] == -1 and row[3] == -1 and row[4] == -1:
            blanket.append(row)
        else:
            specific.append(row)
    if blanket:
        last = {}
        for row in blanket:
            last[(row[0], row[2], row[3], row[4])] = row
        blanket = list(last.values())
    return specific + blanket


def _apply_wpimult(W, IJK, WPIMULT, WI_raw, WI_now):
    """Port of the local ``apply_wpimult``."""
    names = [w.get('name') for w in W]

    # Carry forward any perforation whose index changed since the last
    # control, so a re-completed perforation restarts from its raw value.
    for i, w in enumerate(W):
        if i >= len(WI_raw):
            continue
        current = _np.asarray(w['WI'], dtype=float)
        n = min(current.size, WI_raw[i].size)
        changed = WI_raw[i][:n] != current[:n]
        WI_raw[i][:n][changed] = current[:n][changed]
        WI_now[i][:n][changed] = current[:n][changed]

    for row in WPIMULT or []:
        name, val = row[0], float(row[1])
        I, J, K = (int(row[2]), int(row[3]), int(row[4])) if len(row) >= 5 else (0, 0, 0)
        start = int(row[5]) if len(row) > 5 else 0
        stop = int(row[6]) if len(row) > 6 else 0
        matches = [i for i, n in enumerate(names) if n == name]
        assert len(matches) == 1, 'WPIMULT names exactly one well'
        assert val > 1e-10 and _np.isfinite(val), 'WPIMULT multiplier must be positive'
        well = matches[0]

        N = WI_now[well].size
        wc = _np.atleast_1d(_np.asarray(W[well]['cells'], dtype=int)).ravel()
        if start < 1:
            start = 1
        if stop < 1:
            stop = N
        rng = _np.arange(1, N + 1)
        active = (rng >= start) & (rng <= stop)
        if I > 0:
            active &= IJK[0][wc] == I
        if J > 0:
            active &= IJK[1][wc] == J
        if K > 0:
            active &= IJK[2][wc] == K
        WI_now[well][active] *= val

    for i, w in enumerate(W):
        if i < len(WI_now):
            # The MATLAB writes `w.WI = WI_now{well}.*w.cstatus` and then
            # immediately overwrites it with `w.WI = WI_now{well}`, so the
            # cstatus masking is dead; the unmasked value is what survives.
            w['WI'] = WI_now[i]
    return W, WI_raw, WI_now


def _cat(values):
    return _np.concatenate([_np.atleast_1d(_np.asarray(v)).ravel() for v in values])


def _cell_dims(G):
    """``G.cells.{DX,DY,DZ}`` when present, else the bounding-box dims."""
    cells = G['cells']
    if all(k in cells for k in ('DX', 'DY', 'DZ')):
        return _np.column_stack([_np.asarray(cells[k], dtype=float).ravel()
                                 for k in ('DX', 'DY', 'DZ')])
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _cell_bounding_box_dims
    return _np.asarray(_cell_bounding_box_dims(G, int(cells['num'])), dtype=float)


def _grid_logical_indices(G):
    """Port of ``gridLogicalIndices``: per-cell (I, J, K), one-based."""
    dims = _np.asarray(G['cartDims'], dtype=int).ravel()
    nx, ny = int(dims[0]), int(dims[1])
    index_map = G['cells'].get('indexMap')
    index_map = (_np.arange(int(G['cells']['num']), dtype=int)
                 if index_map is None else _np.asarray(index_map, dtype=int).ravel())
    i = index_map % nx
    j = (index_map // nx) % ny
    k = index_map // (nx * ny)
    return [i + 1, j + 1, k + 1]
