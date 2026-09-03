"""Partial port of MRST initEclipseRock.

This file implements extraction of simple rock fields (`perm`, `poro`,
`ntg`) from the parsed deck and attaches multipliers/faultdata if present.
It intentionally implements a conservative subset sufficient for SPE9
initialization; more MRST behaviors will be added incrementally.
"""
from typing import Any, Dict
import numpy as np


def init_eclipse_rock(deck: Dict[str, Any]) -> Dict[str, Any]:
    rock = {}
    grid = deck.get('GRID', {})

    def _clean_array(key, default=None, ncol=1):
        """Extract a numeric array from grid key, filtering out non-numeric tokens."""
        val = grid.get(key)
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return np.array([[float(val)]])
        if isinstance(val, np.ndarray):
            return val.reshape(-1, ncol)
        # List of mixed tokens: keep only numeric entries
        nums = []
        for v in val:
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                continue  # skip keyword strings and other non-numeric tokens
        if not nums:
            return default
        return np.array(nums, dtype=float).reshape(-1, ncol)

    # Permeability tensor.  MRST initEclipseRock preserves the directional
    # PERMX/PERMY/PERMZ values; collapsing to PERMX changes vertical flow
    # and gives a different SPE1 transmissibility matrix.
    if 'PERM' in grid and not any(key in grid for key in ('PERMX', 'PERMY', 'PERMZ')):
        perm = _clean_array('PERM', ncol=1)
        rock['perm'] = perm
    elif 'PERMX' in grid:
        kx = _clean_array('PERMX', ncol=1).ravel()
        ky = _clean_array('PERMY', ncol=1)
        kz = _clean_array('PERMZ', ncol=1)
        ky = kx.copy() if ky is None else ky.ravel()
        kz = kx.copy() if kz is None else kz.ravel()
        if ky.size == 1 and kx.size > 1:
            ky = np.full(kx.size, float(ky[0]))
        if kz.size == 1 and kx.size > 1:
            kz = np.full(kx.size, float(kz[0]))
        if ky.size != kx.size or kz.size != kx.size:
            raise ValueError('PERMX/PERMY/PERMZ sizes must agree')
        rock['perm'] = np.column_stack((kx, ky, kz))
    else:
        nc = int(np.prod(grid.get('cartDims', [1, 1, 1])))
        rock['perm'] = np.ones((nc, 1))

    # PORO
    if 'PORO' in grid:
        rock['poro'] = _clean_array('PORO', ncol=1)
    else:
        rock['poro'] = np.ones((rock['perm'].shape[0], 1)) * 0.2

    # NTG
    if 'NTG' in grid:
        rock['ntg'] = _clean_array('NTG', ncol=1)

    # Multipliers: only retain numeric GRID multiplier arrays. Eclipse
    # control keywords such as MULTREGT/MULTPVN may contain string tokens.
    multipliers = {}
    for k in list(grid.keys()):
        if k.startswith('MULT'):
            raw = grid[k]
            try:
                arr = _clean_array(k, default=None, ncol=1)
            except (TypeError, ValueError):
                arr = None
            if arr is not None:
                comp = k[4:].lower()
                multipliers[comp] = arr
    if multipliers:
        rock['multipliers'] = multipliers

    # Fault multipliers
    if 'FAULTS' in grid and 'MULTFLT' in grid:
        rock['faultdata'] = {'faults': grid['FAULTS'], 'multflt': grid['MULTFLT']}

    # ROCK keyword: compressibility and pref
    if 'PROPS' in deck and 'ROCK' in deck['PROPS']:
        rock_kw = deck['PROPS']['ROCK']
        # MRST expects rock = [pref cr]; approximate extraction
        try:
            rock['pref'] = float(rock_kw[0])
            rock['cr'] = float(rock_kw[1]) if len(rock_kw) > 1 else 0.0
        except Exception:
            rock['pref'] = 0.0
            rock['cr'] = 0.0

    return rock
