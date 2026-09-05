"""FAHM parameter-table and seven-column ``app.config`` contract.

This module is a direct port of ``FAHM.m``'s ``setDefaultParameterLimits``
and dependent ``get.config`` property.  The UI stores one row per FIPNUM or
SATNUM region, while the optimizer-facing config stores one limit pair per
active cell (or per positive rock entry after applying ``subset``).

MATLAB's seven columns are represented by a Python tuple in exactly this
order::

    name, include, scaling, boxLims, relativeLimits, subset, uniformLimits

Array-valued columns are owned NumPy arrays.  Cell indices in ``subset`` are
zero-based at the Python boundary; their order is MATLAB ``find`` order.
"""

from __future__ import annotations

import math as _math

import numpy as _np


#: ``trainRock`` followed by ``trainFluid`` in FAHM.
PARAMETERS = ('Porv', 'PermX', 'PermY', 'PermZ', 'krw', 'kro', 'krg',
              'Swl', 'Swcr', 'Swu', 'Sowcr', 'Sgl', 'Sgcr', 'Sgu',
              'Sogcr')

#: (region kind, lower, upper), exactly as
#: ``setDefaultParameterLimits`` assigns them.
DEFAULTS = {
    'Porv':  ('fluid', 0.95, 1.05),
    'PermX': ('fluid', 0.10, 10.0),
    'PermY': ('fluid', 0.10, 10.0),
    'PermZ': ('fluid', 0.10, 10.0),
    'krw':   ('saturation', 0.5, 2.0),
    'kro':   ('saturation', 0.5, 2.0),
    'krg':   ('saturation', 0.5, 2.0),
    'Swl':   ('saturation', 0.0, 1.0),
    'Swcr':  ('saturation', 1.0, 1.5),
    'Swu':   ('saturation', 0.8, 1.0),
    'Sowcr': ('saturation', 0.8, 1.2),
    'Sgl':   ('saturation', 0.0, 1.0),
    'Sgcr':  ('saturation', 1.0, 1.5),
    'Sgu':   ('saturation', 0.8, 1.0),
    'Sogcr': ('saturation', 0.8, 1.2),
}

#: Saturation-function quantities read by ``getRelpermScalingPoints``.
SCALING_PARAMETERS = ('krw', 'kro', 'krg', 'Swl', 'Swcr', 'Swu',
                      'Sowcr', 'Sgl', 'Sgcr', 'Sgu', 'Sogcr')

#: Downstream ModelParameter names.
BACKEND_NAME = {
    'Porv': 'porevolume', 'PermX': 'permx', 'PermY': 'permy',
    'PermZ': 'permz',
}

#: SI value of one millidarcy, matching MRST's ``milli*darcy``.
MILLI_DARCY = 9.869232667160128e-16

#: Phase conditions from ``ModelProceedButtonPushed``.  Rock parameters are
#: always available and therefore do not appear here.
PHASE_REQUIREMENTS = {
    'kro': frozenset({'Oil'}),
    'krw': frozenset({'Water'}),
    'Swu': frozenset({'Water'}),
    'Swl': frozenset({'Water'}),
    'Swcr': frozenset({'Water'}),
    'krg': frozenset({'Gas'}),
    'Sgu': frozenset({'Gas'}),
    'Sgl': frozenset({'Gas'}),
    'Sgcr': frozenset({'Gas'}),
    'Sowcr': frozenset({'Water', 'Oil'}),
    'Sogcr': frozenset({'Gas', 'Oil'}),
}


def phase_parameter_availability(*, oil: bool, water: bool,
                                 gas: bool) -> dict[str, bool]:
    """Return the 15 checkbox availability flags used by Model Proceed."""
    active = {name for name, present in (
        ('Oil', oil), ('Water', water), ('Gas', gas)) if present}
    return {
        name: (True if name not in PHASE_REQUIREMENTS
               else PHASE_REQUIREMENTS[name].issubset(active))
        for name in PARAMETERS
    }


def _n_cells(model) -> int:
    if model is None:
        raise ValueError('FAHM parameter config requires a created model')
    try:
        nc = int(model.G['cells']['num'])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('model.G.cells.num is required for FAHM config') \
            from exc
    if nc < 0:
        raise ValueError('model.G.cells.num must be non-negative')
    return nc


def _vector(values, *, name: str, size: int, dtype=float) -> _np.ndarray:
    out = _np.asarray(values, dtype=dtype).reshape(-1, order='F')
    if out.size != size:
        raise ValueError('%s has %d entries; expected %d active cells'
                         % (name, out.size, size))
    return out


def _rock(model) -> dict:
    rock = getattr(model, 'rock', None)
    if not isinstance(rock, dict):
        raise ValueError('model.rock must be a mapping')
    return rock


def region_vector(model, kind: str) -> _np.ndarray:
    """Port FIPNUM/SATNUM lookup, already indexed onto active cells."""
    nc = _n_cells(model)
    regions = _rock(model).get('regions')
    values = (regions or {}).get(kind) if isinstance(regions, dict) else None
    if values is None:
        return _np.ones(nc, dtype=int)
    return _vector(values, name='%s region vector' % kind,
                   size=nc, dtype=int)


def _permeability(model, name: str) -> _np.ndarray:
    nc = _n_cells(model)
    raw = _np.asarray(_rock(model)['perm'], dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    elif raw.ndim != 2:
        raise ValueError("model.rock['perm'] must be a two-dimensional array")
    if raw.shape[0] != nc:
        raise ValueError("model.rock['perm'] has %d rows; expected %d active cells"
                         % (raw.shape[0], nc))
    column = ('PermX', 'PermY', 'PermZ').index(name)
    if raw.shape[1] <= column:
        # FAHM indexes rock.perm(:, ix) without inventing an isotropic
        # fallback.  Stage 5 result models have all three columns.
        raise IndexError("%s requires permeability column %d, got shape %s"
                         % (name, column + 1, raw.shape))
    return raw[:, column].copy()


def parameter_values(model, name: str) -> _np.ndarray | None:
    """Current per-cell values in the units displayed by the FAHM table.

    Permeability is converted from SI to millidarcy here, just as
    ``setDefaultParameterLimits`` calls ``convertTo``.
    """
    if model is None:
        return None
    nc = _n_cells(model)

    if name == 'Porv':
        operators = getattr(model, 'operators', None) or {}
        pv = operators.get('pv')
        if pv is None:
            pv = model._porevolume_vector()
        return _vector(pv, name='pore volume', size=nc)

    if name in ('PermX', 'PermY', 'PermZ'):
        return _permeability(model, name) / MILLI_DARCY

    if name in SCALING_PARAMETERS:
        from PRSTCore.hm.utils.getRelpermScalingPoints import (
            as_dict, getRelpermScalingPoints)
        scaling = as_dict(getRelpermScalingPoints(model))
        for key, values in scaling.items():
            if str(key).lower() == name.lower():
                return _vector(values, name=name, size=nc)
        raise KeyError('getRelpermScalingPoints did not provide %s' % name)
    raise KeyError('unknown FAHM parameter %s' % name)


def default_limits(model, name: str, absolute: bool = False):
    """Port ``setDefaultParameterLimits`` (one row per sorted region ID)."""
    kind, lb, ub = DEFAULTS[name]
    region = region_vector(model, kind)
    ids = _np.unique(region)

    if not absolute:
        return [(int(r), float(lb), float(ub)) for r in ids]

    values = parameter_values(model, name)
    rows = []
    for r in ids:
        inside = values[region == r]
        rows.append((int(r), float(_np.min(inside) * lb),
                     float(_np.max(inside) * ub)))
    return rows


def matlab_num2str(value: float) -> str:
    """R2022b scalar ``num2str`` formatting used by the limits UI."""
    value = float(value)
    if not _math.isfinite(value):
        return 'NaN' if _math.isnan(value) else ('-Inf' if value < 0 else 'Inf')
    if value == 0:
        return '0'
    magnitude = abs(value)
    if magnitude < 1e-4:
        return format(value, '.4e')
    decimals = max(4, 4 - int(_math.floor(_math.log10(magnitude))))
    return format(value, '.%df' % decimals).rstrip('0').rstrip('.')


def expand_region_limits(model, name: str, limits) -> _np.ndarray:
    """Expand UI region rows to ``G.cells.num x 2`` in active-cell order."""
    nc = _n_cells(model)
    kind = DEFAULTS[name][0]
    region = region_vector(model, kind)
    table = _np.asarray(list(limits), dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] == 0:
        raise ValueError('%s limits must have shape (nregion, 3)' % name)

    ids, minimum, maximum = table[:, 0], table[:, 1], table[:, 2]
    # MATLAB initializes ``index = ones(nc,1)``.  A malformed table that
    # omits a region therefore uses its first row; the region-ID column is
    # non-editable in the real App, so valid UI state always covers all IDs.
    index = _np.zeros(nc, dtype=int)
    for row, region_id in enumerate(ids):
        index[region == region_id] = row
    return _np.column_stack((minimum[index], maximum[index]))


def parameter_subset(model, name: str) -> _np.ndarray | None:
    """Return Python's zero-based equivalent of the FAHM ``find`` column."""
    nc = _n_cells(model)
    if name == 'Porv':
        values = parameter_values(model, name)
    elif name in ('PermX', 'PermY', 'PermZ'):
        values = _permeability(model, name)
    else:
        return None
    values = _vector(values, name=name, size=nc)
    return _np.flatnonzero(values > 0).astype(_np.int64, copy=False)


def config_row(model, name: str, enabled: bool, absolute: bool, limits):
    """Build one exact seven-column row of FAHM's dependent ``config``."""
    backend_name = BACKEND_NAME.get(name, name.lower())
    if not enabled:
        # MATLAB executes ``continue`` immediately after columns 1 and 2.
        return (backend_name, False, None, None, None, None, None)

    expanded = expand_region_limits(model, name, limits)
    if absolute and name in ('PermX', 'PermY', 'PermZ'):
        # FAHM.m lines 618-619 already convert mD to SI.  Its second common
        # conversion at line 629 is FAHM-FIX-003 and is intentionally not
        # copied: selected PRST behavior performs exactly one conversion.
        expanded = expanded * MILLI_DARCY

    subset = parameter_subset(model, name)
    if subset is not None:
        expanded = expanded[subset, :]

    scaling = 'log' if name in ('PermX', 'PermY', 'PermZ') else 'linear'
    box_limits = expanded.copy() if absolute else None
    relative_limits = None if absolute else expanded.copy()
    return (backend_name, True, scaling, box_limits, relative_limits,
            None if subset is None else subset.copy(), bool(absolute))
