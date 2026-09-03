"""The parameter table behind FAHM's Parameter tab.

Port of ``FAHM_M.m``'s ``setDefaultParameterLimits`` and the config rows
``StartButtonPushed`` builds from it.

Two things it decides for each tunable quantity:

**Which regions it varies over.** Pore volume and permeability are
tabulated per FIPNUM region; every saturation-function quantity -- the
endpoints and the relative permeabilities -- is per SATNUM, because
that is the region a saturation table belongs to. A deck without those
arrays has one region and one row.

**What its limits mean.** In *relative* mode the pair is a multiplier on
the existing value, so ``[0.1, 10]`` on permeability means a tenth to
ten times whatever the cell already has. In *absolute* mode the same
pair is applied to the region's own range to give a fixed interval:
``min(value in region) * lb`` to ``max(value in region) * ub``. The
defaults differ per quantity and are MRST's, not invented here -- pore
volume gets a deliberately narrow +-5%, permeability a decade either
way, and the saturation endpoints ranges chosen so a tuned curve stays
physical.

**A defect reproduced from MRST.** Multiplying by the value makes no
sense for a quantity that can legitimately *be* zero. Connate gas is
usually zero, and ``Sgl``'s multipliers are ``(0.0, 1.0)``, so absolute
mode gives it the empty interval ``[0, 0]`` -- the endpoint is pinned at
zero and cannot be tuned at all. ``Swl`` has the same multipliers and
escapes only because connate water is rarely zero. MRST does this and
runs, so this does too; use relative mode for the connate endpoints.
"""

import numpy as _np

#: (region kind, lower, upper) per parameter, exactly as
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
    'Sgl':   ('saturation', 0.0, 1.0),
    'Swcr':  ('saturation', 1.0, 1.5),
    'Sgcr':  ('saturation', 1.0, 1.5),
    'Sowcr': ('saturation', 0.8, 1.2),
    'Sogcr': ('saturation', 0.8, 1.2),
    'Swu':   ('saturation', 0.8, 1.0),
    'Sgu':   ('saturation', 0.8, 1.0),
}

#: The saturation-function quantities, which read their values from the
#: model's relperm scaling points rather than from rock or operators.
SCALING_PARAMETERS = ('krw', 'kro', 'krg', 'Swl', 'Sgl', 'Swcr', 'Sgcr',
                      'Sowcr', 'Sogcr', 'Swu', 'Sgu')

#: What each parameter is called downstream, where names are lower case.
BACKEND_NAME = {
    'Porv': 'porevolume', 'PermX': 'permx', 'PermY': 'permy',
    'PermZ': 'permz',
}

MILLI_DARCY = 9.869232667160128e-16


def region_vector(model, kind):
    """Port of the FIPNUM / SATNUM lookup, defaulting to one region."""
    nc = int(model.G['cells']['num']) if model is not None else 1
    rock = getattr(model, 'rock', None) if model is not None else None
    regions = rock.get('regions') if isinstance(rock, dict) else None
    values = (regions or {}).get(kind)
    if values is None:
        return _np.ones(nc, dtype=int)
    return _np.atleast_1d(_np.asarray(values, dtype=int)).ravel()


def parameter_values(model, name):
    """The current value of ``name`` per cell, in the units the table
    shows -- permeability in millidarcy, as FAHM displays it."""
    if model is None:
        return None

    if name == 'Porv':
        operators = getattr(model, 'operators', None) or {}
        pv = operators.get('pv')
        if pv is None:
            pv = model._porevolume_vector()
        return _np.atleast_1d(_np.asarray(pv, dtype=float)).ravel()

    if name in ('PermX', 'PermY', 'PermZ'):
        perm = _np.atleast_2d(_np.asarray(model.rock['perm'], dtype=float))
        column = ('PermX', 'PermY', 'PermZ').index(name)
        if perm.shape[1] == 1:
            # An isotropic rock stores one column, and all three
            # directions read it. MRST indexes rock.perm(:,ix)
            # unconditionally and errors here; since that cannot run, the
            # evident intent is filled in rather than reproduced.
            column = 0
        return perm[:, column] / MILLI_DARCY

    if name in SCALING_PARAMETERS:
        from PRSTCore.hm.utils.getRelpermScalingPoints import (
            as_dict, getRelpermScalingPoints)
        scaling = as_dict(getRelpermScalingPoints(model))
        for key, values in scaling.items():
            if str(key).lower() == name.lower():
                return _np.atleast_1d(_np.asarray(values,
                                                  dtype=float)).ravel()
    return None


def default_limits(model, name, absolute=False):
    """Port of ``setDefaultParameterLimits``: one row per region.

    Returns a list of ``(region, minimum, maximum)``. In relative mode
    every row carries the same multiplier pair; in absolute mode each
    row is scaled by its own region's range.
    """
    kind, lb, ub = DEFAULTS[name]
    region = region_vector(model, kind)
    ids = _np.unique(region)

    if not absolute:
        return [(int(r), float(lb), float(ub)) for r in ids]

    values = parameter_values(model, name)
    if values is None or values.size != region.size:
        # Nothing to scale against; fall back to the multipliers rather
        # than inventing an interval.
        return [(int(r), float(lb), float(ub)) for r in ids]

    rows = []
    for r in ids:
        inside = values[region == r]
        rows.append((int(r), float(_np.min(inside) * lb),
                     float(_np.max(inside) * ub)))
    return rows


def config_row(name, enabled, absolute, limits):
    """One row of FAHM's ``app.config``.

    The columns ``StartButtonPushed`` reads::

        name, enabled, scaling, boxLims, relativeLimits, subset,
        uniformLimits

    A parameter in relative mode supplies ``relativeLimits`` and leaves
    ``boxLims`` empty, and the other way round -- which is what
    ``addParameter`` distinguishes on. Permeability is tuned
    logarithmically because it ranges over decades; everything else is
    linear.
    """
    pairs = [(lo, hi) for _r, lo, hi in limits]
    scaling = 'log' if name.startswith('Perm') else 'linear'
    return (BACKEND_NAME.get(name, name.lower()), bool(enabled), scaling,
            pairs if absolute else None,
            None if absolute else pairs,
            None, False)
