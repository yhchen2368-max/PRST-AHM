"""Port of MRST ``getRelpermScalingPoints.m`` (mrst-2026a/hm/utils).

Collects a model's relative-permeability scaling points as the flat
``{name, value, name, value, ...}`` list MRST's parameter machinery
consumes -- first from the fluid's tabulated ``krPts`` (indexed per cell by
the saturation region), then overridden cell-wise by whatever endpoint
keywords the input deck's PROPS section carries.

The MATLAB cell array becomes a Python list of ``(name, values)`` pairs;
``as_dict`` gives the same content keyed by name.
"""

import numpy as _np

# Column order of fluid.krPts.<phase>: [connate, critical, max-sat, max-kr].
_L, _CR, _U, _KM = 0, 1, 2, 3

VALID_KEYWORDS = (
    'SWL', 'SWCR', 'SWU', 'SGL', 'SGCR', 'SGU', 'SOWCR', 'SOGCR',
    'KRW', 'KRO', 'KRG', 'KRWR', 'KRORW', 'KRORG', 'KRGR',
    'ISWL', 'ISWCR', 'ISWU', 'ISGL', 'ISGCR', 'ISGU', 'ISOWCR', 'ISOGCR',
    'IKRW', 'IKRO', 'IKRG', 'IKRWR', 'IKRORW', 'IKRORG', 'IKRGR',
    'SSWL', 'SSWCR', 'SSWU', 'SSGL', 'SSGCR', 'SSOWCR', 'SSOGCR',
    'SKRW', 'SKRO',
)


def getRelpermScalingPoints(model):
    """Return a list of ``(keyword, per-cell values)`` pairs."""
    nc = int(model.G['cells']['num'])
    pts = _krpts(model)

    regions = {}
    rock = getattr(model, 'rock', None)
    if isinstance(rock, dict):
        regions = rock.get('regions', {}) or {}
    reg = regions.get('saturation')
    reg = (_np.zeros(nc, dtype=int) if reg is None
           else _np.asarray(reg, dtype=int).ravel() - 1)

    scaling = []
    if 'w' in pts:
        w = _rows(pts['w'], reg)
        scaling += [('SWL', w[:, _L]), ('SWCR', w[:, _CR]),
                    ('SWU', w[:, _U]), ('KRW', w[:, _KM])]
    if 'ow' in pts:
        ow = _rows(pts['ow'], reg)
        scaling += [('SOWCR', ow[:, _CR]), ('KRO', ow[:, _KM])]
    if 'og' in pts:
        og = _rows(pts['og'], reg)
        # KRO is shared between the ow and og curves: the MATLAB only adds
        # it from og when ow did not already supply it.
        if any(name == 'KRO' for name, _ in scaling):
            scaling += [('SOGCR', og[:, _CR])]
        else:
            scaling += [('SOGCR', og[:, _CR]), ('KRO', og[:, _KM])]
    if 'g' in pts:
        g = _rows(pts['g'], reg)
        scaling += [('SGL', g[:, _L]), ('SGCR', g[:, _CR]),
                    ('SGU', g[:, _U]), ('KRG', g[:, _KM])]

    # A coarse/extracted grid keeps only the tabulated points.
    if isinstance(model.G, dict) and model.G.get('parent') is not None:
        return scaling

    deck = getattr(model, 'inputdata', None)
    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    if not props:
        return scaling

    act = model.G['cells'].get('indexMap')
    act = (_np.arange(nc, dtype=int) if act is None
           else _np.asarray(act, dtype=int).ravel())

    names = [name for name, _ in scaling]
    for field in props:
        if field not in VALID_KEYWORDS:
            continue
        raw = _np.asarray(props[field], dtype=float).ravel()
        v = raw[act] if raw.size > int(act.max(initial=-1)) else raw[:nc]
        if v.size != nc:
            continue
        if field in names:
            # Only the finite deck entries override the tabulated value.
            ix = names.index(field)
            merged = _np.asarray(scaling[ix][1], dtype=float).copy()
            replace = _np.isfinite(v)
            merged[replace] = v[replace]
            scaling[ix] = (field, merged)
        else:
            scaling.append((field, v))
            names.append(field)
    return scaling


def as_dict(scaling):
    """The same content keyed by keyword."""
    return {name: values for name, values in scaling}


def _krpts(model):
    fluid = getattr(model, 'fluid', None)
    if isinstance(fluid, dict):
        return fluid.get('krPts', {}) or {}
    return getattr(fluid, 'krPts', {}) or {}


def _rows(table, reg):
    """``pts.<phase>(reg, :)`` -- one row per cell, selected by region."""
    table = _np.atleast_2d(_np.asarray(table, dtype=float))
    reg = _np.clip(reg, 0, table.shape[0] - 1)
    return table[reg, :]
