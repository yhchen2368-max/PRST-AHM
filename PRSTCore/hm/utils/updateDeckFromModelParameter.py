"""Port of MRST ``updateDeckFromModelParameter.m`` (mrst-2026a/hm/utils).

Writes a tuned parameter vector back into the deck keywords it came from,
so the calibrated model can be exported as a runnable ECLIPSE case.

Each parameter's value lands in the deck section that owns it: pore volume
in EDIT, permeability in GRID, saturation endpoints in PROPS. A keyword the
deck does not yet carry is first materialised full-length from the fluid's
tabulated ``krPts`` (indexed by SATNUM), then overwritten on the active
cells -- so inactive cells keep a physically meaningful default.
"""

import numpy as _np

_EDIT_FIELDS = ('porv',)
_GRID_FIELDS = ('permx', 'permy', 'permz')
_PROPS_FIELDS = ('swl', 'swcr', 'swu', 'sowcr', 'sogcr', 'sgl', 'sgcr',
                 'sgu', 'krw', 'kro', 'krg')


def updateDeckFromModelParameter(deck, setup, parameters):
    """Return ``deck`` with each parameter's current value written back."""
    dims = _np.asarray(deck['RUNSPEC']['cartDims'], dtype=int).ravel()
    n_total = int(_np.prod(dims))

    model = setup['model'] if isinstance(setup, dict) else setup.model
    act = model.G['cells'].get('indexMap')
    act = (_np.arange(int(model.G['cells']['num']), dtype=int)
           if act is None else _np.asarray(act, dtype=int).ravel())

    pts = _krpts(model)

    regions = deck.get('REGIONS', {}) or {}
    if 'SATNUM' in regions:
        reg = _np.asarray(regions['SATNUM'], dtype=int).ravel() - 1
    else:
        reg = _np.zeros(n_total, dtype=int)

    for p in parameters:
        name = _name(p)
        value = _value_of(p, setup)
        field = 'porv' if name == 'porevolume' else name

        if field in _EDIT_FIELDS:
            section = deck.setdefault('EDIT', {})
        elif field in _GRID_FIELDS:
            section = deck.setdefault('GRID', {})
        elif field in _PROPS_FIELDS:
            section = deck.setdefault('PROPS', {})
        else:
            continue

        key = field.upper()
        if key not in section:
            section[key] = _initial_field(field, p, pts, reg, n_total)
        arr = _np.asarray(section[key], dtype=float).ravel().copy()
        arr[act] = _np.asarray(value, dtype=float).ravel()
        section[key] = arr
    return deck


def _initial_field(field, p, pts, reg, n_total):
    """A keyword absent from the deck starts from the tabulated krPts.

    MATLAB reads ``getfield(pts, p.location{4}, p.location{5})``, i.e. the
    phase and column the parameter points at, then expands it by region.
    Only the PROPS endpoints have such a table; the others start at zero.

    ``location{5}`` is a *subscript*, ``{':', col}`` -- MATLAB's way of
    saying "column col of that table". PRSTCore spells the same thing
    ``(slice(None), col)``, so the column index is the subscript's last
    element rather than the element itself.
    """
    if field not in _PROPS_FIELDS:
        return _np.zeros(n_total, dtype=float)
    location = _location(p)
    if len(location) >= 5 and pts:
        phase, subscript = location[3], location[4]
        table = pts.get(phase) if isinstance(pts, dict) else getattr(pts, phase, None)
        if table is not None:
            table = _np.atleast_2d(_np.asarray(table, dtype=float))
            col = _column_of(subscript)
            if col is None or col >= table.shape[1]:
                return _np.zeros(n_total, dtype=float)
            values = table[:, col]
            return values[_np.clip(reg, 0, values.size - 1)]
    return _np.zeros(n_total, dtype=float)


def _column_of(subscript):
    """The integer column a location's final subscript selects."""
    if isinstance(subscript, (tuple, list)):
        for entry in reversed(subscript):
            if isinstance(entry, (int, _np.integer)) and not isinstance(entry, bool):
                return int(entry)
        return None
    if isinstance(subscript, (int, _np.integer)) and not isinstance(subscript, bool):
        return int(subscript)
    return None


def _krpts(model):
    fluid = getattr(model, 'fluid', None)
    if isinstance(fluid, dict):
        return fluid.get('krPts', {}) or {}
    return getattr(fluid, 'krPts', {}) or {}


def _name(p):
    return str(p['name'] if isinstance(p, dict) else p.name).lower()


def _location(p):
    loc = p['location'] if isinstance(p, dict) else getattr(p, 'location', [])
    return list(loc or [])


def _value_of(p, setup):
    """``p.getfun(setup.(p.belongsTo), p.location{:})``."""
    getfun = p['getfun'] if isinstance(p, dict) else getattr(p, 'getfun')
    belongs = p['belongsTo'] if isinstance(p, dict) else getattr(p, 'belongsTo')
    owner = setup[belongs] if isinstance(setup, dict) else getattr(setup, belongs)
    return getfun(owner, *_location(p))
