"""Port of MRST ``getCapPressScalingPoints.m`` (mrst-2026a/hm/utils).

The capillary-pressure counterpart of ``getRelpermScalingPoints``: unlike
that one it has no tabulated fallback, so it returns only the endpoint
keywords the deck's PROPS section actually carries.
"""

import numpy as _np

VALID_KEYWORDS = ('SWLPC', 'PCW', 'SGLPC', 'PCG')


def getCapPressScalingPoints(model):
    """Return a list of ``(keyword, per-cell values)`` pairs."""
    scaling = []
    deck = getattr(model, 'inputdata', None)
    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    if not props:
        return scaling

    nc = int(model.G['cells']['num'])
    act = model.G['cells'].get('indexMap')
    act = (_np.arange(nc, dtype=int) if act is None
           else _np.asarray(act, dtype=int).ravel())

    for field in props:
        if field not in VALID_KEYWORDS:
            continue
        raw = _np.asarray(props[field], dtype=float).ravel()
        v = raw[act] if raw.size > int(act.max(initial=-1)) else raw[:nc]
        if v.size == nc:
            scaling.append((field, v))
    return scaling


def as_dict(scaling):
    """The same content keyed by keyword."""
    return {name: values for name, values in scaling}
