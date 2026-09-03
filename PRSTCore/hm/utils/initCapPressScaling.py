"""Port of MRST ``initCapPressScaling.m`` (mrst-2026a/hm/utils).

Capillary-pressure endpoint scaling, the Pc counterpart of
``initRelpermScaling.m``: two points per phase, the connate saturation
``S<phase>LPC`` and the maximum capillary pressure ``PC<phase>``, read for
both the drainage (no prefix) and imbibition (``I`` prefix) branches.
Absent keywords stay ``NaN``, which downstream code reads as "fall back to
the tabulated value" (``SaturationProperty.getPair``).
"""

import numpy as _np


def initCapPressScaling(deck, nc):
    """Return ``{'drainage': {...}, 'imbibition': {...}}``."""
    drain, ok_d = _getCapPressScaling(deck, nc, '')
    imb, ok_i = _getCapPressScaling(deck, nc, 'I')
    return {'drainage': drain, 'imbibition': imb}


def _getCapPressScaling(deck, nc, prefix):
    w, okw = _getPCScaling(deck, nc, prefix, 'W')
    g, okg = _getPCScaling(deck, nc, prefix, 'G')
    return {'w': w, 'g': g}, (okw or okg)


def _getPCScaling(deck, nc, prefix, phase):
    """``pts(:,1) = <prefix>S<phase>LPC``, ``pts(:,2) = <prefix>PC<phase>``."""
    pts = _np.full((int(nc), 2), _np.nan, dtype=float)
    connate = '%sS%sLPC' % (prefix, phase)
    maxv = '%sPC%s' % (prefix, phase)

    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    present = False
    for i, field in enumerate((connate, maxv)):
        if field in props:
            values = _np.asarray(props[field], dtype=float).ravel()
            if values.size == 1:
                pts[:, i] = values[0]
            else:
                pts[:values.size, i] = values[:int(nc)]
            present = True
    return pts, present
