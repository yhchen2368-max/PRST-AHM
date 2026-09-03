"""Port of MRST ``initRelpermScaling.m``
(mrst-2026a/model-io/deckformat/params/rock).

Reads the relative-permeability endpoint-scaling keywords from a deck into
per-cell arrays, for both the drainage (no prefix) and imbibition (``I``
prefix) branches.

Each phase gets an ``(nc, 4)`` array of ``[connate, critical, max-sat,
max-kr]``.  A keyword the deck does not carry stays ``NaN``, which
``SaturationProperty.getPair`` reads as "fall back to the tabulated value".
"""

import numpy as _np

PHASES = ('W', 'OW', 'OG', 'G')


def initRelpermScaling(deck, nc):
    """Return ``{'drainage': {...}, 'imbibition': {...}}``."""
    drain, _ = _getThreePhaseScaling(deck, nc, '')
    imb, _ = _getThreePhaseScaling(deck, nc, 'I')
    # ``mis`` is MRST-0's `% edited by zhang` third table: the saturation
    # functions that apply at full surfactant concentration, keyed by the
    # ``S`` prefix (SSWL, SSWCR, ...). Leaving it out means a surfactant
    # model's scalers have nowhere to live -- imposeRelpermScaling accepts
    # those keywords and would write them into a table that is not there.
    mis, _ = _getThreePhaseScaling(deck, nc, 'S')
    return {'drainage': drain, 'imbibition': imb, 'miscible': mis}


def _getThreePhaseScaling(deck, nc, prefix):
    out, present = {}, False
    for phase in PHASES:
        pts, ok = _getRelPermScaling(deck, nc, prefix, phase)
        out[phase.lower()] = pts
        present = present or ok
    return out, present


def _getRelPermScaling(deck, nc, prefix, phase):
    """``[<p>S<ph>L, <p>S<ph>CR, <p>S<ph>U, <p>KR<ph>, <p>KR<ph>R]``.

    Five columns, not four. The last is the *residual-endpoint* relative
    permeability -- KRWR/KRGR for the water and gas curves, KRORW/KRORG
    for the two oil curves -- which three-point scaling needs and which a
    four-column table simply does not have room for.

    The oil curves take their maximum from KRO rather than KROW/KROG,
    matching the MATLAB's ``maxv`` construction.
    """
    pts = _np.full((int(nc), 5), _np.nan, dtype=float)

    connate = '%sS%sL' % (prefix, phase)
    crit = '%sS%sCR' % (prefix, phase)
    maxs = '%sS%sU' % (prefix, phase)
    if phase in ('OW', 'OG'):
        maxv = '%sKRO' % prefix
        crtv = '%sKROR%s' % (prefix, phase[-1])
    else:
        maxv = '%sKR%s' % (prefix, phase)
        crtv = '%sKR%sR' % (prefix, phase)

    props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
    present = False
    for i, field in enumerate((connate, crit, maxs, maxv, crtv)):
        if field in props:
            values = _np.asarray(props[field], dtype=float).ravel()
            if values.size == 1:
                pts[:, i] = values[0]
            else:
                n = min(values.size, int(nc))
                pts[:n, i] = values[:n]
            present = True
    return pts, present
