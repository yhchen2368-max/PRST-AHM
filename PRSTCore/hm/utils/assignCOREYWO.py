"""Port of MRST ``assignCOREYWO.m`` (mrst-2026a/hm/utils).

Analytical Corey water/oil relative permeability and capillary pressure,
used in place of a tabulated SWOF when history matching the Corey
exponents and endpoints directly.

The 13-column ``COREYWO`` record, one row per saturation region:

    1  SWL     connate water
    2  SWU     maximum water
    3  SWCR    critical water
    4  SOWCR   critical oil (in water)
    5  krOLW   oil kr at connate water
    6  krORW   oil kr at critical water        (NaN -> two-segment form)
    7  krWR    water kr at critical oil        (NaN -> two-segment form)
    8  krWU    water kr at maximum water
    9  pcOW    entry capillary pressure
    10 nOW     oil Corey exponent
    11 nW      water Corey exponent
    12 np      capillary-pressure exponent     (0 -> no Pc)
    13 SpcO    saturation at which Pc vanishes
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_select as _ad_select

# 0-based column indices for the 1-based MATLAB record above.
SWL, SWU, SWCR, SOWCR = 0, 1, 2, 3
KROLW, KRORW, KRWR, KRWU = 4, 5, 6, 7
PCOW, NOW, NW, NP, SPCO = 8, 9, 10, 11, 12


def assignCOREYWO(f, coreywo, SGL, reg):
    """Attach ``krW``/``krOW``/``pcOW``/``krPts`` to the fluid dict ``f``.

    ``coreywo`` is ``(nreg, 13)``; ``SGL`` is one connate-gas value per
    region; ``reg`` carries ``reg['sat']``, the number of regions.
    """
    krW, krOW, pcOW, pts_w, pts_ow, hasPC = _getFunctions(coreywo, SGL, reg)
    f['krW'] = krW
    f['krOW'] = krOW
    f.setdefault('krPts', {})
    f['krPts']['w'] = pts_w
    f['krPts']['ow'] = pts_ow
    f['coreywo'] = _np.asarray(coreywo, dtype=float)

    # A region with np == 0 contributes no Pc; MATLAB then falls back to
    # any pcOW already on the fluid for that region.
    existing = f.get('pcOW')
    for i, entry in enumerate(pcOW):
        if entry is None and isinstance(existing, (list, tuple)) and i < len(existing):
            pcOW[i] = existing[i]
            hasPC = True
    if hasPC:
        f['pcOW'] = pcOW
    return f


def _getFunctions(COREYWO, SGL, reg):
    COREYWO = _np.atleast_2d(_np.asarray(COREYWO, dtype=float))
    SGL = _np.atleast_1d(_np.asarray(SGL, dtype=float)).ravel()
    nreg = int(reg['sat'] if isinstance(reg, dict) else reg.sat)

    pts_w = _np.zeros((nreg, 4))
    pts_ow = _np.zeros((nreg, 4))
    krW, krOW, pcOW = [None] * nreg, [None] * nreg, [None] * nreg
    hasPC = False

    for i in range(nreg):
        row, sgl = COREYWO[i, :], float(SGL[i])
        pts_w[i, :], pts_ow[i, :] = _getPoints(row, sgl)
        krW[i] = _make(CoreyKrW, row, sgl)
        krOW[i] = _make(CoreyKrOW, row, sgl)
        if row[NP] != 0.0:
            pcOW[i] = _make_pc(CoreyPcOW, row)
            hasPC = True
    return krW, krOW, pcOW, pts_w, pts_ow, hasPC


def _make(fn, row, sgl):
    return lambda s: fn(s, row, sgl)


def _make_pc(fn, row):
    return lambda s: fn(s, row)


def CoreyKrOW(SO, COREYWO, SGL):
    """Oil relative permeability as a function of oil saturation."""
    c = _np.asarray(COREYWO, dtype=float)
    SW = 1.0 - SO - SGL
    denom = 1.0 - c[SWCR] - c[SOWCR] - SGL
    SWn = (SW - c[SWCR]) / denom
    S = _value(SW)
    out = _zeros_like(SO, S.size)

    if not _np.isnan(c[KRORW]):
        ix1 = (S >= c[SWL]) & (S < c[SWCR])
        ix2 = (S >= c[SWCR]) & (S <= 1.0 - c[SOWCR] - SGL)
        ix3 = (S > 1.0 - c[SOWCR] - SGL) & (S <= c[SWU])
        if _np.any(ix1):
            out = _put(out, ix1,
                       c[KRORW] + (c[KROLW] - c[KRORW]) * (c[SWCR] - SW) / (c[SWCR] - c[SWL]))
        if _np.any(ix2):
            out = _put(out, ix2, c[KRORW] * (1.0 - SWn) ** c[NOW])
        if _np.any(ix3):
            out = _put(out, ix3, 0.0)
    else:
        ix1 = (S >= c[SWL]) & (S <= 1.0 - c[SOWCR] - SGL)
        ix2 = (S > 1.0 - c[SOWCR] - SGL) & (S <= c[SWU])
        if _np.any(ix1):
            out = _put(out, ix1, c[KROLW] * (1.0 - SWn) ** c[NOW])
        if _np.any(ix2):
            out = _put(out, ix2, 0.0)

    below = S < c[SWL]
    if _np.any(below):
        out = _put(out, below, c[KROLW])
    return out


def CoreyKrW(SW, COREYWO, SGL):
    """Water relative permeability as a function of water saturation."""
    c = _np.asarray(COREYWO, dtype=float)
    denom = 1.0 - c[SWCR] - c[SOWCR] - SGL
    SWn = (SW - c[SWCR]) / denom
    S = _value(SW)
    out = _zeros_like(SW, S.size)

    if not _np.isnan(c[KRWR]):
        ix1 = (S >= c[SWL]) & (S < c[SWCR])
        ix2 = (S >= c[SWCR]) & (S <= 1.0 - c[SOWCR] - SGL)
        ix3 = (S > 1.0 - c[SOWCR] - SGL) & (S <= c[SWU])
        if _np.any(ix1):
            out = _put(out, ix1, 0.0)
        if _np.any(ix2):
            out = _put(out, ix2, c[KRWR] * SWn ** c[NW])
        if _np.any(ix3):
            span = c[SOWCR] + SGL + c[SWU] - 1.0
            out = _put(out, ix3,
                       c[KRWU] - (c[KRWU] - c[KRWR]) * (c[SWU] - SW) / span)
    else:
        ix1 = (S >= c[SWL]) & (S < c[SWCR])
        ix2 = (S >= c[SWCR]) & (S <= c[SWU])
        if _np.any(ix1):
            out = _put(out, ix1, 0.0)
        if _np.any(ix2):
            out = _put(out, ix2, c[KRWU] * SWn ** c[NW])

    above = S > c[SWU]
    if _np.any(above):
        out = _put(out, above, c[KRWU])
    return out


def CoreyPcOW(SW, COREYWO):
    """Oil/water capillary pressure as a function of water saturation."""
    c = _np.asarray(COREYWO, dtype=float)
    S = _value(SW)
    out = _zeros_like(SW, S.size)
    ix1 = (S >= c[SWL]) & (S <= c[SPCO])
    ix2 = (S > c[SPCO]) & (S < c[SWU])
    if _np.any(ix1):
        out = _put(out, ix1,
                   c[PCOW] * ((c[SPCO] - SW) / (c[SPCO] - c[SWCR])) ** c[NP])
    if _np.any(ix2):
        out = _put(out, ix2, 0.0)
    return out


def _getPoints(coreywo, SGL):
    """Port of the local ``getPoints``: the four scaling points per curve."""
    c = _np.asarray(coreywo, dtype=float)
    pts = _np.array([c[SWL], c[SWCR], c[SWU], c[KRWU]], dtype=float)
    pts_o = _np.array([0.0, c[SOWCR] - SGL, 1.0, c[KROLW]], dtype=float)
    return pts, pts_o


# --------------------------------------------------------------- helpers --

def _value(x):
    return x.val if isinstance(x, _SparseADI) else _np.atleast_1d(
        _np.asarray(x, dtype=float)).ravel()


def _zeros_like(reference, n):
    """``zeros(numel(S),1)``, promoted to ADI when the argument is ADI."""
    if isinstance(reference, _SparseADI):
        return _SparseADI.constant(_np.zeros(n), reference.nvar)
    return _np.zeros(n, dtype=float)


def _put(out, mask, values):
    """``out(mask) = values(mask)`` for both plain arrays and ADI."""
    if isinstance(out, _SparseADI) or isinstance(values, _SparseADI):
        nvar = out.nvar if isinstance(out, _SparseADI) else values.nvar
        if not isinstance(values, _SparseADI):
            values = _SparseADI.constant(
                _np.broadcast_to(_np.asarray(values, dtype=float),
                                 (out.val.size,)).copy(), nvar)
        if not isinstance(out, _SparseADI):
            out = _SparseADI.constant(_np.asarray(out, dtype=float), nvar)
        return _ad_select(mask, values, out)
    out = _np.asarray(out, dtype=float).copy()
    values = _np.broadcast_to(_np.asarray(values, dtype=float), out.shape)
    out[mask] = values[mask]
    return out
