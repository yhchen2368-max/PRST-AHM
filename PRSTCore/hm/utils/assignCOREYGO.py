"""Port of MRST ``assignCOREYGO.m`` (mrst-2026a/hm/utils).

The gas/oil counterpart of ``assignCOREYWO``: the same three-segment Corey
construction with water and gas exchanged, parametrised by connate *water*
``SWL`` instead of connate gas.

The 13-column ``COREYGO`` record, one row per saturation region:

    1  SGL     connate gas
    2  SGU     maximum gas
    3  SGCR    critical gas
    4  SOGCR   critical oil (in gas)
    5  krOLG   oil kr at connate gas
    6  krORG   oil kr at critical gas          (NaN -> two-segment form)
    7  krGR    gas kr at critical oil          (NaN -> two-segment form)
    8  krGU    gas kr at maximum gas
    9  pcOG    entry capillary pressure
    10 nOG     oil Corey exponent
    11 nG      gas Corey exponent
    12 np      capillary-pressure exponent     (0 -> no Pc)
    13 SpcG    saturation at which Pc vanishes

``CoreyPcOG`` differs from its water twin beyond renaming: it takes
``SWL``, rises with ``(SG - SpcG)`` rather than falling with
``(SpcO - SW)``, and normalises by ``1 - SpcG - SOGCR - SWL``.
"""

import numpy as _np

from .assignCOREYWO import _put, _value, _zeros_like

SGL, SGU, SGCR, SOGCR = 0, 1, 2, 3
KROLG, KRORG, KRGR, KRGU = 4, 5, 6, 7
PCOG, NOG, NG, NP, SPCG = 8, 9, 10, 11, 12


def assignCOREYGO(f, coreygo, SWL, reg):
    """Attach ``krG``/``krOG``/``pcOG``/``krPts`` to the fluid dict ``f``."""
    krG, krOG, pcOG, pts_g, pts_og, hasPC = _getFunctions(coreygo, SWL, reg)
    f['krG'] = krG
    f['krOG'] = krOG
    f.setdefault('krPts', {})
    f['krPts']['g'] = pts_g
    f['krPts']['og'] = pts_og
    f['coreygo'] = _np.asarray(coreygo, dtype=float)

    existing = f.get('pcOG')
    for i, entry in enumerate(pcOG):
        if entry is None and isinstance(existing, (list, tuple)) and i < len(existing):
            pcOG[i] = existing[i]
            hasPC = True
    if hasPC:
        f['pcOG'] = pcOG
    return f


def _getFunctions(COREYGO, SWL_, reg):
    COREYGO = _np.atleast_2d(_np.asarray(COREYGO, dtype=float))
    SWL_ = _np.atleast_1d(_np.asarray(SWL_, dtype=float)).ravel()
    nreg = int(reg['sat'] if isinstance(reg, dict) else reg.sat)

    pts_g = _np.zeros((nreg, 4))
    pts_og = _np.zeros((nreg, 4))
    krG, krOG, pcOG = [None] * nreg, [None] * nreg, [None] * nreg
    hasPC = False

    for i in range(nreg):
        row, swl = COREYGO[i, :], float(SWL_[i])
        pts_g[i, :], pts_og[i, :] = _getPoints(row, swl)
        krG[i] = (lambda r, s: (lambda sg: CoreyKrG(sg, r, s)))(row, swl)
        krOG[i] = (lambda r, s: (lambda so: CoreyKrOG(so, r, s)))(row, swl)
        if row[NP] != 0.0:
            pcOG[i] = (lambda r, s: (lambda sg: CoreyPcOG(sg, r, s)))(row, swl)
            hasPC = True
    return krG, krOG, pcOG, pts_g, pts_og, hasPC


def CoreyKrOG(SO, COREYGO, SWL_):
    """Oil relative permeability as a function of oil saturation."""
    c = _np.asarray(COREYGO, dtype=float)
    SG = 1.0 - SO - SWL_
    SGn = (SG - c[SGCR]) / (1.0 - c[SGCR] - c[SOGCR] - SWL_)
    S = _value(SG)
    out = _zeros_like(SO, S.size)

    if not _np.isnan(c[KRORG]):
        ix1 = (S >= c[SGL]) & (S < c[SGCR])
        ix2 = (S >= c[SGCR]) & (S <= 1.0 - c[SOGCR] - SWL_)
        ix3 = (S > 1.0 - c[SOGCR] - SWL_) & (S <= c[SGU])
        if _np.any(ix1):
            out = _put(out, ix1,
                       c[KRORG] + (c[KROLG] - c[KRORG]) * (c[SGCR] - SG) / (c[SGCR] - c[SGL]))
        if _np.any(ix2):
            out = _put(out, ix2, c[KRORG] * (1.0 - SGn) ** c[NOG])
        if _np.any(ix3):
            out = _put(out, ix3, 0.0)
    else:
        ix1 = (S >= c[SGL]) & (S <= 1.0 - c[SOGCR] - SWL_)
        ix2 = (S > 1.0 - c[SOGCR] - SWL_) & (S <= c[SGU])
        if _np.any(ix1):
            out = _put(out, ix1, c[KROLG] * (1.0 - SGn) ** c[NOG])
        if _np.any(ix2):
            out = _put(out, ix2, 0.0)

    below = S < c[SGL]
    if _np.any(below):
        out = _put(out, below, c[KROLG])
    return out


def CoreyKrG(SG, COREYGO, SWL_):
    """Gas relative permeability as a function of gas saturation."""
    c = _np.asarray(COREYGO, dtype=float)
    SGn = (SG - c[SGCR]) / (1.0 - c[SGCR] - c[SOGCR] - SWL_)
    S = _value(SG)
    out = _zeros_like(SG, S.size)

    if not _np.isnan(c[KRGR]):
        ix1 = (S >= c[SGL]) & (S < c[SGCR])
        ix2 = (S >= c[SGCR]) & (S <= 1.0 - c[SOGCR] - SWL_)
        ix3 = (S > 1.0 - c[SOGCR] - SWL_) & (S <= c[SGU])
        if _np.any(ix1):
            out = _put(out, ix1, 0.0)
        if _np.any(ix2):
            out = _put(out, ix2, c[KRGR] * SGn ** c[NG])
        if _np.any(ix3):
            span = c[SOGCR] + SWL_ + c[SGU] - 1.0
            out = _put(out, ix3,
                       c[KRGU] - (c[KRGU] - c[KRGR]) * (c[SGU] - SG) / span)
    else:
        ix1 = (S >= c[SGL]) & (S < c[SGCR])
        ix2 = (S >= c[SGCR]) & (S <= c[SGU])
        if _np.any(ix1):
            out = _put(out, ix1, 0.0)
        if _np.any(ix2):
            out = _put(out, ix2, c[KRGU] * SGn ** c[NG])

    above = S > c[SGU]
    if _np.any(above):
        out = _put(out, above, c[KRGU])
    return out


def CoreyPcOG(SG, COREYGO, SWL_):
    """Oil/gas capillary pressure as a function of gas saturation."""
    c = _np.asarray(COREYGO, dtype=float)
    S = _value(SG)
    out = _zeros_like(SG, S.size)
    ix1 = (S >= c[SGL]) & (S < c[SPCG])
    ix2 = (S >= c[SPCG]) & (S <= c[SGU])
    if _np.any(ix1):
        out = _put(out, ix1,
                   c[PCOG] * ((SG - c[SPCG]) / (1.0 - c[SPCG] - c[SOGCR] - SWL_)) ** c[NP])
    if _np.any(ix2):
        out = _put(out, ix2, 0.0)
    return out


def _getPoints(coreygo, SWL_):
    c = _np.asarray(coreygo, dtype=float)
    pts = _np.array([c[SGL], c[SGCR], c[SGU], c[KRGU]], dtype=float)
    pts_o = _np.array([0.0, c[SOGCR] - SWL_, 1.0, c[KROLG]], dtype=float)
    return pts, pts_o
