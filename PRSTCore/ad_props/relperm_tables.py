"""Python port of MRST's ``assignSWFN.m``/``assignSGFN.m``/``assignSOF2.m``/
``assignSOF3.m`` (mrst-2026a/autodiff/ad-props/props) plus the Corey
analytical relative-permeability correlation (``ad-props/simple/
coreyPhaseRelpermAD.m``).

Complements the already-wired SWOF/SGOF table path
(``GenericBlackOilModel``'s ``as_table``/relperm evaluation) with readers for
the alternative ECLIPSE keyword combinations: ``SWFN``+``SOF2``/``SOF3``
(water relperm/Pc and oil relperm as separate single-phase-saturation
tables) instead of ``SWOF`` (combined water/oil-relative-to-water table),
and ``SGFN``+``SOF2``/``SOF3`` instead of ``SGOF``. Each ``assign_*``
function returns simple callables (1D piecewise-linear interpolation,
matching MRST's default ``interpTable``), independent of
``GenericBlackOilModel``'s internal table format so they're usable directly
by any caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as _np


def split_table_regions(table):
    """Split a stacked multi-region saturation table into its regions.

    ECLIPSE repeats a saturation-function keyword once per region (NTSFUN),
    and the deck parser concatenates those blocks into one array.  Every
    block is individually ascending in its saturation column, so a *drop*
    in column 0 marks a region boundary -- the same rule MRST's
    ``readRelPermTable`` uses to slice the keyword into per-region tables.

    Returns a list of arrays; a single-region table gives a one-element
    list.
    """
    if table is None:
        return []
    table = _np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0:
        return [table]
    cuts = _np.flatnonzero(_np.diff(table[:, 0]) < 0.0) + 1
    if cuts.size == 0:
        return [table]
    return _np.split(table, cuts)


def build_swof_sgof_tables(props, region: int = 0):
    """Return ``(swof, sgof)`` 4-column tables (``[Sw,Krw,Krow,Pcow]`` /
    ``[Sg,Krg,Krog,Pcog]``), read directly from ``SWOF``/``SGOF`` when
    present, else synthesized from the Family II keywords ``SWFN``/``SGFN``
    + ``SOF3``/``SOF2``.

    Mirrors how MRST's ``assignSWFN``/``assignSGFN``/``assignSOF3``/
    ``assignSOF2`` independently build ``fluid.krW``/``krG``/``pcOW``/
    ``pcOG`` (functions of Sw/Sg) and ``fluid.krOW``/``krOG`` (functions of
    So), which ``assignRelPerm``/``BlackOilCapillaryPressure`` then combine
    -- ``krOW`` evaluated at ``So = 1 - Sw``, ``krOG`` at
    ``So = 1 - Sg - Swco`` (``assignSGOF.m``'s convention). ``swof`` is
    ``None`` if neither family is usable (a black-oil model always needs a
    water/oil curve); ``sgof`` may legitimately be ``None`` for two-phase
    oil-water decks.
    """
    def _table(name, ncol):
        raw = props.get(name, [])
        size = _np.asarray(raw, dtype=object).size
        if size == 0 or size % ncol != 0:
            return None
        # A multi-region keyword arrives as its regions concatenated. They
        # must be sliced apart before any interpolation: the stacked array
        # is not monotonic in saturation, and interpTable sorts by the
        # saturation column, which interleaves the regions into a single
        # nonsense curve instead of failing.
        regions = split_table_regions(resolve_table_defaults(raw, ncol))
        if not regions:
            return None
        return regions[min(int(region), len(regions) - 1)]

    def _so_krcol(col_sof3, col_sof2):
        sof3 = _table('SOF3', 3)
        if sof3 is not None:
            return sof3[:, 0], sof3[:, col_sof3]
        sof2 = _table('SOF2', 2)
        if sof2 is not None:
            return sof2[:, 0], sof2[:, col_sof2]
        return None

    swof = _table('SWOF', 4)
    if swof is None:
        swfn = _table('SWFN', 3)
        so_kr = _so_krcol(1, 1)  # SOF3 col 1 = Krow; SOF2's single Kro doubles as Krow.
        if swfn is not None and so_kr is not None:
            so_col, krow_col = so_kr
            sw, krw, pcow = swfn[:, 0], swfn[:, 1], swfn[:, 2]
            krow = _np.interp(1.0 - sw, so_col, krow_col, left=krow_col[0], right=krow_col[-1])
            swof = _np.column_stack([sw, krw, krow, pcow])

    sgof = _table('SGOF', 4)
    if sgof is None:
        slgof = _table('SLGOF', 4)
        if slgof is not None:
            # assignSLGOF.m: SGOF's Sg column is max(1-Sl, 0); columns
            # 2:end (Krg, Krog, Pcog) carry over unchanged, then the row
            # order is flipped (Sl descends where Sg ascends).
            sgof = _np.column_stack([_np.maximum(1.0 - slgof[:, 0], 0.0), slgof[:, 1:]])[::-1]

    if sgof is None:
        sgfn = _table('SGFN', 3)
        so_kr = _so_krcol(2, 1)  # SOF3 col 2 = Krog; SOF2's single Kro doubles as Krog.
        if sgfn is not None and so_kr is not None:
            so_col, krog_col = so_kr
            swco = float(swof[0, 0]) if swof is not None else 0.0
            sg, krg, pcog = sgfn[:, 0], sgfn[:, 1], sgfn[:, 2]
            so = 1.0 - sg - swco
            krog = _np.interp(so, so_col, krog_col, left=krog_col[0], right=krog_col[-1])
            sgof = _np.column_stack([sg, krg, krog, pcog])

    return swof, sgof


def resolve_table_defaults(raw, ncol: int) -> _np.ndarray:
    """Port of ``readRelPermTable.m``'s ``convertTable``/``insertDefaultValues``.

    ECLIPSE table keywords (SWOF, SGOF, SWFN, SGFN, SOF2, SOF3, ...) allow
    entries in columns 2..ncol to be defaulted with ``1*``; ECLIPSE/MRST
    then fills each defaulted entry by linear interpolation (with
    extrapolation) of that column against column 1 (the saturation),
    using only the table's non-defaulted rows.  A column that is entirely
    defaulted is treated as all-zero (this happens for capillary-pressure
    columns left out of a table).  ``raw`` may be a flat sequence mixing
    numeric values and ``'1*'``/``'N*'`` default tokens, as produced by the
    deck tokenizer.
    """
    flat = []
    for v in _np.asarray(raw, dtype=object).ravel():
        if isinstance(v, str):
            # A record terminator can be glued onto the last token with no
            # separating whitespace (e.g. ``30.0/``), matching the same
            # quirk ``read_grid.py``'s ``_flatten_tokens`` strips.
            v = v.rstrip('/') if v != '/' else v
            if v == '/' or v == '':
                continue
            if '*' in v:
                flat.append(_np.nan)
                continue
        flat.append(float(v))
    table = _np.asarray(flat, dtype=float).reshape((-1, ncol))

    sat = table[:, 0]
    for c in range(1, ncol):
        col = table[:, c]
        missing = _np.isnan(col)
        if not _np.any(missing):
            continue
        if _np.all(missing):
            table[:, c] = 0.0
            continue
        known = ~missing
        x, y = sat[known], col[known]
        order = _np.argsort(x)
        x, y = x[order], y[order]
        col[missing] = _np.interp(sat[missing], x, y)
        # np.interp clamps outside [x[0], x[-1]]; MRST's interp1(...,'extrap')
        # linearly extrapolates instead, so patch the out-of-range points.
        below = missing & (sat < x[0])
        above = missing & (sat > x[-1])
        if _np.any(below) and x.size > 1:
            slope = (y[1] - y[0]) / (x[1] - x[0])
            col[below] = y[0] + slope * (sat[below] - x[0])
        if _np.any(above) and x.size > 1:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            col[above] = y[-1] + slope * (sat[above] - x[-1])
        table[:, c] = col
    return table


def _interp_table(x_table: _np.ndarray, y_table: _np.ndarray) -> Callable[[_np.ndarray], _np.ndarray]:
    """Port of MRST ``interpTable``: piecewise-linear interpolation, with
    the table's end values held constant outside its saturation range
    (matching ``interp1``'s default extrapolation-by-clamping behavior for
    a monotone table)."""
    x_table = _np.asarray(x_table, dtype=float)
    y_table = _np.asarray(y_table, dtype=float)

    def f(x):
        return _np.interp(_np.asarray(x, dtype=float), x_table, y_table)

    return f


@dataclass(slots=True)
class WaterFunctionTable:
    """Port of ``assignSWFN``'s per-region output."""
    krW: Callable
    pcOW: Callable | None
    points: _np.ndarray  # [swco, sw_at_last_immobile, sw_max, krW_max]


@dataclass(slots=True)
class GasFunctionTable:
    """Port of ``assignSGFN``'s per-region output."""
    krG: Callable
    pcOG: Callable | None
    points: _np.ndarray  # [sgco, sg_at_last_immobile, sg_max, krG_max]


@dataclass(slots=True)
class OilFunctionTable2Phase:
    """Port of ``assignSOF2``'s per-region output (oil relperm vs. so,
    single hydrocarbon system e.g. oil-water or oil-gas only)."""
    krO: Callable


@dataclass(slots=True)
class OilFunctionTable3Phase:
    """Port of ``assignSOF3``'s per-region output (oil relperm vs. so,
    separately for the oil-water and oil-gas subsystems, combined later by
    e.g. Stone's model for three-phase black-oil)."""
    krOW: Callable
    krOG: Callable


def assign_swfn(swfn_table) -> WaterFunctionTable:
    """``swfn_table``: (n, 2) or (n, 3) array, columns ``[Sw, krW, (PcOW)]``."""
    raw = _np.asarray(swfn_table, dtype=object)
    t = resolve_table_defaults(raw, raw.shape[-1])
    sw, kr = t[:, 0], t[:, 1]
    pc = t[:, 2] if t.shape[1] > 2 else _np.zeros_like(sw)
    has_pc = bool(_np.any(pc != 0.0))

    last_immobile = _np.flatnonzero(kr == 0.0)
    sw_immobile = sw[last_immobile[-1]] if last_immobile.size else sw[0]
    points = _np.array([sw[0], sw_immobile, sw[-1], kr[-1]])

    return WaterFunctionTable(
        krW=_interp_table(sw, kr),
        pcOW=_interp_table(sw, pc) if has_pc else None,
        points=points,
    )


def assign_sgfn(sgfn_table, swco: float = 0.0) -> GasFunctionTable:
    """``sgfn_table``: (n, 2) or (n, 3) array, columns ``[Sg, krG, (PcOG)]``.
    ``swco`` (connate water saturation, from the paired water table's
    endpoints) is recorded for parity with MRST's point set but does not
    affect the interpolants themselves."""
    raw = _np.asarray(sgfn_table, dtype=object)
    t = resolve_table_defaults(raw, raw.shape[-1])
    sg, kr = t[:, 0], t[:, 1]
    pc = t[:, 2] if t.shape[1] > 2 else _np.zeros_like(sg)
    has_pc = bool(_np.any(pc != 0.0))

    last_immobile = _np.flatnonzero(kr == 0.0)
    sg_immobile = sg[last_immobile[-1]] if last_immobile.size else sg[0]
    points = _np.array([sg[0], sg_immobile, sg[-1], kr[-1]])

    return GasFunctionTable(
        krG=_interp_table(sg, kr),
        pcOG=_interp_table(sg, pc) if has_pc else None,
        points=points,
    )


def assign_sof2(sof2_table) -> OilFunctionTable2Phase:
    """``sof2_table``: (n, 2) array, columns ``[So, krO]``."""
    raw = _np.asarray(sof2_table, dtype=object)
    t = resolve_table_defaults(raw, raw.shape[-1])
    return OilFunctionTable2Phase(krO=_interp_table(t[:, 0], t[:, 1]))


def assign_sof3(sof3_table) -> OilFunctionTable3Phase:
    """``sof3_table``: (n, 3) array, columns ``[So, krOW, krOG]``."""
    raw = _np.asarray(sof3_table, dtype=object)
    t = resolve_table_defaults(raw, raw.shape[-1])
    return OilFunctionTable3Phase(krOW=_interp_table(t[:, 0], t[:, 1]), krOG=_interp_table(t[:, 0], t[:, 2]))


def corey_relperm(s, *, n: float, sr: float, sr_tot: float, kr_max: float = 1.0):
    """Port of MRST ``coreyPhaseRelpermAD.m``: the Corey power-law relative
    permeability correlation

        sat = clip((s - sr) / (1 - sr_tot), 0, 1)
        kr(s) = kr_max * sat ** n

    ``sr`` is this phase's own residual saturation; ``sr_tot`` is the *sum*
    of residual saturations over every phase present (so the normalizing
    span ``1 - sr_tot`` is the total mobile saturation range, not just this
    phase's own ``[sr, 1]``).
    """
    span = max(1.0 - sr_tot, 1.0e-12)
    sat = _np.clip((_np.asarray(s, dtype=float) - sr) / span, 0.0, 1.0)
    return kr_max * sat**n
