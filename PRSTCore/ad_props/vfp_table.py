"""Python port of MRST's ``VFPTable.m`` + ``getMultiDimInterpolator.m``
(mrst-2026a/autodiff/ad-core/{models/facilities/vfp,utils}): vertical flow
performance (VFP) table interpolation for the ``VFPPROD``/``VFPINJ`` deck
keywords, used to convert a tubing-head-pressure (THP) well control into an
implicit bottom-hole-pressure (BHP) constraint.

Scope: table construction + ``evaluate_bhp`` (flow, THP, [WFR, GFR, [ALQ]])
-> BHP via N-D linear interpolation with linear extrapolation outside the
table's range, matching MRST's ``griddedInterpolant(..., 'linear',
'linear')``. Not ported: the deck-keyword parser (``readVFPPROD.m`` /
``readVFPINJ.m`` -- no example deck in this repository exercises these
keywords) or the AD/well-equation wiring
(``setupWellControlEquationsSingleWell.m``'s ``'thp'`` control case, which
differentiates through the interpolant via a finite-difference chain-rule
bump). Callers needing a THP-controlled well in an AD residual should
evaluate on the current iterate's numeric values (``.val``) and treat the
result as a converged-iteration constant, the same convention already used
for status-flag classification in
:mod:`PRSTCore.ad_core.models.multisegment_well`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as _np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class VFPProdTable:
    """Port of ``VFPTable.m``'s producer branch (``d.Q`` present in the
    deck record). Axes, in interpolation order: FLO, THP, WFR, GFR, [ALQ]."""

    flo: _np.ndarray
    thp: _np.ndarray
    wfr: _np.ndarray
    gfr: _np.ndarray
    alq: _np.ndarray
    bhp_table: _np.ndarray  # (n_flo, n_thp, n_wfr, n_gfr, n_alq)
    flow_type: str = "oil"
    water_ratio_type: str = "wor"
    gas_ratio_type: str = "gor"
    ref_depth: float = 0.0
    _interp: RegularGridInterpolator = field(init=False, repr=False)
    _has_alq: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        flo, thp, wfr, gfr, alq = (
            _np.asarray(a, dtype=float) for a in (self.flo, self.thp, self.wfr, self.gfr, self.alq)
        )
        table = _np.asarray(self.bhp_table, dtype=float)
        expected = (flo.size, thp.size, wfr.size, gfr.size, alq.size)
        if table.shape != expected:
            raise ValueError(f"bhp_table shape {table.shape} does not match axes {expected}")

        self._has_alq = alq.size > 1
        axes = [flo, thp, wfr, gfr]
        if self._has_alq:
            axes.append(alq)
        else:
            # MRST: numel(d.ALQ) <= 1 -> squeeze away the (unused) ALQ axis
            # and select its single table slice.
            table = table[:, :, :, :, 0]

        self._interp = RegularGridInterpolator(
            tuple(axes), table, method="linear", bounds_error=False, fill_value=None,
        )

    def evaluate_bhp(self, flow, thp, wfr, gfr, alq=None) -> _np.ndarray:
        """Port of ``VFPTable.evaluateBHP`` for a producer table."""
        flow, thp, wfr, gfr = (
            _np.atleast_1d(_np.asarray(a, dtype=float)) for a in (flow, thp, wfr, gfr)
        )
        n = max(flow.size, thp.size, wfr.size, gfr.size)
        flow, thp, wfr, gfr = (_np.broadcast_to(a, (n,)) for a in (flow, thp, wfr, gfr))
        cols = [flow, thp, wfr, gfr]
        if self._has_alq:
            alq_arr = _np.broadcast_to(
                _np.atleast_1d(_np.asarray(0.0 if alq is None else alq, dtype=float)), (n,)
            )
            cols.append(alq_arr)
        return self._interp(_np.column_stack(cols))


@dataclass
class VFPInjTable:
    """Port of ``VFPTable.m``'s injector branch (``d.BHP`` present in the
    deck record). Axes, in interpolation order: FLO, THP."""

    flo: _np.ndarray
    thp: _np.ndarray
    bhp_table: _np.ndarray  # (n_flo, n_thp)
    flow_type: str = "oil"
    ref_depth: float = 0.0
    _interp: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        flo, thp = (_np.asarray(a, dtype=float) for a in (self.flo, self.thp))
        table = _np.asarray(self.bhp_table, dtype=float)
        if table.shape != (flo.size, thp.size):
            raise ValueError(f"bhp_table shape {table.shape} does not match axes {(flo.size, thp.size)}")
        self._interp = RegularGridInterpolator(
            (flo, thp), table, method="linear", bounds_error=False, fill_value=None,
        )

    def evaluate_bhp(self, flow, thp) -> _np.ndarray:
        """Port of ``VFPTable.evaluateBHP`` for an injector table."""
        flow, thp = (_np.atleast_1d(_np.asarray(a, dtype=float)) for a in (flow, thp))
        n = max(flow.size, thp.size)
        flow, thp = (_np.broadcast_to(a, (n,)) for a in (flow, thp))
        return self._interp(_np.column_stack([flow, thp]))


def assign_vfpprod(flo, thp, wfr, gfr, alq, bhp_table, *, flowtype="OIL", wfrtype="WOR",
                    gfrtype="GOR", ref_depth=0.0) -> VFPProdTable:
    """Build a :class:`VFPProdTable` from raw ``VFPPROD`` record arrays
    (``flo, thp, wfr, gfr, alq`` axis vectors and the ``(n_flo, n_thp,
    n_wfr, n_gfr, n_alq)`` ``bhp_table``, as produced by
    ``readVFPPROD.m``'s ``FLO/THP/WFR/GFR/ALQ/Q`` fields)."""
    return VFPProdTable(
        flo=_np.asarray(flo, dtype=float), thp=_np.asarray(thp, dtype=float),
        wfr=_np.asarray(wfr, dtype=float), gfr=_np.asarray(gfr, dtype=float),
        alq=_np.atleast_1d(_np.asarray(alq, dtype=float)), bhp_table=bhp_table,
        flow_type=str(flowtype).lower(), water_ratio_type=str(wfrtype).lower(),
        gas_ratio_type=str(gfrtype).lower(), ref_depth=float(ref_depth),
    )


def assign_vfpinj(flo, thp, bhp_table, *, flowtype="OIL", ref_depth=0.0) -> VFPInjTable:
    """Build a :class:`VFPInjTable` from raw ``VFPINJ`` record arrays
    (``flo, thp`` axis vectors and the ``(n_flo, n_thp)`` ``bhp_table``, as
    produced by ``readVFPINJ.m``'s ``FLO/THP/BHP`` fields)."""
    return VFPInjTable(
        flo=_np.asarray(flo, dtype=float), thp=_np.asarray(thp, dtype=float),
        bhp_table=bhp_table, flow_type=str(flowtype).lower(), ref_depth=float(ref_depth),
    )
