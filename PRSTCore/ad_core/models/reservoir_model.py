"""Python port of MRST's ``ReservoirModel.m`` (mrst-2026a/autodiff/ad-core/models).

Adds the fields and reusable primitives every reservoir (as opposed to
generic AD) model needs on top of :class:`PhysicalModel`: fluid/rock,
phase-presence flags, pressure/saturation increment limits
(``dpMaxRel``/``dpMaxAbs``/``dsMaxAbs``), pressure clamping bounds, and the
CNV/MB convergence tolerances plus the per-phase CNV/MB computation itself
(the numerical core of MRST's ``getConvergenceValuesCNV``, shared here so
concrete models don't each reimplement the same reduction).
"""

from __future__ import annotations

import numpy as _np

from .physical_model import PhysicalModel


class ReservoirModel(PhysicalModel):
    """Base class for reservoir simulation models (black-oil, compositional,
    ...). Mirrors MRST's ``ReservoirModel``: phase flags, rock/fluid, and the
    dpMax/dsMax + CNV/MB primitives concrete models compose their
    ``updateState``/``checkConvergence`` from."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fluid = kwargs.get('fluid', None)
        self.rock = kwargs.get('rock', None)

        self.water = bool(kwargs.get('water', False))
        self.oil = bool(kwargs.get('oil', False))
        self.gas = bool(kwargs.get('gas', False))

        self.dpMaxRel = float(kwargs.get('dpMaxRel', _np.inf))
        self.dpMaxAbs = float(kwargs.get('dpMaxAbs', _np.inf))
        self.dsMaxAbs = float(kwargs.get('dsMaxAbs', 0.2))
        self.minimumPressure = float(kwargs.get('minimumPressure', -_np.inf))
        self.maximumPressure = float(kwargs.get('maximumPressure', _np.inf))

        self.useCNVConvergence = bool(kwargs.get('useCNVConvergence', False))
        self.toleranceCNV = float(kwargs.get('toleranceCNV', 1.0e-3))
        self.toleranceMB = float(kwargs.get('toleranceMB', 1.0e-7))

    # ------------------------------------------------------------------
    # dpMax / dsMax primitives (ReservoirModel.updateState / updateSaturations)
    # ------------------------------------------------------------------
    def limit_pressure_increment(self, p0, dp):
        """Port of the pressure branch of ``ReservoirModel.updateState``:
        apply ``dpMaxRel``/``dpMaxAbs`` via :meth:`limit_increment`, then
        clamp to ``[minimumPressure, maximumPressure]`` via ``capProperty``."""
        p = self.limit_increment(p0, dp, rel_max=self.dpMaxRel, abs_max=self.dpMaxAbs)
        return self.cap_property(p, self.minimumPressure, self.maximumPressure)

    def limit_saturation_increment(self, s0, ds):
        """Port of the saturation branch of ``ReservoirModel.updateSaturations``:
        cap the update by ``dsMaxAbs`` (an *absolute* limit only -- MRST does
        not apply a relative saturation limit, since saturations can be 0)."""
        return self.limit_increment(s0, ds, rel_max=None, abs_max=self.dsMaxAbs)

    def shared_saturation_scale(self, *ds_components):
        """Single scale factor, shared across several saturation components
        (e.g. dSw/dSg/dSo for a 3-phase update), capped by ``dsMaxAbs`` and
        driven by whichever component changes the most -- the joint-limit
        pattern MRST's ``ThreePhaseBlackOilModel.updateState`` uses so that
        Sw/Sg/So move together rather than each phase saturating at a
        different rate. Returns the scale array; apply it to each component
        individually (``s0_i + ds_i * scale``)."""
        worst = _np.zeros_like(_np.asarray(ds_components[0], dtype=float))
        for ds in ds_components:
            worst = _np.maximum(worst, _np.abs(_np.asarray(ds, dtype=float)))
        if not _np.isfinite(self.dsMaxAbs):
            return _np.ones_like(worst)
        return _np.divide(
            self.limit_increment(_np.zeros_like(worst), worst, abs_max=self.dsMaxAbs),
            _np.where(worst > 0, worst, 1.0),
        )

    # ------------------------------------------------------------------
    # CNV/MB convergence primitive (ReservoirModel.getConvergenceValuesCNV core)
    # ------------------------------------------------------------------
    @staticmethod
    def cnv_mb_from_residual(residual_phase, b_factor, rho_surface, pv, dt):
        """Per-phase CNV and MB convergence values from a single phase's
        residual. Port of the inner reduction of MRST
        ``getConvergenceValuesCNV``:

            eq   = residual / rho_surface
            Bavg = mean(1 / b_factor)
            CNV  = Bavg * dt * max(|eq| / pv)
            MB   = |Bavg * dt * sum(eq)| / sum(pv)

        ``residual_phase``, ``b_factor``, and ``pv`` are per-cell arrays for
        this phase; ``rho_surface``/``dt`` are scalars. Returns ``(cnv, mb)``.
        """
        eq = _np.asarray(residual_phase, dtype=float) / rho_surface
        pv = _np.asarray(pv, dtype=float)
        b_avg = float(_np.mean(1.0 / _np.asarray(b_factor, dtype=float)))
        cnv = b_avg * dt * float(_np.max(_np.abs(eq) / pv))
        mb = abs(b_avg * dt * float(_np.sum(eq))) / float(_np.sum(pv))
        return cnv, mb
