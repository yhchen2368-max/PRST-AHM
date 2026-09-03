"""Python port of MRST's ``SimpleWell.m`` (mrst-2026a/autodiff/ad-core/models/wells).

A standard (single-segment) well: Peaceman perforation-to-reservoir coupling
(drawdown-weighted mobility, injection-phase-mixing for crossflow), aggregated
into per-cell reservoir source terms and per-well surface-rate/control
equations. Generalized over an arbitrary number of active phases so the same
code serves both the two-phase (water/oil) and three-phase (water/oil/gas)
black-oil paths -- previously two ~130-line near-duplicates inlined in
``GenericBlackOilModel``.

Does not (yet) model wellbore friction or segment/node topology -- see
:class:`PRSTCore.ad_core.models.multisegment_well.MultisegmentWell` for that.
"""

from __future__ import annotations

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.adi import as_sparse as _as_sparse


class SimpleWell:
    """Port of MRST ``SimpleWell``/``computeWellContributionsSingleWell.m``.

    Stateless: one instance's :meth:`compute_contributions` handles one
    well's perforations for one residual evaluation. Held by
    :class:`PRSTCore.ad_core.models.facility_model.FacilityModel`, which
    loops over all wells and combines the results.
    """

    def compute_contributions(self, *, w, cells, p, bhp, cdp, lam_phases,
                               rhoS_phases, component_mass_fn, nc, nvar, n_component_phases):
        """Assemble one well's perforation contributions.

        Parameters
        ----------
        w : dict
            Well spec (``WI``, ``cstatus``, ``compi``, ``sign``, ...).
        cells : sequence of int
            0-based reservoir cell index per perforation.
        p : SparseADI
            Reservoir pressure (nc-sized).
        bhp : SparseADI
            This well's bottom-hole pressure (scalar-valued ADI).
        cdp : array
            Per-perforation hydrostatic pressure drop (from
            :meth:`update_connection_pressure_drop`), same length as ``cells``.
        lam_phases : sequence of SparseADI
            Per-phase mobility (nc-sized), one entry per *mobility* phase
            (water/oil/gas).
        rhoS_phases : sequence of float
            Surface density per mobility phase, same order as ``lam_phases``.
        component_mass_fn : callable(cell, qph_list) -> list[SparseADI]
            Maps a cell index and the (possibly injection-mix-adjusted) list
            of per-phase volumetric perforation fluxes to a list of
            *component* mass fluxes (length ``n_component_phases``) -- the
            identity map for water/oil, plus the disgas/vapoil rs/rv cross
            terms for the three-phase black-oil case.
        n_component_phases : int
            Number of mass-conservation components (2 for water/oil, 3 for
            water/oil/gas with dissolved gas / vaporized oil).

        Returns
        -------
        src : list[SparseADI] (length n_component_phases)
            Per-cell reservoir source contributions (to subtract from the
            component-mass residual), each nc-sized (via ``scatter``).
        surface : list[SparseADI] (length n_component_phases)
            This well's total surface-rate contribution per component
            (mass/ ``rhoS``), scalar-valued (as an nw-sized scatter -- caller
            supplies ``nw``/well index via ``scatter``).
        perf_phase_flux, perf_component_flux : ndarray
            Per-perforation phase/component volumetric flux values (for
            caching into ``state`` -- used by the next ministep's
            :meth:`update_connection_pressure_drop`).
        """
        nph = len(lam_phases)
        wis = _np.asarray(w.get("WI", []), dtype=float).ravel()
        if wis.size == 1 and len(cells) > 1:
            wis = _np.full(len(cells), wis[0])
        if wis.size < len(cells):
            wis = _np.pad(wis, (0, len(cells) - wis.size), constant_values=0.0)
        cstatus = _np.asarray(w.get("cstatus", _np.ones(len(cells), dtype=bool)), dtype=bool).ravel()
        if cstatus.size < len(cells):
            cstatus = _np.pad(cstatus, (0, len(cells) - cstatus.size), constant_values=False)
        compi = _np.asarray(w.get("compi", _np.zeros(nph)), dtype=float).ravel()
        if compi.size < nph:
            compi = _np.pad(compi, (0, nph - compi.size))

        # Every perforation of this well at once.  Doing it one at a time
        # meant an AD scalar per perforation per phase, and each of those
        # operations carried the *system's* full width -- a 1-by-27000 sparse
        # row built, scaled and multiplied to hold three numbers.  Gathering
        # the cells first makes each step one operation on an nperf-long
        # vector instead, and the well terms stop dominating the assembly.
        AD = type(p)
        cells = _np.asarray(cells, dtype=int).ravel()
        active = (cells >= 0) & (cells < nc) & cstatus[:cells.size]
        perf_cells = cells[active]

        if perf_cells.size == 0:
            empty_perf = [AD.constant(_np.zeros(0), nvar) for _ in range(n_component_phases)]
            empty_surf = [AD.constant(_np.zeros(1), nvar) for _ in range(n_component_phases)]
            return perf_cells, empty_perf, empty_surf, \
                _np.zeros((0, nph)), _np.zeros((0, n_component_phases))

        nperf = perf_cells.size
        perf_wi = wis[:cells.size][active]
        # cdp arrives short, or absent, before the first connection-pressure
        # update has run; the perforations it does not cover take a zero
        # drop, which is what the per-perforation ``ip < len(cdp)`` guard
        # this replaces did.
        full_cdp = _np.zeros(cells.size, dtype=float)
        if cdp is not None:
            given = _np.asarray(cdp, dtype=float).ravel()[:cells.size]
            full_cdp[:given.size] = given
        perf_cdp = full_cdp[active]

        # bhp is one unknown shared by every perforation, so it broadcasts.
        tdp = (p[perf_cells] - (bhp._broadcast(nperf) + perf_cdp)) * (-perf_wi)
        perf_mobility = [lam_phases[k][perf_cells] for k in range(nph)]
        base = [perf_mobility[k] * tdp for k in range(nph)]

        base_values = _np.stack([b.val for b in base], axis=1)
        tdp_values = tdp.val

        def mix_toward_compi(values: _np.ndarray, injecting_mask: _np.ndarray, default_mix: _np.ndarray):
            """Port of ``crossFlowMixture.m``, gated as its callers gate it.

            ``WellComponentPhaseFlux`` calls this whenever ``any(perfIsInjector)``
            -- it does *not* additionally require the well to be a producer.
            An injector with cross-flow (some perforation flowing back into
            the wellbore while the well injects overall) therefore reinjects
            the blended wellbore composition in MRST; restricting the blend
            to ``sign < 0`` as this used to left such a well injecting raw
            ``compi`` and ignoring the back-produced fluid entirely.

            ``crossFlowMixture`` itself returns ``compi`` untouched when
            there is no inflow (``all(flux_in == 0)``), which is what makes
            the unrestricted gate safe for an ordinary injector.
            """
            # WellComponentPhaseFlux normalizes compi before the call:
            # compi./max(sum(compi, 2), 1e-10).
            mix = _np.asarray(default_mix, dtype=float).copy()
            total_mix = float(_np.sum(mix))
            mix = mix / max(total_mix, 1.0e-10)
            if not _np.any(injecting_mask):
                return mix
            inflow = -_np.minimum(values, 0.0)
            if not _np.any(inflow > 0.0):
                # all(flux_in == 0): no cross-flow, compi stands.
                return mix
            net_injection = max(float(_np.sum(values)), 0.0)
            mixture = _np.sum(inflow, axis=0) + net_injection * mix
            # active = compT > 0
            if float(_np.sum(mixture)) > 0.0:
                mix = mixture / float(_np.sum(mixture))
            return mix

        # A perforation is "injecting" based on the sign of its raw drawdown
        # (tdp), not any individual phase's mobility-weighted value -- using
        # a phase's own base value here would misclassify a perforation with
        # exactly zero mobility in that phase (e.g. connate water) regardless
        # of the true flow direction.
        injecting = _np.asarray(tdp_values, dtype=float) > 0.0
        phase_mix = mix_toward_compi(base_values, injecting, compi[:nph])

        # An injecting perforation delivers the wellbore mixture rather than
        # the reservoir's own phase split.  Selecting between the two
        # branches per perforation is what the loop did one at a time;
        # ``ad_select`` does it for the whole well, choosing by value exactly
        # as the ``if`` did.  Where no perforation injects -- the common case
        # for a producer -- the second branch is not formed at all.
        if _np.any(injecting):
            total_mobility = perf_mobility[0]
            for k in range(1, nph):
                total_mobility = total_mobility + perf_mobility[k]
            injected = [total_mobility * tdp * phase_mix[k] for k in range(nph)]
            if injecting.all():
                phase_flux = injected
            else:
                phase_flux = [_ad_select(injecting, injected[k], base[k])
                              for k in range(nph)]
        else:
            phase_flux = base

        # ``component_mass_fn`` indexes the per-cell properties it closes
        # over, so an array of cells and vector fluxes go through the same
        # code a single cell and scalars did.
        component_flux = list(component_mass_fn(perf_cells, phase_flux))

        component_values = _np.stack([q.val for q in component_flux], axis=1)
        component_injecting = _np.sum(component_values, axis=1) > 0.0
        component_mix = mix_toward_compi(component_values, component_injecting, compi[:n_component_phases]
                                          if compi.size >= n_component_phases else _np.pad(compi, (0, n_component_phases - compi.size)))

        if _np.any(component_injecting):
            total_component = component_flux[0]
            for k in range(1, n_component_phases):
                total_component = total_component + component_flux[k]
            mixed = [total_component * component_mix[k]
                     for k in range(n_component_phases)]
            if component_injecting.all():
                component_flux = mixed
            else:
                component_flux = [_ad_select(component_injecting, mixed[k],
                                             component_flux[k])
                                  for k in range(n_component_phases)]

        # Per-component reservoir source contributions as *unscattered*
        # per-perforation AD values alongside the cell indices they land in.
        # Scattering is deliberately left to the caller: FacilityModel
        # batches every well's perforations into one grid-sized scatter per
        # component, instead of one per well (each of which built a full
        # nc-row sparse AD value to place a handful of numbers).
        per_perf_src = []
        surface = []
        for k in range(n_component_phases):
            per_perf = component_flux[k]
            per_perf_src.append(per_perf)
            surface.append((per_perf / rhoS_phases[k]).sum())

        perf_phase_flux = _np.stack([q.val for q in phase_flux], axis=1)
        perf_component_flux = _np.stack([q.val for q in component_flux], axis=1)
        return perf_cells, per_perf_src, surface, perf_phase_flux, perf_component_flux
