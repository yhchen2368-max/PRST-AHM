"""Python port of MRST's ``FacilityModel.m``/``GenericFacilityModel.m``
(mrst-2026a/autodiff/ad-core/models/facilities).

Owns the list of per-well models (currently
:class:`PRSTCore.ad_core.models.well_model.SimpleWell`; multi-segment wells
are a planned extension) and combines their perforation contributions into
the reservoir source terms and well control-equation residuals that
``GenericBlackOilModel``'s component-mass equations consume.
"""

from __future__ import annotations

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI

from .well_model import SimpleWell


class FacilityModel:
    """Port of MRST ``FacilityModel``/``GenericFacilityModel``: combines all
    active wells' :class:`SimpleWell` contributions for one residual
    evaluation.

    Parameters
    ----------
    well_cells_fn : callable(well_dict) -> list[int]
        Resolves a well's perforation cell list (0-based). Kept as an
        injected callback rather than a direct model reference so this class
        doesn't depend on ``GenericBlackOilModel``'s deck/grid conventions.
    """

    #: What ``getProp`` accepts, and where each lives on a state.
    WELL_PROPERTIES = ('qWs', 'qOs', 'qGs', 'bhp')

    def __init__(self, well_cells_fn):
        self.well_cells_fn = well_cells_fn
        self._well = SimpleWell()  # stateless; one instance serves every well

    def getProp(self, state, name):
        """Port of MRST ``FacilityModel.getProp``: one well quantity,
        carrying whatever derivatives the state carries.

        History-matching objectives ask for ``qWs``, ``qOs``, ``qGs`` or
        ``bhp`` and differentiate the result. On a state from
        ``getStateAD`` these are AD variables seeded at the facility
        columns, so the objective comes back with its Jacobian row
        already assembled -- which is exactly the adjoint's
        ``dg_n/dx_n``. On a plain state they are numbers, and the same
        objective returns a value.

        The well solutions are the fallback so this also works on a
        state read back from a simulator's restart file, which carries
        ``wellSol`` but none of the ``facility_*`` primaries.
        """
        if name not in self.WELL_PROPERTIES:
            raise ValueError(
                'FacilityModel.getProp knows %s, not %r'
                % (', '.join(self.WELL_PROPERTIES), name))

        key = 'facility_' + name
        if key in state:
            return state[key]

        sols = state.get('wellSol') or []
        return _np.asarray([float(w.get(name, 0.0)) for w in sols],
                           dtype=float)

    def compute_well_contributions(self, *, wells, state, p, bhp, lam_phases, rhoS_phases,
                                    component_mass_fn, nc, nw, nvar, n_component_phases):
        """Combine every active well's perforation contributions.

        Parameters
        ----------
        wells : list[dict]
        state : dict
            Read for ``state['facility_cdp']`` (per-well, per-perforation
            hydrostatic pressure drop from
            :func:`update_connection_pressure_drop`).
        p : SparseADI
            Reservoir pressure (nc-sized).
        bhp : SparseADI
            All wells' bottom-hole pressures (nw-sized).
        lam_phases : sequence of SparseADI
            Per-phase mobility (nc-sized each), in mobility-phase order
            (water, oil[, gas]).
        rhoS_phases : sequence of float
            Surface density per mobility phase, same order as ``lam_phases``.
        component_mass_fn : callable(cell, qph_list) -> list[SparseADI]
            See :meth:`SimpleWell.compute_contributions`.
        n_component_phases : int
            Number of mass-conservation components (see ``SimpleWell``).

        Returns
        -------
        src : list[SparseADI] (length n_component_phases), each nc-sized
            Reservoir-cell source contributions, to subtract from the
            component-mass residual.
        surface : list[SparseADI] (length n_component_phases), each nw-sized
            Per-well total surface rate per component (mass / rhoS) -- feeds
            the ``fW``/``fO``/``fG`` "declared vs. realized surface rate"
            equations.
        perf_phase_all, perf_component_all : list[ndarray]
            Per-well arrays of per-perforation phase/component volumetric
            flux values, for caching into ``state`` (consumed by the next
            ministep's :func:`update_connection_pressure_drop`).
        """
        facility_cdp = state.get("facility_cdp", [])
        src = [_SparseADI.constant(_np.zeros(nc), nvar) for _ in range(n_component_phases)]
        surface = [_SparseADI.constant(_np.zeros(nw), nvar) for _ in range(n_component_phases)]
        perf_phase_all, perf_component_all = [], []

        # Batch every well's perforations into one grid-sized scatter per
        # component.  Each per-well scatter used to build a full nc-row
        # sparse AD value, so 119 wells meant 238 grid-sized sparse
        # constructions per residual evaluation -- and they dominated the
        # assembly once the model had many wells (T142: ~700 ms/Newton at
        # one well, ~2 s/Newton at 111).
        all_cells = []
        all_perf = [[] for _ in range(n_component_phases)]
        all_surface = [[] for _ in range(n_component_phases)]

        for iw, w in enumerate(wells):
            cells = self.well_cells_fn(w)
            cdp_i = _np.asarray(facility_cdp[iw], dtype=float).ravel() if iw < len(facility_cdp) else _np.zeros(0)

            perf_cells, per_perf_src, well_surface, perf_phase, perf_component = \
                self._well.compute_contributions(
                    w=w, cells=cells, p=p, bhp=bhp[iw], cdp=cdp_i,
                    lam_phases=lam_phases, rhoS_phases=rhoS_phases,
                    component_mass_fn=component_mass_fn, nc=nc, nvar=nvar,
                    n_component_phases=n_component_phases,
                )
            if perf_cells.size:
                all_cells.append(perf_cells)
                for k in range(n_component_phases):
                    all_perf[k].append(per_perf_src[k])
            for k in range(n_component_phases):
                all_surface[k].append(well_surface[k])
            perf_phase_all.append(perf_phase)
            perf_component_all.append(perf_component)

        for k in range(n_component_phases):
            if all_perf[k]:
                cells_all = _np.concatenate(all_cells)
                perf_all = _SparseADI.concat(all_perf[k])
                src[k] = _SparseADI.scatter(cells_all, perf_all, nc)
            else:
                src[k] = _SparseADI.constant(_np.zeros(nc), nvar)
            # One scalar per well, in well order: concat is exactly the
            # nw-wide surface-rate vector the control equations read.
            surface[k] = (_SparseADI.concat(all_surface[k])
                          if all_surface[k] else surface[k])

        return src, surface, perf_phase_all, perf_component_all

    @staticmethod
    def compute_control_equations(wells, *, qs_phases, bhp, phase_order):
        """Port of ``setupWellControlEquationsSingleWell.m``'s control-type
        dispatch (bhp/rate/orat/wrat/grat/lrat/vrat), one equation per well.

        ``qs_phases`` is a dict from a one-letter phase code (``'w'``/``'o'``/
        ``'g'``) to that phase's nw-sized surface-rate ADI variable;
        ``phase_order`` lists which codes are active (e.g. ``['w','o']`` or
        ``['w','o','g']``).
        """
        closure = []
        for iw, w in enumerate(wells):
            typ = str(w.get("type", "")).lower()
            target = float(w.get("val", 0.0))
            if typ == "bhp":
                closure.append((bhp[iw] - target) / (86400.0 * 1.0e5))
            elif typ in ("rate", "vrat"):
                closure.append(sum(qs_phases[ph][iw] for ph in phase_order) - target)
            elif typ == "orat":
                closure.append(qs_phases["o"][iw] - target)
            elif typ == "wrat":
                closure.append(qs_phases["w"][iw] - target)
            elif typ == "grat":
                closure.append(qs_phases["g"][iw] - target)
            elif typ == "lrat":
                closure.append(qs_phases["w"][iw] + qs_phases["o"][iw] - target)
            elif typ == "resv":
                # ctrl_eq(is_resv) = sum_ph q_s{ph}.*rho(:, ph) - target, with
                # rho the per-phase surface-to-reservoir conversion frozen for
                # the report step by ``updateRESVControls``.  Its absence is
                # not something to guess around: without the factors the
                # equation would be a surface-rate control wearing a
                # reservoir-rate target.
                factors = w.get("ControlDensity")
                if factors is None:
                    raise ValueError(
                        "RESV well %r has no ControlDensity; "
                        "prepareReportstep must run updateRESVControls first"
                        % w.get("name", iw))
                factors = _np.asarray(factors, dtype=float).ravel()
                reservoir_rate = None
                for k, ph in enumerate(phase_order):
                    if k >= factors.size:
                        break
                    term = qs_phases[ph][iw] * float(factors[k])
                    reservoir_rate = term if reservoir_rate is None else reservoir_rate + term
                closure.append(reservoir_rate - target)
            else:
                raise ValueError("Unsupported MRST well control type %r" % typ)
        return closure

    # ------------------------------------------------------ group control --
    #
    # MRST-0's GenericFacilityModel carries these; 2026a's does not. A
    # GCONPROD/GCONINJE target constrains a group, and these turn it into
    # per-well controls. See PRSTCore.ad_core.models.well_group_control.

    @staticmethod
    def getWellLimits(wells):
        """Port of ``GenericFacilityModel.getWellLimits``."""
        from .well_group_control import get_well_limits
        return get_well_limits(wells)

    @staticmethod
    def updateWellGroupControl(wellSol, drivingForces, q_p, wells=None):
        """Port of ``GenericFacilityModel.updateWellGroupControl``.

        Called after the well potentials are known, which is why it takes
        ``q_p`` rather than computing it: MRST evaluates the potential in
        the same loop that updates each well's connection pressure drop.
        """
        from .well_group_control import update_well_group_control
        groups = drivingForces.get('G') if isinstance(drivingForces, dict) \
            else getattr(drivingForces, 'G', None)
        if not groups:
            return wellSol
        if wells is None:
            wells = drivingForces.get('W') if isinstance(drivingForces, dict) \
                else getattr(drivingForces, 'W', None)
        return update_well_group_control(wellSol, groups, wells or [], q_p)
