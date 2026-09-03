"""Python port of MRST's ``MultisegmentWell.m`` + ``setupMSWellEquationSingleWell.m``
(mrst-2026a/autodiff/ad-core/models/facilities): a well discretized as a
one-dimensional network of nodes and segments rather than
:class:`PRSTCore.ad_core.models.well_model.SimpleWell`'s single bottom-hole
pressure.

Extra primary variables per well (beyond ``SimpleWell``'s ``qWs/qOs/qGs/bhp``):
  - ``pN``:  node pressure, one per *internal* node (``n_nodes - 1``; the top
    node's pressure is the well's ``bhp``).
  - ``rW``/``rO``/``rG``: node mixture mass fractions, same count as ``pN``.
  - ``vmix``: mixture mass flux per segment.

Extra equations: node mass conservation (one set per active phase, size
``n_nodes - 1``), segment pressure-drop (hydrostatic + friction via
:func:`PRSTCore.ad_core.models.wellbore_friction.well_bore_friction`, size
``n_segments``), and a node composition-closure equation (``sum(r) == 1``,
size ``n_nodes - 1``).

Scope notes (unlike most of this package, there is no pre-existing working
Python implementation to extract from, and none of PRSTCore's SPE1/SPE9/
Norne/Egg regression decks use multi-segment wells, so this is validated by
direct unit tests -- mass conservation across the node network, closure
identities -- rather than an end-to-end MRST trace comparison):

  - The gas-oil-ratio *status* classification at each node (which phase is
    locally saturated, driving whether ``rs``/``rv`` is capped) is evaluated
    at the current Newton iterate's value only (not carried through as an
    ADI-differentiable quantity), matching this codebase's existing
    convention for such status/classification flags elsewhere in
    ``GenericBlackOilModel`` (e.g. ``oil_saturated_override=(sg.val > 0.0)``).
    ``SparseADI`` has no differentiable ``abs``/``min`` primitive to carry
    that coupling through exactly as MRST's ADI class does; freezing it is a
    standard quasi-Newton simplification -- it affects Newton's convergence
    rate, not the converged residual, since the equations themselves are
    unchanged.
  - The wellbore-friction pressure drop is differentiated through
    ``vmix``/``rhoSeg`` via
    :func:`PRSTCore.ad_core.models.wellbore_friction.well_bore_friction_adi`
    (the Moody friction-factor formula itself -- laminar ``16/Re``, the
    Colebrook-White-style turbulent correlation, and the linear
    transitional interpolation between them -- is full ADI arithmetic).
    Only the laminar/transitional/turbulent *regime classification* and the
    zero-flow fallback are evaluated at the frozen current-iterate value,
    the same convention as the gor/ogr status freeze above: this affects
    Newton's convergence rate only (exactly at a regime boundary), not the
    converged residual, since the regime formulas agree there by
    construction (the transitional interpolation is built to match the
    laminar/turbulent formulas at its endpoints).
  - The segment mixture viscosity used for the friction Reynolds number
    (``mu_seg``) is a caller-supplied scalar/array, not derived from the
    node mixture composition/PVT -- callers wanting a composition-dependent
    viscosity should compute it themselves (e.g. a mole- or mass-weighted
    mixing rule over the node's phase viscosities) and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_abs as _abs_adi

from .msw_operators import build_msw_operators
from .wellbore_friction import well_bore_friction_adi as _well_bore_friction_adi


@dataclass(slots=True)
class MultisegmentWellTopology:
    """Static (non-ADI) geometry/topology for one multi-segment well.
    Port of the fields ``MultisegmentWell`` reads off ``W.nodes``/``W.segments``.
    """
    node_depth: _np.ndarray          # (n_nodes,) node 0 = top/bhp node
    node_volume: _np.ndarray         # (n_nodes,) wellbore volume represented by each node
    cell2node: _np.ndarray           # (n_perf,) which node each perforation attaches to (0-based)
    segments_topo: _np.ndarray       # (n_segments, 2) 0-based (from, to) node index pairs
    segment_length: _np.ndarray      # (n_segments,)
    segment_diameter: _np.ndarray    # (n_segments,)
    segment_roughness: _np.ndarray   # (n_segments,)

    @property
    def n_nodes(self) -> int:
        return self.node_depth.size

    @property
    def n_internal_nodes(self) -> int:
        return self.node_depth.size - 1

    def build_operators(self) -> dict:
        return build_msw_operators(self.n_nodes, self.segments_topo)


def _scatter_perf_to_nodes(cell2node: _np.ndarray, values, n_nodes: int):
    """Port of ``W.cell2node * cq_s{ph}``: sum per-perforation ADI values
    onto their attached node."""
    out = _SparseADI.constant(_np.zeros(n_nodes), values.nvar)
    for i, node in enumerate(cell2node):
        out = out + _SparseADI.scatter([int(node)], values[i:i + 1], n_nodes)
    return out


def _sum_adi(x):
    """Sum an ADI vector to a scalar ADI (linear map, so the Jacobian is
    just the column-sum of ``x``'s Jacobian)."""
    return _SparseADI(_np.array([x.val.sum()]), x.jac.sum(axis=0))


class MultisegmentWell:
    """Port of MRST ``MultisegmentWell``: assembles the node/segment
    equations for one multi-segment well, given already-evaluated reservoir
    connection properties (mirroring ``SimpleWell``'s perforation coupling)
    plus the well's own local ADI variables.
    """

    def __init__(self, topology: MultisegmentWellTopology):
        self.topology = topology
        self.operators = topology.build_operators()

    def compute_node_mix(self, *, bhp, pN, q_s, alpha, rhoS_phases, b_factors=None, model=None,
                          disgas: bool = False, vapoil: bool = False):
        """Port of ``getNodeMix``/``computeNodeProps``: node mixture density
        (size ``n_nodes``, including the top/bhp node) from the
        per-internal-node mass-fraction primary variables ``alpha`` (list of
        ADI, each size ``n_nodes - 1``) plus the top node's mixture (derived
        from the declared surface rates ``q_s``).

        For the water+oil (``n_phase == 2``) case, or the three-phase case
        without disgas/vapoil, pass ``b_factors`` (a list of per-phase
        formation volume factors, ADI or plain arrays of size ``n_nodes``,
        e.g. evaluated by the caller at ``p_full = concat([bhp, pN])``);
        defaults to incompressible (``b == 1``) if omitted. For three-phase
        with disgas/vapoil, pass ``model`` exposing
        ``_phase_pvt_from_phase_pressures_adi`` (reused rather than
        reimplemented; well nodes are evaluated at a single node pressure
        for all three phases, i.e. no capillary pressure at the wellbore) --
        ``b_factors`` is ignored in that case.
        """
        nph = len(alpha)
        p_full = _SparseADI.concat([bhp, pN])

        qt_s = _abs_adi(q_s[0])
        for k in range(1, nph):
            qt_s = qt_s + _abs_adi(q_s[k])

        top_mix = [_abs_adi(q_s[k]) / qt_s for k in range(nph)]
        mix_s = [_SparseADI.concat([top_mix[k], alpha[k]]) for k in range(nph)]

        if nph == 3 and (disgas or vapoil) and model is not None:
            rW, rO, rG = mix_s
            eps = 1.0e-12
            gor_val = _np.abs(rG.val / _np.where(_np.abs(rO.val) > eps, rO.val, eps))
            ogr_val = _np.abs(rO.val / _np.where(_np.abs(rG.val) > eps, rG.val, eps))

            pvt = model._phase_pvt_from_phase_pressures_adi(
                p_full, p_full, p_full,
                rs_override=gor_val if disgas else _np.zeros(p_full.val.shape),
                rv_override=ogr_val if vapoil else _np.zeros(p_full.val.shape),
                sG_override=_np.ones(p_full.val.shape) if disgas else _np.zeros(p_full.val.shape),
                oil_saturated_override=_np.ones(p_full.val.shape, dtype=bool) if disgas else None,
                gas_saturated_override=_np.ones(p_full.val.shape, dtype=bool) if vapoil else None,
            )
            b = [pvt["bw"], pvt["bo"], pvt["bg"]]
            rs = pvt["rs"] if disgas else _SparseADI.constant(_np.zeros(p_full.val.shape), p_full.nvar)
            rv = pvt["rv"] if vapoil else _SparseADI.constant(_np.zeros(p_full.val.shape), p_full.nvar)

            d = 1.0 - rs * rv
            xO = (rO - rs * rG) / d if disgas else rO
            xG = (rG - rv * rO) / d if vapoil else rG
            x = [rW, xO, xG]
        elif b_factors is not None:
            b = b_factors
            x = mix_s
        else:
            b = [_SparseADI.constant(_np.ones(p_full.val.shape), p_full.nvar) for _ in range(nph)]
            x = mix_s

        vol_ratio = x[0] / b[0]
        for k in range(1, nph):
            vol_ratio = vol_ratio + x[k] / b[k]

        rhom = mix_s[0] * rhoS_phases[0]
        for k in range(1, nph):
            rhom = rhom + mix_s[k] * rhoS_phases[k]
        rhom = rhom / vol_ratio

        return mix_s, rhom

    def compute_equations(self, *, bhp, pN, alpha, vmix, q_s, cq_s, rhoS_phases, rhom, dt,
                           alpha0, rhom0, gravity: float = 9.80665, mu_seg=None,
                           assume_turbulent: bool = False):
        """Port of the equation-assembly body of ``setupMSWellEquationSingleWell.m``
        (node-property computation is a separate step; see
        :meth:`compute_node_mix`).

        Parameters
        ----------
        cq_s : list[SparseADI] (length n_phase)
            Per-perforation surface-rate contributions (already computed by
            a :class:`~PRSTCore.ad_core.models.well_model.SimpleWell`-style
            Peaceman perforation model applied to this well's connections).
        rhom, rhom0 : SparseADI
            Node mixture densities (size ``n_nodes``, from
            :meth:`compute_node_mix`) at the current and previous timestep.
        alpha, alpha0 : list[SparseADI]
            Internal-node mass fractions (size ``n_nodes - 1`` each) at the
            current and previous timestep.
        mu_seg : array or None
            Segment mixture viscosity for the friction Reynolds number
            (see module docstring); defaults to a nominal 1 cP if omitted.

        Returns
        -------
        eqs : list[SparseADI] (length n_phase)
            Top-level "declared vs. realized surface rate" equations
            (mirrors ``SimpleWell``'s ``fW``/``fO``/``fG``).
        eqs_ms : dict
            ``{'node_<phase>': SparseADI, ..., 'pDropSeg': SparseADI,
            'segMassClosure': SparseADI}``.
        """
        ops = self.operators
        topo = self.topology
        nph = len(alpha)
        vols = topo.node_volume
        n_nodes = topo.n_nodes

        up = vmix.val >= 0.0

        eqs = []
        eqs_ms = {}
        for ph in range(nph):
            upstream_alpha = ops["segment_upstream"](up, alpha[ph])
            ec = ops["div"](upstream_alpha * vmix) + _scatter_perf_to_nodes(topo.cell2node, cq_s[ph], n_nodes) * rhoS_phases[ph]
            accumulation = (alpha[ph] * rhom[1:] - alpha0[ph] * rhom0[1:]) * (vols[1:] / dt)
            # setupMSWellEquationSingleWell.m updates ec in place --
            #   ec(2:end) = ec(2:end) + accumulation
            # -- and only then forms eqs{ph} = q_s{ph} - sum(ec)/rho_s(ph).
            # Summing the pre-accumulation ec instead drops the wellbore
            # storage term from the top-level surface-rate equation, which
            # is invisible at steady state (alpha0 == alpha, rhom0 == rhom)
            # but wrong whenever the wellbore composition or density is
            # changing over the step.
            ec = ec + _SparseADI.scatter(_np.arange(1, n_nodes), accumulation, n_nodes)
            eqs.append(q_s[ph] - _sum_adi(ec) / rhoS_phases[ph])
            eqs_ms[f"node_{ph}"] = ec[1:]

        ddz = ops["grad"](topo.node_depth)
        rho_seg = ops["aver"](rhom)
        dph = rho_seg * (gravity * ddz)

        mu_seg_arr = _np.full(ops["n_segments"], 1.0e-3) if mu_seg is None else _np.asarray(mu_seg, dtype=float)
        friction = _well_bore_friction_adi(
            vmix, rho_seg, mu_seg_arr, topo.segment_diameter, topo.segment_length,
            topo.segment_roughness, flowtype="massRate", assume_turbulent=assume_turbulent,
        )
        p_full = _SparseADI.concat([bhp, pN])
        bhp_val = float(bhp.val[0])
        p_drop_seg = (ops["grad"](p_full) - dph - friction) / bhp_val

        seg_mass_closure = _SparseADI.constant(_np.ones(topo.n_internal_nodes), alpha[0].nvar) - alpha[0]
        for k in range(1, nph):
            seg_mass_closure = seg_mass_closure - alpha[k]

        eqs_ms["pDropSeg"] = p_drop_seg
        eqs_ms["segMassClosure"] = seg_mass_closure
        return eqs, eqs_ms
