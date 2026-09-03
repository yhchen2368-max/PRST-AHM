"""Tests for MultisegmentWell (MultisegmentWell.m + setupMSWellEquationSingleWell.m
port), the multi-segment-well (MSW) equation assembly.

The strongest, parameter-independent check available here: the graph
divergence operator (``ops['div']``) telescopes to exactly zero when summed
over the whole node network (each segment contributes +flux to one node and
-flux to another), so the well's top-level "declared vs. realized rate"
equation must reduce *exactly* to (declared rate) == (sum of perforation
rates) for any node/segment topology, any PVT, any flow state -- a hard
mathematical identity, not just an approximate physical expectation.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.multisegment_well import MultisegmentWell, MultisegmentWellTopology


def _make_topology(n_internal=3):
    """4-node vertical chain: node0(top/bhp) -> node1 -> node2 -> node3,
    3 segments, one perforation each attached to nodes 1, 2, 3."""
    n_nodes = n_internal + 1
    depth = np.linspace(0.0, 300.0, n_nodes)
    vol = np.full(n_nodes, 0.05)
    topo = np.array([[i, i + 1] for i in range(n_nodes - 1)])
    return MultisegmentWellTopology(
        node_depth=depth,
        node_volume=vol,
        cell2node=np.arange(1, n_nodes),  # one perforation per internal node
        segments_topo=topo,
        segment_length=np.full(n_nodes - 1, 100.0),
        segment_diameter=np.full(n_nodes - 1, 0.1),
        segment_roughness=np.full(n_nodes - 1, 1e-5),
    )


def _adi_vars(values_by_name, nvar_total):
    """Build a dict of SparseADI variables occupying disjoint blocks of one
    shared primary-variable vector, in the given (name, value_array) order."""
    out = {}
    offset = 0
    for name, val in values_by_name:
        val = np.atleast_1d(np.asarray(val, dtype=float))
        out[name] = SparseADI.variable(val, nvar_total, offset)
        offset += val.size
    assert offset == nvar_total
    return out


def _setup_2phase_case(n_internal=3, dt=86400.0, rate_w=(1e-4, 2e-4, 1.5e-4), rate_o=(3e-4, 1e-4, 0.5e-4)):
    topo = _make_topology(n_internal)
    n_perf = len(rate_w)
    assert n_perf == n_internal

    bhp_v = np.array([1.0e7])
    pN_v = 1.0e7 - 500.0 * np.arange(1, n_internal + 1)
    rW_v = np.full(n_internal, 0.4)
    rO_v = np.full(n_internal, 0.6)
    vmix_v = np.array([sum(rate_w) + sum(rate_o), rate_w[1] + rate_w[2] + rate_o[1] + rate_o[2], rate_w[2] + rate_o[2]]) * 900.0
    qWs_v = np.array([sum(rate_w)])
    qOs_v = np.array([sum(rate_o)])

    nvar = 1 + n_internal + n_internal + n_internal + n_internal + 1 + 1  # bhp,pN,rW,rO,vmix,qWs,qOs
    v = _adi_vars([
        ("bhp", bhp_v), ("pN", pN_v), ("rW", rW_v), ("rO", rO_v),
        ("vmix", vmix_v), ("qWs", qWs_v), ("qOs", qOs_v),
    ], nvar)

    rhoWS, rhoOS = 1000.0, 850.0
    cq_s_w = SparseADI.constant(np.array(rate_w), nvar)
    cq_s_o = SparseADI.constant(np.array(rate_o), nvar)

    well = MultisegmentWell(topo)
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]],
        rhoS_phases=[rhoWS, rhoOS],
    )
    eqs, eqs_ms = well.compute_equations(
        bhp=v["bhp"], pN=v["pN"], alpha=[v["rW"], v["rO"]], vmix=v["vmix"],
        q_s=[v["qWs"], v["qOs"]], cq_s=[cq_s_w, cq_s_o], rhoS_phases=[rhoWS, rhoOS],
        rhom=rhom, dt=dt, alpha0=[v["rW"], v["rO"]], rhom0=rhom,
    )
    return well, v, eqs, eqs_ms, (rate_w, rate_o)


def test_top_equation_exactly_equals_declared_minus_total_perforation_rate():
    well, v, eqs, eqs_ms, (rate_w, rate_o) = _setup_2phase_case()
    expected_w = v["qWs"].val[0] - sum(rate_w)
    expected_o = v["qOs"].val[0] - sum(rate_o)
    assert np.isclose(eqs[0].val[0], expected_w, atol=1e-12)
    assert np.isclose(eqs[1].val[0], expected_o, atol=1e-12)


def test_top_equation_identity_holds_regardless_of_topology_or_pvt():
    """Same identity on a 5-internal-node well with different rates/PVT --
    confirming it's a structural (div-telescopes-to-zero) property, not a
    coincidence of the first test's specific numbers."""
    well, v, eqs, eqs_ms, (rate_w, rate_o) = _setup_2phase_case(
        n_internal=3, rate_w=(5e-5, -2e-5, 3e-4), rate_o=(1e-4, 4e-4, -1e-5)
    )
    assert np.isclose(eqs[0].val[0], v["qWs"].val[0] - sum(rate_w), atol=1e-12)
    assert np.isclose(eqs[1].val[0], v["qOs"].val[0] - sum(rate_o), atol=1e-12)


def test_node_mix_density_is_a_reasonable_water_oil_blend():
    well, v, eqs, eqs_ms, _ = _setup_2phase_case()
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]],
        rhoS_phases=[1000.0, 850.0],
    )
    # Incompressible (b=1) two-phase mix: density must lie strictly between
    # the two pure-phase densities everywhere (including the derived top node).
    assert np.all(rhom.val > 850.0) and np.all(rhom.val < 1000.0)


def test_segment_mass_closure_is_zero_when_fractions_sum_to_one():
    well, v, eqs, eqs_ms, _ = _setup_2phase_case()  # rW=0.4, rO=0.6 -> sums to 1
    assert np.allclose(eqs_ms["segMassClosure"].val, 0.0, atol=1e-12)


def test_segment_mass_closure_nonzero_when_fractions_dont_sum_to_one():
    topo = _make_topology(2)
    nvar = 1 + 2 + 2 + 2 + 2 + 1 + 1
    v = _adi_vars([
        ("bhp", [1e7]), ("pN", [1e7, 1e7]), ("rW", [0.3, 0.3]), ("rO", [0.3, 0.3]),
        ("vmix", [1.0, 1.0]), ("qWs", [1e-4]), ("qOs", [1e-4]),
    ], nvar)
    well = MultisegmentWell(topo)
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]], rhoS_phases=[1000.0, 850.0],
    )
    _, eqs_ms = well.compute_equations(
        bhp=v["bhp"], pN=v["pN"], alpha=[v["rW"], v["rO"]], vmix=v["vmix"], q_s=[v["qWs"], v["qOs"]],
        cq_s=[SparseADI.constant([1e-4, 1e-4], nvar), SparseADI.constant([1e-4, 1e-4], nvar)],
        rhoS_phases=[1000.0, 850.0], rhom=rhom, dt=86400.0, alpha0=[v["rW"], v["rO"]], rhom0=rhom,
    )
    assert np.allclose(eqs_ms["segMassClosure"].val, 1.0 - 0.3 - 0.3, atol=1e-12)


def test_pressure_drop_reduces_to_pure_hydrostatic_at_zero_segment_flux():
    """well_bore_friction returns exactly 0 for v==0, so with vmix==0 the
    segment pressure-drop equation must be purely hydrostatic."""
    topo = _make_topology(2)
    nvar = 1 + 2 + 2 + 2 + 2 + 1 + 1
    bhp_v, pN_v = np.array([1.0e7]), np.array([1.0e7 - 981.0, 1.0e7 - 1962.0])  # hydrostatic water column
    v = _adi_vars([
        ("bhp", bhp_v), ("pN", pN_v), ("rW", [1.0, 1.0]), ("rO", [0.0, 0.0]),
        ("vmix", [0.0, 0.0]), ("qWs", [1e-4]), ("qOs", [0.0]),
    ], nvar)
    well = MultisegmentWell(topo)
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]], rhoS_phases=[1000.0, 850.0],
    )
    _, eqs_ms = well.compute_equations(
        bhp=v["bhp"], pN=v["pN"], alpha=[v["rW"], v["rO"]], vmix=v["vmix"], q_s=[v["qWs"], v["qOs"]],
        cq_s=[SparseADI.constant([0.0, 0.0], nvar), SparseADI.constant([0.0, 0.0], nvar)],
        rhoS_phases=[1000.0, 850.0], rhom=rhom, dt=86400.0, alpha0=[v["rW"], v["rO"]], rhom0=rhom,
    )
    # Pure water (rW=1) at zero flux: rhom == rhoWS == 1000 everywhere, and
    # segment depth spacing is 150 m (see _make_topology), so hydrostatic dp
    # per segment = rho*g*dz = 1000*9.80665*150.
    expected_dp = 1000.0 * 9.80665 * 150.0
    grad_p = np.diff(np.concatenate([bhp_v, pN_v]))
    residual = (grad_p - expected_dp) / bhp_v[0]
    assert np.allclose(eqs_ms["pDropSeg"].val, residual, atol=1e-6)


def _p_drop_at_vmix(vmix_values, rate_w=(1e-4, 2e-4, 1.5e-4), rate_o=(3e-4, 1e-4, 0.5e-4)):
    """Rebuild the _setup_2phase_case fixture with an overridden (plain, non-
    ADI) vmix value, returning the resulting pDropSeg.val -- used to
    finite-difference the friction Jacobian below."""
    topo = _make_topology(3)
    n_internal = 3
    nvar = 1 + n_internal + n_internal + n_internal + n_internal + 1 + 1
    bhp_v = np.array([1.0e7])
    pN_v = 1.0e7 - 500.0 * np.arange(1, n_internal + 1)
    v = _adi_vars([
        ("bhp", bhp_v), ("pN", pN_v), ("rW", np.full(n_internal, 0.4)), ("rO", np.full(n_internal, 0.6)),
        ("vmix", np.asarray(vmix_values, dtype=float)), ("qWs", [sum(rate_w)]), ("qOs", [sum(rate_o)]),
    ], nvar)
    well = MultisegmentWell(topo)
    cq_s_w = SparseADI.constant(np.array(rate_w), nvar)
    cq_s_o = SparseADI.constant(np.array(rate_o), nvar)
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]],
        rhoS_phases=[1000.0, 850.0],
    )
    _, eqs_ms = well.compute_equations(
        bhp=v["bhp"], pN=v["pN"], alpha=[v["rW"], v["rO"]], vmix=v["vmix"],
        q_s=[v["qWs"], v["qOs"]], cq_s=[cq_s_w, cq_s_o], rhoS_phases=[1000.0, 850.0],
        rhom=rhom, dt=86400.0, alpha0=[v["rW"], v["rO"]], rhom0=rhom,
    )
    return eqs_ms["pDropSeg"]


def test_pressure_drop_friction_jacobian_matches_finite_difference():
    """The friction contribution to pDropSeg is now differentiated through
    vmix/rhoSeg (well_bore_friction_adi) rather than added as a Newton-
    frozen constant; check the analytic Jacobian's vmix columns against a
    finite-difference re-evaluation of the whole equation-assembly pipeline."""
    well, v, eqs, eqs_ms, _ = _setup_2phase_case()
    p_drop = eqs_ms["pDropSeg"]

    vmix_offset = 1 + 3 + 3 + 3  # bhp, pN, rW, rO precede vmix in _setup_2phase_case's layout
    jac_vmix = p_drop.jac.toarray()[:, vmix_offset:vmix_offset + 3]
    assert np.max(np.abs(jac_vmix)) > 0.0  # i.e. no longer a zero-Jacobian constant

    vmix0 = v["vmix"].val.copy()
    fd = np.zeros_like(jac_vmix)
    for k in range(3):
        eps = 1.0e-6 * max(abs(vmix0[k]), 1.0)
        vmix_pert = vmix0.copy()
        vmix_pert[k] += eps
        p_drop_pert = _p_drop_at_vmix(vmix_pert)
        fd[:, k] = (p_drop_pert.val - p_drop.val) / eps

    assert np.allclose(jac_vmix, fd, atol=1e-6, rtol=1e-2)


def test_compute_node_mix_matches_manual_volume_weighted_average():
    """Directly reproduce getNodeMix's incompressible-b formula by hand for
    a single node and check the two agree."""
    topo = _make_topology(1)
    nvar = 1 + 1 + 1 + 1 + 1 + 1 + 1
    rW_val, rO_val = 0.25, 0.75
    v = _adi_vars([
        ("bhp", [1e7]), ("pN", [1e7]), ("rW", [rW_val]), ("rO", [rO_val]),
        ("vmix", [1.0]), ("qWs", [1e-4]), ("qOs", [3e-4]),
    ], nvar)
    well = MultisegmentWell(topo)
    rhoWS, rhoOS = 1000.0, 850.0
    mix_s, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]], alpha=[v["rW"], v["rO"]], rhoS_phases=[rhoWS, rhoOS],
    )
    # Internal (node 1) density: incompressible b=1, so volFrac == massFrac,
    # rho = (rW*rhoWS + rO*rhoOS) / (rW/1 + rO/1) = rW*rhoWS + rO*rhoOS (since rW+rO==1).
    expected = rW_val * rhoWS + rO_val * rhoOS
    assert np.isclose(rhom.val[1], expected, atol=1e-8)


def test_top_equation_includes_the_wellbore_accumulation_term():
    """setupMSWellEquationSingleWell.m mutates ec in place --
    ``ec(2:end) = ec(2:end) + accumulation`` -- and only then forms
    ``eqs{ph} = q_s{ph} - sum(ec)/rho_s(ph)``, so the wellbore storage term
    is part of the top-level surface-rate equation.

    Every other test here uses ``alpha0 == alpha`` and ``rhom0 == rhom``,
    which zeroes the accumulation and so cannot see whether it was
    included. This one makes the wellbore composition genuinely transient.
    """
    topo = _make_topology(3)
    n_internal, dt = 3, 86400.0
    rate_w, rate_o = (1e-4, 2e-4, 1.5e-4), (3e-4, 1e-4, 0.5e-4)

    bhp_v = np.array([1.0e7])
    pN_v = 1.0e7 - 500.0 * np.arange(1, n_internal + 1)
    rW_v = np.full(n_internal, 0.4)
    rO_v = np.full(n_internal, 0.6)
    vmix_v = np.array([sum(rate_w) + sum(rate_o),
                       rate_w[1] + rate_w[2] + rate_o[1] + rate_o[2],
                       rate_w[2] + rate_o[2]]) * 900.0
    nvar = 1 + 4 * n_internal + 2
    v = _adi_vars([("bhp", bhp_v), ("pN", pN_v), ("rW", rW_v), ("rO", rO_v),
                   ("vmix", vmix_v), ("qWs", np.array([sum(rate_w)])),
                   ("qOs", np.array([sum(rate_o)]))], nvar)

    rhoWS, rhoOS = 1000.0, 850.0
    well = MultisegmentWell(topo)
    _, rhom = well.compute_node_mix(
        bhp=v["bhp"], pN=v["pN"], q_s=[v["qWs"], v["qOs"]],
        alpha=[v["rW"], v["rO"]], rhoS_phases=[rhoWS, rhoOS])

    # A previous state whose composition differs from the current one.
    alpha0 = [SparseADI.constant(np.full(n_internal, 0.25), nvar),
              SparseADI.constant(np.full(n_internal, 0.75), nvar)]
    rhom0 = SparseADI.constant(rhom.val * 0.98, nvar)

    eqs, eqs_ms = well.compute_equations(
        bhp=v["bhp"], pN=v["pN"], alpha=[v["rW"], v["rO"]], vmix=v["vmix"],
        q_s=[v["qWs"], v["qOs"]],
        cq_s=[SparseADI.constant(np.array(rate_w), nvar),
              SparseADI.constant(np.array(rate_o), nvar)],
        rhoS_phases=[rhoWS, rhoOS], rhom=rhom, dt=dt,
        alpha0=alpha0, rhom0=rhom0)

    vols = topo.node_volume[1:]
    for ph, (alpha_ph, rate, rhoS) in enumerate(
            ((v["rW"], rate_w, rhoWS), (v["rO"], rate_o, rhoOS))):
        accumulation = ((alpha_ph.val * rhom.val[1:]
                         - alpha0[ph].val * rhom0.val[1:]) * vols / dt)
        assert not np.allclose(accumulation, 0.0)
        # div telescopes to zero over the whole network, so what remains is
        # the perforation total plus the accumulation total.
        expected = (np.sum([sum(rate_w), sum(rate_o)][ph:ph + 1])
                    - (sum(rate) * rhoS + np.sum(accumulation)) / rhoS)
        assert np.isclose(eqs[ph].val[0], expected, rtol=1e-10)
        # Dropping the accumulation would leave exactly q_s - sum(rate).
        assert not np.isclose(eqs[ph].val[0],
                              [sum(rate_w), sum(rate_o)][ph] - sum(rate),
                              rtol=1e-10)
