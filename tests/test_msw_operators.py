"""Tests for the multi-segment-well graph operators (MultisegmentWell.m
constructor port), a simple 4-node vertical chain: node0(top/bhp) ->
node1 -> node2 -> node3, i.e. 3 segments."""

from __future__ import annotations

import numpy as np

from PRSTCore.ad_core.models.msw_operators import build_msw_operators


def _chain(n_nodes=4):
    topo = np.array([[i, i + 1] for i in range(n_nodes - 1)])
    return build_msw_operators(n_nodes, topo)


def test_grad_matches_finite_difference_on_a_chain():
    ops = _chain(4)
    x = np.array([10.0, 7.0, 3.0, 0.0])  # node values, e.g. depth
    g = ops["grad"](x)
    # segment i connects node i -> node i+1; grad = x[to]-x[from]
    assert np.allclose(g, [x[1] - x[0], x[2] - x[1], x[3] - x[2]])


def test_div_is_the_adjoint_of_grad_incidence():
    ops = _chain(4)
    q = np.array([2.0, -1.0, 3.0])  # per-segment flux
    d = ops["div"](q)
    # node0 ("from" of seg0 only): +q0
    # node1 ("to" of seg0, "from" of seg1): -q0 + q1
    # node2 ("to" of seg1, "from" of seg2): -q1 + q2
    # node3 ("to" of seg2 only): -q2
    expected = np.array([q[0], -q[0] + q[1], -q[1] + q[2], -q[2]])
    assert np.allclose(d, expected)


def test_aver_is_simple_two_point_average():
    ops = _chain(4)
    x = np.array([100.0, 80.0, 60.0, 40.0])
    a = ops["aver"](x)
    assert np.allclose(a, [(100 + 80) / 2, (80 + 60) / 2, (60 + 40) / 2])


def test_segment_upstream_selects_from_or_to_by_flag():
    ops = _chain(4)
    # val is internal-node-indexed: val[0] is node1's value, val[1] node2's, val[2] node3's.
    val = np.array([1.0, 2.0, 3.0])
    # All segments flowing "downstream" (from->to): upstream = "from" node.
    flag_down = np.array([True, True, True])
    up = ops["segment_upstream"](flag_down, val)
    # seg0: from=node0 (top, no entry) -> falls back to node1's value (val[0]=1.0)
    # seg1: from=node1 -> val[0]=1.0
    # seg2: from=node2 -> val[1]=2.0
    assert np.allclose(up, [1.0, 1.0, 2.0])

    flag_up = np.array([False, False, False])
    up2 = ops["segment_upstream"](flag_up, val)
    # seg0: to=node1 -> val[0]=1.0 ; seg1: to=node2 -> val[1]=2.0 ; seg2: to=node3 -> val[2]=3.0
    assert np.allclose(up2, [1.0, 2.0, 3.0])


def test_operators_on_a_branching_topology():
    # node0(top) -> node1 -> {node2, node3} (a Y-branch), 3 segments.
    topo = np.array([[0, 1], [1, 2], [1, 3]])
    ops = build_msw_operators(4, topo)
    x = np.array([0.0, 10.0, 20.0, 30.0])
    g = ops["grad"](x)
    assert np.allclose(g, [10.0, 10.0, 20.0])
    q = np.array([5.0, 2.0, 3.0])
    d = ops["div"](q)
    # node1 is "to" of seg0 and "from" of both seg1,seg2: -5 + 2 + 3 = 0
    assert np.isclose(d[1], -5.0 + 2.0 + 3.0)
