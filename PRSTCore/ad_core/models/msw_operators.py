"""Segment/node graph operators for multi-segment wells, ported from the
constructor of MRST's ``MultisegmentWell.m`` (mrst-2026a/autodiff/ad-core/
models/facilities).

Node 0 is the well's top node (bottom-hole/``bhp`` location, no primary
variable of its own); nodes ``1..nn-1`` are internal nodes with their own
``pN`` pressure primary variable. ``segments_topo`` (0-based node index
pairs) defines the segment graph connecting them.
"""

from __future__ import annotations

import numpy as _np
import scipy.sparse as _sp

from PRSTCore.ad_core.adi import SparseADI as _SparseADI


def _apply(matrix, x):
    """``matrix @ x``, dispatching to ``SparseADI.linear_map`` so the
    Jacobian propagates correctly when ``x`` is an ADI value."""
    if isinstance(x, _SparseADI):
        return x.linear_map(matrix)
    return matrix @ _np.asarray(x, dtype=float)


def build_msw_operators(n_nodes: int, segments_topo) -> dict:
    """Port of the operator setup inside ``MultisegmentWell``'s constructor.

    Returns a dict with:
      - ``grad(x)``: node-valued array (size ``n_nodes``) -> segment-valued
        pressure difference (``x[to] - x[from]``), one per segment.
      - ``div(x)``: segment-valued array -> node-valued array (graph
        divergence: outflow positive at the "from" node).
      - ``aver(x)``: node-valued array -> segment-valued array, the simple
        average of each segment's two endpoint values.
      - ``segment_upstream(flag, val)``: ``flag`` a per-segment boolean
        (True = flow from "from"->"to"), ``val`` an *internal-node-indexed*
        array (size ``n_nodes - 1``, i.e. ``val[k]`` is node ``k+1``'s
        value); returns the per-segment upstream value. The top node (index
        0) has no entry in ``val``, so a segment whose upstream is the top
        node falls back to node 1's value, matching MRST's
        ``segmentUpstreamValue`` clamp.
      - ``C``: the ``(n_segments, n_nodes)`` signed incidence matrix.
    """
    topo = _np.asarray(segments_topo, dtype=_np.int64).reshape(-1, 2)
    ns = topo.shape[0]
    rows = _np.repeat(_np.arange(ns), 2)
    cols = topo.ravel()

    C = _sp.csr_matrix((_np.tile([1.0, -1.0], ns), (rows, cols)), shape=(ns, n_nodes))
    neg_C = -C

    def grad(x):
        return _apply(neg_C, x)

    def div(x):
        return _apply(C.T, x)

    aver_mat = _sp.csr_matrix((_np.tile([1.0, 1.0], ns), (rows, cols)), shape=(ns, n_nodes))
    row_sums = _np.asarray(aver_mat.sum(axis=1)).ravel()
    aver_mat = _sp.diags(1.0 / row_sums) @ aver_mat

    def aver(x):
        return _apply(aver_mat, x)

    def segment_upstream(flag, val):
        flag = _np.asarray(flag, dtype=bool)
        frm, to = topo[:, 0], topo[:, 1]
        ix = _np.where(flag, frm, to)
        idx_internal = _np.maximum(ix - 1, 0)
        return val[idx_internal]

    return {"grad": grad, "div": div, "aver": aver, "segment_upstream": segment_upstream, "C": C, "n_segments": ns}
