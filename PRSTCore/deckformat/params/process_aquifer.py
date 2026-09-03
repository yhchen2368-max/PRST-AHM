"""Python port of MRST's ``processAquifer.m`` (mrst-2026a/model-io/deckformat/
params/rock): parses the ``AQUANCON``/``AQUFETP`` deck keywords (Fetkovich
analytic aquifer) into a per-connection table.

Scope: the common case where every ``AQUANCON`` box cell is itself active
and the aquifer attaches directly to that cell's own boundary face in the
given direction (MRST's own "aqcellin" path). Not ported: the "aqcellout"
path for a box specified one cell *outside* the active domain (requires a
neighbor-cell lookup back into the domain) -- not exercised by the local
MSW.data reference deck this port is validated against.
"""
from __future__ import annotations

import numpy as _np

_DIR_FLAGS = {"I-": 1, "I+": 2, "J-": 3, "J+": 4, "K-": 5, "K+": 6}
# 0-based axis + which side (0=low/negative-face, 1=high/positive-face)
_DIR_AXIS_SIDE = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1), 5: (2, 0), 6: (2, 1)}


def _strip_quotes(tok) -> str:
    s = str(tok)
    return s[1:-1] if len(s) >= 2 and s[0] == "'" and s[-1] == "'" else s


def process_aquifer(deck: dict, G: dict) -> dict:
    """Port of ``processAquifer.m``.

    Returns ``{'aquifers' (nconn, 8), 'aquind' (dict, 0-based column
    indices: aquid, conn, pvttbl, J, C, alpha, depthconn, depthaq),
    'initval' ({'pressures', 'volumes'}, one per aquifer id),
    'aquiferprops' ({'depthaq', 'C', 'J', 'pvttbl'}, one per aquifer id)}``.
    ``conn`` is a 0-based active-cell index.
    """
    aquind = {"aquid": 0, "conn": 1, "pvttbl": 2, "J": 3, "C": 4,
              "alpha": 5, "depthconn": 6, "depthaq": 7}

    solution = deck["SOLUTION"]
    aquancon = [r for r in solution["AQUANCON"] if r]
    aqufetp = _np.atleast_2d(_np.asarray(solution["AQUFETP"], dtype=float))
    if aqufetp.shape[1] < 7:
        # AQUFETP's trailing PVT-table-number field defaults to 1 when omitted.
        pad = _np.ones((aqufetp.shape[0], 7 - aqufetp.shape[1]))
        aqufetp = _np.hstack([aqufetp, pad])
    # MRST's corresponding rejection is commented out.  FAHM deliberately
    # accepts NaN initial pressure here and fills it from restart aquiferSol
    # (ECLIPSE) or connected reservoir pressures (tNavigator) after state0
    # has been imported.

    # AQUANCON/AQUFETP aren't covered by convert_deck_units' general
    # per-keyword table, so convert their physical columns here directly
    # from the deck's own recorded unit factors (matching the exact
    # AQUFETP column semantics: depth [length], pressure [press], volume
    # [reservoir volume], compressibility [1/press], J [reservoir volume
    # / time / press]).
    uf = deck.get("_unit_factors", {})
    f_length = float(uf.get("length", 1.0))
    f_press = float(uf.get("press", 1.0))
    f_resvol = float(uf.get("resvolume", uf.get("liqvol_r", 1.0)))
    f_compr = float(uf.get("compressibility", 1.0))
    f_J = f_resvol / (float(uf.get("time", 1.0)) * f_press)
    aqufetp = aqufetp.copy()
    aqufetp[:, 1] *= f_length          # depth
    aqufetp[:, 2] *= f_press           # initial pressure
    aqufetp[:, 3] *= f_resvol          # initial volume
    aqufetp[:, 4] *= f_compr           # total compressibility
    aqufetp[:, 5] *= f_J               # productivity index

    cart_dims = _np.asarray(G["cartDims"], dtype=int)
    nx, ny, nz = (int(cart_dims[d]) for d in range(3))
    nfull = nx * ny * nz
    cart_to_active = _np.full(nfull, -1, dtype=_np.int64)
    cart_to_active[G["cells"]["indexMap"]] = _np.arange(G["cells"]["num"], dtype=_np.int64)

    rows_i, rows_j, rows_k = [], [], []
    rows_aquid, rows_influx, rows_influxmult, rows_faceflag = [], [], [], []
    for rec in aquancon:
        aquid = int(rec[0])
        imin, imax = int(rec[1]), int(rec[2])
        jmin, jmax = int(rec[3]), int(rec[4])
        kmin, kmax = int(rec[5]), int(rec[6])
        dirstr = _strip_quotes(rec[7]).upper()
        flag = _DIR_FLAGS[dirstr]
        try:
            influxcoef = float(rec[8])
        except (TypeError, ValueError):
            influxcoef = _np.nan
        try:
            influxmultcoef = float(rec[9])
        except (TypeError, ValueError):
            influxmultcoef = 1.0
        if influxmultcoef != influxmultcoef:  # AQUANCON's defaulted multiplier is 1.0, not "use area"
            influxmultcoef = 1.0

        ii, jj, kk = _np.meshgrid(_np.arange(imin, imax + 1), _np.arange(jmin, jmax + 1),
                                   _np.arange(kmin, kmax + 1), indexing="ij")
        n = ii.size
        rows_i.append(ii.ravel()); rows_j.append(jj.ravel()); rows_k.append(kk.ravel())
        rows_aquid.append(_np.full(n, aquid))
        rows_influx.append(_np.full(n, influxcoef))
        rows_influxmult.append(_np.full(n, influxmultcoef))
        rows_faceflag.append(_np.full(n, flag))

    I = _np.concatenate(rows_i); J = _np.concatenate(rows_j); K = _np.concatenate(rows_k)
    aquid_col = _np.concatenate(rows_aquid)
    influxcoef = _np.concatenate(rows_influx)
    influxmultcoef = _np.concatenate(rows_influxmult)
    faceflag = _np.concatenate(rows_faceflag)

    cart_idx = (I - 1) + nx * (J - 1) + nx * ny * (K - 1)
    active = cart_to_active[cart_idx]
    keep = active >= 0
    conn = active[keep]
    aquid_col = aquid_col[keep]
    influxcoef = influxcoef[keep]
    influxmultcoef = influxmultcoef[keep]
    faceflag = faceflag[keep]

    nconn = conn.size
    naq = int(_np.max(aqufetp[:, 0]))

    # Find, for each connection cell, its own half-face matching the
    # AQUANCON direction (MRST's face-tag convention: 1..6 = I-,I+,J-,J+,K-,K+).
    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    aqfaces = _np.zeros(nconn, dtype=_np.int64)
    for i in range(nconn):
        c = conn[i]
        hf = cell_faces[face_pos[c]:face_pos[c + 1]]
        tag = hf[:, 1] if hf.shape[1] > 1 else None
        match = hf[tag == faceflag[i]] if tag is not None else _np.empty((0, hf.shape[1]))
        aqfaces[i] = match[0, 0] if match.shape[0] else -1

    valid = aqfaces >= 0
    aqareas = _np.zeros(nconn)
    aqareas[valid] = G["faces"]["areas"][aqfaces[valid]]

    use_area = _np.isnan(influxcoef)
    influxcoef = _np.where(use_area, aqareas, influxcoef)
    alpha_raw = influxmultcoef * influxcoef

    # Normalize alpha to sum to 1 within each aquifer id.
    alpha = _np.zeros(nconn)
    for aq in _np.unique(aquid_col):
        m = aquid_col == aq
        total = _np.sum(alpha_raw[m])
        alpha[m] = alpha_raw[m] / total if total != 0 else 0.0

    aquiferprops_by_id = {
        "depthaq": {}, "C": {}, "J": {}, "pvttbl": {},
    }
    for row in aqufetp:
        aq = int(row[0])
        aquiferprops_by_id["depthaq"][aq] = row[1]
        aquiferprops_by_id["C"][aq] = row[4]
        aquiferprops_by_id["J"][aq] = row[5]
        aquiferprops_by_id["pvttbl"][aq] = row[6]

    depthaq = _np.array([aquiferprops_by_id["depthaq"][a] for a in aquid_col])
    C = _np.array([aquiferprops_by_id["C"][a] for a in aquid_col])
    Jv = _np.array([aquiferprops_by_id["J"][a] for a in aquid_col])
    pvttbl = _np.array([aquiferprops_by_id["pvttbl"][a] for a in aquid_col])
    depthconn = G["cells"]["centroids"][conn, 2]

    aquifers = _np.zeros((nconn, 8))
    aquifers[:, aquind["aquid"]] = aquid_col
    aquifers[:, aquind["conn"]] = conn
    aquifers[:, aquind["pvttbl"]] = pvttbl
    aquifers[:, aquind["J"]] = Jv
    aquifers[:, aquind["C"]] = C
    aquifers[:, aquind["alpha"]] = alpha
    aquifers[:, aquind["depthconn"]] = depthconn
    aquifers[:, aquind["depthaq"]] = depthaq

    order = _np.argsort(aqufetp[:, 0])
    initval = {
        "pressures": aqufetp[order, 2],
        "volumes": aqufetp[order, 3],
    }
    aquiferprops = {
        "depthaq": aqufetp[order, 1],
        "C": aqufetp[order, 4],
        "J": aqufetp[order, 5],
        "pvttbl": aqufetp[order, 6],
    }

    return {"aquifers": aquifers, "aquind": aquind, "initval": initval, "aquiferprops": aquiferprops}
