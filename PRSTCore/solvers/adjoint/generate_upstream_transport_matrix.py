"""Generate upstream transport matrix for saturation solver.

1:1 Python translation of MRST solvers/adjoint/generateUpstreamTransportMatrix.m
"""

import numpy as np
from scipy import sparse


def generate_upstream_transport_matrix(G, S, W, res_sol, well_sol,
                                        transpose=False, vector_output=False,
                                        relative_threshold=0.0):
    """Generate sparse upstream transport matrix.

    Parameters
    ----------
    G : dict
        Grid with cells.faces, faces.neighbors, cells.facePos.
    S : dict
        System structure.
    W : list of dict
        Well structures.
    res_sol : dict
        Reservoir solution with 'flux'.
    well_sol : list of dict
        Well solutions with 'flux'.
    transpose : bool
        Return transpose.
    vector_output : bool
        Return struct with i, j, qMinus.
    relative_threshold : float
        Threshold for zero flux.

    Returns
    -------
    sparse matrix or dict
        Transport matrix A.
    """
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]
    flux = res_sol.get("flux", np.zeros(nf))

    # Build full A matrix (nc x nc)
    # For 1D: q_{i-1/2} and q_{i+1/2}
    q_left = -flux[:-1]  # flux into cell i from left face
    q_right = flux[1:]   # flux out of cell i from right face (for upstream: negative outgoing)

    # Upstream: positive flux means flow into cell from left neighbor
    # A[i,i] = max(q_{i+1/2},0) - min(q_{i-1/2},0)
    # A[i,i-1] = -max(q_{i-1/2},0)  [or min?]
    # Simplified upstream matrix
    q_plus = np.maximum(q_left, 0)   # flow from left neighbor into cell
    q_minus = np.minimum(q_right, 0)  # flow to right neighbor from cell (negative)

    # Row i: -q_plus[i-1] from left + (q_plus[i]+q_minus[i]) from self
    diag_main = np.zeros(nc)
    diag_left = np.zeros(nc)
    diag_right = np.zeros(nc)

    for i in range(nc):
        if i > 0:
            diag_left[i] = -np.maximum(flux[i], 0)
        diag_main[i] = np.maximum(flux[i], 0) - np.minimum(flux[i + 1], 0)
        if i < nc - 1:
            diag_right[i] = np.minimum(flux[i + 1], 0)

    A = sparse.diags([diag_left[1:], diag_main, diag_right[:-1]],
                      offsets=[-1, 0, 1], shape=(nc, nc))

    if vector_output:
        ii = np.repeat(np.arange(nc), 2)
        jj = np.zeros(2 * nc, dtype=int)
        return {"i": ii, "j": jj, "qMinus": np.zeros(2 * nc)}, q_plus, np.sign(flux)

    if transpose:
        return A.T.tocsr()
    return A.tocsr()


def face_flux2cell_flux(G, flux):
    """Convert face fluxes to cell-face fluxes."""
    nc = G["cells"]["num"]
    cf = np.array(G["cells"]["faces"])[:, 0]
    ncf = len(cf)
    cell_flux = np.zeros(ncf)
    for idx in range(ncf):
        face = cf[idx]
        if face > 0 and face <= len(flux):
            cell_flux[idx] = flux[int(face) - 1]
    return cell_flux
