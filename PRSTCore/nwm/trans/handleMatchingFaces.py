"""Port of MRST ``handleMatchingFaces``: compute the intersection relation
between the *layered* (matching) boundaries of two subgrids of the
near-wellbore model.

The faces on the layered boundaries are matching, and only the common areas
are obtained from the cells and boundary nodes of ``G1``.
"""

import numpy as np

from ..utils.tabulate_NWM import tabulate_NWM


def _fun_faces(G, c):
    return G['cells']['faces'][G['cells']['facePos'][c]:G['cells']['facePos'][c + 1], 0]


def _fun_nodes(G, f):
    return G['faces']['nodes'][G['faces']['nodePos'][f]:G['faces']['nodePos'][f + 1]]


def handleMatchingFaces(G1, cells1, bdnodes1, G2):
    """Compute the face intersection relation between the layered boundaries
    of subgrids ``G1`` and ``G2`` (``G2`` located inside ``G1``).

    Returns an ``n x 3`` array:
        column 1 - Face of G1
        column 2 - Face of G2
        column 3 - Areas of intersection subfaces
    """
    g2 = G2['surfGrid']

    # Find boundary faces of the 2D surface grid
    faces2 = np.flatnonzero(~np.all(g2['faces']['neighbors'] >= 0, axis=1))
    nodes2 = [_fun_nodes(g2, f) for f in faces2]

    bdnodes2 = np.asarray(g2['nodes']['boundary'], dtype=np.int64)
    bdnodesTwoPts = np.concatenate([bdnodes2, bdnodes2[:1]])

    bdfaces2 = np.zeros(len(bdnodes2), dtype=np.int64)
    for i in range(len(bdfaces2)):
        idx = [np.all(np.isin(n, bdnodesTwoPts[i:i + 2])) for n in nodes2]
        bdfaces2[i] = faces2[np.array(idx, dtype=bool)][0]

    # Find boundary faces of G1 (layered)
    nRef = np.asarray(G2['layers']['refinement'], dtype=np.int64)
    layerIdx = np.concatenate([[0], np.cumsum(nRef)])   # 0-based layer starts
    nxyfaces = np.count_nonzero(G2['faces']['surfaces'] == 0) / G2['layers']['num']

    relation = []
    for k in range(len(cells1)):
        faces1 = np.concatenate([_fun_faces(G1, c) for c in cells1[k]])
        tab = tabulate_NWM(faces1)
        faces1 = tab[tab[:, 1] == 1, 0]
        nodes1 = [_fun_nodes(G1, f) for f in faces1]
        # MATLAB: bdnFourPts = [bdnodes1{k}, bdnodes1{k+1}]; -- horizontal
        # concatenation of two column vectors gives an (n, 2) matrix, row i
        # = [bdnodes1{k}(i), bdnodes1{k+1}(i)] (pairing the i-th boundary
        # node of layer k with the i-th of layer k+1); column_stack is the
        # correct NumPy equivalent, not vstack (which would instead stack
        # the two arrays as two ROWS of one long vector).
        bdnFourPts = np.column_stack([bdnodes1[k], bdnodes1[k + 1]])
        bdnFourPts = np.vstack([bdnFourPts, bdnFourPts[0]])
        ca0 = []
        for i in range(len(bdnodes1[k])):
            idx = [np.all(np.isin(n, bdnFourPts[i:i + 2])) for n in nodes1]
            bdface1 = faces1[np.array(idx, dtype=bool)][0]
            faces2_i = bdfaces2[i] + (np.arange(layerIdx[k], layerIdx[k + 1])
                                      * int(nxyfaces))
            areas_i = G2['faces']['areas'][faces2_i]
            ca0.append(np.column_stack([np.full(faces2_i.size, bdface1),
                                        faces2_i, areas_i]))
        if ca0:
            relation.append(np.vstack(ca0))
    if relation:
        return np.vstack(relation)
    return np.empty((0, 3))
