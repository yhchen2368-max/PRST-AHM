"""Port of MRST ``computeRadTransFactor``: radial half-transmissibility
factors for a 2D radial grid (``halfTrans = perm .* ft``), assuming
steady-state flow and using the 'transmissibility centre' obtained by
integrating the pressure over the cell area."""

import numpy as np

from .._core import computeGeometry, mergeOptions
from ..utils.tri_area import tri_area


def _fAngle(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1, 2) / np.linalg.norm(v2, 2))


def computeI(pA, pB, pC):
    """Integral coefficients for the triangle ``(pA, pB, pC)``.

    Returns ``(IR, IA, S, thA, b, c)``.
    """
    thA = _fAngle(pB - pA, pC - pA)
    thB = _fAngle(pA - pB, pC - pB)
    thC = _fAngle(pB - pC, pA - pC)

    c = np.linalg.norm(pA - pB, 2)
    b = np.linalg.norm(pA - pC, 2)
    a = np.linalg.norm(pB - pC, 2)

    IR = (c / a * np.sin(thB) * thA
          + b / a * np.cos(thC) * np.log(b)
          + c / a * np.cos(thB) * np.log(c)
          - 1.5)

    IA = (b / a * np.sin(thC)
          * (np.log(np.sin(thC) / np.sin(thB))
             + (np.pi / 2 - thC) * np.cos(thC) / np.sin(thC)
             - (np.pi / 2 - thB) * np.cos(thB) / np.sin(thB))
          + np.pi / 2 - thB)

    S = tri_area(pA, pB, pC)
    return IR, IA, S, thA, b, c


def computeRadTransFactor(G, pW, skin, **kwargs):
    """Compute the radial half transmissibility factor ``ft`` for the 2D
    radial grid ``G``, corresponding to ``G['cells']['faces']``.

    Parameters
    ----------
    G : dict
        Radial grid structure built by ``buildRadialGrid``; requires
        ``G['radDims']`` = ``[nA, nR]`` or ``[nA, nR1, nR2]`` (only cells
        with r-indices within ``1 - nR1`` are involved in the computation).
    pW : array_like
        2D coordinate of the well point.
    skin : float
        Skin factor of the well.
    nodeCoords : array_like, optional
        Provided 2D coordinates of the grid nodes (default: ``G.nodes``).
    """
    opt = mergeOptions({'nodeCoords': None}, **kwargs)
    if opt['nodeCoords'] is not None:
        p = np.asarray(opt['nodeCoords'], dtype=float)
    else:
        p = np.asarray(G['nodes']['coords'], dtype=float)
    G = dict(G)
    G['nodes']['coords'] = p
    G = computeGeometry(G)

    nA = int(G['radDims'][0])
    nRT = np.asarray(G['radDims'][1:], dtype=np.int64).ravel()
    ft = np.full(nA * int(np.sum(nRT)) * 4, np.nan)
    pA = np.asarray(pW, dtype=float)

    # nodes:
    # B1 : R- & A+, 1 & 2
    # C1 : R- & A-, 1 & 4
    # C2 : R+ & A-, 3 & 4
    # B2 : R+ & A+, 2 & 3
    N = np.array([[0, 1], [0, 3], [2, 3], [1, 2]])

    for c in range(nA * int(nRT[0])):
        fPos = np.arange(G['cells']['facePos'][c], G['cells']['facePos'][c + 1])
        f = G['cells']['faces'][fPos, 0]
        n = [G['faces']['nodes'][G['faces']['nodePos'][fi]:G['faces']['nodePos'][fi + 1]]
             for fi in f]
        nBC = np.zeros(4, dtype=np.int64)
        for i in range(4):
            nBC[i] = np.intersect1d(n[N[i, 0]], n[N[i, 1]])[0]
        pB1 = p[nBC[0]]
        pC1 = p[nBC[1]]
        pC2 = p[nBC[2]]
        pB2 = p[nBC[3]]
        IR1, IA1, S1, _, b1, c1 = computeI(pA, pB1, pC1)
        IR2, IA2, S2, thA, b2, c2 = computeI(pA, pB2, pC2)
        S0 = S2 - S1
        r0 = np.exp(S2 / S0 * IR2 - S1 / S0 * IR1)
        dth0 = S2 / S0 * IA2 - S1 / S0 * IA1
        r_ = np.linalg.norm(G['faces']['centroids'][f[0]] - pA, 2)
        r = np.linalg.norm(G['faces']['centroids'][f[2]] - pA, 2)
        if c < nA:
            # Equivalent radius
            r_ = r_ * np.exp(-skin)
        ft_r_ = thA / np.log(r0 / r_)          # r-
        ft_r = thA / np.log(r / r0)            # r+
        ft_a_B = np.log(b2 / b1) / (thA - dth0)  # th B
        ft_a_C = np.log(c2 / c1) / dth0          # th C

        # face type:
        # 1 - Radial -
        # 2 - Angular B1B2
        # 3 - Radial +
        # 4 - Angular C1C2
        ft[fPos] = np.array([ft_r_, ft_a_B, ft_r, ft_a_C])
    return ft
