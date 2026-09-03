"""Port of MRST ``generateHWGridNodes``: generate the 3D points of all radial
HW grid surfaces and the 2D planar points."""

import numpy as np

from .._core import griddata, inpolygon
from ..utils.bisection import bisection
from ..utils.computePD import computePD
from ..utils.convertTo3DPlane import convertTo3DPlane
from ..utils.convertToXYPlane import convertToXYPlane


def _cart2pol(x, y):
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    return theta, r


def _pol2cart(theta, r):
    return r * np.cos(theta), r * np.sin(theta)


def generateHWGridNodes(GV, packed, well, radPara):
    """Generate 3D points of all radial HW grid surfaces and 2D planar points.

    Returns ``(pSurfs, pSurfXY, wellbores)``:
      pSurfs    - 3D points of all HW grid surfaces
      pSurfXY   - pSurfs in the xy plane, used to compute the radial
                  transmissibility factors
      wellbores - Structure of the casing and screen, used to generate nodes
                  and segments for the multi-segment well
    """
    gridType = radPara['gridType']
    if gridType == 'pureCircular':
        generator = CircularRadialPoints
    elif gridType == 'gradual':
        generator = GradualRadialPoints
    else:
        raise ValueError(f'generateHWGridNodes: Unknown radial grid type: {gridType}')

    ns = well['segmentNum']
    results = [generator(GV, packed, well, radPara, ii) for ii in range(ns + 1)]
    pSurfs = [r[0] for r in results]
    pSurfXY = [r[1] for r in results]
    wellbores = [r[2] for r in results]
    return pSurfs, pSurfXY, wellbores


def _assignWellbore(p, bdn, origin, T, R, well, ii, theta, xx, yy, zz):
    """Store the borewall (and screen) nodes, prepared for the multi-segment
    well."""
    nA = len(bdn)
    pW = p[:nA, :]
    wellbore = {'wall': {'radius': well['radius'][ii], 'coords': pW},
                'reservoirCells': np.arange(nA)}
    if 'screenRadius' in well:
        rS = well['screenRadius'][ii]
        assert rS < well['radius'][ii], \
            'The screen radius must be less than the casing radius!'
        pxS, pyS = _pol2cart(theta, rS)
        pxS = pxS + origin[0]
        pyS = pyS + origin[1]
        pzS = griddata(xx, yy, zz, pxS, pyS)
        pS = np.column_stack([pxS, pyS, pzS])
        pS = convertTo3DPlane(pS, T, R)
        wellbore['screen'] = {'radius': rS, 'coords': pS}
    return wellbore


def CircularRadialPoints(GV, packed, well, radPara, ii):
    """Generate pure circular radial points for a single well node ``ii``."""
    # Assign parameters
    rM = radPara['maxRadius']
    nRT = np.atleast_1d(np.asarray(radPara['nRadCells'], dtype=int))
    bdn = packed['bdyNodes'][ii]
    vxID = packed['vertexID']
    origin = np.asarray(well['trajectory'][ii], dtype=float)
    rW = well['radius'][ii]

    # Coordinate transformation for outer boundary nodes and well center
    pob0, origin, T, R, _ = convertToXYPlane(GV['nodes']['coords'][bdn], vxID, origin)
    origin = origin.ravel()
    pob = pob0 - origin
    pob = pob[:, :2]

    # Theta and radii of the boundary nodes
    theta, rob = _cart2pol(pob[:, 0], pob[:, 1])
    assert np.all(rob > rM), 'Max radius of the radial grid is greater ' \
                             'than the boundary radius, please reduce the value'

    # Generate circular radial points
    rr = np.exp(np.linspace(np.log(rW), np.log(rM), nRT[0]))
    RR, THETA = np.meshgrid(rr, theta)
    px, py = _pol2cart(THETA, RR)
    px = px.ravel(order='F')
    py = py.ravel(order='F')

    # Add the boundary points
    px = np.concatenate([px, pob[:, 0]])
    py = np.concatenate([py, pob[:, 1]])

    # Return the planar points
    pxy = np.column_stack([px, py])

    # Map back to the original coordinate
    px = px + origin[0]
    py = py + origin[1]

    # Get pz by interpolation
    xx = np.concatenate([pob0[:, 0], [origin[0]]])
    yy = np.concatenate([pob0[:, 1], [origin[1]]])
    zz = np.concatenate([pob0[:, 2], [origin[2]]])
    pz = griddata(xx, yy, zz, px, py)
    pz[:len(bdn)] = pob0[:, 2]

    # Return the 3D points
    p = np.column_stack([px, py, pz])
    p = convertTo3DPlane(p, T, R)

    # Store borewall and screen nodes
    wellbore = _assignWellbore(p, bdn, origin, T, R, well, ii, theta, xx, yy, zz)
    return p, pxy, wellbore


def GradualRadialPoints(GV, packed, well, radPara, ii):
    """Generate gradual radial points for a single well node ``ii``."""
    # Assign parameters
    boxRatio = np.asarray(radPara['boxRatio'], dtype=float)
    nRT = np.atleast_1d(np.asarray(radPara['nRadCells'], dtype=int))
    pDMult = radPara['pDMult']
    offCenter = radPara['offCenter']
    bdn = packed['bdyNodes'][ii]
    vxID = packed['vertexID']
    origin = np.asarray(well['trajectory'][ii], dtype=float)
    rW = well['radius'][ii]

    # Coordinate transformation for outer boundary nodes and well center
    pob0, origin, T, R, _ = convertToXYPlane(GV['nodes']['coords'][bdn], vxID, origin)
    origin = origin.ravel()
    pob = pob0 - origin
    pob = pob[:, :2]

    # Rectangular box size
    a = boxRatio[0] * (np.max(pob[:, 0]) - np.min(pob[:, 0]))
    b = boxRatio[1] * (np.max(pob[:, 1]) - np.min(pob[:, 1]))

    # Get four box vertices
    if offCenter:
        pobm = np.mean(pob, axis=0) / 2
        pbv = np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) / 2
        pbv = pbv + pobm
        # Well center distance to the boundary
        xw = pbv[1, 0]
        yw = -pbv[1, 1]
    else:
        pbv = np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) / 2
        # Distance of well center to the boundary
        xw = a / 2
        yw = b / 2
    assert np.all(inpolygon(pbv[:, 0], pbv[:, 1], pob[:, 0], pob[:, 1])), \
        'Box vertexes outside the well region were detected, try to reduce ' \
        'the box size'

    # Number of angular cells
    nA = len(bdn)

    # Get points on the box boundary
    pbb = np.zeros((nA, 2))
    sg1 = np.sign(pob[vxID, :2])
    sg2 = np.sign(pbv)
    idx = np.array([np.flatnonzero(np.all(sg1 == sg2[i, :], axis=1))[0]
                    for i in range(4)])
    pbv = pbv[idx]

    # Assign four box vertices
    pbb[vxID, :] = pbv
    assert vxID[0] == 0
    for i in range(4):
        i1 = vxID[i]
        if i < 3:
            i2 = vxID[i + 1]
            im = np.arange(i1 + 1, i2)
        else:
            i2 = 0
            im = np.arange(i1 + 1, nA)
        p1 = pob[i1, :]
        p2 = pob[i2, :]
        pm12 = pob[im, :]
        l1m = np.sqrt(np.sum((pm12 - p1) ** 2, axis=1))
        l1m = l1m / np.linalg.norm(p2 - p1)
        p3 = pbb[i1, :]
        p4 = pbb[i2, :]
        p34m = p3 + l1m[:, None] * (p4 - p3)
        pbb[im, :] = p34m  # Assign points in box edges

    # Get points between pob and pbb
    space = np.linspace(0, 1, nRT[1] + 1)
    space = space[1:-1]
    px1 = pob[:, 0][:, None] + (pbb[:, 0] - pob[:, 0])[:, None] * space[None, :]
    py1 = pob[:, 1][:, None] + (pbb[:, 1] - pob[:, 1])[:, None] * space[None, :]

    # Get points inside the box
    pD = getPD(rW, a, b, xw, yw, nRT, pDMult)
    px2 = np.zeros((nA, nRT[0] - 1))
    py2 = np.zeros((nA, nRT[0] - 1))
    rij = np.zeros((nA, nRT[0] - 1))
    theta, rM = _cart2pol(pbb[:, 0], pbb[:, 1])
    for i in range(nA):
        for j in range(nRT[0] - 1):
            fun = lambda r: computePD(r * np.cos(theta[i]), r * np.sin(theta[i]),
                                      a, b, xw, yw) - pD[j]
            rij[i, j] = bisection(fun, rW, rM[i], 1e-5)[0]
            px2[i, j], py2[i, j] = _pol2cart(theta[i], rij[i, j])

    # Wellbore (Casing) points
    pxW, pyW = _pol2cart(theta, rW)

    # Put together, reverse for numbering
    px = np.column_stack([pxW, px2[:, ::-1], pbb[:, 0], px1[:, ::-1], pob[:, 0]])
    py = np.column_stack([pyW, py2[:, ::-1], pbb[:, 1], py1[:, ::-1], pob[:, 1]])

    # Return the planar points
    pxy = np.column_stack([px.ravel(order='F'), py.ravel(order='F')])

    # Map back to the original coordinate
    px = px.ravel(order='F') + origin[0]
    py = py.ravel(order='F') + origin[1]

    # Get pz by interpolation
    xx = np.concatenate([pob0[:, 0], [origin[0]]])
    yy = np.concatenate([pob0[:, 1], [origin[1]]])
    zz = np.concatenate([pob0[:, 2], [origin[2]]])
    pz = griddata(xx, yy, zz, px, py)
    pz[:len(bdn)] = pob0[:, 2]

    # Return the 3D points
    p = np.column_stack([px, py, pz])
    p = convertTo3DPlane(p, T, R)

    # Store borewall and screen nodes
    wellbore = _assignWellbore(p, bdn, origin, T, R, well, ii, theta, xx, yy, zz)
    return p, pxy, wellbore


def getPD(rW, a, b, xw, yw, nRT, pDMult):
    """Get pD inside the box."""
    pDM = computePD(rW, 0, a, b, xw, yw)
    pD = np.linspace(pDM / pDMult, pDM, nRT[0])
    pD = pD[:-1]
    return pD
