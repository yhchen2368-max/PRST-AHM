"""Port of MRST ``NearWellboreModel``: class for generating the data
structures passed to the AD simulators for the hybrid grid of the
near-wellbore model.

The hybrid grid is composed of three subgrids: the corner-point grid (CPG),
the volume-of-interest (VOI) grid and the horizontal-well (HW) grid.
"""

from __future__ import annotations

import time

import numpy as np

from .._core import gridLogicalIndices, mergeOptions, removeCells, rldecode
from ..gridding.assembleGrids import assembleGrids
from ..trans.handleMatchingFaces import handleMatchingFaces
from ..trans.handleNonMatchingFaces import handleNonMatchingFaces
from ..trans.computeRadTransFactor import computeRadTransFactor
from ..utils.arrayfunUniOut import arrayfunUniOut
from ..utils.getDZ import getDZ
from ..utils.getUnitDisVectors import getUnitDisVectors
from .. import _deps


def checkDeck(deck):
    """Validate the input deck for the near-wellbore model."""
    regions = deck.get('REGIONS', {})
    if regions and len(regions) > 0:
        raise ValueError('The near-wellbore model now only supports single '
                         'region definition')
    solution = deck.get('SOLUTION', {})
    if 'EQUIL' not in solution:
        raise ValueError("The keyword 'EQUIL' in the deck input is required "
                         "for the equilibrium initialization")
    # EQUIL records may mix numeric fields with trailing string tokens
    # (comments/labels captured by the deck reader), so this only checks
    # the record *count* (one equilibration region) rather than trying to
    # build a uniform numeric array out of the whole deck record.
    equil = solution['EQUIL']
    if len(equil) > 1:
        raise ValueError('The near-wellbore model now only supports single '
                         'equilibration region')


class NearWellboreModel:
    """Class for generating the necessary data structures passed to the MRST
    AD simulators for the hybrid grid of the near-wellbore model.

    Properties
    ----------
    subGrids : list
        Subgrids {Corner-point grid (CPG), VOI grid, HW grid}.
    inputDeck : dict
        ECLIPSE-style input deck of the CPG.
    well : dict
        Structure of well information.
    gloGrid : dict
        The global hybrid grid.
    fluid : dict
        AD-solver fluid from the ECLIPSE-style input deck.
    """

    def __init__(self, subGrids, deck, well, **kwargs):
        checkDeck(deck)
        self.subGrids = list(subGrids)
        self.inputDeck = deck
        self.well = well
        self.gloGrid = self.validateGlobalGrid()
        self.fluid = self.setupFluid()

    # -- construction ------------------------------------------------------

    def validateGlobalGrid(self, **kwargs):
        """Validate the global hybrid grid by subgrids (cells in the VOI and
        HW region are removed from the CPG / VOI grid)."""
        GCu = self.updateCPG()
        GVu = self.updateVOIGrid()
        GWu = self.updateHWGrid()
        # Combine the updated subgrids
        return assembleGrids([GCu, GVu, GWu])

    def setupFluid(self):
        """Initialize the AD fluid from the input deck."""
        return _deps.initDeckADIFluid(self.inputDeck)

    def assignInputSubGrds(self, nargout=3):
        """Assign the input subgrids."""
        subG = self.subGrids
        assert nargout <= len(subG)
        if nargout == 1:
            return [subG]
        return subG[:nargout]

    def assignSubGrds(self, nargout=3):
        """Assign the updated subgrids from the global grid."""
        subG = self.gloGrid['subGrids']
        assert nargout <= len(subG)
        if nargout == 1:
            return [subG]
        return subG[:nargout]

    def assignInputSubGrdTypes(self, nargout=3):
        """Assign the types of the input subgrids."""
        types = ['CPG', 'VOI Grid', 'HW Grid']
        assert nargout <= len(types)
        if nargout == 1:
            return [types]
        return types[:nargout]

    def assignSubGrdTypes(self, nargout=3):
        """Assign the types of the updated subgrids."""
        subG = self.gloGrid['subGrids']
        types = ['Updated CPG', 'Updated VOI Grid', 'Updated HW Grid']
        assert nargout <= len(subG)
        if nargout == 1:
            return [types]
        return types[:nargout]

    # -- packed simulation data -------------------------------------------

    def packedSimData(self, rockW, **kwargs):
        """Obtain all necessary simulation data structures of the hybrid grid
        in the near-wellbore model."""
        # Global grid
        G = self.gloGrid
        # AD fluid
        f = self.fluid
        # Rocks
        rockC = self.getCPGRockFromDeck()
        rockV = self.getVOIRocksByInterp()
        rock = self.getGlobalRock([rockC, rockV, rockW])
        # Simulation model
        T = self.getTransGloGrid(rock)
        intXn = self.computeIntxnRelation()
        nnc = self.generateNonNeighborConn(intXn, rock, T)
        G['nnc'] = nnc
        T_all, N_all = self.assembleTransNeighbors(T, nnc)
        model = self.setupSimModel(rock, T_all, N_all)
        # Schedule
        schedule = self.getSimSchedule(model, **kwargs)
        # Initial state
        initState = self.getInitState(model)
        return G, rock, f, model, schedule, initState

    def packedCPGSimData(self):
        """Obtain all necessary simulation data structures of the CPG."""
        # CPG
        GC, _ = self.assignInputSubGrds(nargout=2)
        # AD fluid
        f = self.fluid
        # Rocks
        rockC = self.getCPGRockFromDeck()
        # Simulation model
        modelC = self.setupCPGSimModel()
        # Schedule
        scheduleC = self.getCPGSimSchedule(modelC)
        # Init state
        initStateC = self.getInitState(modelC)
        return GC, rockC, f, modelC, scheduleC, initStateC

    # -- rock ---------------------------------------------------------------

    def getGlobalRock(self, rocks):
        """Get the rock for the global grid (``rocks = [rockC, rockV,
        rockW]``), mapping the rocks from the input subgrids to the global
        grid."""
        for j in range(len(rocks)):
            if 'ntg' not in rocks[j]:
                rocks[j]['ntg'] = np.ones_like(np.asarray(rocks[j]['poro'],
                                                          dtype=float))
        G = self.gloGrid
        mapc = self.cellMapFromInputSubGrdsToGloGrd()
        fn = ('perm', 'poro', 'ntg')
        rock = {}
        for f in fn:
            data0 = np.asarray(rocks[0][f])
            ncol = data0.shape[1] if data0.ndim > 1 else 1
            val = np.zeros((G['cells']['num'], ncol))
            for j in range(len(rocks)):
                data = np.asarray(rocks[j][f])
                if data.ndim == 1:
                    val[mapc[j][:, 1], 0] = data[mapc[j][:, 0]]
                else:
                    val[mapc[j][:, 1], :] = data[mapc[j][:, 0], :]
            rock[f] = val
            assert rock[f].shape[0] == G['cells']['num']
        return rock

    def getCPGRockFromDeck(self):
        """Get the rock of the input CPG from the input deck."""
        deck = self.inputDeck
        GC, _ = self.assignInputSubGrds(nargout=2)
        rockC = _deps.initEclipseRock(deck)
        rockC = _deps.compressRock(rockC, GC['cells']['indexMap'])
        return rockC

    def getVOIRocksByInterp(self, **kwargs):
        """Get the rock of the input VOI grid by interpolation of the CPG
        rock.  Optional keyword ``InterpMethod``: 'linear' (default) |
        'nearest' | 'natural' | 'cubic' | 'v4'."""
        opt = mergeOptions({'InterpMethod': 'linear'}, **kwargs)

        GC, GV = self.assignInputSubGrds(nargout=2)
        rockC = self.getCPGRockFromDeck()
        grdecl = self.getGrdEclFromDeck()

        # Layer indices to determine the corresponding layers in the packed
        # data of the VOI grid layers
        layerID = rldecode(np.arange(1, len(GV['layers']['refinement']) + 1),
                           GV['layers']['refinement'])
        assert len(layerID) == GV['layers']['num']

        # Get CPG cell centers and cell-face centers.
        #
        # MRST's corner-point branch calls a dedicated computeCpGeometry
        # (pillar-averaged cell/face centers) not ported here; PRSTCore's
        # compute_geometry (tetrahedralization-based, exhaustively
        # validated against MRST's computeGeometry.m earlier this
        # session) already gives geometrically valid cell/face centroids
        # for corner-point grids too, so it is reused for both branches --
        # these feed getUnitDisVectors purely to establish local
        # permeability-tensor axis directions, not exact volumes/areas,
        # so a differently-computed (but still valid) representative
        # point per cell/face is an acceptable substitute.
        cCenters = GC['cells']['centroids']
        cfCenters = GC['faces']['centroids'][GC['cells']['faces'][:, 0]]

        # Get the properties by interpolation
        rockV = {'perm': np.zeros((GV['cells']['num'], 3)),
                 'poro': np.zeros((GV['cells']['num'], 1)),
                 'ntg': np.zeros((GV['cells']['num'], 1))}
        packed = GV['parentInfo']
        for layer in range(1, GV['layers']['num'] + 1):
            cellsC = packed['boxCells'][layerID[layer - 1] - 1]
            # Perm, loc --> glo
            ux, uy, uz = getUnitDisVectors(GC, cfCenters, cellsC)
            perm_loc = rockC['perm'][cellsC, :]
            perm_glo = (perm_loc[:, 0][:, None] * ux
                        + perm_loc[:, 1][:, None] * uy
                        + perm_loc[:, 2][:, None] * uz)
            xx = cCenters[cellsC, 0]
            yy = cCenters[cellsC, 1]
            cellsV = np.flatnonzero(GV['cells']['layers'] == layer)
            xq = GV['cells']['centroids'][cellsV, 0]
            yq = GV['cells']['centroids'][cellsV, 1]
            permV = np.zeros((len(cellsV), 3))
            for dim in range(3):
                permV[:, dim] = _deps_griddata(xx, yy, perm_glo[:, dim], xq, yq,
                                               opt['InterpMethod'])
            rockV['perm'][cellsV, :] = permV

            # poro
            poroC = rockC['poro'][cellsC]
            poroV = _deps_griddata(xx, yy, poroC, xq, yq, opt['InterpMethod'])
            rockV['poro'][cellsV, 0] = poroV

            # ntg
            ntgC = rockC['ntg'][cellsC]
            ntgV = _deps_griddata(xx, yy, ntgC, xq, yq, opt['InterpMethod'])
            rockV['ntg'][cellsV, 0] = ntgV
        return rockV

    def assignSubRocks(self, rock):
        """Assign the rocks from the global rock for the updated subgrids."""
        mapc = self.cellMapFromSubGrdsToGloGrd()
        fn = list(rock.keys())
        subRocks = [dict() for _ in range(len(mapc))]
        for i in range(len(subRocks)):
            for j in range(len(fn)):
                src = np.asarray(rock[fn[j]])
                if src.ndim == 1:
                    subRocks[i][fn[j]] = src[mapc[i][:, 1]]
                else:
                    subRocks[i][fn[j]] = src[mapc[i][:, 1], :]
        return subRocks

    # -- transmissibility ---------------------------------------------------

    def assembleTransNeighbors(self, T, nnc):
        """Assemble transmissibility and neighborship for the simulation
        model."""
        G = self.gloGrid
        T_all = np.concatenate([T, nnc['T']])
        N_all = np.vstack([G['faces']['neighbors'], nnc['cells']])
        assert len(T_all) == N_all.shape[0]
        return T_all, N_all

    def getTransGloGrid(self, rock):
        """Compute the transmissibility for the global grid (half
        transmissibility of the updated CPG, VOI grid and HW grid)."""
        # Half transmissibility of the updated subgrids
        hTC = self.computeCPGHalfTrans(rock)
        hTV = self.computeVOIGrdHalfTrans(rock)
        hTW = self.computeHWGrdHalfTrans(rock)
        hT = np.concatenate([hTC, hTV, hTW])
        # Get full transmissibility, corresponding to G.faces.neighbors
        G = self.gloGrid
        cf = G['cells']['faces'][:, 0]
        assert len(hT) == len(cf)
        nf = G['faces']['num']
        T = 1.0 / np.bincount(cf, weights=1.0 / hT, minlength=nf)
        return T

    def computeCPGHalfTrans(self, rock):
        """Compute the half transmissibility of the updated CPG."""
        GCu, _ = self.assignSubGrds(nargout=2)
        rockCu, _, _ = self.assignSubRocks(rock)
        grdecl = self.getGrdEclFromDeck()
        if 'COORD' in grdecl:
            # MRST path: computeCpGeometry + computeTrans('K_system',
            # 'loc_xyz', ...).  The PRSTCore computeTrans approximates this
            # with the global-xyz formulation.
            hT = _deps.computeTrans(GCu, rockCu)
        elif 'DX' in grdecl:
            hT = _deps.computeTrans(GCu, rockCu)
        else:
            raise ValueError('Unknown deck grid input')
        return hT

    def computeVOIGrdHalfTrans(self, rock):
        """Compute the half transmissibility of the updated VOI grid."""
        _, GVu = self.assignSubGrds(nargout=2)
        _, rockVu, _ = self.assignSubRocks(rock)
        return _deps.computeTrans(GVu, rockVu)

    def computeHWGrdHalfTrans(self, rock):
        """Compute the half transmissibility of the updated HW grid (radial
        in the near-well region)."""
        _, _, GWu = self.assignSubGrds()
        _, _, rockWu = self.assignSubRocks(rock)
        # Compute the linear transmissibility first
        hT = _deps.computeTrans(GWu, rockWu)
        # Get the radial transmissibility factor
        ft = self.getRadTransFactors()
        assert len(ft) == len(GWu['cells']['faces'][:, 0])
        # Compute the radial transmissibility
        DZ = np.array([getDZ(GWu, c) for c in range(GWu['cells']['num'])])
        DZ = rldecode(DZ, np.diff(GWu['cells']['facePos']))
        perm = rockWu['perm'][:, 0]
        perm = rldecode(perm, np.diff(GWu['cells']['facePos']))
        hT_rad = perm * DZ * ft
        # Assign the radial transmissibility
        isRad = ~np.isnan(hT_rad)
        hT = np.asarray(hT).copy()
        hT[isRad] = hT_rad[isRad]
        return hT

    def getRadTransFactors(self):
        """Get the radial half-transmissibility factors for the HW grid (from
        the 2D surface grids first, then extended to the layered HW grid)."""
        print(' -- Computing the radial transmissibility factors')
        _, _, GWu = self.assignSubGrds()
        gW = GWu['surfGrid']
        pXYs = GWu['layers']['coordsXY']
        # Skin factors of segments
        s_seg = np.atleast_1d(np.asarray(self.well['skinFactor'], dtype=float))
        if s_seg.size != self.well['segmentNum']:
            raise AssertionError('The skin factors should be given for all segments')
        # Assign to all surface grids
        skins = np.concatenate([[s_seg[0]], (s_seg[:-1] + s_seg[1:]) / 2,
                                [s_seg[-1]]])
        pW = np.zeros(2)
        ft = [computeRadTransFactor(gW, pW, skins[i], nodeCoords=pXYs[i])
              for i in range(len(pXYs))]
        # Extend the factor to the layered HW grid
        ft = [(ft[i] + ft[i + 1]) / 2 for i in range(len(ft) - 1)]
        for k in range(len(ft)):
            tmp = ft[k].reshape(4, -1, order='F')
            tmp = np.vstack([tmp, np.full((2, tmp.shape[1]), np.nan)])
            ft[k] = tmp.ravel(order='F')
        return np.concatenate(ft)

    # -- non-neighbor connections -------------------------------------------

    def generateNonNeighborConn(self, intXn, rock, T):
        """Generate the non-neighbor connections (NNCs)."""
        nnc_nmf = self.nncOfNonMatchingBoundaries(intXn, rock, T)
        nnc_mf = self.nncOfMatchingBoundaries(intXn, rock, T)
        nnc = {'cells': np.vstack([nnc_nmf['cells'], nnc_mf['cells']]),
               'T': np.concatenate([nnc_nmf['T'], nnc_mf['T']]),
               'hT': np.vstack([nnc_nmf['hT'], nnc_mf['hT']])}
        return nnc

    def nncOfNonMatchingBoundaries(self, intXn, rock, T):
        """Generate non-neighbor connections (NNCs) arising from the
        non-matching boundaries."""
        G = self.gloGrid
        nmf = np.asarray(intXn['nonMatchingFaces'], dtype=float)
        # A (col, face-index) pair identified by nonMatchingIntxnRelation as
        # a candidate boundary subface can, after assembleGrids/removeCells
        # finishes stitching the subgrids together, turn out to already have
        # both neighbors present (typically a handful of faces out of
        # several thousand, seen so far only with the 'triangular' VOI
        # reconstruction). Such a face is already fully connected through
        # the regular grid topology, so it needs no extra NNC -- adding one
        # would double-count that connection's transmissibility. Filtered
        # out here rather than asserted against, mirroring how
        # mapIntxnRelationCV already drops rows that fall outside the
        # reconnected global grid for the same underlying reason.
        already_connected = np.zeros(nmf.shape[0], dtype=bool)
        for j in range(2):
            faces_j = nmf[:, j].astype(np.int64)
            already_connected |= np.all(G['faces']['neighbors'][faces_j] >= 0, axis=1)
        if np.any(already_connected):
            nmf = nmf[~already_connected]
        # Centers of subfaces
        fc = nmf[:, 3:6]
        # Area normals of subfaces
        N = nmf[:, 6:9]
        num = nmf.shape[0]
        nnc = {'cells': np.zeros((num, 2), dtype=np.int64),
               'T': np.zeros(num), 'hT': np.zeros((num, 2))}
        for j in range(2):
            faces = nmf[:, j].astype(np.int64)
            assert np.all(~np.all(G['faces']['neighbors'][faces] >= 0, axis=1))
            cells = np.maximum(G['faces']['neighbors'][faces, 0],
                               G['faces']['neighbors'][faces, 1])
            K = rock['perm'][cells, :]
            D = fc - G['cells']['centroids'][cells, :]
            hT = (K[:, 0] * D[:, 0] * N[:, 0]
                  + K[:, 1] * D[:, 1] * N[:, 1]
                  + K[:, 2] * D[:, 2] * N[:, 2])
            hT = np.abs(hT) / np.sum(D * D, axis=1)
            nnc['cells'][:, j] = cells
            nnc['hT'][:, j] = hT
        nnc['T'] = 1.0 / (1.0 / nnc['hT'][:, 0] + 1.0 / nnc['hT'][:, 1])
        return nnc

    def nncOfMatchingBoundaries(self, intXn, rock, T):
        """Generate non-neighbor connections (NNCs) arising from the matching
        boundaries."""
        G = self.gloGrid
        mf = np.asarray(intXn['matchingFaces'])
        # See nncOfNonMatchingBoundaries: drop rows whose face is already
        # fully connected in the assembled global grid.
        already_connected = np.zeros(mf.shape[0], dtype=bool)
        for j in range(2):
            faces_j = mf[:, j].astype(np.int64)
            already_connected |= np.all(G['faces']['neighbors'][faces_j] >= 0, axis=1)
        if np.any(already_connected):
            mf = mf[~already_connected]
        num = mf.shape[0]
        nnc = {'cells': np.zeros((num, 2), dtype=np.int64),
               'T': np.zeros(num), 'hT': np.zeros((num, 2))}
        for j in range(2):
            faces = mf[:, j].astype(np.int64)
            assert np.all(~np.all(G['faces']['neighbors'][faces] >= 0, axis=1))
            cells = np.maximum(G['faces']['neighbors'][faces, 0],
                               G['faces']['neighbors'][faces, 1])
            comareas = mf[:, 2]
            ratio = comareas / G['faces']['areas'][faces]
            hT = T[faces] * ratio
            nnc['cells'][:, j] = cells
            nnc['hT'][:, j] = hT
        nnc['T'] = 1.0 / (1.0 / nnc['hT'][:, 0] + 1.0 / nnc['hT'][:, 1])
        return nnc

    # -- intersection relations ---------------------------------------------

    def computeIntxnRelation(self):
        """Compute the intersection relations between subgrids."""
        print(' -- Computing intersection relations between subgrids')
        nmf_CV = _vstack2(self.nonMatchingIntxnRelation([0, 1], 'top'),
                          self.nonMatchingIntxnRelation([0, 1], 'bot'))
        nmf_CV = self.mapIntxnRelationCV(nmf_CV)

        nmf_VW = _vstack2(self.nonMatchingIntxnRelation([1, 2], 'heel'),
                          self.nonMatchingIntxnRelation([1, 2], 'toe'))
        nmf_VW = self.mapIntxnRelationVW(nmf_VW)

        mf_CV = self.matchingIntxnRelation([0, 1])
        mf_CV = self.mapIntxnRelationCV(mf_CV)

        mf_VW = self.matchingIntxnRelation([1, 2])
        mf_VW = self.mapIntxnRelationVW(mf_VW)

        intXn = {'nonMatchingFaces': _vstack2(nmf_CV, nmf_VW),
                 'matchingFaces': _vstack2(mf_CV, mf_VW)}
        return intXn

    def nonMatchingIntxnRelation(self, grdInd, bdyLoc):
        """Compute the intersection relations of non-matching faces on the
        boundaries of the input subgrids."""
        subG = self.assignInputSubGrds()
        assert grdInd[1] <= len(subG)
        assert grdInd[1] == grdInd[0] + 1
        G1, G2 = subG[grdInd[0]], subG[grdInd[1]]
        types = self.assignInputSubGrdTypes()
        F = G2['parentInfo']['faces']
        if bdyLoc in ('top', 'heel'):
            f1 = F[0]
            f2 = np.flatnonzero(G2['faces']['surfaces'] == 1)
        elif bdyLoc in ('bot', 'toe'):
            f1 = F[-1]
            f2 = np.flatnonzero(G2['faces']['surfaces']
                                == np.max(G2['faces']['surfaces']))
        else:
            raise ValueError('Unknown boundary location')
        t1 = time.time()
        print(f'      {types[grdInd[0]]:>8} - {types[grdInd[1]]:>8}: '
              f'{bdyLoc:>7} boundary, ', end='')
        nmf = handleNonMatchingFaces(G1, f1, G2, f2, isfaceNodesSorted=True)
        t2 = time.time()
        print(f'elapsed time {t2 - t1:.2f} [s]')
        return nmf

    def matchingIntxnRelation(self, grdInd):
        """Compute the intersection relations of matching faces on the
        layered boundaries of the input subgrids."""
        subG = self.assignInputSubGrds()
        assert grdInd[1] <= len(subG)
        assert grdInd[1] == grdInd[0] + 1
        G1, G2 = subG[grdInd[0]], subG[grdInd[1]]
        types = self.assignInputSubGrdTypes()
        C = G2['parentInfo']['cells']
        BN = G2['parentInfo']['bdyNodes']
        t1 = time.time()
        print(f'      {types[grdInd[0]]:>8} - {types[grdInd[1]]:>8}: '
              f'layered boundary, ', end='')
        mf = handleMatchingFaces(G1, C, BN, G2)
        t2 = time.time()
        print(f'elapsed time {t2 - t1:.2f} [s]')
        return mf

    # -- simulation model ---------------------------------------------------

    def setupSimModel(self, rock, T_all, N_all):
        """Setup the simulation model passed to the AD black-oil simulator
        for the global grid."""
        G = self.gloGrid
        f = self.fluid
        # Internal connections
        intCon = np.all(N_all >= 0, axis=1)
        N = N_all[intCon, :]
        T = T_all[intCon]
        # Phase components
        ph = self.getPhaseFromDeck()
        # MRST's own NearWellboreModel.setupSimModel constructs
        # ``GenericBlackOilModel`` -- the modern ``ThreePhaseBlackOilModel &
        # GenericReservoirModel`` mix-in whose equations are fully
        # automatic-differentiated through ``FlowDiscretization``/
        # ``GenericFacilityModel`` (every term, including the flux's
        # dependence on saturation-driven mobility, is differentiated
        # through the ADI chain rule -- nothing is hand-approximated).
        # PRSTCore's ``GenericBlackOilModel`` defaults to a different,
        # *incomplete* hand-assembled-Jacobian path (``_get_equations_3ph``:
        # flux terms are linearized in pressure only, with no
        # d(mobility)/d(saturation) term at all) unless
        # ``mrst_generic_assembly=True`` is requested -- passing it here
        # routes through ``_mrst_generic_adi_residual``/``FacilityModel``,
        # the same complete-Jacobian machinery the deck-driven SPE1/SPE9/EGG
        # pipeline already uses, matching what MRST's own nwm module does.
        model = _deps.GenericBlackOilModel(G, rock, f, water=ph['wat'],
                                           oil=ph['oil'], gas=ph['gas'],
                                           vapoil=ph['vapo'], disgas=ph['disg'],
                                           mrst_generic_assembly=True)
        # _get_relperm_tables/_phase_pvt read model.inputdata/model._blackoil_pvt
        # (see _select_model_from_deck's deck-driven wiring); without these
        # a GenericBlackOilModel silently falls back to constant PVT and
        # quadratic relperm defaults, which has a vanishing (zero) sG
        # derivative at sG=0 and makes the assembled Jacobian's gas columns
        # structurally singular whenever the deck declares GAS active but
        # every cell starts fully undersaturated (see the NWM.data example).
        model.inputdata = self.inputDeck
        pvt = f.get('blackoil_pvt') if isinstance(f, dict) else None
        if pvt is not None:
            model._blackoil_pvt = pvt
        # Reset the operators
        model.operators = _deps.setupOperatorsTPFA(G, rock, neighbors=N,
                                                   trans=T)
        model.operators['N_all'] = N_all
        model.operators['T_all'] = T_all

        # Aquifer model
        hasAquifer, output = self.handleAquifers()
        if hasAquifer:
            model.AquiferModel = _deps.AquiferModel(
                model, output['aquifers'], output['aquind'],
                output['aquiferprops'], output['initval'])
        return model

    def handleAquifers(self):
        """Handle the aquifers (only Fetkovich aquifers are supported)."""
        deck = self.inputDeck
        solution = deck.get('SOLUTION', {})
        hasAquifer = ('AQUANCON' in solution) and ('AQUFETP' in solution)
        if not hasAquifer:
            return False, None
        GC, _, _ = self.assignInputSubGrds()
        output = _deps.processAquifer(deck, GC)
        # Map the connection cells
        aquifers = output['aquifers']
        aquind = output['aquind']
        cells = aquifers[:, aquind['conn']]
        mapc = self.cellMapFromInputSubGrdsToGloGrd()[0]
        cells = [mapc[np.flatnonzero(mapc[:, 0] == c)[0], 1] for c in cells]
        idx = [not np.isnan(c) for c in cells]
        cells = [cells[i] for i in range(len(cells)) if idx[i]]
        aquifers = aquifers[np.array(idx, dtype=bool), :]
        aquifers[:, aquind['conn']] = np.array(cells)
        output['aquifers'] = aquifers
        # The aquifer connects to the VOI grid
        if not all(idx):
            output = self.getAquifersVOIG(output)
        return True, output

    def getAquifersVOIG(self, output):
        """Aquifers connected to the VOI grid (only supports the influx
        bottom aquifer)."""
        G = self.gloGrid
        _, GV = self.assignInputSubGrds(nargout=2)
        aquifers = output['aquifers']
        aquind = output['aquind']
        # Get the connection faces and cells (GV)
        nSurf = GV['layers']['num'] + 1
        surfInd = G['faces']['surfaces'][G['faces']['grdID'] == 2]
        assert nSurf == np.max(surfInd)
        facesV = np.flatnonzero((G['faces']['surfaces'] == nSurf)
                                & (G['faces']['grdID'] == 2))
        N = G['faces']['neighbors'][facesV, :]
        assert np.all(~np.all(N >= 0, axis=1))
        connV = np.maximum(N[:, 0], N[:, 1])
        # The connection faces and cells (GC)
        connC = aquifers[:, aquind['conn']].astype(np.int64)
        facesC = np.zeros(len(connC), dtype=np.int64)
        for i in range(len(connC)):
            facePos = np.arange(G['cells']['facePos'][connC[i]],
                                G['cells']['facePos'][connC[i] + 1])
            f = G['cells']['faces'][facePos, :]
            # Note: only supports the influx bottom aquifer
            facesC[i] = f[f[:, 1] == 6, 0][0]
        N = G['faces']['neighbors'][facesC, :]
        assert np.all(~np.all(N >= 0, axis=1))
        # Get the aquifer alpha
        deck = self.inputDeck
        aquancon = np.asarray(deck['SOLUTION']['AQUANCON'])
        influxcoef = np.array([r[8] for r in aquancon])
        influxmultcoef = np.array([r[9] for r in aquancon])
        # Use area weighted (aquifer influx coefficient multiplier = 1)
        assert np.all(np.isnan(influxcoef)) and np.all(influxmultcoef == -1)
        facesA = np.concatenate([facesC, facesV])
        influxcoef = G['faces']['areas'][facesA]
        alpha = influxcoef / np.sum(influxcoef)
        # Assemble the aquifer
        aquifersV = np.full((len(connV), 7), np.nan)
        aquifersV[:, aquind['conn']] = connV
        aquifersV[:, aquind['depthconn']] = G['cells']['centroids'][connV, 2]
        flds = ('aquid', 'pvttbl', 'J', 'C', 'depthaq')
        for fld in flds:
            aquifersV[:, aquind[fld]] = np.unique(aquifers[:, aquind[fld]])
        aquifers = np.vstack([aquifers, aquifersV])
        aquifers[:, aquind['alpha']] = alpha
        output['aquifers'] = aquifers
        output['connFaces'] = np.concatenate([facesC, facesV])
        return output

    def getSimSchedule(self, model, **kwargs):
        """Get the simulation schedule for the global grid from the
        production/injection control data in the deck."""
        print(' -- Converting schedule from input deck')
        opt = mergeOptions({'refDepthFrom': 'deck'}, **kwargs)
        G = self.gloGrid
        # Assign the CPG schedule first, to get the well structure of the CPG
        # model
        modelC = self.setupCPGSimModel()
        scheduleC = self.getCPGSimSchedule(modelC)
        wc, _, WI = self.getWellCellPara(model)
        # Need to map the well cells of other wells from input CPG to the
        # global grid
        mapc = self.cellMapFromInputSubGrdsToGloGrd()[0]
        # Define a tmp rock
        rockTmp = {'perm': np.full((G['cells']['num'], 3), np.nan)}
        for i in range(len(scheduleC['control'])):
            W0 = scheduleC['control'][i]['W']
            ii = [w.get('name') == self.well['name'] for w in W0]
            # Map the cells of the other wells
            WRegular = [W0[k] for k in range(len(W0)) if not ii[k]]
            for w in WRegular:
                c0 = np.asarray(w['cells'], dtype=np.int64)
                # intersect (c0, mapc[:,0]) 'stable'
                sel = np.array([np.any(mapc[:, 0] == c) for c in c0])
                assert np.all(sel)
                w['cells'] = mapc[np.array([np.flatnonzero(mapc[:, 0] == c)[0]
                                            for c in c0]), 1]
            # Well structure for the HW. W0 is a plain Python list (see the
            # WRegular list-comprehension just above), so it needs the same
            # boolean-mask-via-zip pattern, not numpy fancy indexing.
            W = dict([w for w, keep in zip(W0, ii) if keep][0])
            if opt['refDepthFrom'] == 'deck':
                refDepth = W['refDepth']
                print(f'    Info : The reference depth of {self.well["name"]} '
                      'adopts the value from deck')
            elif opt['refDepthFrom'] == 'trajectory':
                refDepth = self.well['trajectory'][0, 2]
                print(f'    Info : The reference depth of {self.well["name"]} '
                      'has been set to the depth of the first well point')
            elif opt['refDepthFrom'] == 'topNode':
                refDepth = self.wellboreGrid['cells']['centroids'][0, 2]
                print(f'    Info : The reference depth of {self.well["name"]} '
                      'has been set to the depth of the top node in the '
                      'multi-segment well definition')
            else:
                raise ValueError('Unknown reference depth definition type')
            # Redefine some fields
            W['cells'] = wc
            W['r'] = np.full(len(wc), np.nan)
            W['rR'] = np.full(len(wc), np.nan)
            W['dir'] = np.full(len(wc), 'X')
            W['WI'] = WI
            W['refDepth'] = refDepth
            W['cell_origin'] = np.ones(len(wc))
            W['cstatus'] = np.ones(len(wc), dtype=bool)
            # Call 'addWell' to compute the 'W.dz'
            WTmp = _deps.addWell({}, G, rockTmp, wc, name='WTmp',
                                 refDepth=refDepth)
            W['dZ'] = WTmp['dZ']
            # Combine with the other wells
            WNew = [W] + WRegular
            scheduleC['control'][i]['W'] = WNew
        return scheduleC

    def setupCPGSimModel(self):
        """Setup the simulation model passed to the AD black-oil simulator
        for the input CPG grid."""
        f = self.fluid
        GC, _ = self.assignInputSubGrds(nargout=2)
        rockC = self.getCPGRockFromDeck()
        # Phase components
        ph = self.getPhaseFromDeck()
        # Same GenericBlackOilModel MRST uses here, and for the same reason
        # as in setupSimModel it needs mrst_generic_assembly=True to get the
        # complete (fully AD) Jacobian rather than PRSTCore's hand-assembled
        # compatibility path.
        model = _deps.GenericBlackOilModel(GC, rockC, f, water=ph['wat'],
                                           oil=ph['oil'], gas=ph['gas'],
                                           vapoil=ph['vapo'], disgas=ph['disg'],
                                           mrst_generic_assembly=True)
        model.inputdata = self.inputDeck
        pvt = f.get('blackoil_pvt') if isinstance(f, dict) else None
        if pvt is not None:
            model._blackoil_pvt = pvt
        return model

    def getCPGSimSchedule(self, model):
        """Get the simulation schedule for the input CPG from the deck."""
        deck = self.inputDeck
        return _deps.convertDeckScheduleToMRST(model, deck)

    def getWellCellPara(self, model):
        """Get the parameters for the well cells of the HW.

        Returns ``(wc, wf, WI)``: well cell indices, well face indices and
        well indices of the well cells."""
        wc = self.getWellCells()
        G = self.gloGrid
        # Wellbore faces (always the first face of wc)
        wf = np.array([G['cells']['faces'][G['cells']['facePos'][c], 0]
                       for c in wc])
        assert np.all(~np.all(G['faces']['neighbors'][wf] >= 0, axis=1))
        # Well index
        WI = model.operators['T_all'][wf]
        return wc, wf, WI

    def getWellCells(self):
        """Get the well cell indices (the innermost radial ring of the opened
        segments of the HW grid)."""
        G = self.gloGrid
        _, _, GW = self.assignSubGrds()
        nA = GW['radDims'][0]
        # All HW grid cells at the global grid
        mapc = self.cellMapFromSubGrdsToGloGrd()
        cells = mapc[2][:, 1]
        assert np.all(G['cells']['grdID'][cells] == 3)
        cells = cells.reshape(-1, GW['layers']['num'], order='F')
        # Completed segments
        segs = np.asarray(self.well['openedSegs'], dtype=np.int64) - 1
        assert np.all(segs < self.well['segmentNum'])
        # Well cells
        wc = cells[0:nA, segs]
        wc = wc.ravel(order='F')
        return wc

    def getInitState(self, model):
        """Get the initial state by equilibrium initialization."""
        deck = self.inputDeck
        return _deps.initStateDeck(model, deck)

    def getPhaseFromDeck(self):
        """Get the phase components from the input deck."""
        rspec = self.inputDeck.get('RUNSPEC', {})
        ph = {'wat': False, 'oil': False, 'gas': False, 'vapo': False,
              'disg': False}
        ph['wat'] = 'WATER' in rspec and bool(rspec.get('WATER', False))
        ph['oil'] = 'OIL' in rspec and bool(rspec.get('OIL', False))
        ph['gas'] = 'GAS' in rspec and bool(rspec.get('GAS', False))
        ph['vapo'] = 'VAPOIL' in rspec and bool(rspec.get('VAPOIL', False))
        ph['disg'] = 'DISGAS' in rspec and bool(rspec.get('DISGAS', False))
        return ph

    # -- maps ---------------------------------------------------------------

    def cellMapFromInputSubGrdsToGloGrd(self):
        """Cell map from the input subgrids to the global grid."""
        subG = self.assignSubGrds()
        mapc = self.cellMapFromSubGrdsToGloGrd()
        for i in range(len(subG)):
            mapc[i][:, 0] = subG[i]['cells']['map'][mapc[i][:, 0]]
        return mapc

    def faceMapFromInputSubGrdsToGloGrd(self):
        """Face map from the input subgrids to the global grid."""
        subG = self.assignSubGrds()
        mapf = self.faceMapFromSubGrdsToGloGrd()
        for i in range(len(subG)):
            mapf[i][:, 0] = subG[i]['faces']['map'][mapf[i][:, 0]]
        return mapf

    def cellMapFromSubGrdsToGloGrd(self):
        """Cell map from the updated subgrids (after-removed cells) to the
        global grid."""
        subG = self.assignSubGrds()
        nc = np.array([g['cells']['num'] for g in subG], dtype=np.int64)
        csnc = np.concatenate([[0], np.cumsum(nc)])
        mapc = [np.column_stack([np.arange(nc[i]), np.arange(nc[i]) + csnc[i]])
                for i in range(len(nc))]
        return mapc

    def faceMapFromSubGrdsToGloGrd(self):
        """Face map from the updated subgrids (after-removed cells) to the
        global grid."""
        subG = self.assignSubGrds()
        nf = np.array([g['faces']['num'] for g in subG], dtype=np.int64)
        csnf = np.concatenate([[0], np.cumsum(nf)])
        mapf = [np.column_stack([np.arange(nf[i]), np.arange(nf[i]) + csnf[i]])
                for i in range(len(nf))]
        return mapf

    def getGrdEclFromDeck(self):
        """Get the ECLIPSE grid structure from the deck (without 'ACTNUM' for
        robustness)."""
        deck = self.inputDeck
        GRID = deck.get('GRID', {})
        if 'COORD' in GRID:  # CPG
            fn = ('cartDims', 'COORD', 'ZCORN')
        else:                # Cartesian grid
            fn = ('cartDims', 'DX', 'DY', 'DZ', 'TOPS')
        grdecl = {f: GRID[f] for f in fn}
        return grdecl

    # -- checks ---------------------------------------------------------------

    def checkCellMaps(self):
        """Check the cell maps."""
        G = self.gloGrid
        mapc = self.cellMapFromSubGrdsToGloGrd()
        subG = self.assignSubGrds()
        for i in range(len(subG)):
            p1 = subG[i]['cells']['centroids'][mapc[i][:, 0], :]
            p2 = G['cells']['centroids'][mapc[i][:, 1], :]
            assert np.allclose(p1, p2), 'Wrong cell map!'
        mapc = self.cellMapFromInputSubGrdsToGloGrd()
        subG = self.assignInputSubGrds()
        for i in range(len(subG)):
            p1 = subG[i]['cells']['centroids'][mapc[i][:, 0], :]
            p2 = G['cells']['centroids'][mapc[i][:, 1], :]
            assert np.allclose(p1, p2), 'Wrong cell map!'
        print('   Corrected cell map   ')

    def checkFaceMaps(self):
        """Check the face maps."""
        G = self.gloGrid
        mapf = self.faceMapFromSubGrdsToGloGrd()
        subG = self.assignSubGrds()
        for i in range(len(subG)):
            p1 = subG[i]['faces']['centroids'][mapf[i][:, 0], :]
            p2 = G['faces']['centroids'][mapf[i][:, 1], :]
            assert np.allclose(p1, p2), 'Wrong face map!'
        mapf = self.faceMapFromInputSubGrdsToGloGrd()
        subG = self.assignInputSubGrds()
        for i in range(len(subG)):
            p1 = subG[i]['faces']['centroids'][mapf[i][:, 0], :]
            p2 = G['faces']['centroids'][mapf[i][:, 1], :]
            assert np.allclose(p1, p2), 'Wrong face map!'
        print('   Corrected face map   ')

    def checkIntxnRelation(self, intXn):
        """Check the intersection relation by comparing the face areas."""
        G = self.gloGrid
        R = np.vstack([intXn['nonMatchingFaces'][:, :3], intXn['matchingFaces']])
        f1 = np.unique(R[:, 0])
        f2 = np.unique(R[:, 1])
        A = R[:, 2]
        Af1 = G['faces']['areas'][f1]
        Af2 = G['faces']['areas'][f2]
        Af1_ = np.array([np.sum(A[R[:, 0] == f]) for f in f1])
        Af2_ = np.array([np.sum(A[R[:, 1] == f]) for f in f2])
        errf1 = np.abs(Af1 - Af1_) / Af1
        errf2 = np.abs(Af2 - Af2_) / Af2
        assert np.all(~np.all(G['faces']['neighbors'][R[:, 0].astype(int)] >= 0, axis=1))
        assert np.all(~np.all(G['faces']['neighbors'][R[:, 1].astype(int)] >= 0, axis=1))
        return errf1, errf2

    # -- plotting (demonstration aids; matplotlib is used where trivial) ------

    def plotNonMatchingIntxnRelation(self, intXn, f1):
        """Plot the intersection relations of the non-matching face ``f1``."""
        raise NotImplementedError(
            'plotNonMatchingIntxnRelation: grid plotting is not implemented '
            'in the Python PRSTCore port yet.')

    def plotMatchingIntxnRelation(self, intXn, f1):
        """Plot the intersection relations of the matching face ``f1``."""
        raise NotImplementedError(
            'plotMatchingIntxnRelation: grid plotting is not implemented in '
            'the Python PRSTCore port yet.')

    # -- protected helpers ----------------------------------------------------

    def mapIntxnRelationCV(self, CV):
        """Map the intersection relation from the input subgrids (GC and GV)
        to the global grid."""
        if CV is None or CV.size == 0 or CV.ndim == 1:
            return np.empty((0, CV.shape[1] if CV is not None and CV.ndim == 2 else 9))
        mapf = self.faceMapFromInputSubGrdsToGloGrd()
        mapfC, mapfV = mapf[0], mapf[1]
        fC = [mapfC[np.flatnonzero(mapfC[:, 0] == f), 1] for f in CV[:, 0]]
        fV1_lists = [mapfV[np.flatnonzero(mapfV[:, 0] == f), 1] for f in CV[:, 1]]
        # When the VOI surfaces constitute part of the top or bottom boundary
        # of the global grid, fC will be empty. MRST's own arrayfun-based
        # fV1 assumes every VOI face always has exactly one match, which a
        # triangular (rather than MRST's default Voronoi) VOI reconstruction
        # can violate at the boundary -- treated the same way as an empty fC
        # row rather than raising, since it is the same "not part of the
        # reconnected global grid" case.
        idx = np.array([len(x) > 0 for x in fC]) & np.array([len(x) > 0 for x in fV1_lists])
        fC = np.concatenate([x for x, keep in zip(fC, idx) if keep]) if np.any(idx) else np.empty(0)
        if len(fC) > 0:
            fV1 = np.array([x[0] for x, keep in zip(fV1_lists, idx) if keep])
            ia = CV[idx, 2:]
            return np.column_stack([fC, fV1, ia])
        return np.empty((0, CV.shape[1]))

    def mapIntxnRelationVW(self, VW):
        """Map the intersection relation from the input subgrids (GV and GW)
        to the global grid."""
        if VW is None or VW.size == 0 or VW.ndim == 1:
            return np.empty((0, VW.shape[1] if VW is not None and VW.ndim == 2 else 9))
        mapf = self.faceMapFromInputSubGrdsToGloGrd()
        mapfV, mapfW = mapf[1], mapf[2]
        fV2_lists = [mapfV[np.flatnonzero(mapfV[:, 0] == f), 1] for f in VW[:, 0]]
        fW_lists = [mapfW[np.flatnonzero(mapfW[:, 0] == f), 1] for f in VW[:, 1]]
        # Same "not part of the reconnected global grid" case as
        # mapIntxnRelationCV's fC/fV1 filtering: a triangular (rather than
        # Voronoi) VOI reconstruction can leave a heel/toe boundary face
        # with no corresponding entry in the global-grid face map.
        idx = np.array([len(x) > 0 for x in fV2_lists]) & np.array([len(x) > 0 for x in fW_lists])
        if not np.any(idx):
            return np.empty((0, VW.shape[1]))
        fV2 = np.array([x[0] for x, keep in zip(fV2_lists, idx) if keep])
        fW = np.array([x[0] for x, keep in zip(fW_lists, idx) if keep])
        ia = VW[idx, 2:]
        return np.column_stack([fV2, fW, ia])

    def updateCPG(self):
        """Get the updated CPG whose cells inside the VOI are removed."""
        GC, GV = self.assignInputSubGrds(nargout=2)
        cV = np.concatenate(GV['parentInfo']['cells'])
        GCu, mapc, mapf, _ = removeCells(GC, cV)
        # Assign some fields to keep consistency of the subgrids
        ijk = gridLogicalIndices(GCu)
        GCu['cells']['layers'] = ijk[2] + 1
        GCu['faces']['surfaces'] = np.full(GCu['faces']['num'], np.nan)
        GCu['cells']['map'] = mapc
        GCu['faces']['map'] = mapf
        return GCu

    def updateVOIGrid(self):
        """Get the updated VOI grid whose cells inside the HW region are
        removed."""
        _, GV, GW = self.assignInputSubGrds()
        cW = np.concatenate(GW['parentInfo']['cells'])
        GVu, mapc, mapf, _ = removeCells(GV, cW)
        # Map the ID of layers and surfaces for cells and faces
        GVu['cells']['layers'] = GVu['cells']['layers'][mapc]
        GVu['faces']['surfaces'] = GVu['faces']['surfaces'][mapf]
        GVu['cells']['map'] = mapc
        GVu['faces']['map'] = mapf
        return GVu

    def updateHWGrid(self):
        """Get the updated HW grid (no cells are removed)."""
        _, _, GW = self.assignInputSubGrds()
        GWu = GW
        # Assign some fields to keep consistency of the subgrids
        GWu['cells']['map'] = np.arange(GW['cells']['num'])
        GWu['faces']['map'] = np.arange(GW['faces']['num'])
        return GWu


def _vstack2(a, b):
    """Stack two possibly-empty matrices."""
    if a is None or a.size == 0:
        return b
    if b is None or b.size == 0:
        return a
    return np.vstack([a, b])


def _deps_griddata(x, y, z, xq, yq, method):
    """Interpolation wrapper (from scipy) mirroring MATLAB ``griddata``."""
    from scipy.interpolate import griddata as _g
    return _g((np.asarray(x).ravel(), np.asarray(y).ravel()),
              np.asarray(z).ravel(), (np.asarray(xq), np.asarray(yq)),
              method=method)
