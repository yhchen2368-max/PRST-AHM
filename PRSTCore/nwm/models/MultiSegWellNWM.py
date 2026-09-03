"""Port of MRST ``MultiSegWellNWM``: derived class for generating the data
structures passed to the AD simulators for the hybrid grid of the
near-wellbore model coupled with the multi-segment well model."""

from __future__ import annotations

import numpy as np

from .._core import mergeOptions, tessellationGrid
from ..gridding.makeLayeredGridNWM import makeLayeredGridNWM
from .. import _deps
from .NearWellboreModel import NearWellboreModel


class MultiSegWellNWM(NearWellboreModel):
    """Derived class for generating the necessary data structures passed to
    the MRST AD simulators for the hybrid grid of the near-wellbore model
    coupled with the multi-segment well model.

    Additional property
    -------------------
    wellboreGrid : dict
        1D 'wellbore grid' in the void wellbore space which conforms with
        the reservoir grid.
    """

    def __init__(self, subGrids, deck, well, **kwargs):
        super().__init__(subGrids, deck, well, **kwargs)
        assert 'isMS' in well and well['isMS'], \
            'The input well is not a multi-segment well'
        self.wellboreGrid = self.buildWellboreGrid()

    def setupSimModel(self, rock, T_all, N_all):
        """Setup the simulation model passed to the AD black-oil simulator
        for the global grid (the multi-segment well model only supports
        ``ThreePhaseBlackOilModel``)."""
        G = self.gloGrid
        f = self.fluid
        # Internal connections
        intCon = np.all(N_all >= 0, axis=1)
        N = N_all[intCon, :]
        T = T_all[intCon]
        # Phase components
        ph = self.getPhaseFromDeck()
        assert ph['wat'] and ph['oil'] and ph['gas'], \
            "The multi-segment well model now only supports " \
            "'ThreePhaseBlackOilModel'"
        # MRST's MultiSegWellNWM.setupSimModel calls ThreePhaseBlackOilModel
        # rather than GenericBlackOilModel, but that MRST class is *also*
        # fully automatic-differentiated (equationsBlackOil.m builds every
        # equation from ADI primary variables, same as GenericBlackOilModel
        # -- the two only differ in whether properties route through the
        # StateFunction dependency graph, not in Jacobian completeness).
        # PRSTCore's ``_deps.ThreePhaseBlackOilModel`` aliases the same
        # GenericBlackOilModel class ``_deps.GenericBlackOilModel`` does
        # (see nwm/_deps.py); without mrst_generic_assembly=True it falls
        # into PRSTCore's *hand-assembled* Jacobian path
        # (_get_equations_3ph), which -- unlike either MRST model -- omits
        # d(mobility)/d(saturation) from the flux terms entirely (see
        # NearWellboreModel.setupSimModel's identical fix and its commit
        # message for the full diagnosis). Requesting it here gets the
        # complete Jacobian either MRST class would actually produce.
        model = _deps.ThreePhaseBlackOilModel(G, rock, f, water=ph['wat'],
                                              oil=ph['oil'], gas=ph['gas'],
                                              vapoil=ph['vapo'],
                                              disgas=ph['disg'],
                                              mrst_generic_assembly=True)
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

    def getSimSchedule(self, model, **kwargs):
        """Get the multi-segment well simulation schedule from the deck and
        node/segment definitions."""
        opt = mergeOptions({'returnMS': True, 'refDepthFrom': 'topNode'},
                           **kwargs)
        schedule = super().getSimSchedule(model, refDepthFrom='topNode')
        if not opt['returnMS']:
            return schedule
        nodes = self.generateNodes()
        segs = self.generateSegments()
        for i in range(len(schedule['control'])):
            W0 = schedule['control'][i]['W']
            ii = [w.get('name') == self.well['name'] for w in W0]
            WRegular = [W0[k] for k in range(len(W0)) if not ii[k]]
            W = [W0[k] for k in range(len(W0)) if ii[k]][0]
            W = _deps.convert2MSWell(
                W, cell2node=nodes['cell2node'], connDZ=W['dZ'],
                nodeDepth=nodes['depth'], topo=segs['topo'],
                segLength=segs['length'], segRoughness=[], segFlowModel=[],
                segType=[], segDiam=segs['diam'], G=[], vol=nodes['vol'])
            W['segments']['roughness'] = segs['roughness']
            W['segments']['flowModel'] = segs['flowModel']
            if len(WRegular) > 0:
                WNew = _deps.combineMSwithRegularWells(WRegular, W)
            else:
                WNew = W
            schedule['control'][i]['W'] = WNew
        return schedule

    def buildWellboreGrid(self):
        """Build the grid for the void space inside the wellbore."""
        _, _, GW = self.assignSubGrds()
        # Get the borewall (casing) nodes and connectivity list
        wellbores = GW['wellbores']
        pW = [wb['wall']['coords'] for wb in wellbores]
        nA = GW['radDims'][0]
        assert all(len(x) == nA for x in pW)
        p = pW[0][:, :2]
        t = [np.arange(nA, dtype=np.int64)]
        # Build the wellbore grid
        g = tessellationGrid(p, t)
        g['nodes']['coords'] = pW[0]
        gW = makeLayeredGridNWM(g, pW, connectivity=t)
        gW['radDims'] = [nA, 1, gW['layers']['num']]
        return gW

    def generateNodes(self):
        """Generate node definitions from the wellbore grid for the
        multi-segment well."""
        gW = self.wellboreGrid
        nA = gW['radDims'][0]
        n = np.atleast_1d(np.asarray(self.well['openedSegs'], dtype=np.int64))
        n = np.repeat(n, nA)          # 1-based segment id per well cell
        wc = self.getWellCells()
        from scipy.sparse import csr_matrix
        cell2node = csr_matrix((np.ones(len(wc), dtype=float),
                                (n - 1, np.arange(len(wc)))),
                               shape=(gW['cells']['num'], len(wc)))
        nodes = {
            'coords': gW['cells']['centroids'],
            'depth': gW['cells']['centroids'][:, 2],
            'vol': gW['cells']['volumes'],
            'dist': np.full(gW['cells']['num'], np.nan),
            'cell2node': cell2node,
            'resCells': wc,
        }
        return nodes

    def generateSegments(self):
        """Generate segment definitions from the wellbore grid for the
        multi-segment well."""
        gW = self.wellboreGrid
        # Topology
        f = np.flatnonzero(np.all(gW['faces']['neighbors'] >= 0, axis=1))
        t = gW['faces']['neighbors'][f, :]
        t = np.sort(t, axis=1)
        # Length
        dxyz1 = gW['cells']['centroids'][t[:, 0], :] - gW['faces']['centroids'][f, :]
        L1 = np.sqrt(np.sum(dxyz1 ** 2, axis=1))
        dxyz2 = gW['cells']['centroids'][t[:, 1], :] - gW['faces']['centroids'][f, :]
        L2 = np.sqrt(np.sum(dxyz2 ** 2, axis=1))
        L = L1 + L2
        # Roughness
        roughness = np.asarray(self.well['roughness'], dtype=float)
        roughness = (roughness[:-1] + roughness[1:]) / 2
        # Diameter
        D = 2 * np.asarray(self.well['radius'], dtype=float)[1:-1]
        # Area
        area = gW['faces']['areas'][f]
        # Segment number
        ns = len(L)

        def fm(v, rho, mu):
            return _deps.wellBoreFriction(v, rho, mu, D, L, roughness,
                                          'massRate')

        segs = {'length': L, 'roughness': roughness, 'diam': D, 'topo': t,
                'area': area, 'num': ns, 'flowModel': fm}
        return segs

    def plotSegments(self, nodes, segs, S):
        """Plot the nodes and reservoir cells associated with segment ``S``."""
        raise NotImplementedError(
            'plotSegments: grid plotting is not implemented in the Python '
            'PRSTCore port yet.')
