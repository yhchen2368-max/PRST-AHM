"""Port of MRST ``WellPlacementOptimization.m`` (mrst-2026a/hm/test).

Optimises where a producer sits, rather than how it is operated. The
well's trajectory becomes the tunable parameter via
:class:`WellPositionControl`, and the objective is oil revenue net of
water handling (:func:`npv_ow`), maximised by adjoint gradients.

``nPoints=2`` makes the trajectory a straight line, and ``verticalTol``
holds it vertical; the perturbation and update caps keep each iteration
from moving the well further than the gradient is trustworthy over.

Unlike :mod:`HistoryMatching` this runs entirely inside PRSTCore -- a
20x20x3 box model, no external simulator -- so it needs no configuration
beyond what is here.

``initSimpleADIFluid`` has no PRSTCore port yet (206 lines in MRST's
ad-props), so :func:`build_fluid` assembles the same two-phase fluid
directly. That is the one place this file departs from the MATLAB.
"""

import numpy as np

DARCY = 9.869232667160128e-13
MILLI = 1e-3
CENTI_POISE = 1e-3
BARSA = 1e5
DAY = 86400.0
METER = 1.0
YEAR = 365.0 * DAY

NXYZ = (20, 20, 3)
DXYZ = (400.0, 400.0, 15.0)
P_REF = 200 * BARSA

PRICES = {'OilPrice': 70.0, 'WaterProductionCost': 3.0,
          'WaterInjectionCost': 5.0, 'DiscountFactor': 0.1}

#: Endpoint scaling imposed on the simple fluid.
SCALING = {'SWL': 0.1, 'SWCR': 0.2, 'SWU': 0.9, 'SOWCR': 0.1,
           'KRW': 0.9, 'KRO': 0.8}


def build_fluid():
    """The two-phase fluid MRST builds with initSimpleADIFluid.

    Quadratic Corey curves, constant water compressibility, exponential
    oil formation-volume factor.
    """
    mu = np.array([0.3, 3.0]) * CENTI_POISE
    rho = np.array([1014.0, 859.0])
    c = 5e-5 / BARSA

    fluid = {
        'phases': 'WO',
        'muW': lambda p: np.full_like(np.asarray(p, dtype=float), mu[0]),
        'muO': lambda p: np.full_like(np.asarray(p, dtype=float), mu[1]),
        'rhoWS': rho[0], 'rhoOS': rho[1],
        'krW': lambda s: np.asarray(s, dtype=float) ** 2,
        'krO': lambda s: np.asarray(s, dtype=float) ** 2,
        'bW': lambda p: np.ones_like(np.asarray(p, dtype=float)),
        'bO': lambda p: np.exp((np.asarray(p, dtype=float) - P_REF) * c),
        'krPts': {'w': [0, 0, 1, 1], 'ow': [0, 0, 1, 1]},
    }
    return fluid


def build_case():
    """Grid, rock, fluid, model, wells and schedule."""
    from PRSTCore.ad_core.models.generic_black_oil_model import \
        GenericBlackOilModel
    from PRSTCore.ad_core.timesteps import rampup_timesteps
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    from PRSTCore.gridprocessing.add_bounding_box_fields import \
        add_bounding_box_fields
    from PRSTCore.gridprocessing.cart_grid import cart_grid
    from PRSTCore.gridprocessing.compute_geometry import compute_geometry
    from PRSTCore.solvers.incomp.make_rock import make_rock
    from PRSTCore.solvers.incomp.vertical_well import vertical_well

    G = compute_geometry(cart_grid(NXYZ, DXYZ))
    G = add_bounding_box_fields(G)
    rock = make_rock(G, 1 * MILLI * DARCY, 0.2)

    model = GenericBlackOilModel(G, rock, build_fluid(), gas=False)
    model = impose_relperm_scaling(model, **SCALING)
    model.toleranceCNV = 1e-6
    model = model.validateModel()

    # Injectors at the four corners, one producer near a corner.
    W = []
    wx = (1, NXYZ[0], 1, NXYZ[0])
    wy = (1, NXYZ[1], NXYZ[1], 1)
    for k in range(4):
        W = vertical_well(W, G, rock, wx[k], wy[k], range(1, NXYZ[2] + 1),
                          type='rate', val=300.0 / DAY, name='I%d' % (k + 1),
                          comp_i=[1, 0])
    W = vertical_well(W, G, rock, 5, 5, range(1, NXYZ[2] + 1), type='bhp',
                      val=100 * BARSA, name='P5', comp_i=[1, 0])
    # MRST's verticalWell takes 'Sign'; PRSTCore's port does not, so it is
    # set afterwards. Every well here is positive-signed in the MATLAB,
    # producer included.
    for well in W:
        well['sign'] = 1

    from PRSTCore.ad_core.utils.simple_schedule import simple_schedule
    steps = rampup_timesteps(2 * YEAR, 30 * DAY, 5)
    schedule = simple_schedule(steps, [{'W': W} for _ in steps])
    return G, rock, model, W, schedule


def main():
    """Run the placement optimisation. Returns ``(u, setupNew)``."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        simulate_schedule_ad
    from PRSTCore.hm.utils.evaluate.evaluateObjective import evaluateObjective
    from PRSTCore.hm.utils.optimWellPlacementSimple import \
        optimWellPlacementSimple
    from PRSTCore.optimization.objectives.npv_ow import npv_ow
    from PRSTCore.optimization.utils.parameters import (
        add_parameter, get_scaled_parameter_vector,
        update_setup_from_scaled_parameters)
    from PRSTCore.solvers.incomp.init_state import init_state
    from PRSTCore.visualization.diagnostics.utils.trajectory.\
        well_position_control import WellPositionControl

    G, rock, model, W, schedule = build_case()
    state0 = init_state(G, W, P_REF, [[0.0, 1.0]])
    ws, states = simulate_schedule_ad(state0, model, schedule)[:2]

    # Only the producer's position is tuned.
    #
    # MRST builds this as WellPositionControl(G, 'w', W, 'perturbationSize',
    # ..., 'maxUpdatePoint', ..., 'verticalTol', ..., 'nPoints', 2) -- it
    # derives the trajectory from the well's cells itself. PRSTCore's port
    # is a 38-line stub against MRST's 401-line class: it takes the
    # trajectory points directly and carries everything else in a
    # parameters dict, with no vertical constraint and no update caps. The
    # settings are passed through so they are not lost, but this optimiser
    # will not honour them until that class is filled in.
    wno = [4]
    for ctrl in schedule['control']:
        for k in wno:
            well = ctrl['W'][k]
            cells = np.atleast_1d(np.asarray(well['cells'], dtype=int)).ravel()
            points = np.asarray(G['cells']['centroids'], dtype=float)[cells]
            well['posControl'] = WellPositionControl(
                points[[0, -1], :],          # nPoints=2: a straight line
                {'perturbationSize': [20, 20, 5],
                 'maxUpdatePoint': [80, 80, 10],
                 'verticalTol': 10 * METER, 'nPoints': 2})

    def obj(model_, states_, schedule_, tt, tstep, state, ff):
        return npv_ow(model_, states_, schedule_, compute_partials=tt,
                      tstep=tstep, state=state, from_states=ff, **PRICES)

    objVal = float(np.sum([np.sum(x) for x in
                           obj(model, states, schedule, False, None, None,
                               True)]))

    setup = {'model': model, 'schedule': schedule, 'state0': state0}
    params = add_parameter([], setup, name='posControl', subset=wno)
    u0 = get_scaled_parameter_vector(setup, params)

    def f(u):
        return evaluateObjective(u, obj, setup, params, enforceBounds=False,
                                 objScaling=objVal, NonLinearSolver=None,
                                 AdjointLinearSolver=None,
                                 Gradient='AdjointAD')

    u = optimWellPlacementSimple(u0, f, W)

    setupNew = update_setup_from_scaled_parameters(setup, params, u)
    ws_opt = simulate_schedule_ad(setupNew['state0'], setupNew['model'],
                                  setupNew['schedule'])[0]
    _plot(ws, ws_opt, schedule)
    return u, setupNew


def _plot(ws, ws_opt, schedule):
    from PRSTCore.ad_core.plotting.plot_well_sols import plot_well_sols
    plot_well_sols([ws, ws_opt], schedule['step']['val'],
                   linestyles=[':', '-'], markerstyles=['o', ''],
                   datasetnames=['Initial', 'Optimized'])


if __name__ == '__main__':
    main()
