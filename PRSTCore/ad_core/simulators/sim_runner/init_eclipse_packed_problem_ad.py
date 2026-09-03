"""Python counterpart of MRST initEclipsePackedProblemAD under PRSTCore.ad_core.
"""
from typing import Any

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.sim_runner.pack_simulation_problem import pack_simulation_problem


def init_eclipse_packed_problem_ad(deck: Any, **opts) -> Any:
    try:
        state0, model, schedule, nls = init_eclipse_problem_ad(deck, **opts)
    except NotImplementedError as e:
        raise NotImplementedError(
            "initEclipsePackedProblemAD requires further initEclipseProblemAD support: %s" % str(e)
        )

    name = None
    if isinstance(model, dict) and 'inputdata' in model and isinstance(model['inputdata'], dict):
        name = model['inputdata'].get('RUNSPEC', {}).get('TITLE')
    if name is None:
        name = opts.get('BaseName') or opts.get('Name') or 'PRSTCorePackedProblem'

    problem = pack_simulation_problem(state0, model, schedule, name, NonLinearSolver=nls, **opts)
    return problem
