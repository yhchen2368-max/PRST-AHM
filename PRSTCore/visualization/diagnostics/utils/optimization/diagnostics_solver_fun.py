"""MRST ``diagnosticsSolverFun.m`` counterpart."""


def diagnostics_solver_fun(problem, objective, **kwargs):
    W = kwargs.get("W") or problem.get("W") or problem.get("schedule", {}).get("control", [{}])[0].get("W", [])
    state = kwargs.get("state", problem.get("state0", {}))
    return objective.compute(W, state=state, **kwargs) if hasattr(objective, "compute") else objective(problem, **kwargs)


diagnosticsSolverFun = diagnostics_solver_fun

