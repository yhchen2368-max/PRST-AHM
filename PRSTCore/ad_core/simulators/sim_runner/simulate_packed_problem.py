from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad


def simulate_packed_problem(problem):
    setup = problem["SimulatorSetup"]
    solver_kwargs = {k: v for k, v in setup.items() if k not in ("state0", "model", "schedule")}
    solver_kwargs["return_report"] = True
    well_sols, states, schedulereport = simulate_schedule_ad(
        setup["state0"], setup["model"], setup["schedule"], **solver_kwargs
    )
    problem["_simulated"] = {"wellSols": well_sols, "states": states, "report": schedulereport}
    return well_sols, states
