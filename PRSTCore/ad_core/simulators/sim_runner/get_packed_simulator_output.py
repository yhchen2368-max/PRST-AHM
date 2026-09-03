from PRSTCore.ad_core.simulators.sim_runner.simulate_packed_problem import simulate_packed_problem


def get_packed_simulator_output(problem):
    if problem.get("_simulated") is None:
        return simulate_packed_problem(problem)
    return problem["_simulated"]["wellSols"], problem["_simulated"]["states"]


def get_packed_simulator_report(problem):
    if problem.get("_simulated") is None:
        simulate_packed_problem(problem)
    return problem["_simulated"].get("report", None)
