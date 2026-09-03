import numpy as np
from PRSTCore.ad_core.simulators import simulate_schedule_ad
from PRSTCore.ad_core.simulators.sim_runner import (
    get_packed_simulator_output,
    get_packed_simulator_report,
    pack_simulation_problem,
    simulate_packed_problem,
)


def test_pack_and_simulate():
    schedule = {"step": {"val": np.array([1.0]), "control": np.array([0], dtype=int)},
                "control": [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True}]}]}
    setup = {"state0": {"time": 0.0}, "model": {"porevolume": np.array([1.0])}, "schedule": schedule}
    problem = pack_simulation_problem(setup["state0"], setup["model"], setup["schedule"], "test")
    wells, states = simulate_packed_problem(problem)
    assert len(wells) == 1
    assert len(states) == 1
    wells2, states2 = get_packed_simulator_output(problem)
    assert wells2 == wells
    assert states2 == states


def test_schedule_report_shape():
    schedule = {"step": {"val": np.array([1.0]), "control": np.array([0], dtype=int)},
                "control": [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True}]}]}
    setup = {"state0": {"time": 0.0}, "model": {"porevolume": np.array([1.0])}, "schedule": schedule}
    problem = pack_simulation_problem(setup["state0"], setup["model"], setup["schedule"], "test")
    simulate_packed_problem(problem)
    report = get_packed_simulator_report(problem)
    assert isinstance(report, dict)
    assert "ControlstepReports" in report
    assert len(report["ControlstepReports"]) == 1
    cr = report["ControlstepReports"][0]
    assert "StepReports" in cr
    assert isinstance(cr["StepReports"], list)
    assert len(cr["StepReports"]) >= 1
    sr = cr["StepReports"][0]
    assert "Timestep" in sr
    assert "Converged" in sr


def test_simulate_schedule_return_report():
    schedule = {"step": {"val": np.array([1.0]), "control": np.array([0], dtype=int)},
                "control": [{"W": [{"type": "rate", "val": 1.0, "sign": -1, "status": True}]}]}
    state0 = {"time": 0.0}
    model = {"porevolume": np.array([1.0])}
    wells, states, schedulereport = simulate_schedule_ad(state0, model, schedule, return_report=True)
    assert len(wells) == 1
    assert len(states) == 1
    assert schedulereport["NumControlSteps"] == 1
    assert "ControlstepReports" in schedulereport
