def pack_simulation_problem(state0, model, schedule, base_name, **kwargs):
    simulator_setup = {
        "state0": state0,
        "model": model,
        "schedule": schedule,
        "NonLinearSolver": kwargs.get("NonLinearSolver", None),
    }
    simulator_setup.update({k: v for k, v in kwargs.items() if k not in ("NonLinearSolver",)})
    return {
        "BaseName": base_name,
        "Name": kwargs.get("Name", type(model).__name__ if model is not None else "Problem"),
        "Description": kwargs.get("Description", ""),
        "SimulatorSetup": simulator_setup,
        "Modules": kwargs.get("Modules", []),
        "OutputHandlers": {},
        "_simulated": None,
    }
