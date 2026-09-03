%% Native MRST EGG first-report-step reference (two-phase oil/water).
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, solver] = initEclipseProblemAD( ...
    fullfile(cgnet_root, 'examples', 'EGG', 'Egg_Model_ECL.DATA'));
controlId = schedule.step.control(1);
control = schedule.control(controlId);
dt = schedule.step.val(1);
[forces, fstruct] = model.getDrivingForces(control);
model = model.validateModel(fstruct, false);
model.checkStateFunctionDependencies();
schedule = model.validateSchedule(schedule);
state0 = model.validateState(state0);
[model, state0] = model.updateForChangedControls(state0, fstruct);
[initialProblem, initialState] = model.getEquations(state0, state0, dt, fstruct, ...
    'iteration', 1, 'resOnly', false);
initialProblem = initialProblem.assembleSystem();
initial_residual = initialProblem.b;
initial_jacobian = initialProblem.A;
[state, report, ministates] = solver.solveTimestep(state0, dt, model, forces{:}, ...
    'initialGuess', state0, 'controlId', controlId);
save(fullfile(cgnet_root, 'mrst_egg_first_step.mat'), ...
    'state0', 'state', 'report', 'ministates', 'dt', 'initialState', ...
    'initial_residual', 'initial_jacobian', 'controlId');
fprintf('model=%s converged=%d iterations=%d ministeps=%d\n', ...
    class(model), report.Converged, report.Iterations, numel(ministates));
