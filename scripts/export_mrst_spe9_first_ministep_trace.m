%% Native MRST Newton-state trace for SPE9's first 1-day mini-step.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, solver] = initEclipseProblemAD( ...
    fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA'));
control = schedule.control(schedule.step.control(1));
[~, forces] = model.getDrivingForces(control);
model = model.validateModel(forces, false);
model.checkStateFunctionDependencies();
state0 = model.validateState(state0);
[model, state0] = model.updateForChangedControls(state0, forces);

% SimpleTimeStepSelector chooses this first mini-step from the 10-day
% report step.  Execute PhysicalModel.stepFunction once per MRST Newton
% iteration and retain the state after each linear solve/update.
dt = day;
[model, state] = model.prepareTimestep(state0, state0, dt, forces);
solver.relaxationParameter = solver.maxRelaxation;
solver.convergenceIssues = false;
states = cell(solver.maxIterations + 1, 1);
reports = cell(solver.maxIterations + 1, 1);
for i = 1:(solver.maxIterations + 1)
    [state, report] = model.stepFunction(state, state0, dt, forces, ...
        solver.LinearSolver, solver, i);
    states{i} = state;
    reports{i} = report;
    if report.Converged || report.Failure
        states = states(1:i);
        reports = reports(1:i);
        break
    end
end
save(fullfile(cgnet_root, 'mrst_spe9_first_ministep_trace.mat'), ...
    'state0', 'states', 'reports', 'dt');
fprintf('steps=%d finalConverged=%d\n', numel(reports), reports{end}.Converged);
