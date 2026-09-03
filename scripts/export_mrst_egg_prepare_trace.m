%% EGG first-mini-step pre-Newton reference after FacilityModel preparation.
% This captures the exact state/equation pair used by
% NonLinearSolver.solveMinistep after GenericFacilityModel.prepareTimestep.
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
dt = schedule.step.val(1) / 10;
[forces, fstruct] = model.getDrivingForces(control);
model = model.validateModel(fstruct, false);
model.checkStateFunctionDependencies();
state0 = model.validateState(state0);
[model, state0] = model.updateForChangedControls(state0, fstruct);
[model, preparedState] = model.prepareTimestep(state0, state0, dt, fstruct);
preparedBeforeEquations = preparedState;
[problem, preparedState] = model.getEquations(state0, preparedState, dt, fstruct, ...
    'iteration', 1, 'resOnly', false);
problem = problem.assembleSystem();
prepared_residual = problem.b;
prepared_jacobian = problem.A;
prepared_phase_flux = value(model.FacilityModel.getProp(preparedState, 'PhaseFlux'));
prepared_component_flux = value(model.FacilityModel.getProp(preparedState, 'ComponentTotalFlux'));
gravity = model.gravity;
W = schedule.control(controlId).W;
save(fullfile(cgnet_root, 'mrst_egg_prepare_trace.mat'), ...
    'state0', 'preparedBeforeEquations', 'preparedState', 'prepared_residual', 'prepared_jacobian', ...
    'dt', 'controlId', 'gravity', 'W', 'prepared_phase_flux', ...
    'prepared_component_flux');
