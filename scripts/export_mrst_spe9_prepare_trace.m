%% Native MRST GenericFacilityModel.prepareTimestep reference for SPE9.
% This isolates the exact pre-Newton well-bore pressure-drop and control
% switching path used by NonLinearSolver.solveMinistep.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, ~] = initEclipseProblemAD( ...
    fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA'));
controlId = schedule.step.control(1);
control = schedule.control(controlId);
dt = schedule.step.val(1);
[~, fstruct] = model.getDrivingForces(control);
model = model.validateModel(fstruct, false);
model.checkStateFunctionDependencies();
schedule = model.validateSchedule(schedule);
state0 = model.validateState(state0);
[model, state0] = model.updateForChangedControls(state0, fstruct);

% GenericFacilityModel.prepareTimestep (MRST source) updates cdp first and
% then calls applyWellLimits.  Keep the original state0 as its reference.
state_prepared = state0;
[model, state_prepared] = model.prepareTimestep( ...
    state_prepared, state0, dt, fstruct);
[prepared_problem, prepared_equation_state] = model.getEquations( ...
    state0, state_prepared, dt, fstruct, 'iteration', 1, 'resOnly', false);
prepared_problem = prepared_problem.assembleSystem();
prepared_residual = prepared_problem.b;
prepared_jacobian = prepared_problem.A;
[prepared_density, prepared_mobility] = model.getProps( ...
    state_prepared, 'Density', 'Mobility');
prepared_density = value(prepared_density);
prepared_mobility = cellfun(@value, prepared_mobility, 'UniformOutput', false);
prepared_mobility = [prepared_mobility{:}];
forcing_wells = fstruct.W;
gravity = model.gravity;

save(fullfile(cgnet_root, 'mrst_spe9_prepare_trace.mat'), ...
    'state0', 'state_prepared', 'forcing_wells', 'prepared_density', ...
    'prepared_mobility', 'prepared_equation_state', 'prepared_residual', ...
    'prepared_jacobian', 'gravity', 'dt', 'controlId');
fprintf('prepared active wells=%d\n', numel(state_prepared.wellSol));
for i = 1:numel(state_prepared.wellSol)
    ws = state_prepared.wellSol(i);
    fprintf('%s %s %.15g cdp=[', ws.name, ws.type, ws.val);
    fprintf(' %.15g', ws.cdp);
    fprintf(' ]\n');
end
