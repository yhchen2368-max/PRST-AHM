%% Export an exact MRST baseline for the first SPE1 report step.
%
% This script intentionally follows initEclipseProblemAD and the same
% validation path as simulateScheduleAD.  It is the reference used by the
% Python parity checks; do not substitute a manually constructed state or
% schedule here.

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
deck_path = fullfile(cgnet_root, 'examples', 'SpE1', 'BENCH_SPE1.DATA');
output_file = fullfile(cgnet_root, 'spe1_mrst_first_step.mat');

assert(isfolder(mrst_root), 'MRST root not found: %s', mrst_root);
assert(isfile(deck_path), 'SPE1 deck not found: %s', deck_path);

addpath(mrst_root);
startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, nonlinear] = initEclipseProblemAD(deck_path);

control_id = schedule.step.control(1);
[force_args, control_struct] = model.getDrivingForces(schedule.control(control_id));
model = model.validateModel(control_struct);
state0 = model.validateState(state0);
driving_forces = merge_options(model.getValidDrivingForces(), force_args{:});
dt = schedule.step.val(1);

[problem, assembled_state] = model.getEquations( ...
    state0, state0, dt, driving_forces, 'iteration', 0);
[jacobian, rhs] = problem.getLinearSystem();
residual = -rhs;

% Use the standard MRST nonlinear solver supplied by initEclipseProblemAD.
[state, report] = nonlinear.solveTimestep( ...
    state0, dt, model, force_args{:}, 'initialGuess', state0);

% Evaluate the converged state through the same GenericBlackOilModel path.
% These fields let the Python port compare each state-function layer rather
% than inferring a correction from the final pressure alone.
[problem_final, assembled_state_final] = model.getEquations( ...
    state0, state, dt, driving_forces, 'iteration', report.Iterations);
[jacobian_final, rhs_final] = problem_final.getLinearSystem();
residual_final = -rhs_final;
[mobility_final, density_final, shrinkage_final, rsmax_final, pv_final] = ...
    model.getProps(assembled_state_final, 'Mobility', 'Density', ...
                   'ShrinkageFactors', 'RsMax', 'PoreVolume');
mobility_final = cellfun(@value, mobility_final, 'UniformOutput', false);
density_final = cellfun(@value, density_final, 'UniformOutput', false);
shrinkage_final = cellfun(@value, shrinkage_final, 'UniformOutput', false);
rsmax_final = value(rsmax_final);
pv_final = value(pv_final);
mobility_final = [mobility_final{:}];
density_final = [density_final{:}];
shrinkage_final = [shrinkage_final{:}];

cell_volumes = model.G.cells.volumes;
cell_centroids = model.G.cells.centroids;
transmissibility = model.operators.T;
neighbors = model.operators.N;
wells = schedule.control(control_id).W;

save(output_file, ...
    'jacobian', 'rhs', 'residual', ...
    'state0', 'assembled_state', 'state', 'report', ...
    'jacobian_final', 'rhs_final', 'residual_final', 'assembled_state_final', ...
    'mobility_final', 'density_final', 'shrinkage_final', 'rsmax_final', 'pv_final', ...
    'dt', 'control_id', 'cell_volumes', 'cell_centroids', ...
    'transmissibility', 'neighbors', 'wells', ...
    '-v7');

fprintf('MRST SPE1 reference written to %s\n', output_file);
fprintf('cells=%d dt=%.17g A=%dx%d nnz=%d residual_norm=%.17g converged=%d\n', ...
    model.G.cells.num, dt, size(jacobian, 1), size(jacobian, 2), ...
    nnz(jacobian), norm(residual), report.Converged);
