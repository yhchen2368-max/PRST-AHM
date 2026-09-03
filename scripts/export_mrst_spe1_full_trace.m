%% Export full MRST 2026a SPE1 nonlinear/Jacobian trace.
%
% This is a diagnostic exporter. It follows simulateScheduleAD's control-step
% loop and uses the normal NonLinearSolver for the actual solution. After each
% report step, it reconstructs the pre-Newton states from reportLevel=3 states
% and assembles the corresponding MRST linearized systems.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(cgnet_root, 'mrst-2026a');
deck_path = fullfile(cgnet_root, 'examples', 'SpE1', 'BENCH_SPE1.DATA');
output_file = fullfile(cgnet_root, 'spe1_mrst_full_trace.mat');

assert(isfolder(mrst_root), 'MRST root not found: %s', mrst_root);
assert(isfile(deck_path), 'SPE1 deck not found: %s', deck_path);

addpath(mrst_root);
startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state, model, schedule, nonlinear] = initEclipseProblemAD(deck_path);
nonlinear.reportLevel = 3;
nonlinear.timeStepSelector.reset();

ctrl = schedule.control(schedule.step.control(1));
[force_args, fstruct] = model.getDrivingForces(ctrl);
model = model.validateModel(fstruct);
model.checkStateFunctionDependencies();
schedule = model.validateSchedule(schedule);
state = model.validateState(state);

nsteps = numel(schedule.step.val);
trace_step = [];
trace_ministep = [];
trace_iteration = [];
trace_dt = [];
trace_assembly_time = [];
trace_linear_iterations = [];
trace_linear_residual = [];
trace_linear_time = [];
trace_linear_solution_time = [];
trace_linear_converged = [];
trace_relaxation = [];
trace_solved = [];
trace_residual_values = {};
trace_residual_converged = {};
trace_jac_rows = {};
trace_jac_cols = {};
trace_jac_vals = {};
trace_residual_vector = {};
trace_jac_shape = [];
trace_residual_size = [];
step_iterations = zeros(nsteps, 1);
step_ministeps = zeros(nsteps, 1);
step_times = zeros(nsteps, 1);

prev_control = nan;
record_count = 0;
for step = 1:nsteps
    state0_control = state;
    curr_control = schedule.step.control(step);
    if prev_control ~= curr_control
        [force_args, fstruct] = model.getDrivingForces(schedule.control(curr_control));
        [model, state0_control] = model.updateForChangedControls(state, fstruct);
        prev_control = curr_control;
    end
    driving_forces = merge_options(model.getValidDrivingForces(), force_args{:});
    dt = schedule.step.val(step);
    timer = tic();
    [state, report, ministates] = nonlinear.solveTimestep( ...
        state0_control, dt, model, force_args{:}, 'controlId', curr_control);
    step_times(step) = toc(timer);
    if ~report.Converged
        error('SPE1 did not converge at report step %d', step);
    end
    step_iterations(step) = report.Iterations;
    step_ministeps(step) = numel(report.StepReports);

    mini_state0 = state0_control;
    for ms = 1:numel(report.StepReports)
        step_report = report.StepReports{ms};
        dt_ms = step_report.Timestep;
        guess = mini_state0;
        guess.time = mini_state0.time + dt_ms;
        [~, prepared_state] = model.prepareTimestep( ...
            guess, mini_state0, dt_ms, driving_forces);
        pre_state = prepared_state;
        nonlinear_reports = step_report.NonlinearReport;
        for it = 1:numel(nonlinear_reports)
            nr = nonlinear_reports{it};
            timer_asm = tic();
            [problem, ~] = model.getEquations( ...
                mini_state0, pre_state, dt_ms, driving_forces, ...
                'iteration', it);
            assembly_time = toc(timer_asm);
            [jacobian, rhs] = problem.getLinearSystem();
            residual = -rhs;
            [conv, values] = model.checkConvergence(problem);
            [ii, jj, vv] = find(jacobian);

            record_count = record_count + 1;
            trace_step(record_count, 1) = step;
            trace_ministep(record_count, 1) = ms;
            trace_iteration(record_count, 1) = it;
            trace_dt(record_count, 1) = dt_ms;
            trace_assembly_time(record_count, 1) = assembly_time;
            trace_jac_rows{record_count, 1} = uint32(ii);
            trace_jac_cols{record_count, 1} = uint32(jj);
            trace_jac_vals{record_count, 1} = vv;
            trace_residual_vector{record_count, 1} = residual;
            trace_jac_shape(record_count, :) = size(jacobian);
            trace_residual_size(record_count, 1) = numel(residual);
            trace_residual_values{record_count, 1} = values;
            trace_residual_converged{record_count, 1} = conv;
            trace_solved(record_count, 1) = isfield(nr, 'Solved') && nr.Solved;

            if isfield(nr, 'LinearSolver') && isstruct(nr.LinearSolver)
                ls = nr.LinearSolver;
                trace_linear_iterations(record_count, 1) = getfield_default(ls, 'Iterations', nan);
                trace_linear_residual(record_count, 1) = getfield_default(ls, 'Residual', nan);
                trace_linear_time(record_count, 1) = getfield_default(ls, 'SolverTime', nan);
                trace_linear_solution_time(record_count, 1) = getfield_default(ls, 'LinearSolutionTime', nan);
                trace_linear_converged(record_count, 1) = getfield_default(ls, 'Converged', false);
            else
                trace_linear_iterations(record_count, 1) = nan;
                trace_linear_residual(record_count, 1) = nan;
                trace_linear_time(record_count, 1) = nan;
                trace_linear_solution_time(record_count, 1) = nan;
                trace_linear_converged(record_count, 1) = false;
            end

            if isfield(nr, 'StabilizeReport') && isstruct(nr.StabilizeReport) && ...
                    isfield(nr.StabilizeReport, 'relaxationParameter')
                trace_relaxation(record_count, 1) = nr.StabilizeReport.relaxationParameter;
            else
                trace_relaxation(record_count, 1) = nan;
            end

            if isfield(nr, 'state')
                pre_state = nr.state;
            end
        end
        mini_state0 = ministates{ms};
    end
    fprintf('STEP=%03d records=%d iterations=%d ministeps=%d wall=%.3fs\n', ...
        step, record_count, report.Iterations, numel(report.StepReports), step_times(step));
end

jac_ptr = zeros(numel(trace_jac_rows) + 1, 1);
residual_ptr = zeros(numel(trace_residual_vector) + 1, 1);
for k = 1:numel(trace_jac_rows)
    jac_ptr(k + 1) = jac_ptr(k) + numel(trace_jac_vals{k});
    residual_ptr(k + 1) = residual_ptr(k) + numel(trace_residual_vector{k});
end
trace_jac_rows = vertcat(trace_jac_rows{:});
trace_jac_cols = vertcat(trace_jac_cols{:});
trace_jac_vals = vertcat(trace_jac_vals{:});
trace_residual_vector = vertcat(trace_residual_vector{:});

save(output_file, ...
    'trace_step', 'trace_ministep', 'trace_iteration', 'trace_dt', ...
    'trace_assembly_time', 'trace_linear_iterations', ...
    'trace_linear_residual', 'trace_linear_time', ...
    'trace_linear_solution_time', 'trace_linear_converged', ...
    'trace_relaxation', 'trace_solved', 'trace_residual_values', ...
    'trace_residual_converged', 'trace_jac_rows', 'trace_jac_cols', ...
    'trace_jac_vals', 'jac_ptr', 'trace_residual_vector', ...
    'residual_ptr', 'trace_jac_shape', 'trace_residual_size', ...
    'step_iterations', 'step_ministeps', 'step_times', '-v7');
fprintf('MRST full SPE1 trace written to %s (%d records)\n', output_file, record_count);

function value = getfield_default(s, name, default)
    if isstruct(s) && isfield(s, name)
        value = s.(name);
    else
        value = default;
    end
end
