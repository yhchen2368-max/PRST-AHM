%% Native MRST NORNE first report-step reference.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(cgnet_root, 'mrst-2026a');
if ~exist(mrst_root, 'dir')
    mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
end
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

deckfile = fullfile(cgnet_root, 'examples', 'Norne', ...
    'Norne_simplified', 'NORNE_ATW2013.DATA');
[state0, model, schedule, solver] = initEclipseProblemAD(deckfile);
solver.verbose = true;

controlId = schedule.step.control(1);
control = schedule.control(controlId);
dt = schedule.step.val(1);

% Match simulateScheduleAD's per-control preparation before solveTimestep.
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

fprintf('MRST NORNE init cells=%d steps=%d dt=%g wells=%d solver=%s linear=%s\n', ...
    model.G.cells.num, numel(schedule.step.val), dt, numel(control.W), ...
    class(solver), class(solver.LinearSolver));

t0 = tic;
[state, report, ministates] = solver.solveTimestep( ...
    state0, dt, model, forces{:}, 'initialGuess', state0, ...
    'controlId', controlId);
elapsed = toc(t0);

ministep_dt = nan(numel(report.StepReports), 1);
ministep_iterations = nan(numel(report.StepReports), 1);
ministep_converged = false(numel(report.StepReports), 1);
trace_residuals = {};
trace_converged = {};
trace_linear_time = {};
trace_linear_iterations = {};
for mini = 1:numel(report.StepReports)
    sr = report.StepReports{mini};
    ministep_dt(mini) = sr.Timestep;
    ministep_iterations(mini) = sr.Iterations;
    ministep_converged(mini) = sr.Converged;
    nr = sr.NonlinearReport;
    r = nan(numel(nr), 1);
    c = false(numel(nr), 1);
    lt = nan(numel(nr), 1);
    li = nan(numel(nr), 1);
    for it = 1:numel(nr)
        if isfield(nr{it}, 'Residuals')
            vals = nr{it}.Residuals;
            if ~isempty(vals)
                r(it) = max(vals);
            end
        end
        if isfield(nr{it}, 'Converged')
            c(it) = nr{it}.Converged;
        end
        if isfield(nr{it}, 'LinearSolver')
            ls = nr{it}.LinearSolver;
            if isfield(ls, 'LinearSolveTime')
                lt(it) = ls.LinearSolveTime;
            end
            if isfield(ls, 'Iterations')
                li(it) = ls.Iterations;
            end
        end
    end
    trace_residuals{mini} = r;
    trace_converged{mini} = c;
    trace_linear_time{mini} = lt;
    trace_linear_iterations{mini} = li;
    fprintf('MRST ministep=%d dt=%g converged=%d iterations=%d residual_last=%g\n', ...
        mini, ministep_dt(mini), ministep_converged(mini), ...
        ministep_iterations(mini), r(end));
end

fprintf('MRST NORNE done converged=%d iterations=%d ministeps=%d elapsed=%g p=[%g,%g] sw=[%g,%g] sg=[%g,%g]\n', ...
    report.Converged, report.Iterations, numel(report.StepReports), elapsed, ...
    min(value(state.pressure)), max(value(state.pressure)), ...
    min(value(state.s(:,1))), max(value(state.s(:,1))), ...
    min(value(state.s(:,3))), max(value(state.s(:,3))));

save(fullfile(cgnet_root, 'mrst_norne_first_step.mat'), ...
    'state0', 'state', 'report', 'ministates', 'dt', 'controlId', ...
    'initialState', 'initial_residual', 'initial_jacobian', ...
    'elapsed', 'ministep_dt', 'ministep_iterations', ...
    'ministep_converged', 'trace_residuals', 'trace_converged', ...
    'trace_linear_time', 'trace_linear_iterations');
