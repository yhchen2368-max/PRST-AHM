%% Export the exact MRST Newton states at the SPE1 parity divergence steps.
% The selected report steps are the only ones where the Python run has a
% different Newton iteration count.  This keeps the comparison tied to the
% ordinary MRST 2026a solver path; no hand-created reference equations are
% used here.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
deck_path = fullfile(cgnet_root, 'examples', 'SPE1', 'BENCH_SPE1.DATA');
output_file = fullfile(cgnet_root, 'spe1_mrst_selected_trace.mat');
selected_steps = [22, 58, 90];

addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props
[state, model, schedule, nonlinear] = initEclipseProblemAD(deck_path);
nonlinear.reportLevel = 3;
last_step = max(selected_steps);
schedule.step.val = schedule.step.val(1:last_step);
schedule.step.control = schedule.step.control(1:last_step);
[~, ~, schedule_report] = simulateScheduleAD(state, model, schedule, ...
    'NonLinearSolver', nonlinear, 'Verbose', false);

all_reports = schedule_report.ControlstepReports;
trace_p = {};
trace_sw = {};
trace_sg = {};
trace_rs = {};
trace_status = {};
trace_bhp = {};
trace_qgs = {};
trace_residuals = {};
trace_converged = {};
trace_step = [];
trace_ministep = [];
trace_dt = [];
trace_iteration = [];
trace_relaxation = [];
for step = selected_steps
    step_reports = all_reports{step}.StepReports;
    for ms = 1:numel(step_reports)
        nr = step_reports{ms}.NonlinearReport;
        for it = 1:numel(nr)
            if ~isfield(nr{it}, 'state')
                continue;
            end
            st = nr{it}.state;
            trace_p{end+1,1} = st.pressure;
            trace_sw{end+1,1} = st.s(:,1);
            trace_sg{end+1,1} = st.s(:,3);
            trace_rs{end+1,1} = st.rs;
            trace_status{end+1,1} = st.status;
            trace_bhp{end+1,1} = vertcat(st.wellSol.bhp);
            trace_qgs{end+1,1} = vertcat(st.wellSol.qGs);
            trace_residuals{end+1,1} = nr{it}.Residuals;
            trace_converged{end+1,1} = nr{it}.ResidualsConverged;
            trace_step(end+1,1) = step;
            trace_ministep(end+1,1) = ms;
            trace_dt(end+1,1) = step_reports{ms}.Timestep;
            trace_iteration(end+1,1) = it;
            stab = nr{it}.StabilizeReport;
            if isstruct(stab) && isfield(stab, 'relaxationParameter')
                trace_relaxation(end+1,1) = stab.relaxationParameter;
            else
                trace_relaxation(end+1,1) = nan;
            end
        end
    end
end
save(output_file, 'trace_p', 'trace_sw', 'trace_sg', 'trace_rs', ...
    'trace_status', 'trace_bhp', 'trace_qgs', 'trace_residuals', ...
    'trace_converged', 'trace_step', 'trace_ministep', 'trace_dt', ...
    'trace_iteration', 'trace_relaxation', 'all_reports', '-v7');
fprintf('Wrote %d MRST nonlinear trace states to %s\n', numel(trace_p), output_file);
