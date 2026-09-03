%% Export every MRST nonlinear state for the first two SPE1 report steps.
clear; clc;
script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
deck_path = fullfile(cgnet_root, 'examples', 'SpE1', 'BENCH_SPE1.DATA');
output_file = fullfile(cgnet_root, 'spe1_mrst_step2_trace.mat');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props
[state, model, schedule, nonlinear] = initEclipseProblemAD(deck_path);
nonlinear.reportLevel = 3;
schedule.step.val = schedule.step.val(1:2);
schedule.step.control = schedule.step.control(1:2);
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
trace_step = [];
trace_dt = [];
trace_iteration = [];
for step = 1:2
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
            trace_step(end+1,1) = step;
            trace_dt(end+1,1) = step_reports{ms}.Timestep;
            trace_iteration(end+1,1) = it;
        end
    end
end
save(output_file, 'trace_p', 'trace_sw', 'trace_sg', 'trace_rs', ...
    'trace_status', 'trace_bhp', 'trace_qgs', 'trace_step', 'trace_dt', ...
    'trace_iteration', 'all_reports', '-v7');
fprintf('Wrote %d MRST nonlinear trace states to %s\n', numel(trace_p), output_file);
