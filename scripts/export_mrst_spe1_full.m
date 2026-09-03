%% Export the complete MRST 2026a SPE1 schedule for Python parity checks.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
deck_path = fullfile(cgnet_root, 'examples', 'SpE1', 'BENCH_SPE1.DATA');
output_file = fullfile(cgnet_root, 'spe1_mrst_full.mat');

assert(isfolder(mrst_root), 'MRST root not found: %s', mrst_root);
assert(isfile(deck_path), 'SPE1 deck not found: %s', deck_path);
addpath(mrst_root);
startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, nonlinear] = initEclipseProblemAD(deck_path);
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule, ...
    'NonLinearSolver', nonlinear, 'Verbose', false);

n = numel(states);
nc = model.G.cells.num;
pressure = zeros(nc, n);
sw = zeros(nc, n);
sg = zeros(nc, n);
rs = zeros(nc, n);
time = zeros(1, n);
bhp = zeros(2, n);
qws = zeros(2, n);
qos = zeros(2, n);
qgs = zeros(2, n);
iterations = zeros(1, n);
ministeps = zeros(1, n);
for i = 1:n
    st = states{i};
    pressure(:, i) = st.pressure;
    sw(:, i) = st.s(:, 1);
    sg(:, i) = st.s(:, 3);
    rs(:, i) = st.rs;
    time(i) = st.time;
    bhp(:, i) = vertcat(st.wellSol.bhp);
    qws(:, i) = vertcat(st.wellSol.qWs);
    qos(:, i) = vertcat(st.wellSol.qOs);
    qgs(:, i) = vertcat(st.wellSol.qGs);
    iterations(i) = report.ControlstepReports{i}.Iterations;
    ministeps(i) = numel(report.ControlstepReports{i}.StepReports);
end

save(output_file, 'pressure', 'sw', 'sg', 'rs', 'time', ...
    'bhp', 'qws', 'qos', 'qgs', 'iterations', 'ministeps', ...
    'state0', 'schedule', '-v7');
fprintf('MRST full SPE1 reference written to %s (%d report steps)\n', output_file, n);
