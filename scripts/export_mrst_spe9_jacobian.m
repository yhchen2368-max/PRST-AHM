% MRST SPE9 first-step Jacobian export for Python comparison
run('C:/Users/junji/Desktop/github/mrst-2026a/startup.m');
mrstModule add ad-core ad-blackoil ad-props deckformat mrst-gui
deck = readEclipseDeck('C:/Users/junji/Desktop/github/Cgnet/examples/SPE9/SPE9_CP.DATA');
deck = convertDeckUnits(deck);
[state0, model, schedule] = initEclipseProblemAD(deck, 'TimestepStrategy', 'none');

% Ensure black-oil state fields are present (SPE9 is gas+oil)
if ~isfield(state0, 'rs')
    state0.rs = zeros(model.G.cells.num, 1);
end
if ~isfield(state0, 'rv')
    state0.rv = zeros(model.G.cells.num, 1);
end

% Get first timestep and driving forces
dt = schedule.step.val(1);
drivingForces = schedule.control(schedule.step.control(1));

% Manually call getEquations to get initial Jacobian (iteration 1, no solve)
[problem, state] = model.getEquations(state0, state0, dt, drivingForces, 'iteration', 1, 'resOnly', false);

% Extract and save
J = problem.A;
r = problem.b;
nc = model.G.cells.num;
fprintf('MRST: nc=%d J=(%d,%d) nnz=%d res_max=%.3e\n', nc, size(J,1), size(J,2), nnz(J), max(abs(r)));

% Save sparse matrix in coordinate format (1-indexed)
[ii, jj, vv] = find(J);
writematrix([ii jj vv], 'spe9_mrst_jacobian.txt', 'Delimiter', ' ');
writematrix(r, 'spe9_mrst_residual.txt');
writematrix(state.pressure, 'spe9_mrst_pressure.txt');
if isfield(state, 's')
    writematrix(state.s, 'spe9_mrst_saturation.txt');
end
fprintf('Saved to spe9_mrst_*.txt\n');
