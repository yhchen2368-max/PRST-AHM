function export_mrst_spe9_full(output_file)
%EXPORT_MRST_SPE9_FULL Reference well-solution + timing data for the full
% 90-step SPE9 run, for comparison against PRSTCore's
% init_eclipse_problem_ad + simulate_schedule_ad on the same deck.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-props deckformat ad-core ad-blackoil

fn = fullfile(repo, 'examples', 'SPE9', 'SPE9_CP.DATA');
deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);

[state0, model, schedule] = initEclipseProblemAD(deck);

nls = NonLinearSolver('useLineSearch', true);

tic
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls); %#ok<ASGLU>
elapsed = toc;

nw = numel(wellSols{1});
nt = numel(wellSols);
names = cell(nw, 1);
for k2 = 1:nw
    names{k2} = wellSols{1}(k2).name;
end
qOs = zeros(nt, nw); qWs = zeros(nt, nw); bhp = zeros(nt, nw);
for kt = 1:nt
    for kw = 1:nw
        qOs(kt, kw) = wellSols{kt}(kw).qOs;
        qWs(kt, kw) = wellSols{kt}(kw).qWs;
        bhp(kt, kw) = wellSols{kt}(kw).bhp;
    end
end

pressure_final = states{end}.pressure;
sw_final = states{end}.s(:,1);

save(output_file, 'names', 'qOs', 'qWs', 'bhp', 'elapsed', ...
     'pressure_final', 'sw_final', '-v7');
fprintf('MRST SPE9 reference written to %s (elapsed=%.2fs)\n', output_file, elapsed);
end
