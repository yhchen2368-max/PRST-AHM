function export_mrst_spe1(output_file)
%EXPORT_MRST_SPE1 Reference well-solution + timing data for the SPE1
% benchmark, for comparison against PRSTCore's init_eclipse_problem_ad +
% simulate_schedule_ad on the same deck.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-props deckformat ad-core ad-blackoil

fn = fullfile(repo, 'examples', 'SpE1', 'BENCH_SPE1.DATA');
deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);

G = initEclipseGrid(deck);
G = computeGeometry(G);
rock = initEclipseRock(deck);
rock = compressRock(rock, G.cells.indexMap);
fluid = initDeckADIFluid(deck);

gravity reset on;

[ijk{1:3}] = ind2sub(G.cartDims, G.cells.indexMap);
k = ijk{3};
p0 = [329.7832774859256; 330.2313357125603; 330.9483500720813];
p0 = convertFrom(p0(k), barsa);
s0 = repmat([0.12, 0.88, 0.0], [G.cells.num, 1]);
rs0 = repmat(226.1966570852417, [G.cells.num, 1]);
state0 = struct('s', s0, 'rs', rs0, 'rv', 0, 'pressure', p0);

model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);

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
qOs = zeros(nt, nw); qWs = zeros(nt, nw); qGs = zeros(nt, nw); bhp = zeros(nt, nw);
for kt = 1:nt
    for kw = 1:nw
        qOs(kt, kw) = wellSols{kt}(kw).qOs;
        qWs(kt, kw) = wellSols{kt}(kw).qWs;
        qGs(kt, kw) = wellSols{kt}(kw).qGs;
        bhp(kt, kw) = wellSols{kt}(kw).bhp;
    end
end

pressure_final = states{end}.pressure;
sw_final = states{end}.s(:,1);
so_final = states{end}.s(:,2);
sg_final = states{end}.s(:,3);

save(output_file, 'names', 'qOs', 'qWs', 'qGs', 'bhp', 'elapsed', ...
     'pressure_final', 'sw_final', 'so_final', 'sg_final', '-v7');
fprintf('MRST SPE1 reference written to %s (elapsed=%.2fs)\n', output_file, elapsed);
end
