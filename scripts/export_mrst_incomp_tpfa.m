function export_mrst_incomp_tpfa(output_file)
%EXPORT_MRST_INCOMP_TPFA Reference + timing data for incompTPFA parity.
%
% Small single-phase case: a 3D cartGrid with one rate-controlled injector,
% one bhp-controlled producer, and a Dirichlet pressure face on one side --
% exercises wells, boundary conditions, and the core TPFA linear algebra
% (gravity off, matching the Python port's scope). Also includes a larger
% timed case (no wells/bc, source-only) for a speed comparison.

if nargin < 1 || isempty(output_file)
    repo = fileparts(fileparts(mfilename('fullpath')));
    output_file = fullfile(repo, 'tests', 'incomp_tpfa_mrst_ref.mat');
end

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add incomp
gravity off

%% Case 1: correctness -- wells + Dirichlet bc
celldim = [6, 5, 4];
physdim = [6, 5, 4];
G = computeGeometry(cartGrid(celldim, physdim));
rock = makeRock(G, 1e-13, 1);
T = computeTrans(G, rock);
fluid = initSingleFluid('mu', 1e-3, 'rho', 1000);

W = [];
W = addWell(W, G, rock, 1, 'Type', 'rate', 'Val', 1e-4, 'Name', 'I1', 'Radius', 0.1);
W = addWell(W, G, rock, G.cells.num, 'Type', 'bhp', 'Val', 1.0e5, 'Name', 'P1', 'Radius', 0.1);

bc = pside([], G, 'YMin', 1.2e5);

state = initResSol(G, 1e5);
state.wellSol = initWellSol(W, 1e5);
state = incompTPFA(state, G, T, fluid, 'wells', W, 'bc', bc);

pressure = state.pressure;
flux = state.flux;
facePressure = state.facePressure;
well_flux = vertcat(state.wellSol.flux);
well_pressure = vertcat(state.wellSol.pressure);
neighbors = G.faces.neighbors;

% Export the inputs actually consumed by incompTPFA (well cells/WI/type/val,
% bc face/type/value, rock perm) so the Python side can reconstruct the exact
% same linear system without needing a ported addWell/pside/makeRock first --
% this test is scoped to incompTPFA's own linear algebra, not those helpers.
well_cells = {W.cells};
well_WI    = {W.WI};
well_type  = {W.type};
well_val   = vertcat(W.val);
bc_face  = bc.face;
bc_type  = bc.type;
bc_value = bc.value;
rock_perm = rock.perm;

%% Timing benchmark
celldim_big = [30, 30, 20];
physdim_big = celldim_big;
Gbig = computeGeometry(cartGrid(celldim_big, physdim_big));
rockBig = makeRock(Gbig, 1e-13, 1);

srcBig = addSource([], 1, 1e-3);
srcBig = addSource(srcBig, Gbig.cells.num, -1e-3);

Twarm = computeTrans(Gbig, rockBig);
stateWarm = initResSol(Gbig, 1e5);
stateWarm = incompTPFA(stateWarm, Gbig, Twarm, fluid, 'src', srcBig); %#ok<NASGU>

nrep = 3;
t_trans = inf; t_solve = inf;
for k = 1:nrep
    tic; Tb = computeTrans(Gbig, rockBig); tt = toc;
    stateBig = initResSol(Gbig, 1e5);
    tic; stateBig = incompTPFA(stateBig, Gbig, Tb, fluid, 'src', srcBig); ts = toc; %#ok<NASGU>
    t_trans = min(t_trans, tt);
    t_solve = min(t_solve, ts);
end
bench_num_cells = prod(celldim_big);

save(output_file, ...
    'celldim', 'physdim', 'pressure', 'flux', 'facePressure', ...
    'well_flux', 'well_pressure', 'neighbors', ...
    'well_cells', 'well_WI', 'well_type', 'well_val', ...
    'bc_face', 'bc_type', 'bc_value', 'rock_perm', ...
    't_trans', 't_solve', 'bench_num_cells', 'celldim_big', ...
    '-v7');

fprintf('MRST incompTPFA reference written to %s\n', output_file);
fprintf('MATLAB timing (%d cells): trans=%.4fs solve=%.4fs\n', bench_num_cells, t_trans, t_solve);
end
