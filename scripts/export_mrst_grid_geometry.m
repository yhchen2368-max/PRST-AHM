function export_mrst_grid_geometry(output_file)
%EXPORT_MRST_GRID_GEOMETRY Reference + timing data for cartGrid/tensorGrid/computeGeometry parity.
%
% Generates three correctness cases (regular 3D cartGrid, a perturbed
% non-uniform 3D tensorGrid to stress-test the general polyhedral geometry
% algorithm, and a 2D cartGrid) plus a larger timed case used to compare
% MATLAB/MRST vs. Python/PRSTCore wall-clock speed for grid construction
% and geometry computation.

if nargin < 1 || isempty(output_file)
    repo = fileparts(fileparts(mfilename('fullpath')));
    output_file = fullfile(repo, 'tests', 'grid_geometry_mrst_ref.mat');
end

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));

%% Case 1: regular 3D cartGrid
G1 = computeGeometry(cartGrid([4, 3, 2], [40, 30, 20]));

%% Case 2: perturbed, non-uniform tensorGrid (stresses the general
%  triangulated-face / tetrahedralized-cell algorithm, not just the
%  trivial cuboid formula that would suffice for an axis-aligned grid).
rng(0);
x = cumsum([0, 1, 1.5, 0.7, 2, 1.2]);
y = cumsum([0, 1, 0.8, 1.3, 1]);
z = cumsum([0, 1, 1.1]);
G2 = tensorGrid(x, y, z);
jitter = 0.15 * (2*rand(G2.nodes.num, 3) - 1);
G2.nodes.coords = G2.nodes.coords + jitter;
G2 = computeGeometry(G2);

%% Case 3: 2D cartGrid
G3 = computeGeometry(cartGrid([5, 4], [5, 4]));

%% Timing benchmark: cartGrid + computeGeometry on a larger grid.
% Warm up once (MATLAB JIT / first-call overhead), then take the min of
% several timed repeats.
celldim = [40, 40, 20];
physdim = [40, 40, 20];

Gwarm = computeGeometry(cartGrid(celldim, physdim)); %#ok<NASGU>

nrep = 3;
t_topo = inf; t_geom = inf; t_total = inf;
for k = 1:nrep
    tic; Gt = cartGrid(celldim, physdim); tt = toc;
    tic; Gg = computeGeometry(Gt); tg = toc; %#ok<NASGU>
    t_topo  = min(t_topo, tt);
    t_geom  = min(t_geom, tg);
    t_total = min(t_total, tt + tg);
end
bench_num_cells = prod(celldim);

save(output_file, 'G1', 'G2', 'G3', ...
    't_topo', 't_geom', 't_total', 'bench_num_cells', 'celldim', 'physdim', ...
    '-v7');

fprintf('MRST grid/geometry reference written to %s\n', output_file);
fprintf('MATLAB timing (%d cells): topo=%.4fs geom=%.4fs total=%.4fs\n', ...
    bench_num_cells, t_topo, t_geom, t_total);
end
