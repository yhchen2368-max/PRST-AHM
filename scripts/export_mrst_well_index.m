function export_mrst_well_index(output_file)
%EXPORT_MRST_WELL_INDEX Reference data for computeWellIndex.m parity.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));

rng(0);
n = 40;
G = cartGrid([n, 1, 1]);
G = computeGeometry(G);
G.cartDims = [4 5 2]; % pretend 3D so griddim-3 Kh path is exercised via override below
G.griddim = 3;

dx = 5 + 10*rand(n,1);
dy = 5 + 10*rand(n,1);
dz = 1 + 5*rand(n,1);
kx = 1e-13 * (0.1 + 5*rand(n,1));
ky = 1e-13 * (0.1 + 5*rand(n,1));
kz = 1e-14 * (0.1 + 5*rand(n,1));
radius = 0.1 * ones(n,1);
skin = [zeros(n/2,1); 0.5*rand(n/2,1)];

rock = struct('perm', [kx, ky, kz]);
cellDims = [dx, dy, dz];
cells = (1:n)';

WI_z = computeWellIndex(G, rock, radius, cells, 'Dir', 'z', 'cellDims', cellDims, 'Skin', skin);
WI_x = computeWellIndex(G, rock, radius, cells, 'Dir', 'x', 'cellDims', cellDims, 'Skin', skin);

save(output_file, 'dx', 'dy', 'dz', 'kx', 'ky', 'kz', 'radius', 'skin', 'WI_z', 'WI_x', '-v7');
fprintf('MRST computeWellIndex reference written to %s\n', output_file);
end
