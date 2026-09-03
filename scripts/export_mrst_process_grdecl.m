function export_mrst_process_grdecl(output_file, deck_path, varname)
%EXPORT_MRST_PROCESS_GRDECL Reference data for processGRDECL/computeGeometry parity.
%
% Reads a real ECLIPSE deck's corner-point GRID section and runs MRST's own
% processGRDECL + computeGeometry, for direct comparison against the Python
% port (PRSTCore.gridprocessing.process_grdecl + compute_geometry).

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add deckformat

deck = readEclipseDeck(deck_path);
deck = convertDeckUnits(deck);
grdecl = deck.GRID;

% SplitDisconnected=false: PRSTCore's process_grdecl port does not (yet)
% implement splitDisconnectedGrid, a separate post-processing step; disable
% it here so this comparison targets processGRDECL's own topology/fault
% handling instead of that unrelated feature.
tic; G = processGRDECL(grdecl, 'SplitDisconnected', false); t_topo = toc;
tic; G = computeGeometry(G); t_geom = toc;

cell_volumes = G.cells.volumes;
cell_centroids = G.cells.centroids;
face_areas = G.faces.areas;
face_normals = G.faces.normals;
face_centroids = G.faces.centroids;
neighbors = G.faces.neighbors;
node_coords = G.nodes.coords;
num_cells = G.cells.num;
num_faces = G.faces.num;
num_nodes = G.nodes.num;

save(output_file, ...
    'cell_volumes', 'cell_centroids', 'face_areas', 'face_normals', ...
    'face_centroids', 'neighbors', 'node_coords', ...
    'num_cells', 'num_faces', 'num_nodes', 't_topo', 't_geom', ...
    '-v7');

fprintf('%s: cells=%d faces=%d nodes=%d topo=%.3fs geom=%.3fs\n', ...
    varname, num_cells, num_faces, num_nodes, t_topo, t_geom);
end
