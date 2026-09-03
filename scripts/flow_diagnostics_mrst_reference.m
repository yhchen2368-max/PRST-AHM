function flow_diagnostics_mrst_reference(output_file)
%FLOW_DIAGNOSTICS_MRST_REFERENCE Generate a small MRST diagnostics reference.
%
% This is a deliberately tiny wells-only case used to verify that the Python
% PRSTCore diagnostics implementation matches MRST's diagnostics algebra.
%
% The case is a 1D three-cell grid with one injector in cell 1, one producer
% in cell 3, unit internal flux from left to right, unit pore volume, and
% unit well rates.  The script saves both the MRST inputs and outputs so the
% Python verifier can run exactly the same diagnostic equations.

if nargin < 1 || isempty(output_file)
    repo = fileparts(fileparts(mfilename('fullpath')));
    output_file = fullfile(repo, 'tests', 'flow_diagnostics_mrst_1d_ref.mat');
end

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add diagnostics
gravity off

G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
rock = makeRock(G, ones(G.cells.num, 1), ones(G.cells.num, 1));

W = [];
W = addWell(W, G, rock, 1, 'Type', 'rate', 'Val',  1, 'Name', 'I1', 'sign',  1);
W = addWell(W, G, rock, 3, 'Type', 'rate', 'Val', -1, 'Name', 'P1', 'sign', -1);

state = struct();
state.pressure = [3; 2; 1];
state.flux = zeros(G.faces.num, 1);
internal = all(G.faces.neighbors > 0, 2);
state.flux(internal) = 1;
state.wellSol = initWellSol(W, 100);
state.wellSol(1).flux =  1;
state.wellSol(2).flux = -1;
state.wellSol(1).pressure = 3;
state.wellSol(2).pressure = 1;
state.wellSol(1).bhp = 3;
state.wellSol(2).bhp = 1;

D = computeTOFandTracer(state, G, rock, 'wells', W, ...
    'computeWellTOFs', true, 'firstArrival', false);
WP = computeWellPairs(state, G, rock, W, D);
[F, Phi] = computeFandPhi(poreVolume(G, rock), D.tof);
Lorenz = computeLorenz(F, Phi);
[Ev, tD] = computeSweep(F, Phi);

salloc = cellfun(@sum, {WP.inj.alloc}, 'UniformOutput', false);
if isempty(salloc)
    wellCommunication = [];
else
    wellCommunication = vertcat(salloc{:});
end

well_cells = {W.cells};
well_names = {W.name};
well_sign = vertcat(W.sign);
well_val = vertcat(W.val);
well_status = vertcat(W.status);
well_refDepth = vertcat(W.refDepth);
well_dZ = {W.dZ};
wellsol_flux = {state.wellSol.flux};
wellsol_pressure = vertcat(state.wellSol.pressure);
wellsol_bhp = vertcat(state.wellSol.bhp);

save(output_file, ...
    'G', 'rock', 'state', 'W', ...
    'well_cells', 'well_names', 'well_sign', 'well_val', 'well_status', ...
    'well_refDepth', 'well_dZ', 'wellsol_flux', 'wellsol_pressure', 'wellsol_bhp', ...
    'D', 'WP', 'F', 'Phi', 'Lorenz', 'Ev', 'tD', 'wellCommunication', ...
    '-v7');

fprintf('MRST flow diagnostics reference written to %s\n', output_file);
end

