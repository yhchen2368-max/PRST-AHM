%% Native MRST Norne initialization reference.
% This is intentionally initialization-only: it exposes the exact
% multi-region EQUIL/RSVD/PVTG state before the Python AD assembly is ported.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, solver] = initEclipseProblemAD( ...
    fullfile(cgnet_root, 'examples', 'Norne', 'Norne_simplified', 'NORNE_ATW2013.DATA'));
controlId = schedule.step.control(1);
W = schedule.control(controlId).W;
gravity = model.gravity;
G = model.G;
model = model.validateModel();
[phasePressure, capillaryPressure, relativePermeability, shrinkageFactors] = ...
    model.getProps(state0, 'PhasePressures', 'CapillaryPressure', ...
    'RelativePermeability', 'ShrinkageFactors');
deck = convertDeckUnits(readEclipseDeck(fullfile(cgnet_root, 'examples', ...
    'Norne', 'Norne_simplified', 'NORNE_ATW2013.DATA')));
initRegions = getInitializationRegionsDeck(model, deck);
rsRegion1 = initRegions{1}.rs(zeros(model.G.cells.num, 1), model.G.cells.centroids(:, 3));
save(fullfile(cgnet_root, 'mrst_norne_initial.mat'), ...
    'state0', 'W', 'gravity', 'G', 'schedule', 'controlId', ...
    'phasePressure', 'capillaryPressure', 'relativePermeability', ...
    'shrinkageFactors', 'rsRegion1');
fprintf('cells=%d wells=%d steps=%d\n', model.G.cells.num, numel(W), numel(schedule.step.val));
