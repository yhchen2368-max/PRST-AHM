%% Native MRST fluid samples for the SPE9 EQUIL hydrostatic ODE.
% This exports exactly the functions used by
% getInitializationRegionsBlackOil/getOilDensity, so the Python port can
% compare its PVT evaluator without reconstructing MRST internals.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

deck_path = fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA');
[state, model] = initEclipseProblemAD(deck_path);
deck = convertDeckUnits(readEclipseDeck(deck_path));
pvto = deck.PROPS.PVTO{1};
p = linspace(min(state.pressure), max(state.pressure), 101)';
rsSat = model.fluid.rsSat(p);
rs = min(rsSat, 247.56957328385892*ones(size(p)));
saturated = rs >= rsSat;
bo = model.fluid.bO(p, rs, saturated);
bw = model.fluid.bW(p);
rhoO = bo.*(model.fluid.rhoOS(1) + rs.*model.fluid.rhoGS(1));
rhoW = bw.*model.fluid.rhoWS(1);
rhoOS = model.fluid.rhoOS(1);
rhoWS = model.fluid.rhoWS(1);
rhoGS = model.fluid.rhoGS(1);
save(fullfile(cgnet_root, 'mrst_spe9_pvt_sample.mat'), ...
     'p', 'rsSat', 'rs', 'saturated', 'bo', 'bw', 'rhoO', 'rhoW', ...
     'rhoOS', 'rhoWS', 'rhoGS', 'pvto');
