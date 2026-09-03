%% Export native MRST initial states for every standalone example deck.
% These values are the reference for Python's initStateDeck / EQUIL port.
% The script intentionally invokes initEclipseProblemAD for each deck, so
% INCLUDE expansion, unit conversion and equilibrium setup stay in MRST.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
output_file = fullfile(cgnet_root, 'mrst_example_initial_states.mat');
names = {'SPE1', 'SPE9', 'EGG', 'NORNE'};
paths = { ...
    fullfile(cgnet_root, 'examples', 'SPE1', 'BENCH_SPE1.DATA'), ...
    fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA'), ...
    fullfile(cgnet_root, 'examples', 'EGG', 'Egg_Model_ECL.DATA'), ...
    fullfile(cgnet_root, 'examples', 'Norne', 'Norne_simplified', 'NORNE_ATW2013.DATA') ...
};

addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props
examples = struct('name', {}, 'pressure', {}, 's', {}, 'rs', {}, 'rv', {}, ...
    'indexMap', {}, 'cartDims', {}, 'centroids', {}, 'schedule_steps', {}, ...
    'schedule_controls', {}, 'wells_per_first_control', {});
for i = 1:numel(paths)
    [state, model, schedule] = initEclipseProblemAD(paths{i});
    ex = struct();
    ex.name = names{i};
    ex.pressure = state.pressure;
    ex.s = state.s;
    if isfield(state, 'rs')
        ex.rs = state.rs;
    else
        ex.rs = [];
    end
    if isfield(state, 'rv')
        ex.rv = state.rv;
    else
        ex.rv = [];
    end
    ex.indexMap = model.G.cells.indexMap;
    ex.cartDims = model.G.cartDims;
    ex.centroids = model.G.cells.centroids;
    ex.schedule_steps = schedule.step.val;
    ex.schedule_controls = schedule.step.control;
    if isempty(schedule.control)
        ex.wells_per_first_control = 0;
    else
        ex.wells_per_first_control = numel(schedule.control(1).W);
    end
    examples(end+1) = ex; %#ok<SAGROW>
    fprintf('%s: %d cells, %d report steps\n', ex.name, numel(ex.pressure), numel(ex.schedule_steps));
end
save(output_file, 'examples', '-v7');
fprintf('Wrote MRST example initial states to %s\n', output_file);
