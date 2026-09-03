%% Inspect the exact MRST PVTG evaluation at the Norne initialization state.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model] = initEclipseProblemAD(fullfile(cgnet_root, 'examples', ...
    'Norne', 'Norne_simplified', 'NORNE_ATW2013.DATA'));
model = model.validateModel();
[pp, b] = model.getProps(state0, 'PhasePressures', 'ShrinkageFactors');
ix = 38999; % MATLAB index corresponding to Python's maximum-difference cell.
p_all = pp{3};
rv_all = state0.rv;
% The regional PVT dispatcher requires a full active-cell vector.
try
    % TABDIMS declares two PVT tables, but PVTNUM selects table one for
    % every active Norne cell.  The raw fluid keeps those region functions
    % in a cell array; StateFunction dispatch normally performs this index.
    bg_fun = model.fluid.bG{1};
    bg_true_all = bg_fun(p_all, rv_all, true(numel(p_all), 1));
    bg_false_all = bg_fun(p_all, rv_all, false(numel(p_all), 1));
catch err
    disp(model.fluid.bG);
    disp(getReport(err, 'extended'));
    rethrow(err)
end
p = p_all(ix);
rv = rv_all(ix);
bg_true = bg_true_all(ix);
bg_false = bg_false_all(ix);
bg_state = b{3}(ix);
phaseNames = model.getPhaseNames();
save(fullfile(cgnet_root, 'mrst_norne_pvt_sample.mat'), ...
    'ix', 'p', 'rv', 'bg_true', 'bg_false', 'bg_state', 'phaseNames');
fprintf('phase=%s p=%.15g rv=%.15g bG(true)=%.15g bG(false)=%.15g state=%.15g\n', ...
    phaseNames, p, rv, bg_true, bg_false, bg_state);
