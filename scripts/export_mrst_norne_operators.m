% Export default-MRST Norne TPFA connections for Python parity checks.
clear; clc;
script_dir = fileparts(mfilename('fullpath'));
root = fileparts(script_dir);
addpath(fullfile(root, 'mrst-2026a')); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props
[~, model] = initEclipseProblemAD(fullfile(root, 'examples', 'Norne', ...
    'Norne_simplified', 'NORNE_ATW2013.DATA'));
N = model.operators.N;
T = model.operators.T;
perm = model.rock.perm;
ntg = model.rock.ntg;
save(fullfile(root, 'mrst_norne_operators.mat'), 'N', 'T', 'perm', 'ntg');
fprintf('connections=%d\n', size(N, 1));
