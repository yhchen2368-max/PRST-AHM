clear; clc;
script_dir = fileparts(mfilename('fullpath'));
root = fileparts(script_dir);
addpath(fullfile(root, 'mrst-2026a')); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props
[~, model] = initEclipseProblemAD(fullfile(root, 'examples', 'Norne', ...
    'Norne_simplified', 'NORNE_ATW2013.DATA'));
ix = [43579; 43603; 29568; 31831];
disp([ix, model.rock.perm(ix, :), model.rock.ntg(ix)]);
