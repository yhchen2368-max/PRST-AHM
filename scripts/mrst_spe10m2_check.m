%% MRST 2026a comparison: SPE10 Model 2 first report step
%  Same deck PRSTCore just failed on, with MRST's own deck path, to see
%  whether the difficulty is PRSTCore's solver or the model itself.
clear; clc;

mrstRoot = 'C:\Users\junji\Desktop\github\PRSTCore\mrst-2026a';
if isempty(strfind(path, mrstRoot)) %#ok<STREMP>
    addpath(mrstRoot);
    startup;
end
try
    mrstModule add deckformat ad-blackoil ad-core ad-props;
    fprintf('MRST modules loaded\n');
catch e
    fprintf('mrstModule add failed: %s\n', e.message);
end

deckFile = 'C:\Users\junji\Desktop\github\PRSTCore\examples\spe10model2\SPE10_MODEL2.DATA';
t0 = tic;
deck = readEclipseDeck(deckFile);
deck = convertDeckUnits(deck);
fprintf('deck read + convert: %.1f s\n', toc(t0));

t0 = tic;
[state0, model, schedule, nonlinear] = initEclipseProblemAD(deck);
fprintf('model init: %.1f s  cells=%d  oil=%d water=%d gas=%d\n', ...
        toc(t0), model.G.cells.num, model.oil, model.water, model.gas);
model.verbose = false;

linsolve = nonlinear.LinearSolver;
fprintf('MRST linear solver: %s\n', class(linsolve));

fprintf('report steps: %d   first dt = %.3g days   wells(ctrl1): %d\n', ...
        numel(schedule.step.val), schedule.step.val(1)/86400, ...
        numel(schedule.control(1).W));

% Truncate to the first report step only.
schedule1 = schedule;
schedule1.step.val     = schedule.step.val(1);
schedule1.step.control = schedule.step.control(1);

t0 = tic;
try
    [~, states, reports] = simulateScheduleAD(state0, model, schedule1, ...
        'LinearSolver', linsolve, 'NonLinearSolver', nonlinear, ...
        'outputHandler', @(varargin) []);
    fprintf('first step finished in %.1f s\n', toc(t0));
    for k = 1:numel(reports)
        rep = reports{k};
        fprintf('step %d: converged=%d  iterations=%d  cutting=%d\n', ...
                k, rep.Converged, rep.Iterations, rep.MinistepCuttingCount);
        if isfield(rep, 'StepReports')
            for j = 1:numel(rep.StepReports)
                sr = rep.StepReports{j};
                if isfield(sr, 'NonlinearReport')
                    fprintf('  ministep %d: dt=%.3g  converged=%d  newton=%d\n', ...
                            j, sr.Timestep, sr.Converged, numel(sr.NonlinearReport));
                end
            end
        end
    end
catch e
    fprintf('simulateScheduleAD failed after %.1f s:\n%s\n', toc(t0), e.message);
    for k = 1:numel(e.stack)
        fprintf('  %s:%d\n', e.stack(k).name, e.stack(k).line);
    end
end
