run('C:/Users/junji/Desktop/github/mrst-2026a/startup.m');
mrstModule add ad-core ad-blackoil ad-props deckformat; gravity on;

deck  = convertDeckUnits(readEclipseDeck('C:/Users/junji/Desktop/github/PRSTCore/examples/SpE1/SPE1CASE2.DATA'));
G     = computeGeometry(initEclipseGrid(deck));
rock  = compressRock(initEclipseRock(deck), G.cells.indexMap);
fluid = initDeckADIFluid(deck);
model = selectModelFromDeck(G, rock, fluid, deck);

schedule = convertDeckScheduleToMRST(model, deck);
dt       = schedule.step.val(1);
[~, forces] = model.getDrivingForces(schedule.control(1));

% Validate *with* the driving forces so the facility model picks the
% wells up -- without this the problem has no well equations at all.
model  = model.validateModel(forces);
state0 = initStateDeck(model, deck);
state0 = model.validateState(state0);

state = state0;
state.s(:,1)   = min(max(state0.s(:,1) + 0.01, 0), 1);
state = model.validateState(state);

fprintf('WSOLN %d\n', numel(state.wellSol));
for i = 1:numel(state.wellSol)
    ws = state.wellSol(i);
    fprintf('WSOL %s bhp %.8e qWs %.8e qOs %.8e qGs %.8e\n', ...
        ws.name, ws.bhp, ws.qWs, ws.qOs, ws.qGs);
end

problem = model.getEquations(state0, state, dt, forces, 'resOnly', false);
Rraw = [];
for i = 1:numel(problem.equations)
    n = numel(value(problem.equations{i}));
    fprintf('EQNAME %d %s %d\n', i, problem.equationNames{i}, n);
    Rraw = [Rraw; value(problem.equations{i})];
end
writematrix(Rraw, 'MRST_RAW3.csv');
fprintf('MRST nraw %d\n', numel(Rraw));
