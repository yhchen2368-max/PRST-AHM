%% Native MRST SPE9 first report-step reference.
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cgnet_root = fileparts(script_dir);
mrst_root = fullfile(fileparts(cgnet_root), 'mrst-2026a');
addpath(mrst_root); startup;
mrstModule add deckformat ad-core ad-blackoil ad-props

[state0, model, schedule, solver] = initEclipseProblemAD( ...
    fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA'));
deck = convertDeckUnits(readEclipseDeck( ...
    fullfile(cgnet_root, 'examples', 'SPE9', 'SPE9_CP.DATA')));
deck_pvtw = deck.PROPS.PVTW;
deck_pvto = deck.PROPS.PVTO;
direct_bw = model.fluid.bW(state0.pressure);
direct_bo = model.fluid.bO(state0.pressure, state0.rs, false(numel(state0.rs), 1));
controlId = schedule.step.control(1);
control = schedule.control(controlId);
dt = schedule.step.val(1);
% These preparation calls are the first-iteration path in
% simulateScheduleAD.m.  Calling solveTimestep directly without them
% leaves GenericFacilityModel without its initialized well state.
[forces, fstruct] = model.getDrivingForces(control);
model = model.validateModel(fstruct, false);
model.checkStateFunctionDependencies();
schedule = model.validateSchedule(schedule);
state0 = model.validateState(state0);
[model, state0] = model.updateForChangedControls(state0, fstruct);
[initialProblem, initialState] = model.getEquations(state0, state0, dt, fstruct, ...
    'iteration', 1, 'resOnly', false);
initialProblem = initialProblem.assembleSystem();
initial_residual = initialProblem.b;
initial_jacobian = initialProblem.A;
initial_component_mass = model.getProps(initialState, 'ComponentTotalMass');
[initial_b, initial_density, initial_mobility] = model.getProps(initialState, ...
    'ShrinkageFactors', 'Density', 'Mobility');
initial_flow_state = model.FlowDiscretization.buildFlowState(model, initialState, state0, dt);
[initial_component_flux, initial_phase_flux, initial_phase_upwind, ...
    initial_phase_potential] = model.getProps(initial_flow_state, ...
    'ComponentTotalFlux', 'PhaseFlux', 'PhaseUpwindFlag', 'PhasePotentialDifference');
initial_component_mass = cellfun(@value, initial_component_mass, 'UniformOutput', false);
initial_b = cellfun(@value, initial_b, 'UniformOutput', false);
initial_density = cellfun(@value, initial_density, 'UniformOutput', false);
initial_mobility = cellfun(@value, initial_mobility, 'UniformOutput', false);
initial_component_flux = cellfun(@value, initial_component_flux, 'UniformOutput', false);
initial_phase_flux = cellfun(@value, initial_phase_flux, 'UniformOutput', false);
initial_phase_potential = cellfun(@value, initial_phase_potential, 'UniformOutput', false);
[facility_surface_rates, facility_surface_density] = model.FacilityModel.getSurfaceRates(initialState);
facility_component_flux = model.FacilityModel.FacilityFlowDiscretization.get( ...
    model.FacilityModel, initialState, 'ComponentTotalFlux');
facility_rho0 = model.getProps(state0, 'Density');
facility_surface_rates = cellfun(@value, facility_surface_rates, 'UniformOutput', false);
facility_surface_density = cellfun(@value, facility_surface_density, 'UniformOutput', false);
facility_component_flux = cellfun(@value, facility_component_flux, 'UniformOutput', false);
facility_rho0 = cellfun(@value, facility_rho0, 'UniformOutput', false);
post_state0_pressure = value(model.getProp(state0, 'pressure'));
post_state0_rs = value(model.getProp(state0, 'rs'));
post_direct_bo = value(model.fluid.bO(post_state0_pressure, post_state0_rs, ...
    false(numelValue(post_state0_rs), 1)));
operators_T = model.operators.T;
operators_N = model.operators.N;
operators_pv = model.operators.pv;
rock_perm = model.rock.perm;
G_cell_centroids = model.G.cells.centroids;
G_cell_volumes = model.G.cells.volumes;
G_faces_neighbors = model.G.faces.neighbors;
G_faces_normals = model.G.faces.normals;
G_faces_centroids = model.G.faces.centroids;
G_cells_faces = model.G.cells.faces;
G_cells_facePos = model.G.cells.facePos;
[state, report, ministates] = solver.solveTimestep( ...
    state0, dt, model, forces{:}, 'initialGuess', state0, ...
    'controlId', controlId);
iterations = report.Iterations;
converged = report.Converged;
save(fullfile(cgnet_root, 'mrst_spe9_first_step.mat'), ...
    'state0', 'state', 'report', 'ministates', 'iterations', 'converged', 'dt', ...
    'initialState', 'initial_residual', 'initial_jacobian', ...
    'initial_component_mass', 'initial_component_flux', ...
    'initial_b', 'initial_density', 'initial_mobility', ...
    'initial_phase_flux', 'initial_phase_upwind', 'initial_phase_potential', ...
    'facility_surface_rates', 'facility_surface_density', ...
    'facility_component_flux', 'facility_rho0', ...
    'post_state0_pressure', 'post_state0_rs', 'post_direct_bo', ...
    'operators_T', 'operators_N', 'operators_pv', 'rock_perm', ...
    'deck_pvtw', 'deck_pvto', ...
    'direct_bw', ...
    'direct_bo', ...
    'G_cell_centroids', 'G_cell_volumes', ...
    'G_faces_neighbors', 'G_faces_normals', 'G_faces_centroids', ...
    'G_cells_faces', 'G_cells_facePos');
fprintf('converged=%d iterations=%d ministeps=%d\n', ...
    converged, iterations, numel(ministates));
