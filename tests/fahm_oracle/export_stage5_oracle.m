function export_stage5_oracle(outputDir)
%EXPORT_STAGE5_ORACLE Export FAHM.m:1753-1824 state as stable raw arrays.
%
% This exporter deliberately executes the same statements as the App, on
% the checked-in SPE9 EGRID/INIT/UNRST result set.  Every numeric array is
% written as little-endian column-major bytes, with its MATLAB shape in a
% JSON manifest.  Re-running it therefore provides an array/byte-level
% oracle without relying on MAT-file headers or timestamps.

    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(here));
    workspaceRoot = fileparts(projectRoot);
    mrstRoot = fullfile(workspaceRoot, 'MRST');
    prefix = fullfile(projectRoot, 'examples', 'SPE9', 'RESULTS', 'SPE9_CP');
    deckFile = fullfile(projectRoot, 'examples', 'SPE9', 'SPE9_CP.DATA');

    old = pwd;
    cleanup = onCleanup(@() cd(old));
    cd(mrstRoot);
    startup;
    mrstModule add deckformat ad-core ad-blackoil ad-props;
    addpath(fullfile(mrstRoot, 'dev', 'utils'));
    gravity on;

    deck = readEclipseDeck(deckFile);
    deck = convertDeckUnits(deck);
    deck = processEclipseDeck(deck);
    if isfield(deck.GRID, 'MULTPV')
        deck.GRID = rmfield(deck.GRID, 'MULTPV');
    end
    if isfield(deck.EDIT, 'MULTPV')
        deck.EDIT = rmfield(deck.EDIT, 'MULTPV');
    end

    init = readEclipseOutputFileUnFmt([prefix, '.INIT']);
    grid = readEclipseOutputFileUnFmt([prefix, '.EGRID']);
    [G, rock, N, T] = initGridFromEclipseOutput( ...
        init, grid, 'outputSimGrid', false);
    G.trans.neighbors = N;
    G.trans.T = T;

    rsspec = processEclipseRestartSpec(prefix, 'all');
    args = {'restartInfo', rsspec, ...
            'splitWellsOnSignChange', false, ...
            'removeClosedWells', false, ...
            'removeCrossflow', false, ...
            'includeWellSols', false, ...
            'includeAquifers', true, ...
            'includeComponents', isfield(deck.RUNSPEC, 'COMPS')};
    states = convertRestartToStates(prefix, G, args{:});
    state0 = states{1};

    fluid = initDeckADIFluid(deck, 'G', G, 'useMex', true);
    model = selectModelFromDeck(G, rock, fluid, deck, 'useNatural', false);
    model.operators = setupOperatorsTPFA( ...
        G, rock, 'neighbors', G.trans.neighbors);

    relperm = getRelpermScalingPoints(model);
    model = imposeRelpermScaling(model, relperm{:});
    cappress = getCapPressScalingPoints(model);
    model = imposeCapPressScaling(model, cappress{:});

    if ~isempty(model.AquiferModel)
        bad = isnan(model.AquiferModel.initvals.pressures);
        if any(bad)
            if isfield(state0, 'aquiferSol')
                id = vertcat(state0.aquiferSol.num);
                p = vertcat(state0.aquiferSol.pressure);
                p(p < 0) = 0;
                [~, ix] = sort(id);
                model.AquiferModel.initvals.pressures = p(ix);
            else
                aqmodel = model.AquiferModel;
                for i = 1:numel(aqmodel.initvals.pressures)
                    ix = aqmodel.aquifers(:, 1) == i;
                    cells = aqmodel.aquifers(ix, 2);
                    model.AquiferModel.initvals.pressures(i) = ...
                        mean(state0.pressure(cells));
                end
            end
        end
    end

    if isfield(deck.PROPS, 'SWATINIT')
        pc = model.getProp(state0, 'capillarypressure');
        pcow = -pc{1};
        mult = state0.pcow./pcow;
        mult(~isfinite(mult)) = 1;
        model.rock.pcowScale = mult;
    end
    model = model.validateModel();
    [phasePressure, shrinkage, viscosity] = model.getProps( ...
        state0, 'PhasePressures', 'ShrinkageFactors', 'Viscosity');

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    records = struct('name', {}, 'path', {}, 'dtype', {}, 'shape', {}, ...
                     'order', {}, 'nbytes', {});
    records(end+1) = put(outputDir, 'grid/cart_dims', int64(G.cartDims));
    records(end+1) = put(outputDir, 'grid/cells_num', int64(G.cells.num));
    records(end+1) = put(outputDir, 'grid/index_map_1based', int64(G.cells.indexMap));
    records(end+1) = put(outputDir, 'grid/index_map_0based', int64(G.cells.indexMap - 1));
    records(end+1) = put(outputDir, 'grid/centroids', G.cells.centroids);
    records(end+1) = put(outputDir, 'grid/volumes', G.cells.volumes);
    records(end+1) = put(outputDir, 'grid/PORV', G.cells.PORV);
    records(end+1) = put(outputDir, 'grid/trans_neighbors_1based', int64(N));
    records(end+1) = put(outputDir, 'grid/trans_neighbors_0based', int64(N - 1));
    records(end+1) = put(outputDir, 'grid/trans_T', T);
    records(end+1) = put(outputDir, 'rock/poro', rock.poro);
    records(end+1) = put(outputDir, 'rock/perm', rock.perm);
    if isfield(rock, 'ntg')
        records(end+1) = put(outputDir, 'rock/ntg', rock.ntg);
    end
    if isfield(rock, 'regions')
        f = fieldnames(rock.regions);
        for i = 1:numel(f)
            records(end+1) = put(outputDir, ['rock/regions/', f{i}], ...
                                 int64(rock.regions.(f{i})));
        end
    end
    records(end+1) = put(outputDir, 'state0/pressure', state0.pressure);
    records(end+1) = put(outputDir, 'state0/s', state0.s);
    records(end+1) = put(outputDir, 'state0/rs', state0.rs);
    records(end+1) = put(outputDir, 'state0/rv', state0.rv);
    records(end+1) = put(outputDir, 'state0/time', state0.time);
    records(end+1) = put(outputDir, 'operators/N_1based', int64(model.operators.N));
    records(end+1) = put(outputDir, 'operators/N_0based', int64(model.operators.N - 1));
    records(end+1) = put(outputDir, 'operators/T', model.operators.T);
    records(end+1) = put(outputDir, 'operators/T_all', model.operators.T_all);
    records(end+1) = put(outputDir, 'operators/pv', model.operators.pv);
    records(end+1) = put(outputDir, 'operators/C_shape', int64(size(model.operators.C)));
    records(end+1) = put(outputDir, 'operators/C_nnz', int64(nnz(model.operators.C)));
    records = put_scaling(outputDir, records, relperm, 'relperm/input');
    records = put_scaling(outputDir, records, cappress, 'capillary/input');
    if isfield(model.rock, 'krscale')
        records = put_tree(outputDir, records, model.rock.krscale, 'relperm/model');
    end
    if isfield(model.rock, 'pcscale')
        records = put_tree(outputDir, records, model.rock.pcscale, 'capillary/model');
    end
    if isfield(model.rock, 'pcowScale')
        records(end+1) = put(outputDir, 'capillary/pcowScale', model.rock.pcowScale);
    end
    records(end+1) = put(outputDir, 'model/phases_WOG', ...
        uint8([model.water, model.oil, model.gas]));
    records(end+1) = put(outputDir, 'model/disgas_vapoil', ...
        uint8([model.disgas, model.vapoil]));
    phaseNames = model.getPhaseNames();
    for i = 1:numel(phaseNames)
        phase = phaseNames(i);
        records(end+1) = put(outputDir, ...
            ['fluid/phase_pressure_', phase], phasePressure{i});
        records(end+1) = put(outputDir, ...
            ['fluid/shrinkage_', phase], shrinkage{i});
        records(end+1) = put(outputDir, ...
            ['fluid/viscosity_', phase], viscosity{i});
    end
    if model.disgas
        records(end+1) = put(outputDir, 'fluid/rs_max', ...
            model.getProp(state0, 'RsMax'));
    end

    manifest = struct();
    manifest.schema_version = 'fahm-stage5-oracle-v1';
    manifest.oracle_id = 'FAHM-SPE9-RESULT-STATE0';
    manifest.reference = 'MRST/dev/APP/FAHM.mlapp';
    manifest.extracted_source_sha256 = ...
        'a1c29aedd7620438b94bbbcb529c7d9fb6fd62fe909def6d419a10385fb493c7';
    manifest.matlab_release = version('-release');
    manifest.case_prefix = 'examples/SPE9/RESULTS/SPE9_CP';
    manifest.source_lines = 'FAHM.m:1753-1824';
    manifest.arrays = records;
    txt = jsonencode(manifest, 'PrettyPrint', true);
    fid = fopen(fullfile(outputDir, 'manifest.json'), 'w');
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, unicode2native(txt, 'UTF-8'), 'uint8');
end

function records = put_scaling(outputDir, records, scaling, prefix)
    for i = 1:2:numel(scaling)
        records(end+1) = put(outputDir, [prefix, '/', scaling{i}], scaling{i+1});
    end
end

function records = put_tree(outputDir, records, value, prefix)
    names = fieldnames(value);
    for i = 1:numel(names)
        child = value.(names{i});
        name = [prefix, '/', names{i}];
        if isstruct(child)
            records = put_tree(outputDir, records, child, name);
        elseif isnumeric(child) || islogical(child)
            records(end+1) = put(outputDir, name, child);
        end
    end
end

function record = put(outputDir, name, value)
    safe = strrep(name, '/', '__');
    path = [safe, '.bin'];
    full = fullfile(outputDir, path);
    if isa(value, 'double')
        precision = 'double'; dtype = '<f8';
    elseif isa(value, 'single')
        precision = 'single'; dtype = '<f4';
    elseif isa(value, 'int64')
        precision = 'int64'; dtype = '<i8';
    elseif isa(value, 'uint8') || islogical(value)
        value = uint8(value); precision = 'uint8'; dtype = '|u1';
    else
        value = double(value); precision = 'double'; dtype = '<f8';
    end
    fid = fopen(full, 'w', 'ieee-le');
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, value(:), precision);
    info = dir(full);
    record = struct('name', name, 'path', path, 'dtype', dtype, ...
                    'shape', int64(size(value)), 'order', 'F', ...
                    'nbytes', int64(info.bytes));
end
