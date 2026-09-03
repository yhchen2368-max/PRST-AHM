function manifest = export_fahm_minimal_oracle(mrstRoot, outDir)
%EXPORT_FAHM_MINIMAL_ORACLE Export a deterministic FAHM/MRST golden case.
%
% The fixture uses the first three report steps of MRST's bundled SPE1
% three-phase black-oil deck.  It exports raw arrays in MATLAB column-major
% order and a JSON manifest.  No PRST implementation is called here.

    arguments
        mrstRoot (1,:) char
        outDir (1,:) char
    end

    mrstRoot = char(java.io.File(mrstRoot).getCanonicalPath());
    outDir = char(java.io.File(outDir).getCanonicalPath());
    assert(isfolder(mrstRoot), 'MRST root does not exist: %s', mrstRoot);
    if isfolder(outDir)
        entries = dir(outDir);
        entries = entries(~ismember({entries.name}, {'.', '..'}));
        assert(isempty(entries), ...
            'Output directory must not exist or must be empty: %s', outDir);
    end
    ensure_dir(outDir);
    ensure_dir(fullfile(outDir, 'arrays'));
    ensure_dir(fullfile(outDir, 'deck'));

    previous = pwd;
    restore = onCleanup(@() cd(previous)); %#ok<NASGU>
    cd(mrstRoot);
    startup;
    mrstModule add ad-core ad-props ad-blackoil deckformat optimization linearsolvers test-suite
    addpath(genpath(fullfile(mrstRoot, 'dev', 'utils')));
    rng(0, 'twister');
    mrstVerbose(false);

    deckPath = fullfile(mrstRoot, 'core', 'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');
    includePath = fullfile(fileparts(deckPath), 'SPE1.GRDECL');
    copyfile(deckPath, fullfile(outDir, 'deck', 'BENCH_SPE1.DATA'));
    copyfile(includePath, fullfile(outDir, 'deck', 'SPE1.GRDECL'));

    [state0, model, schedule] = initEclipseProblemAD(deckPath);
    schedule.step.val = schedule.step.val(1:3);
    schedule.step.control = schedule.step.control(1:3);
    [wellSols, states] = simulateScheduleAD(state0, model, schedule, 'Verbose', false);
    nstep = numel(states);
    for step = 1:nstep
        states{step}.wellSol = wellSols{step};
    end

    observed = make_observed(states, schedule);
    beta = getNormalizationFactors(observed);
    alpha = struct('ww', 1, 'wo', 1, 'wg', 1, 'wp', 1, ...
                   'wt', 0, 'wf', 0, 'ws', 0);
    W = schedule.control(end).W;
    nw = numel(W);
    one = ones(nw, 1);
    omega = struct('ww', one, 'wo', one, 'wg', one, 'wp', one, ...
                   'wt', one, 'wf', one, 'ws', one);
    objective = matchObservedOWGProfile(model, states, schedule, observed, ...
        'ObjectiveWeight', alpha, ...
        'NormalizationFactor', beta, ...
        'WellsWeight', omega);
    objective = cell2mat(objective);

    setup = struct('model', model, 'schedule', schedule, 'state0', state0);
    [configRows, parameters] = make_parameters(setup);
    pvec = getScaledParameterVector(setup, parameters);
    nparam = cellfun(@(parameter) parameter.nParam, parameters(:));
    sliceEnd = cumsum(nparam);
    slices1 = [sliceEnd - nparam + 1, sliceEnd];
    slices0 = [sliceEnd - nparam, sliceEnd];

    history = make_history_trace();

    arrays = empty_array_records();
    arrays = add_array(arrays, outDir, 'grid/cart_dims', int64(model.G.cartDims));
    arrays = add_array(arrays, outDir, 'grid/cells_num', int64(model.G.cells.num));
    arrays = add_array(arrays, outDir, 'grid/active_index_map_1based', int64(model.G.cells.indexMap));
    arrays = add_array(arrays, outDir, 'grid/active_index_map_0based', int64(model.G.cells.indexMap - 1));
    arrays = add_array(arrays, outDir, 'rock/porosity', model.rock.poro);
    arrays = add_array(arrays, outDir, 'rock/permeability', model.rock.perm);
    arrays = add_array(arrays, outDir, 'operators/pore_volume', model.operators.pv);
    arrays = add_array(arrays, outDir, 'state0/pressure', state0.pressure);
    arrays = add_array(arrays, outDir, 'state0/saturation_WOG', state0.s);
    arrays = add_optional_array(arrays, outDir, 'state0/rs', state0, 'rs');
    arrays = add_optional_array(arrays, outDir, 'state0/rv', state0, 'rv');
    arrays = add_array(arrays, outDir, 'schedule/step_val_seconds', schedule.step.val);
    arrays = add_array(arrays, outDir, 'schedule/step_control_1based', int64(schedule.step.control));
    arrays = add_array(arrays, outDir, 'schedule/step_control_0based', int64(schedule.step.control - 1));

    counts = arrayfun(@(well) numel(well.cells), W(:));
    offsets1 = [1; cumsum(counts(:)) + 1];
    wellCells1 = vertcat(W.cells);
    p2w1 = getPerforationToWellMapping(W);
    arrays = add_array(arrays, outDir, 'schedule/well_cells_1based', int64(wellCells1));
    arrays = add_array(arrays, outDir, 'schedule/well_cells_0based', int64(wellCells1 - 1));
    arrays = add_array(arrays, outDir, 'schedule/well_cell_offsets_1based', int64(offsets1));
    arrays = add_array(arrays, outDir, 'schedule/well_cell_offsets_0based', int64(offsets1 - 1));
    arrays = add_array(arrays, outDir, 'schedule/perforation_to_well_1based', int64(p2w1));
    arrays = add_array(arrays, outDir, 'schedule/perforation_to_well_0based', int64(p2w1 - 1));
    arrays = add_array(arrays, outDir, 'schedule/well_sign', vertcat(W.sign));
    arrays = add_array(arrays, outDir, 'schedule/well_status', logical(vertcat(W.status)));
    arrays = add_array(arrays, outDir, 'schedule/well_control_value', vertcat(W.val));

    for step = 1:nstep
        prefix = sprintf('forward/step_%02d', step);
        arrays = add_array(arrays, outDir, [prefix, '/pressure'], states{step}.pressure);
        arrays = add_array(arrays, outDir, [prefix, '/saturation_WOG'], states{step}.s);
        arrays = add_optional_array(arrays, outDir, [prefix, '/rs'], states{step}, 'rs');
        arrays = add_optional_array(arrays, outDir, [prefix, '/rv'], states{step}, 'rv');
        arrays = add_well_solution(arrays, outDir, [prefix, '/well'], states{step}.wellSol);

        prefix = sprintf('observed/step_%02d', step);
        arrays = add_array(arrays, outDir, [prefix, '/dt_seconds'], observed{step}.dt);
        arrays = add_well_solution(arrays, outDir, [prefix, '/well'], observed{step}.wellSol);
    end

    weightFields = {'ww', 'wo', 'wg', 'wp', 'wt', 'wf', 'ws'};
    arrays = add_array(arrays, outDir, 'objective/alpha', ...
        cellfun(@(field) alpha.(field), weightFields));
    arrays = add_array(arrays, outDir, 'objective/beta', ...
        cellfun(@(field) beta_value(beta, field), weightFields));
    arrays = add_array(arrays, outDir, 'objective/omega', ...
        cell2mat(cellfun(@(field) omega.(field), weightFields, 'UniformOutput', false)));
    arrays = add_array(arrays, outDir, 'objective/per_step_positive_misfit', objective);
    arrays = add_array(arrays, outDir, 'objective/total_positive_misfit', sum(objective));
    arrays = add_array(arrays, outDir, 'objective/evaluator_return_value', -sum(objective));

    for index = 1:numel(configRows)
        row = configRows(index);
        if row.include
            arrays = add_array(arrays, outDir, row.relative_limits_array, row.relative_limits);
            arrays = add_array(arrays, outDir, row.subset_1based_array, int64(row.subset));
            arrays = add_array(arrays, outDir, row.subset_0based_array, int64(row.subset - 1));
        end
        configRows(index).relative_limits = [];
        configRows(index).subset = [];
    end
    arrays = add_array(arrays, outDir, 'parameters/pvec_unit_box', pvec);
    arrays = add_array(arrays, outDir, 'parameters/nparam', int64(nparam));
    arrays = add_array(arrays, outDir, 'parameters/slices_1based_inclusive', int64(slices1));
    arrays = add_array(arrays, outDir, 'parameters/slices_0based_half_open', int64(slices0));

    arrays = add_array(arrays, outDir, 'history/val', history.val);
    arrays = add_array(arrays, outDir, 'history/u', cell2mat(history.u));
    arrays = add_array(arrays, outDir, 'history/pg', history.pg);
    arrays = add_array(arrays, outDir, 'history/alpha', history.alpha);
    arrays = add_array(arrays, outDir, 'history/lsit', history.lsit);
    arrays = add_array(arrays, outDir, 'history/lsfl', history.lsfl);
    arrays = add_array(arrays, outDir, 'history/rho', history.rho);
    arrays = add_array(arrays, outDir, 'history/r', history.r);
    arrays = add_array(arrays, outDir, 'history/hess_present', ...
        logical(cellfun(@(value) ~isempty(value), history.hess)));
    arrays = add_array(arrays, outDir, 'history/params_present', ...
        logical(cellfun(@(value) ~isempty(value), history.params)));

    copiedDeck = fullfile(outDir, 'deck', 'BENCH_SPE1.DATA');
    copiedInclude = fullfile(outDir, 'deck', 'SPE1.GRDECL');
    deckFiles = [file_record(outDir, copiedDeck), file_record(outDir, copiedInclude)];
    manifest = struct();
    manifest.schema_version = 'fahm-oracle-v1';
    manifest.oracle_id = 'FAHM-SPE1-WOG-FIRST-3-STEPS';
    manifest.reference = 'MRST/dev/APP/FAHM.mlapp';
    manifest.fixture_scope = ['Three active black-oil phases, first three SPE1 report steps, ', ...
        'FAHM objective, default four per-cell rock parameters, and FAHM optimizer history schema.'];
    manifest.matlab = struct('version', version, 'release', version('-release'));
    manifest.mrst_release = '2024b';
    manifest.model_class = class(model);
    manifest.active_phase_order = char(model.getPhaseNames());
    manifest.deck_files = deckFiles;
    manifest.well_names = {W.name};
    manifest.well_types = {W.type};
    manifest.weight_field_order = weightFields;
    manifest.config_rows = configRows;
    manifest.parameter_order = cellfun(@(parameter) parameter.name, parameters, 'UniformOutput', false);
    manifest.history_fields = fieldnames(history)';
    manifest.contract = struct( ...
        'matlab_index_base', 1, ...
        'python_index_base', 0, ...
        'matlab_linearization', 'A(:)', ...
        'raw_array_storage_order', 'F', ...
        'python_reshape', 'reshape(shape, order="F")', ...
        'unit_box_bounds', [0, 1], ...
        'parameter_slice_order', 'vertical concatenation in config/parameter order', ...
        'objective_sign', 'matchObservedOWGProfile is positive; evaluateMatchFromEclipseRun returns its negative and negative scaled gradient', ...
        'optimizer_sign', 'optimizeBoundConstrainedForFAHM defaults maximize=true and minimizes the negated callback internally');
    manifest.arrays = arrays;

    json = jsonencode(manifest, 'PrettyPrint', true);
    write_text(fullfile(outDir, 'manifest.json'), [json, newline]);
end

function observed = make_observed(states, schedule)
    observed = cell(size(states));
    for step = 1:numel(states)
        observed{step} = states{step};
        observed{step}.dt = schedule.step.val(step);
        control = schedule.control(schedule.step.control(step));
        for well = 1:numel(observed{step}.wellSol)
            observed{step}.wellSol(well).sign = control.W(well).sign;
            observed{step}.wellSol(well).qWs = ...
                observed{step}.wellSol(well).qWs * (1 + 0.01*step);
            observed{step}.wellSol(well).qOs = ...
                observed{step}.wellSol(well).qOs * (1 - 0.005*step);
            observed{step}.wellSol(well).qGs = ...
                observed{step}.wellSol(well).qGs * (1 + 0.002*step);
            observed{step}.wellSol(well).bhp = ...
                observed{step}.wellSol(well).bhp * (1 + 0.0005*step);
        end
    end
end

function [rows, parameters] = make_parameters(setup)
    appNames = {'Porv', 'PermX', 'PermY', 'PermZ', 'krw', 'kro', 'krg', ...
                'Swl', 'Swcr', 'Swu', 'Sowcr', 'Sgl', 'Sgcr', 'Sgu', 'Sogcr'};
    modelNames = {'porevolume', 'permx', 'permy', 'permz', 'krw', 'kro', 'krg', ...
                  'swl', 'swcr', 'swu', 'sowcr', 'sgl', 'sgcr', 'sgu', 'sogcr'};
    defaults = [.95, 1.05; .1, 10; .1, 10; .1, 10; ...
                .5, 2; .5, 2; .5, 2; 0, 1; 1, 1.5; .8, 1; .8, 1.2; ...
                0, 1; 1, 1.5; .8, 1; .8, 1.2];
    template = struct('app_name', '', 'name', '', 'include', false, ...
        'scaling', '', 'box_limits', [], 'relative_limits_array', '', ...
        'subset_1based_array', '', 'subset_0based_array', '', ...
        'uniform_limits', [], 'default_limits', [], ...
        'relative_limits', [], 'subset', []);
    rows = repmat(template, numel(appNames), 1);
    parameters = {};
    nc = setup.model.G.cells.num;
    for index = 1:numel(appNames)
        rows(index).app_name = appNames{index};
        rows(index).name = modelNames{index};
        rows(index).include = index <= 4;
        rows(index).default_limits = defaults(index, :);
        if ~rows(index).include
            continue;
        end
        if index == 1
            rows(index).scaling = 'linear';
            subset = find(setup.model.operators.pv > 0);
        else
            rows(index).scaling = 'log';
            subset = find(setup.model.rock.perm(:, index - 1) > 0);
        end
        rows(index).box_limits = [];
        rows(index).uniform_limits = false;
        rows(index).relative_limits = repmat(defaults(index, :), nc, 1);
        rows(index).subset = subset;
        rows(index).relative_limits_array = ...
            sprintf('config/%02d_%s_relative_limits', index, lower(appNames{index}));
        rows(index).subset_1based_array = ...
            sprintf('config/%02d_%s_subset_1based', index, lower(appNames{index}));
        rows(index).subset_0based_array = ...
            sprintf('config/%02d_%s_subset_0based', index, lower(appNames{index}));
        parameters = addParameter(parameters, setup, ...
            'name', rows(index).name, ...
            'scaling', rows(index).scaling, ...
            'boxLims', rows(index).box_limits, ...
            'relativeLimits', rows(index).relative_limits, ...
            'subset', rows(index).subset, ...
            'uniformLimits', rows(index).uniform_limits);
    end
end

function history = make_history_trace()
    scratch = tempname;
    mkdir(scratch);
    cleanup = onCleanup(@() remove_scratch(scratch)); %#ok<NASGU>
    directory = struct('work', scratch);
    [~, ~, history] = optimizeBoundConstrainedForFAHM([0.2; 0.8], ...
        @quadratic_callback, directory, ...
        'maxIt', 2, ...
        'gradTol', -1, ...
        'objChangeTol', -1, ...
        'objChangeTolRel', -1, ...
        'plotEvolution', false, ...
        'outputHessian', false, ...
        'params', {});
end

function [value, gradient] = quadratic_callback(u, ~)
    target = [0.65; 0.35];
    delta = u - target;
    value = -sum(delta.^2);
    gradient = -2*delta;
end

function remove_scratch(path)
    if isfolder(path)
        rmdir(path, 's');
    end
end

function value = beta_value(beta, field)
    if isfield(beta, field)
        value = beta.(field);
    else
        value = 0;
    end
end

function records = add_well_solution(records, root, prefix, wellSol)
    fields = {'qWs', 'qOs', 'qGs', 'bhp', 'sign', 'status'};
    for index = 1:numel(fields)
        field = fields{index};
        if isfield(wellSol, field)
            value = vertcat(wellSol.(field));
            if strcmp(field, 'status')
                value = logical(value);
            end
            records = add_array(records, root, [prefix, '/', field], value);
        end
    end
end

function records = add_optional_array(records, root, name, value, field)
    if isfield(value, field)
        records = add_array(records, root, name, value.(field));
    end
end

function records = empty_array_records()
    records = struct('name', {}, 'path', {}, 'dtype', {}, 'shape', {}, ...
                     'order', {}, 'nbytes', {}, 'sha256', {});
end

function records = add_array(records, root, name, value)
    safeName = strrep(name, '/', '__');
    relativePath = ['arrays/', safeName, '.bin'];
    path = fullfile(root, strrep(relativePath, '/', filesep));
    if islogical(value)
        value = uint8(value);
        precision = 'uint8';
        dtype = '|u1';
    elseif isa(value, 'int64')
        precision = 'int64';
        dtype = '<i8';
    else
        value = double(value);
        precision = 'double';
        dtype = '<f8';
    end
    fid = fopen(path, 'wb', 'ieee-le');
    assert(fid >= 0, 'Unable to open %s', path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, value(:), precision);
    clear cleanup;
    shape = size(value);
    record = struct('name', name, 'path', relativePath, 'dtype', dtype, ...
        'shape', shape, 'order', 'F', 'nbytes', dir(path).bytes, ...
        'sha256', sha256_file(path));
    records(end + 1) = record;
end

function record = file_record(root, path)
    relative = strrep(path((numel(root) + 2):end), filesep, '/');
    record = struct('path', relative, 'nbytes', dir(path).bytes, ...
                    'sha256', sha256_file(path));
end

function hash = sha256_file(path)
    fid = fopen(path, 'rb');
    assert(fid >= 0, 'Unable to open %s', path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid, Inf, '*uint8');
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(typecast(bytes, 'int8'));
    raw = typecast(digest.digest(), 'uint8');
    hash = lower(reshape(dec2hex(raw, 2).', 1, []));
end

function ensure_dir(path)
    if ~isfolder(path)
        mkdir(path);
    end
end

function write_text(path, value)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    assert(fid >= 0, 'Unable to open %s', path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, value, 'char');
end
