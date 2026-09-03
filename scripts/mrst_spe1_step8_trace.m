function trace = mrst_spe1_step8_trace()
%MRST_SPE1_STEP8_TRACE Export every stored Newton state for SPE1 report step 8.
%
% This is a read-only MRST diagnostic.  It runs the first eight report steps
% of BENCH_SPE1.DATA with the MRST-2026a defaults, asks NonLinearSolver to
% retain intermediate Newton states (reportLevel = 3), and writes a compact
% CSV plus a MAT file next to this script.

    scriptDir = fileparts(mfilename('fullpath'));
    cgnetRoot = fileparts(scriptDir);
    mrstRoot = fullfile(fileparts(cgnetRoot), 'mrst-2026a');
    deckFile = fullfile(cgnetRoot, 'examples', 'SpE1', 'BENCH_SPE1.DATA');

    assert(isfile(deckFile), 'Deck not found: %s', deckFile);
    assert(isfolder(mrstRoot), 'MRST root not found: %s', mrstRoot);

    cd(mrstRoot);
    run(fullfile(mrstRoot, 'startup.m'));
    mrstModule add ad-core deckformat ad-blackoil ad-props

    [state0, model, schedule, solver] = initEclipseProblemAD(deckFile);
    reportStep = 8;
    schedule.step.val = schedule.step.val(1:reportStep);
    schedule.step.control = schedule.step.control(1:reportStep);

    % NonLinearSolver.solveMinistep stores state after each stepFunction
    % call when reportLevel > 2 (MRST NonLinearSolver.m, lines 369-384).
    solver.reportLevel = 3;
    solver.verbose = 0;

    [~, ~, scheduleReport] = simulateScheduleAD(state0, model, schedule, ...
        'NonLinearSolver', solver, 'Verbose', false, 'OutputMinisteps', false);

    reports = scheduleReport.ControlstepReports;
    assert(numel(reports) == reportStep, 'Expected %d report steps.', reportStep);
    r8 = reports{reportStep};
    W = schedule.control(schedule.step.control(reportStep)).W;
    injCell = getWellCell(W, 'INJECTOR');
    prodCell = getWellCell(W, 'PRODUCER');

    timeStart = sum(schedule.step.val(1:reportStep-1));
    timeNow = timeStart;
    records = repmat(emptyRecord(), 0, 1);
    for mini = 1:numel(r8.StepReports)
        miniReport = r8.StepReports{mini};
        miniDt = miniReport.Timestep;
        nreports = miniReport.NonlinearReport;
        for iteration = 1:numel(nreports)
            nr = nreports{iteration};
            assert(isfield(nr, 'state'), ...
                'No intermediate state stored; solver.reportLevel must exceed 2.');
            st = nr.state;
            sg = st.s(:, 3);
            rs = st.rs;
            vo = model.getCellStatusVO(st, st.s(:, 2), st.s(:, 1), sg);
            voCode = voStatusCodes(vo);
            rec = emptyRecord();
            rec.report_step = reportStep;
            rec.ministep = mini;
            rec.dt_s = miniDt;
            rec.time_start_s = timeNow;
            rec.time_end_s = timeNow + miniDt;
            rec.newton_iteration = iteration;
            rec.converged = getLogical(nr, 'Converged');
            rec.failure = getLogical(nr, 'Failure');
            rec.solved = getLogical(nr, 'Solved');
            rec.residual_max = getResidualMax(nr);
            rec.pressure_min_Pa = min(st.pressure);
            rec.pressure_max_Pa = max(st.pressure);
            rec.sg_min = min(sg);
            rec.sg_max = max(sg);
            rec.sg_injector = sg(injCell);
            rec.sg_producer = sg(prodCell);
            rec.rs_min = min(rs);
            rec.rs_max = max(rs);
            rec.rs_injector = rs(injCell);
            rec.rs_producer = rs(prodCell);
            rec.vo_oil_only_cells = nnz(voCode == 1);
            rec.vo_gas_only_cells = nnz(voCode == 2);
            rec.vo_oil_gas_cells = nnz(voCode == 3);
            rec.injector_phase_status = phaseStatus(voCode(injCell));
            rec.producer_phase_status = phaseStatus(voCode(prodCell));
            records(end + 1, 1) = rec; %#ok<AGROW>
        end
        timeNow = timeNow + miniDt;
    end

    trace = struct();
    trace.deck = deckFile;
    trace.report_step = reportStep;
    trace.report_dt_s = schedule.step.val(reportStep);
    trace.report_time_start_s = timeStart;
    trace.report_time_end_s = timeNow;
    trace.injector_cell = injCell;
    trace.producer_cell = prodCell;
    trace.records = records;
    trace.report = r8;

    csvFile = fullfile(scriptDir, 'mrst_spe1_step8_newton_trace.csv');
    matFile = fullfile(scriptDir, 'mrst_spe1_step8_newton_trace.mat');
    writetable(struct2table(records), csvFile);
    save(matFile, 'trace');
    fprintf('Wrote %s\nWrote %s\n', csvFile, matFile);
end

function cellNo = getWellCell(W, name)
    ix = find(strcmp({W.name}, name), 1);
    assert(~isempty(ix), 'Well %s not found.', name);
    cellNo = W(ix).cells(1);
end

function value = getLogical(s, field)
    if isfield(s, field)
        value = logical(s.(field));
    else
        value = false;
    end
end

function value = getResidualMax(s)
    if isfield(s, 'Residuals') && ~isempty(s.Residuals)
        value = max(abs(s.Residuals));
    else
        value = nan;
    end
end

function code = voStatusCodes(vo)
    code = zeros(numel(vo{3}), 1);
    code(vo{1}) = 1;
    code(vo{2}) = 2;
    code(vo{3}) = 3;
end

function status = phaseStatus(code)
    if code == 3
        status = 'oil_gas';
    elseif code == 1
        status = 'oil_only';
    elseif code == 2
        status = 'gas_only';
    else
        status = 'neither';
    end
end

function rec = emptyRecord()
    rec = struct( ...
        'report_step', 0, 'ministep', 0, 'dt_s', nan, ...
        'time_start_s', nan, 'time_end_s', nan, 'newton_iteration', 0, ...
        'converged', false, 'failure', false, 'solved', false, ...
        'residual_max', nan, 'pressure_min_Pa', nan, 'pressure_max_Pa', nan, ...
        'sg_min', nan, 'sg_max', nan, 'sg_injector', nan, 'sg_producer', nan, ...
        'rs_min', nan, 'rs_max', nan, 'rs_injector', nan, 'rs_producer', nan, ...
        'vo_oil_only_cells', 0, 'vo_gas_only_cells', 0, 'vo_oil_gas_cells', 0, ...
        'injector_phase_status', '', 'producer_phase_status', '');
end
