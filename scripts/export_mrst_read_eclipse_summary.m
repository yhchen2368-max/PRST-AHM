function export_mrst_read_eclipse_summary(output_file)
%EXPORT_MRST_READ_ECLIPSE_SUMMARY Reference data for readEclipseSummaryUnFmt.m /
% convertSummaryToWellSols.m parity, using the real SPE9 case's ECLIPSE
% summary output files (examples/SPE9/RESULTS/SPE9_CP.SMSPEC/.UNSMRY).

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add deckformat

prefix = fullfile(repo, 'examples', 'SPE9', 'RESULTS', 'SPE9_CP');

[smry, spec] = readEclipseSummaryUnFmt(prefix);

time_ref = smry.get(repmat(':+', [1, 4]), 'TIME', ':');
wopr_ref = smry.get('PRODU2', 'WOPR', ':');
wbhp_ref = smry.get('PRODU2', 'WBHP', ':');
wwct_ref = smry.get('PRODU2', 'WWCT', ':');
injwbhp_ref = smry.get('INJE1', 'WBHP', ':');
unit_wbhp = smry.getUnit('PRODU2', 'WBHP');
nsteps_ref = size(smry.data, 2);
nlist_ref = size(smry.data, 1);

[wellSols, wsTime] = convertSummaryToWellSols(prefix);

nw = numel(wellSols{1});
nt = numel(wellSols);
names = cell(nw, 1);
for k = 1:nw
    names{k} = wellSols{1}(k).name;
end
qOs = zeros(nt, nw);
qWs = zeros(nt, nw);
qGs = zeros(nt, nw);
bhp = zeros(nt, nw);
sgn = zeros(nt, nw);
for kt = 1:nt
    for kw = 1:nw
        qOs(kt, kw) = wellSols{kt}(kw).qOs;
        qWs(kt, kw) = wellSols{kt}(kw).qWs;
        qGs(kt, kw) = wellSols{kt}(kw).qGs;
        bhp(kt, kw) = wellSols{kt}(kw).bhp;
        sgn(kt, kw) = wellSols{kt}(kw).sign;
    end
end

save(output_file, 'time_ref', 'wopr_ref', 'wbhp_ref', 'wwct_ref', ...
    'injwbhp_ref', 'unit_wbhp', 'nsteps_ref', 'nlist_ref', ...
    'names', 'qOs', 'qWs', 'qGs', 'bhp', 'sgn', 'wsTime', '-v7');
fprintf('MRST readEclipseSummary/convertSummaryToWellSols reference written to %s\n', output_file);
end
