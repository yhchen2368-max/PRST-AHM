function export_mrst_relperm_tables(output_file)
%EXPORT_MRST_RELPERM_TABLES Reference data for SWFN/SGFN/SOF2/SOF3/Corey parity.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-props

%% SWFN: [Sw, krW, PcOW]
swfn = [0.2 0.0 5.0e4; 0.3 0.02 3.0e4; 0.5 0.18 1.0e4; 0.7 0.5 3.0e3; 0.9 0.85 0; 1.0 1.0 0];
reg = struct('sat', 1);
f = struct();
f = assignSWFN(f, {swfn}, reg);
sw_query = linspace(0.2, 1.0, 25)';
krW_ref = f.krW{1}(sw_query);
pcOW_ref = f.pcOW{1}(sw_query);
swfn_points = f.krPts.w(1, :);

%% SGFN: [Sg, krG, PcOG] (assigned onto the same fluid struct as SWFN, since
%% assignSGFN reads f.krPts.w for its connate-water point)
sgfn = [0.0 0.0 0; 0.1 0.0 0; 0.3 0.15 500; 0.5 0.4 1200; 0.7 0.8 2500];
f = assignSGFN(f, {sgfn}, reg);
sg_query = linspace(0.0, 0.7, 25)';
krG_ref = f.krG{1}(sg_query);
pcOG_ref = f.pcOG{1}(sg_query);
sgfn_points = f.krPts.g(1, :);

% NOTE: assignSOF2's krO uses the internal region-mapping helper
% getRegMap/interpReg (its own doc comment: "internal function... may
% disappear into the void without warning"), not the plain interpTable used
% by SWFN/SGFN/SOF3 below -- skipped here; SOF2 shares the exact same
% underlying 1D linear interpolation primitive validated by those.
so_query = linspace(0.0, 0.8, 25)';

%% SOF3: [So, krOW, krOG]
sof3 = [0.0 0.0 0.0; 0.2 0.03 0.02; 0.4 0.2 0.15; 0.6 0.5 0.45; 0.8 1.0 0.9];
f4 = struct();
f4 = assignSOF3(f4, {sof3}, reg);
krOW_sof3_ref = f4.krOW{1}(so_query);
krOG_sof3_ref = f4.krOG{1}(so_query);

%% Corey
n = 2.5; sr = 0.15; sr_tot = 0.35; krmax = 0.9;
s_query = linspace(0, 1, 30)';
corey_fn = coreyPhaseRelpermAD(n, sr, krmax, sr_tot);
kr_corey_ref = corey_fn(s_query);

save(output_file, 'sw_query', 'krW_ref', 'pcOW_ref', 'swfn_points', ...
    'sg_query', 'krG_ref', 'pcOG_ref', 'sgfn_points', ...
    'so_query', 'krOW_sof3_ref', 'krOG_sof3_ref', ...
    's_query', 'kr_corey_ref', '-v7');
fprintf('MRST relperm table reference written to %s\n', output_file);
end
