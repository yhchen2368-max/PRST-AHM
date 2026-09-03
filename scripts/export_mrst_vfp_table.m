function export_mrst_vfp_table(output_file)
%EXPORT_MRST_VFP_TABLE Reference data for VFPTable.m parity (VFPPROD/VFPINJ).
%
% Builds synthetic VFPPROD/VFPINJ table structs directly (bypassing deck
% parsing, since no example deck in this repo exercises these keywords) and
% evaluates VFPTable.evaluateBHP against them, both inside the table's grid
% (pure interpolation) and outside it (linear extrapolation).

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-core

% ---- VFPPROD: separable affine function so exact multilinear
% interpolation/extrapolation values are independently verifiable. ----
flo = [100 200 300];
thp = [10 20 30];
wfr = [0.1 0.5];
gfr = [50 150];
alq = [0 10];

[FLO, THP, WFR, GFR, ALQ] = ndgrid(flo, thp, wfr, gfr, alq);
Q = 100 + 0.01*FLO + 0.5*THP + 2*WFR + 3*GFR + 0.1*ALQ;

dprod = struct('FLO', flo, 'THP', thp, 'WFR', wfr, 'GFR', gfr, 'ALQ', alq, ...
                'Q', Q, 'FLOID', 'OIL', 'WFRID', 'WOR', 'GFRID', 'GOR', ...
                'ALQID', 'GRAT', 'depth', 1000);
tprod = VFPTable(dprod);

flo_q  = [100 150 250 300 50  350]';
thp_q  = [10  15  25  30  5   35]';
wfr_q  = [0.1 0.3 0.4 0.5 0.05 0.6]';
gfr_q  = [50  100 120 150 40  160]';
alq_q  = [0   5   8   10  -2  12]';
bhp_prod_ref = tprod.evaluateBHP(flo_q, thp_q, wfr_q, gfr_q, alq_q);

% ---- VFPPROD without ALQ dimension (single ALQ slice, squeezed away). ----
Q1 = Q(:, :, :, :, 1);
dprod1 = struct('FLO', flo, 'THP', thp, 'WFR', wfr, 'GFR', gfr, 'ALQ', alq(1), ...
                 'Q', Q1, 'FLOID', 'LIQ', 'WFRID', 'WCT', 'GFRID', 'GLR', ...
                 'depth', 1000);
tprod1 = VFPTable(dprod1);
bhp_prod1_ref = tprod1.evaluateBHP(flo_q, thp_q, wfr_q, gfr_q);

% ---- VFPINJ: separable affine function. ----
floi = [50 150 250];
thpi = [5 15 25];
[FLOI, THPI] = ndgrid(floi, thpi);
BHP = 50 + 0.02*FLOI + 0.8*THPI;

dinj = struct('FLO', floi, 'THP', thpi, 'BHP', BHP, 'FLOID', 'WAT', 'depth', 900);
tinj = VFPTable(dinj);

flo_qi = [50 100 200 250 25 300]';
thp_qi = [5  10  20  25  2  30]';
bhp_inj_ref = tinj.evaluateBHP(flo_qi, thp_qi);

save(output_file, 'flo', 'thp', 'wfr', 'gfr', 'alq', 'Q', ...
     'flo_q', 'thp_q', 'wfr_q', 'gfr_q', 'alq_q', 'bhp_prod_ref', ...
     'Q1', 'bhp_prod1_ref', ...
     'floi', 'thpi', 'BHP', 'flo_qi', 'thp_qi', 'bhp_inj_ref', '-v7');
fprintf('MRST VFPTable reference written to %s\n', output_file);
end
