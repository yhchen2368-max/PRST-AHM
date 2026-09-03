function export_mrst_wellbore_friction(output_file)
%EXPORT_MRST_WELLBORE_FRICTION Reference data for wellBoreFriction.m parity.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-core

rng(0);
n = 200;
% Span laminar, transitional, and turbulent regimes by construction.
v_massrate = 10.^(linspace(-6, 2, n))' .* (2*(rand(n,1)>0.3)-1);
rho = 700 + 400*rand(n,1);
mu  = 10.^(linspace(-4, -2, n))';
Do  = 0.05 + 0.15*rand(n,1);
L   = 5 + 20*rand(n,1);
roughness = 1e-5 + 1e-4*rand(n,1);

dp_massrate = wellBoreFriction(v_massrate, rho, mu, Do, L, roughness, 'massRate', false);
dp_massrate_turb = wellBoreFriction(v_massrate, rho, mu, Do, L, roughness, 'massRate', true);

v_vol = 10.^(linspace(-5, 1, n))' .* (2*(rand(n,1)>0.5)-1);
dp_vol = wellBoreFriction(v_vol, rho, mu, Do, L, roughness, 'volumeRate', false);

v_vel = 10.^(linspace(-3, 2, n))' .* (2*(rand(n,1)>0.5)-1);
dp_vel = wellBoreFriction(v_vel, rho, mu, Do, L, roughness, 'velocity', false);

% Annulus (Di, Do) case -- MRST's numel(D)==2 branch is for a single scalar
% (Di, Do) pair applied to a whole well, not a per-segment array.
Di_scalar = 0.03;
Do_scalar = 0.1;
dp_annulus = wellBoreFriction(v_massrate, rho, mu, [Di_scalar, Do_scalar], L, roughness, 'massRate', false);

save(output_file, 'v_massrate', 'rho', 'mu', 'Do', 'L', 'roughness', ...
    'Di_scalar', 'Do_scalar', ...
    'dp_massrate', 'dp_massrate_turb', 'v_vol', 'dp_vol', 'v_vel', 'dp_vel', 'dp_annulus', '-v7');
fprintf('MRST wellBoreFriction reference written to %s\n', output_file);
end
