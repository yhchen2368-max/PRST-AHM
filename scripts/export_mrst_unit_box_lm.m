function export_mrst_unit_box_lm(output_file)
%EXPORT_MRST_UNIT_BOX_LM Reference data for unitBoxLM.m parity.
%
% A small fixed linear-least-squares problem (residual v(u) = A*u - b,
% constant Jacobian J = A) confined to the unit box, so the optimizer's
% full iteration trace is deterministic and directly comparable.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add optimization

A = [ 1.0  0.5 -0.3  0.2; ...
      0.4 -1.0  0.6  0.1; ...
     -0.2  0.3  1.2 -0.5; ...
      0.6  0.2 -0.4  0.9; ...
     -0.5  0.7  0.1  0.3; ...
      0.3 -0.2  0.5  1.1];
b = [0.8; -0.4; 0.6; 1.0; -0.2; 0.5];
u0 = [0.2; 0.3; 0.4; 0.5];

f = @(u) deal(A*u - b, A);

[v, u, history] = unitBoxLM(u0, f, 'verbose', false, 'plotEvolution', false);

nIt = numel(history.val);
u_hist = zeros(nIt, numel(u0));
for k = 1:nIt
    u_hist(k, :) = history.u{k}(:)';
end

% Also a second case where the box constraint is active (unconstrained
% optimum falls outside [0,1] along one coordinate), to exercise the
% active-set / clamping logic.
A2 = [ 2.0  0.1; ...
       0.1  2.0; ...
      -1.0  0.5];
b2 = [5.0; -5.0; 0.0];
u02 = [0.5; 0.5];
f2 = @(u) deal(A2*u - b2, A2);
[v2, u2, history2] = unitBoxLM(u02, f2, 'verbose', false, 'plotEvolution', false);

save(output_file, 'v', 'u', 'u_hist', 'v2', 'u2', '-v7');
fprintf('MRST unitBoxLM reference written to %s\n', output_file);
end
