function export_mrst_rampup_timesteps(output_file)
%EXPORT_MRST_RAMPUP_TIMESTEPS Reference data for rampupTimesteps.m parity.

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add ad-core

dT1 = rampupTimesteps(365*day, 30*day);
dT2 = rampupTimesteps(365*day, 30*day, 5);
dT3 = rampupTimesteps(100*day, 10*day, 3);
dT4 = rampupTimesteps(45*day, 30*day);       % ramp-up alone exceeds total time
dT5 = rampupTimesteps(1000, 100, 0);          % n=0: no ramp-up steps

save(output_file, 'dT1', 'dT2', 'dT3', 'dT4', 'dT5', '-v7');
fprintf('MRST rampupTimesteps reference written to %s\n', output_file);
end
