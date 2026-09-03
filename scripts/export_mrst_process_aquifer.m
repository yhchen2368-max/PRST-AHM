function export_mrst_process_aquifer(output_file)
%EXPORT_MRST_PROCESS_AQUIFER Reference data for processAquifer.m parity,
% using the real MSW.data deck (which has AQUANCON/AQUFETP).

repo = fileparts(fileparts(mfilename('fullpath')));
mrst_root = fullfile(repo, 'mrst-2026a');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add deckformat ad-core ad-blackoil

fn = fullfile(mrst_root, 'modules', 'nwm', 'data', 'MSW.data');
deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);

G = initEclipseGrid(deck);
G = computeGeometry(G);

output = processAquifer(deck, G);

aquifers = output.aquifers;
aquind = output.aquind;
initval = output.initval;
aquiferprops = output.aquiferprops;

save(output_file, 'aquifers', 'aquind', 'initval', 'aquiferprops', '-v7');
fprintf('MRST processAquifer reference written to %s\n', output_file);
end
