# FAHM Stage 2 oracle

This directory freezes facts from `MRST/dev/APP/FAHM.mlapp`, not from
`dev/test/HistoryMatching.m` and not from the current PRST implementation.

## Contents

- `tools/extract_fahm_static_oracle.py`: extracts MLAPP/source fingerprints,
  the 373 public App components, their source assignments, 95 callback
  bindings and the key event flows.
- `matlab/export_fahm_minimal_oracle.m`: runs the first three report steps of
  the bundled SPE1 W/O/G case and exports deck, schedule, state, observed,
  config, per-cell parameter vector, per-step objective and optimizer history.
- `tools/verify_fahm_oracle.py`: validates hashes, dtypes, shapes and
  column-major binary arrays, or compares two fixture trees byte for byte.
- `tools/run_stage2_oracle.py`: regenerates the static oracle and performs two
  independent MATLAB exports before comparing both with the committed golden.
- `test_stage2_fahm_oracle.py`: fast, MATLAB-free contract checks used by
  pytest after the golden has been produced.
- `../fixtures/fahm_oracle/v1`: versioned static, contract and dynamic data.

## Raw array format

Every array is stored as a headerless `.bin` file. `manifest.json` supplies:

- explicit little-endian `dtype`;
- MATLAB `shape`;
- `order: "F"`, corresponding to MATLAB `A(:)`;
- byte count and SHA-256;
- a semantic array name independent of the physical file name.

Python must read an array with `numpy.fromfile` and reshape it with
`order="F"`. Index arrays are exported in paired 1-based and 0-based forms so
that conversion errors cannot be hidden by numerical tolerances.

## Re-run the full Stage 2 gate

From `E:\19.PRST\RSTCore-main`:

```powershell
.\.venv\Scripts\python.exe -m tests.fahm_oracle.tools.run_stage2_oracle
```

Fast checks without launching MATLAB:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\fahm_oracle\test_stage2_fahm_oracle.py
```

The full gate passes only when the regenerated static snapshot, two fresh
MATLAB exports and the committed golden are all byte-identical.
