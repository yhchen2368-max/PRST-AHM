"""Run the PRSTCore-vs-MRST flow diagnostics parity example."""

from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from PRSTCore.visualization.diagnostics.examples.mrst_parity_1d import main


if __name__ == "__main__":
    raise SystemExit(main())
