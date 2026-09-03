from pathlib import Path
import shutil

import pytest

from PRSTCore.solvers.incomp.examples.mrst_parity_incomp_tpfa import (
    assert_parity,
    format_results,
    generate_mrst_reference,
    run_parity,
)


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_incomp_tpfa_matches_mrst(tmp_path: Path):
    reference = tmp_path / "incomp_tpfa_mrst_ref.mat"
    generate_mrst_reference(reference)
    results, timing = run_parity(reference)
    print(format_results(results, timing))
    assert_parity(results)
