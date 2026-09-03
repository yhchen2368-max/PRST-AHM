from pathlib import Path
import shutil

import pytest

from PRSTCore.gridprocessing.examples.mrst_parity_cart_grid import (
    assert_parity,
    format_results,
    generate_mrst_reference,
    run_parity,
)


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_cart_grid_and_compute_geometry_match_mrst(tmp_path: Path):
    reference = tmp_path / "grid_geometry_mrst_ref.mat"
    generate_mrst_reference(reference)
    results, timing = run_parity(reference)
    print(format_results(results, timing))
    assert_parity(results)
