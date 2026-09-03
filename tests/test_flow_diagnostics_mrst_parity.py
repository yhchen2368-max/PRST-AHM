from pathlib import Path
import shutil

import pytest

from PRSTCore.visualization.diagnostics.examples.mrst_parity_1d import (
    assert_parity,
    generate_mrst_reference,
    run_parity,
)


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_flow_diagnostics_match_mrst_1d_reference(tmp_path: Path):
    reference = tmp_path / "flow_diagnostics_mrst_1d_ref.mat"
    generate_mrst_reference(reference)
    assert_parity(run_parity(reference))

