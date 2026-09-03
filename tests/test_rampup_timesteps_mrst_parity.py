from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import loadmat

from PRSTCore.ad_core.timesteps import rampup_timesteps

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


@pytest.mark.skipif(shutil.which("matlab") is None, reason="MATLAB is required to generate the MRST reference")
def test_rampup_timesteps_matches_mrst(tmp_path: Path):
    reference = tmp_path / "rampup_timesteps_mrst_ref.mat"
    matlab_exe = shutil.which("matlab")
    cmd = (
        f"addpath('{_matlab_path(REPO_ROOT / 'scripts')}'); "
        f"export_mrst_rampup_timesteps('{_matlab_path(reference)}')"
    )
    subprocess.run([matlab_exe, "-batch", cmd], cwd=REPO_ROOT, check=True)
    ref = loadmat(reference, simplify_cells=True, squeeze_me=True)

    day = 86400.0
    cases = [
        ("dT1", rampup_timesteps(365 * day, 30 * day)),
        ("dT2", rampup_timesteps(365 * day, 30 * day, 5)),
        ("dT3", rampup_timesteps(100 * day, 10 * day, 3)),
        ("dT4", rampup_timesteps(45 * day, 30 * day)),
        ("dT5", rampup_timesteps(1000, 100, 0)),
    ]
    for name, py in cases:
        mat = np.atleast_1d(np.asarray(ref[name], dtype=float).ravel())
        assert py.shape == mat.shape
        assert np.allclose(py, mat, rtol=1e-10)
