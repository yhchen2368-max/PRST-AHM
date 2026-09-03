"""Compute normalization factors for observed data weighting.

Corresponds to the pattern used in MRST examples:
    beta = getNormalizationFactors(observed);
    weighting = {'WaterRateWeight', beta.ww, 'OilRateWeight', beta.wo, 'BHPWeight', beta.wp};

Computes appropriate weighting factors so that water rate, oil rate,
and BHP contributions are of comparable magnitude in the mismatch.
"""

import numpy as np


def get_normalization_factors(observed):
    """Compute normalization factors from observed well data.

    Parameters
    ----------
    observed : list of dict
        Observed states, each with wellSol containing qWs, qOs, bhp.

    Returns
    -------
    dict
        Dictionary with 'ww', 'wo', 'wp' weighting factors.
    """
    all_qWs = []
    all_qOs = []
    all_bhp = []

    for obs in observed:
        if "wellSol" in obs and abs:
            for ws in obs["wellSol"]:
                if ws.get("status", True):
                    all_qWs.append(abs(float(ws.get("qWs", 0))))
                    all_qOs.append(abs(float(ws.get("qOs", 0))))
                    all_bhp.append(abs(float(ws.get("bhp", 0))))

    qWs_arr = np.array(all_qWs) if all_qWs else np.array([1.0])
    qOs_arr = np.array(all_qOs) if all_qOs else np.array([1.0])
    bhp_arr = np.array(all_bhp) if all_bhp else np.array([1.0])

    # Normalization: 1 / (mean of absolute values)
    ww = 1.0 / np.mean(qWs_arr) if np.mean(qWs_arr) > 1e-12 else 1.0
    wo = 1.0 / np.mean(qOs_arr) if np.mean(qOs_arr) > 1e-12 else 1.0
    wp = 1.0 / np.mean(bhp_arr) if np.mean(bhp_arr) > 1e-12 else 1.0

    return {"ww": ww, "wo": wo, "wp": wp}
