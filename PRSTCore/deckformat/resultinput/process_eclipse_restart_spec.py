"""Process ECLIPSE restart specification file.

1:1 Python translation of MRST model-io/deckformat/resultinput/processEclipseRestartSpec.m
"""

import numpy as np
import os


def process_eclipse_restart_spec(prefix, restart_amount="default"):
    """Read and process RSSPEC file for restart reading.

    Parameters
    ----------
    prefix : str
        File prefix (with or without .RSSPEC extension).
    restart_amount : str
        'default' or 'all'.

    Returns
    -------
    spec : dict
        Specification structure.
    spec_lgr : list
        LGR specifications.
    """
    pth, nm = os.path.split(prefix)
    name, ext = os.path.splitext(nm)

    if not ext:
        ext = ".RSSPEC"

    filename = os.path.join(pth, name + ext)

    from .read_eclipse_output_file_unfmt import read_eclipse_output_file_unfmt
    rsspec = read_eclipse_output_file_unfmt(filename)

    ih = rsspec["INTEHEAD"]["values"]
    unit_list = ["metric", "field", "lab"]
    unit_idx = int(ih[2]) - 1 if len(ih) > 2 else 0
    unit = unit_list[max(0, min(unit_idx, 2))]

    # Time and date
    t = rsspec.get("TIME", {}).get("values", np.zeros(1))
    d = rsspec.get("ITIME", {}).get("values", np.zeros(3))
    n_steps = len(t)
    date = d.reshape(-1, n_steps).T if len(d) >= n_steps * 3 else np.zeros((n_steps, 3))
    rep_num = date[:, 0].astype(int)
    date = date[:, 1:4]

    # Determine restart type
    names = rsspec.get("NAME", {}).get("values", [])
    first_field = names[0] if len(names) > 0 else "SEQNUM"

    if first_field == "SEQNUM":
        rtype = "unified"
    elif first_field == "INTEHEAD":
        rtype = "multiple"
    else:
        rtype = "unknown"

    # Keywords and pointers
    name_vals = np.array(names)
    nf_list = [i for i, n in enumerate(names) if n == first_field]

    keywords = []
    pointers = []
    prec = []
    num = []

    for k in range(n_steps):
        if k < len(nf_list):
            start = nf_list[k]
            end = nf_list[k + 1] if k + 1 < len(nf_list) else len(name_vals)
            sub_ix = list(range(start, end))

            if restart_amount == "default":
                sub_ix = [i for i in sub_ix if _is_default_field(name_vals[i])]

            keywords.append([name_vals[i] for i in sub_ix])

            ptr_b = rsspec.get("POINTERB", {}).get("values", np.zeros(len(name_vals)))
            ptr = rsspec.get("POINTER", {}).get("values", np.zeros(len(name_vals)))
            pointers.append(2.0**31 * np.array([ptr_b[i] for i in sub_ix]) +
                            np.array([ptr[i] for i in sub_ix]))

            types = rsspec.get("TYPE", {}).get("values", [""] * len(name_vals))
            prec.append([_map_precision(types[i]) for i in sub_ix])

            nums = rsspec.get("NUMBER", {}).get("values", np.zeros(len(name_vals)))
            num.append(np.array([nums[i] for i in sub_ix]))

    spec = {
        "time": t,
        "date": date,
        "unit": unit,
        "type": rtype,
        "keywords": keywords,
        "pointers": pointers,
        "prec": prec,
        "num": num,
        "repNum": rep_num,
    }

    spec_lgr = []
    return spec, spec_lgr


def _is_default_field(name):
    """Check if field is in default set."""
    default_names = {
        "PRESSURE", "SWAT", "SOIL", "SGAS", "RS", "RV",
        "PORV", "PORO", "PERMX", "PERMY", "PERMZ",
        "DEPTH", "FLROIL", "FLRWAT", "FLRGAS",
    }
    return name.upper() in default_names


def _map_precision(typ):
    """Map ECLIPSE type to precision string."""
    mapping = {"INTE": "int32", "REAL": "float32", "DOUB": "float64", "LOGI": "int32"}
    return mapping.get(typ.strip(), "float32")
