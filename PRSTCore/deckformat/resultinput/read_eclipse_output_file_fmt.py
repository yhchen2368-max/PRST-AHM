"""Read formatted (ASCII) ECLIPSE output file.

1:1 Python translation of MRST model-io/deckformat/resultinput/readEclipseOutputFileFmt.m
"""

import numpy as np
import re


def read_eclipse_output_file_fmt(fname):
    """Read formatted (ASCII) ECLIPSE output file.

    Parameters
    ----------
    fname : str
        Path to formatted output file.

    Returns
    -------
    dict
        Data structure with field data.
    """
    with open(fname, "r") as f:
        content = f.read()

    output = {}
    # Split by keyword headers
    # Pattern: 8-char keyword followed by whitespace and number
    pattern = r"^(.{8})\s+(\d+)\s+(\w+)\s*$"

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        match = re.match(pattern, line)
        if match:
            keyword = match.group(1).strip()
            num_vals = int(match.group(2))
            dtype_str = match.group(3)

            dtype_map = {
                "INTE": np.int32,
                "REAL": np.float32,
                "DOUB": np.float64,
                "LOGI": np.int32,
                "CHAR": str,
            }

            values = []
            i += 1
            while len(values) < num_vals and i < len(lines):
                vals_line = lines[i].strip()
                if vals_line:
                    for v in vals_line.split():
                        if dtype_str == "CHAR":
                            values.append(v.strip("'").strip('"'))
                        else:
                            values.append(float(v))
                i += 1

            output[keyword] = {"values": np.array(values[:num_vals]), "type": dtype_str}
        else:
            i += 1

    return output
