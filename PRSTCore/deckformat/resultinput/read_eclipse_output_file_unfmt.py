"""Read unformatted (binary) ECLIPSE output files.

This reader handles the standard ECLIPSE Fortran sequential-record layout:
each keyword is stored as a 16-byte header record followed by one or more
payload records.  Repeated keywords are concatenated in the top-level output
and the ordered per-record data is kept in ``"__records__"`` for restart
files where report-step grouping matters.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


_NUMERIC_DTYPES = {
    "INTE": (">i4", np.int32),
    "LOGI": (">i4", np.int32),
    "REAL": (">f4", np.float32),
    "DOUB": (">f8", np.float64),
}


def read_eclipse_output_file_unfmt(fname, cell_output=False, max_cell_size=None):
    """Read an unformatted ECLIPSE binary output file.

    Parameters
    ----------
    fname : str
        Path to a binary ECLIPSE output file (.INIT, .EGRID, .RSSPEC, .UNRST,
        etc.).
    cell_output : bool
        Kept for API compatibility with the MRST-style reader.  The Python
        implementation always returns keyword arrays.
    max_cell_size : int, optional
        Kept for API compatibility.

    Returns
    -------
    dict
        Mapping from keyword to ``{"values": ..., "type": ...}``.  Duplicate
        keywords are concatenated.  The special ``"__records__"`` entry stores
        the ordered, non-concatenated keyword records.
    """
    del cell_output, max_cell_size

    path = Path(fname)
    grouped: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    with path.open("rb") as handle:
        while True:
            header_offset = handle.tell()
            header = _read_fortran_record(handle, path)
            if header is None:
                break
            if len(header) != 16:
                raise ValueError(
                    f"{path}: expected a 16-byte ECLIPSE keyword header at byte "
                    f"{header_offset}, got {len(header)} bytes"
                )

            keyword = header[:8].decode("ascii", errors="replace").strip()
            count = struct.unpack(">i", header[8:12])[0]
            ecl_type = header[12:16].decode("ascii", errors="replace").strip()
            if count < 0:
                raise ValueError(f"{path}: negative item count {count} for keyword {keyword!r}")

            raw = _read_payload(handle, path, keyword, count, ecl_type)
            values = _decode_payload(raw, count, ecl_type)
            record = {
                "name": keyword,
                "values": values,
                "type": ecl_type,
                "count": int(count),
                "header_offset": int(header_offset),
            }
            records.append(record)

            bucket = grouped.setdefault(keyword, {"type": ecl_type, "chunks": []})
            bucket["chunks"].append(values)

    output: dict[str, Any] = {"__records__": records}
    for keyword, bucket in grouped.items():
        output[keyword] = {
            "values": _combine_chunks(bucket["chunks"]),
            "type": bucket["type"],
        }
    return output


def _read_fortran_record(handle, path: Path) -> bytes | None:
    marker = handle.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise ValueError(f"{path}: truncated Fortran record marker")
    nbytes = struct.unpack(">i", marker)[0]
    if nbytes < 0:
        raise ValueError(f"{path}: invalid negative Fortran record length {nbytes}")

    payload = handle.read(nbytes)
    trailer = handle.read(4)
    if len(payload) != nbytes or len(trailer) != 4:
        raise ValueError(f"{path}: truncated Fortran record payload")
    trailer_nbytes = struct.unpack(">i", trailer)[0]
    if trailer_nbytes != nbytes:
        raise ValueError(
            f"{path}: mismatched Fortran record markers ({nbytes} != {trailer_nbytes})"
        )
    return payload


def _read_payload(handle, path: Path, keyword: str, count: int, ecl_type: str) -> bytes:
    item_size = _item_size(ecl_type)
    total = count * item_size
    if total == 0:
        return b""

    chunks = []
    remaining = total
    while remaining > 0:
        payload = _read_fortran_record(handle, path)
        if payload is None:
            raise ValueError(f"{path}: unexpected end of file while reading {keyword}")
        if len(payload) > remaining:
            raise ValueError(
                f"{path}: payload chunk for {keyword} is larger than the remaining "
                f"declared byte count"
            )
        chunks.append(payload)
        remaining -= len(payload)
    return b"".join(chunks)


def _item_size(ecl_type: str) -> int:
    ecl_type = ecl_type.strip().upper()
    if ecl_type in _NUMERIC_DTYPES:
        return np.dtype(_NUMERIC_DTYPES[ecl_type][0]).itemsize
    if ecl_type == "CHAR":
        return 8
    if ecl_type.startswith("C") and ecl_type[1:].isdigit():
        return int(ecl_type[1:])
    if ecl_type == "MESS":
        return 1
    raise ValueError(f"Unsupported ECLIPSE output data type {ecl_type!r}")


def _decode_payload(raw: bytes, count: int, ecl_type: str):
    ecl_type = ecl_type.strip().upper()
    if ecl_type in _NUMERIC_DTYPES:
        dtype, native = _NUMERIC_DTYPES[ecl_type]
        return np.frombuffer(raw, dtype=dtype, count=count).astype(native, copy=True)

    if ecl_type == "CHAR" or (ecl_type.startswith("C") and ecl_type[1:].isdigit()):
        width = _item_size(ecl_type)
        return [
            raw[i * width:(i + 1) * width].decode("ascii", errors="replace").strip()
            for i in range(count)
        ]

    if ecl_type == "MESS":
        return raw.decode("ascii", errors="replace")

    return np.frombuffer(raw, dtype=np.uint8).copy()


def _combine_chunks(chunks):
    if not chunks:
        return np.zeros(0)
    first = chunks[0]
    if isinstance(first, np.ndarray):
        if len(chunks) == 1:
            return first
        return np.concatenate(chunks)
    if isinstance(first, list):
        values = []
        for chunk in chunks:
            values.extend(chunk)
        return values
    if isinstance(first, str):
        return "".join(chunks)
    return chunks
