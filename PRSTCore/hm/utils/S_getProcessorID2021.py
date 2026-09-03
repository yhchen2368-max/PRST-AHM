"""Port of MRST ``S_getProcessorID2021.m`` (mrst-2026a/hm/utils).

Reads the CPU's ProcessorId, used by the FAHM app for node-locked
licensing. Windows-only in the original -- it shells out to PowerShell's
``Get-CimInstance Win32_Processor`` and takes the fourth line of the
result, which is where the value lands under that cmdlet's table header.
"""

import subprocess as _subprocess

_COMMAND = ('powershell -command "Get-CimInstance Win32_Processor | '
            'Select-Object ProcessorId"')


def S_getProcessorID2021():
    """Return the processor ID string, or ``''`` when unavailable.

    The MATLAB indexes ``fields{4}`` unconditionally and errors on any
    other platform or output shape; the port returns an empty string
    instead, since a missing ID is not a reason to abort a history match.
    """
    try:
        result = _subprocess.run(_COMMAND, shell=True, capture_output=True,
                                 text=True, timeout=30)
    except (OSError, _subprocess.SubprocessError):
        return ''
    lines = [line.strip() for line in (result.stdout or '').splitlines()]
    lines = [line for line in lines if line]
    # Get-CimInstance prints a blank line, the header, an underline, then
    # the value -- the fourth line of the raw output.
    raw = [line.strip() for line in (result.stdout or '').splitlines()]
    if len(raw) >= 4 and raw[3]:
        return raw[3]
    return lines[-1] if lines else ''
