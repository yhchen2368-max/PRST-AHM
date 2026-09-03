"""Port of MRST ``getObservedFromFile.m`` (mrst-2026a/hm/utils/observed).

Dispatches a list of measurement files to the reader for their kind and
concatenates the results.

The MATLAB reports a missing file through ``errordlg`` (a GUI dialog) and
then returns; there is no dialog here, so a missing file raises -- silently
returning a short list would make the mismatch look like a good match.
"""

import os as _os

_READERS = {
    'rates': 'readProductionHistory',
    'bhp': 'readProductionHistory',
    'profile': 'readProfileTest',
    'tracer': 'readTracerTest',
    'saturation': 'readSaturationTest',
}


def getObservedFromFile(filename, fn):
    """Read every file in ``filename`` with the reader for kind ``fn``."""
    kind = str(fn).lower()
    if kind not in _READERS:
        raise ValueError('Unsupported data type: %s' % fn)
    reader = _reader(_READERS[kind])

    names = [filename] if isinstance(filename, (str, _os.PathLike)) else list(filename)
    data = []
    for name in names:
        if not _os.path.exists(name):
            raise FileNotFoundError('Cannot find file %s' % name)
        data.extend(reader(name))
    return data


def _reader(name):
    from importlib import import_module
    module = import_module('PRSTCore.hm.utils.observed.%s' % name)
    return getattr(module, name)
