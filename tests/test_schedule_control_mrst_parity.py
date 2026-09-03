"""MRST parity for the SCHEDULE control splitting.

Runs MRST-0's ``readEclipseDeck`` on a deck, dumps ``SCHEDULE.control``
and ``SCHEDULE.step``, and asserts PRSTCore's
:mod:`PRSTCore.deckformat.deckinput.schedule_control` reproduces both --
the control count, the step lengths, the step-to-control map, and every
item of every record.

Companion to ``scripts/export_mrst_schedule_control.m``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The keywords the MATLAB helper dumps, in its order.
_KEYWORDS = ('WELSPECS', 'COMPDAT', 'WCONHIST', 'WCONINJH', 'WCONPROD',
             'WCONINJE', 'WCONINJ', 'WELTARG', 'GRUPTREE')

#: Decks that exercise a different corner of the splitting and still read
#: in MATLAB in well under a minute.  QIEDIE is deliberately absent: its
#: 54080-cell grid makes readEclipseDeck take the better part of an hour,
#: which does not belong in a test run.
_DECKS = (
    'examples/SPE9/SPE9_CP.DATA',          # three controls, WELSPECS+WCONPROD
    'examples/SPE1/SPE1CASE2.DATA',        # one control, WCONPROD+WCONINJE
    'examples/SPE3/SPE3CASE1.DATA',        # two controls, retargeting
    # Norne is the one that earns its runtime: 247 controls and 108692
    # records, WELOPEN, WELTARG, VFPPROD/VFPINJ, TUNING, and the COMPDAT
    # overlap that handleOverlapCompdat duplicates rather than replaces.
    'examples/Norne/NORNE_ATW2013.DATA',
)


def _matlab_path(path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def _mrst_root() -> Path:
    return Path.home() / 'Desktop' / 'github' / 'MRST-0' / 'MRST'


def _format(value) -> str:
    """Format one item the way the MATLAB helper's fprintf does.

    MATLAB's ``%g`` spells the non-finite values ``Inf``/``-Inf``/``NaN``
    where Python's spells them ``inf``/``-inf``/``nan``; a defaulted
    numeric item is MATLAB's empty matrix and Python's ``None``.
    """
    if value is None:
        return '<empty>'
    if isinstance(value, str):
        return value
    value = float(value)
    if np.isnan(value):
        return 'NaN'
    if np.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return '%.10g' % value


def _parse_reference(text):
    """Read the helper's output back into the shapes to compare."""
    ncontrol = nstep = None
    stepval, stepctrl = [], []
    rows = {}
    nrows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        head, _, rest = line.partition(' ')
        if head == 'ncontrol':
            ncontrol = int(rest)
        elif head == 'nstep':
            nstep = int(rest)
        elif head == 'stepval':
            stepval = [float(v) for v in rest.split()]
        elif head == 'stepctrl':
            stepctrl = [int(v) for v in rest.split()]
        elif head == 'nrow':
            cno, kw, count = rest.split()
            nrows[(int(cno), kw)] = int(count)
        elif head == 'row':
            spec, _, items = rest.partition('|')
            cno, kw, _rno = spec.split()
            rows.setdefault((int(cno), kw), []).append(items.split())
    return dict(ncontrol=ncontrol, nstep=nstep, stepval=stepval,
                stepctrl=stepctrl, rows=rows, nrows=nrows)


@pytest.mark.skipif(shutil.which('matlab') is None,
                    reason='MATLAB is required to generate the MRST reference')
@pytest.mark.skipif(not _mrst_root().exists(),
                    reason='MRST-0 checkout not present')
@pytest.mark.parametrize('deck_rel', _DECKS)
def test_schedule_control_matches_mrst(deck_rel, tmp_path: Path):
    deck_path = REPO_ROOT / deck_rel
    if not deck_path.exists():
        pytest.skip('%s not present' % deck_rel)

    reference = tmp_path / 'schedule_control_mrst_ref.txt'
    command = (
        "addpath('%s'); export_mrst_schedule_control('%s', '%s', '%s')"
        % (_matlab_path(REPO_ROOT / 'scripts'), _matlab_path(deck_path),
           _matlab_path(reference), _matlab_path(_mrst_root()))
    )
    subprocess.run([shutil.which('matlab'), '-batch', command],
                   cwd=str(REPO_ROOT), check=True)
    ref = _parse_reference(reference.read_text())

    schedule = read_eclipse_deck(str(deck_path))['SCHEDULE']
    control = schedule.get('control') or []
    step = schedule.get('step') or {'val': [], 'control': []}

    assert len(control) == ref['ncontrol']
    assert len(step['val']) == ref['nstep']
    assert np.allclose(step['val'], ref['stepval'], rtol=0, atol=1e-9)

    # MRST numbers controls from 1, PRSTCore from 0.
    assert [c + 1 for c in step['control']] == ref['stepctrl']

    for cno, entry in enumerate(control, start=1):
        for kw in _KEYWORDS:
            expected = ref['rows'].get((cno, kw), [])
            got = entry.get(kw) or []
            assert len(got) == ref['nrows'].get((cno, kw), 0), (
                'control %d %s: %d rows, MRST has %d'
                % (cno, kw, len(got), ref['nrows'].get((cno, kw), 0)))
            for rno, (want, have) in enumerate(zip(expected, got), start=1):
                assert [_format(v) for v in have] == want, (
                    'control %d %s row %d' % (cno, kw, rno))
