"""PRSTCore against MRST's own numbers, on the same case.

A finite-difference check tells you a derivative is consistent with its
own residual. It cannot tell you the residual is the one MRST computes.
These compare against MRST output recorded in ``tests/mrst_crosscheck``,
so the comparison runs without MATLAB; regenerating it needs MATLAB and
is described in that directory's README.

The comparison state perturbs **saturation only**. Perturbing pressure
would move the well cells, and a wellSol carried from an earlier state
keeps its bhp -- so the two codes would be comparing different well
drawdowns rather than different source terms. See the README.
"""

import csv
import os

import numpy as np
import pytest

from PRSTCore.ad_core.simulators import adjoint_verification as V

DECK = 'examples/SpE1/SPE1CASE2.DATA'
HERE = os.path.join('tests', 'mrst_crosscheck')
NC = 300

#: Cell 0 holds the injector, cell 299 the producer.
WELL_CELLS = (0, 299)


def _mrst():
    path = os.path.join(HERE, 'MRST_RAW.csv')
    if not os.path.exists(path):
        pytest.skip('MRST output not recorded')
    with open(path) as handle:
        return np.array([float(row[0]) for row in csv.reader(handle) if row])


@pytest.fixture(scope='module')
def case():
    if not os.path.exists(DECK):
        pytest.skip('SPE1 deck not present')
    model, state0, forces, dt = V.build_case(DECK)

    state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in state0.items()}
    state['sW'] = np.clip(np.asarray(state0['sW'], dtype=float) + 0.01, 0, 1)
    state['time'] = float(state0.get('time', 0.0)) + 1.0
    return model, state0, state, dt, forces


def _ours(case):
    model, state0, state, dt, forces = case
    return V.residual(model, state0, state, dt, forces)


def _block(vector, name):
    return {'water': vector[0:NC], 'oil': vector[NC:2 * NC],
            'gas': vector[2 * NC:3 * NC]}[name]


# ------------------------------------------------------- the comparison --

def test_both_codes_produce_the_same_number_of_equations(case):
    """908 = three conservation equations over 300 cells, plus four well
    equations for each of two wells. Three equations instead of seven
    means MRST was built without its wells and the comparison is
    meaningless."""
    assert _ours(case).size == _mrst().size == 908


@pytest.mark.parametrize('block', ['water', 'oil', 'gas'])
def test_the_residual_matches_mrst_everywhere(case, block):
    """Every cell, including the two holding wells."""
    ours = _block(_ours(case), block)
    theirs = _block(_mrst(), block)
    scale = max(float(np.max(np.abs(theirs))), 1e-300)
    assert np.max(np.abs(ours - theirs)) / scale < 1e-10, block


def test_no_cell_disagrees(case):
    """Stated as a count so a later change that moves one cell is visible
    even if the block norm still passes."""
    ours, theirs = _ours(case)[:3 * NC], _mrst()[:3 * NC]
    scale = max(float(np.max(np.abs(theirs))), 1e-300)
    differing = np.flatnonzero(np.abs(ours - theirs) > 1e-9 * scale)
    assert differing.size == 0, (differing % NC).tolist()


def test_the_well_cells_agree_too(case):
    """The source terms were the last thing to be confirmed, and only
    after MRST was validated with its driving forces and both codes
    initialised the well bhp from the same state."""
    ours, theirs = _ours(case), _mrst()
    for offset in (0, NC, 2 * NC):
        for cell in WELL_CELLS:
            i = offset + cell
            scale = max(abs(theirs[i]), 1e-12)
            assert abs(ours[i] - theirs[i]) / scale < 1e-8, (offset, cell)


def test_the_well_cells_carry_a_real_source_term(case):
    """Otherwise the agreement above would be agreement about nothing."""
    model, state0, state, dt, forces = case
    with_wells = V.residual(model, state0, state, dt, forces)[:3 * NC]

    # The model prefers a state-attached well list over the driving
    # forces, so both have to be cleared to get a well-free residual.
    bare0 = {k: v for k, v in state0.items() if k != 'facility_wells'}
    bare = {k: v for k, v in state.items() if k != 'facility_wells'}
    without = V.residual(model, bare0, bare, dt, {'W': []})[:3 * NC]

    source = with_wells - without
    assert np.max(np.abs(source)) > 0.1
    assert set(np.flatnonzero(np.abs(source) > 1e-12).tolist()) \
        <= {c + o for o in (0, NC, 2 * NC) for c in WELL_CELLS}


# ------------------------------------------------------ the preconditions --

def test_gravity_is_on(case):
    """MRST's startup does not enable it -- its own hm drivers all say
    `gravity on` first -- and without it the equilibrated pressure is
    flat at the datum value, 0.2% away. PRSTCore applies it unasked, so
    the two only agree once MRST is told to."""
    model = case[0]
    g = np.asarray(getattr(model, 'gravity', [0, 0, 9.80665]), dtype=float)
    assert abs(g[-1]) > 9.0


def test_the_comparison_leaves_the_well_pressures_alone(case):
    """A pressure perturbation moves the well cells, and a wellSol
    carried from an earlier state keeps its bhp -- so the two codes would
    differ in drawdown rather than in physics. That produced source terms
    in the exact ratios 5/6 and 5/4 before it was understood."""
    _, state0, state, _, _ = case
    assert np.allclose(state['pressure'], state0['pressure'])
    assert not np.allclose(state['sW'], state0['sW'])
