# Cross-checking PRSTCore against MRST itself

A finite-difference check says a derivative is consistent with its own
residual. It says nothing about whether that residual is the one MRST
computes. `jacobian.m` closes that gap: it builds the same case in MRST,
dumps the residual, and `test_mrst_crosscheck.py` compares it entry by
entry.

Running `jacobian.m` needs MATLAB and an MRST tree. Its output is
committed, so the comparison itself runs without either.

## Running

```
matlab -batch "cd('tests/mrst_crosscheck'); run('jacobian.m')"
pytest tests/test_mrst_crosscheck.py
```

## The result

**Every equation agrees to machine precision, in every cell.**

| block | relative difference |
|-------|--------------------|
| water | 1.9e-14 |
| oil   | 3.0e-12 |
| gas   | 8.6e-13 |

Over all 300 cells of SPE1, including the two holding wells. Flux,
accumulation, PVT, capillary pressure, gravity and the well source terms
are the same computation in both codes, not merely similar.

## Three things that had to be right first

Each of these produced a difference that looked like a defect in
PRSTCore and was not.

**Gravity is off by default in MRST.** `startup.m` does not enable it;
MRST's own hm drivers all open with `gravity on`. Without it the
equilibrated pressure is uniform at the datum value and everything is
out by 0.2%. With it, the initial states match to 6.8e-16 in pressure,
saturation and Rs alike.

**The model must be validated with the driving forces.** Calling
`model.validateModel()` before the wells are known leaves the facility
model empty: `getEquations` then returns three equations instead of
seven, with no well source terms at all. The first comparison run this
way appeared to show a large well-source discrepancy. It was comparing
PRSTCore's wells against no wells.

**The well bhp must be initialised from the same state.** MRST's
`initWellSolAD` sets `bhp = p(first perforated cell) + 5*sign*barsa`,
and PRSTCore does the same. But a wellSol already attached to a state is
kept rather than recomputed, so validating `state0` and then a perturbed
state leaves the bhp at state0's value while PRSTCore's is taken from
the perturbed one. With a perturbation that raised every cell pressure
by 1 bar, the effective drawdown differed by exactly that: the producer
saw 6 bar against 5, the injector 4 against 5, and the source terms came
out in the exact ratios 5/6 and 5/4. Perturbing saturation instead
leaves the well cells' pressure alone and the two agree exactly.

The rational ratios were the clue. A modelling difference does not
produce 0.833333 and 1.250000 across three phases; a difference of one
bar in a five-bar drawdown does.
