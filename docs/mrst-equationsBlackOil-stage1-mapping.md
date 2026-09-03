# MRST equationsBlackOil Stage-1 Mapping (SPE9-first)

This checklist maps current Python implementation to MRST reference functions.

## Reference Anchors

- MRST equation entry: `autodiff/ad-blackoil/models/GenericBlackOilModel.m:getModelEquations`
- MRST equation utility: `autodiff/ad-blackoil/utils/equationsBlackOil.m`
- MRST facility coupling: `autodiff/ad-core/models/facilities/FacilityModel.m`
- MRST per-well contributions: `autodiff/ad-core/models/facilities/computeWellContributionsSingleWell.m`

## Variable And Equation Mapping

### Primary variables

- MRST: pressure, saturations, optional Rs/Rv and facility variables.
- PRSTCore stage-1:
  - pressure and water saturation are solved unknowns.
  - Rs/Rv are pressure-dependent state functions (computed, not yet solved as independent unknowns).

### Equation blocks

- MRST conservation blocks (W/O/G components):
  - PRSTCore stage-1 equivalent:
    - pressure equation: total balance surrogate using compressibility + phase flux divergence + well sources.
    - water equation: water accumulation + water flux divergence + well source.
- MRST facility equations appended to reservoir equations:
  - PRSTCore stage-1: simplified well source insertion only (full facility equations pending).

## Implemented In Stage-1

- [x] Deck-driven PVT evaluator object attached during initialization.
- [x] PVTO-driven oil `Bo(p), mu_o(p), Rs(p)` evaluation.
- [x] PVDG-driven gas `Bg(p), mu_g(p)` evaluation.
- [x] PVTW-driven water `Bw(p), mu_w(p)` evaluation.
- [x] `Rv(p)` exposed (defaults to zero when no volatile-oil table is present).
- [x] Reservoir equation path consumes PVT in phase mobilities.
- [x] Reservoir source terms convert rate controls with formation volume factors (first-pass scaling).
- [x] State carries `rs` and `rv` arrays each nonlinear assembly.

## Explicitly Deferred To Stage-2+

- [ ] Full three-component conservation equations with dissolved/vaporized transfer terms in residuals.
- [ ] Rs/Rv as fully coupled primary variables with switching logic.
- [ ] Gravity potential in phase fluxes (depth-based potential difference).
- [ ] Complete FacilityModel equations and control switching parity.
- [ ] Boundary/source treatment parity with dissolved component matrix.
- [ ] CNV/well convergence metric parity.

## SPE9 Scope Of Stage-1

- SPE9 keywords used by this stage: `PVTO`, `PVDG`, `PVTW`, `DENSITY`, `ROCK`, `SWOF`, `SGOF`.
- The implementation is designed to be robust for SPE9 table layout and provides a stable baseline for subsequent 1:1 parity work.
