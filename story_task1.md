# Task 1: Design the optimizer

Design and implement an optimizer for Story 1 in `story.md`.

The optimizer should search for feasible plant operating parameters that minimize total plant cost while producing approximately 82 mol/s propene in the distillate with at least 99.5 mol% propene purity.

## Current references

Use `optimize.py` and `optimizetest.py` as previous attempts, but do not treat either as final.

Prefer the cleaner separation used in `optimizetest.py`:

- one function that simulates the plant for a given parameter set
- one function that evaluates the optimizer objective
- one function that unpacks optimizer parameters into readable values
- one function that reports the final result

## Decision variables

The optimizer should consider the following variables:

- number of reactors, `n`
- reactor inlet temperatures
- per-reactor propane conversions, with the final reactor conversion calculated to meet the total target conversion
- flash 1 temperature and pressure
- flash 2 temperature and pressure
- distillation feed temperature
- distillation pressure
- distillation reflux ratio
- distillate fraction or distillate flowrate
- distillation stage count and feed stage
- fresh propane and water feed rates, if feed optimization is included

The search ranges should be documented in the optimizer code so the selected bounds can be reviewed and adjusted.

## Process constraints

A candidate solution is feasible only if:

- reactor calculations do not exceed equilibrium conversion
- flash 1 removes a water-rich liquid stream
- flash 2 removes a hydrogen-rich vapor stream
- the distillate contains approximately 82 mol/s propene
- the distillate propene purity is at least 99.5 mol%
- temperatures and pressures remain inside physically meaningful bounds

Infeasible candidates should return a large penalty or be rejected cleanly. The optimizer should not crash because a unit operation receives an infeasible input.

## Cost model

The objective should use one consistent cost basis, preferably SEK/s or annualized SEK/year. The result report must state which cost basis is used.

Include these costs where the current classes support them:

- reactor CAPEX
- process oven CAPEX and fuel cost
- flash tank CAPEX, using `FlashTank.calculate_economics`
- distillation column CAPEX and utility costs, if available
- heat exchanger, compressor, and pump costs only when those classes are implemented and reliable enough for optimization

If a cost is excluded because the class is incomplete or not yet integrated, document that explicitly in the result output or code comments.

## Known issues in previous attempts

- `optimize.py` uses a 95% propene purity check in the distillate, but Story 1 requires 99.5%.
- `optimize.py` uses a fixed fresh feed and fixed distillation stage setup, so it only solves a narrower problem.
- `optimizetest.py` includes fresh feed rates, stage count, and feed stage in the search, but its raw cost currently covers only reactor and oven costs.
- `FlashTank.calculate_economics` exists, but the optimizer attempts do not currently call it.
- `Pump` is currently empty, so pump placement should either be excluded from Task 1 or implemented before being included in the optimizer.
- `Compressor` and `HeatExchanger` contain cost logic, but they should only be added to the optimizer after their utility/cost keys and operating assumptions are verified.

## Required output

The optimizer should print or return:

- best objective value
- cost basis used
- number of reactors
- reactor conversions and inlet temperatures
- flash temperatures and pressures
- column pressure, stage count, feed stage, reflux ratio, and distillate fraction or flowrate
- fresh feed rates, if optimized
- distillate total flowrate
- propene flowrate in the distillate
- propene mole fraction in the distillate
- cost breakdown by equipment type
- constraint violations or penalties, if any

The final report should contain enough information to reproduce and inspect the best candidate.

## Acceptance criteria

The task is done when:

- running the optimizer completes without unhandled exceptions
- the best solution satisfies the 82 mol/s propene target within a documented tolerance
- the best solution satisfies at least 99.5 mol% propene purity
- the code separates plant simulation, objective evaluation, parameter unpacking, and result reporting
- infeasible unit-operation states are handled cleanly by the objective function
- the optimizer reports the best parameter set and a cost breakdown

## Non-goals for this task

Automatic placement of arbitrary heat exchangers, pumps, and compressors between every unit operation is out of scope until those unit classes are complete and cost-compatible. For Task 1, it is acceptable to optimize the main reactor, flash, and distillation operating parameters first, then add auxiliary equipment in a later task.
