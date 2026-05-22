# Task 3: Optimize with binary separation assumptions

Write an optimization script for the plant from Story 1, but simplify every separation calculation by treating the separation problem as binary propane/propene separation.

The goal is to keep the optimizer structure from `optimizeCodex.py` while replacing expensive four-component separation calculations with a faster and easier-to-debug binary separation approximation. This task should make it possible to test optimization logic, cost logic, and feasibility handling without the full four-component distillation solve dominating runtime.

## Current references

Use these files as references:

- `optimize.py`
- `optimizeCodex.py`
- `optimizetest.py`
- `time_complexity_report.md`

Prefer the structure in `optimizeCodex.py`:

- one function that builds optimizer bounds
- one function that unpacks parameters
- one function that simulates the plant
- one function that evaluates the objective
- one function that reports the final result

The new script should be separate from the current optimizer, for example:

- `optimize_binary_separation.py`

Do not replace `optimizeCodex.py` unless explicitly requested.

## Binary separation model

Whenever a stream enters separation equipment, first construct a proxy stream containing only the propane and propene from the real stream.

The proxy stream should preserve:

- temperature
- pressure
- phase, unless the separation model requires a specific phase
- propane flowrate
- propene flowrate

The proxy stream should set these components to zero:

- `H2`
- `H2O`

Run the separation calculation on the proxy stream.

After the binary separation, reconstruct real outlet streams by combining:

- the propane and propene split produced by the binary separation
- all hydrogen sent to the vapor outlet
- all water sent to the liquid outlet

This means each separation step follows this rule:

- vapor outlet: binary vapor propane/propene + all incoming `H2` + zero incoming `H2O`
- liquid outlet: binary liquid propane/propene + zero incoming `H2` + all incoming `H2O`

The total propane and propene flow across both outlets must match the propane and propene flow entering the separator.

The total hydrogen and water flow across both outlets must match the hydrogen and water flow entering the separator.

## Separation equipment

Apply the binary separation assumption to:

- flash tank 1
- flash tank 2
- the distillation column or simplified column model

For flash tanks, it is acceptable to reuse the existing `FlashTank.run` on a propane/propene-only proxy stream, then reconstruct the real liquid and vapor streams afterward.

For distillation:

- reuse the existing distillation model on a propane/propene-only proxy stream

## Decision variables

The optimizer should consider the same core variables as Task 1 where they remain meaningful:

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

If a decision variable is removed because the binary model does not use it, document that in the script comments and final output.

## Process constraints

A candidate solution is feasible only if:

- reactor calculations do not exceed equilibrium conversion
- separator mass balances close for all four species
- water is routed to the liquid stream after each binary separation
- hydrogen is routed to the vapor stream after each binary separation
- the final distillate contains approximately 82 mol/s propene within a documented tolerance
- the final distillate propene purity is at least 99.5 mol% on a total-stream basis
- temperatures and pressures remain inside physically meaningful bounds

Infeasible candidates should return a large penalty or be rejected cleanly. The optimizer should not crash because a unit operation receives an infeasible input.

## Cost model

Use the same cost basis as `optimizeCodex.py`, preferably SEK/s of operation.

Include costs where they remain meaningful:

- reactor CAPEX
- process oven CAPEX and fuel cost
- flash tank CAPEX, using `FlashTank.calculate_economics` where possible
- distillation utility costs if the existing column model is used
- simplified distillation cost assumptions if a simplified binary split model is used

If a cost is excluded because the simplified binary model does not support it, document that explicitly in the result output.

## Required helper functions

Add small helper functions for the binary separation logic. Suggested functions:

- `make_binary_proxy_stream(stream)`
- `reconstruct_binary_separator_outlets(original_stream, binary_liquid, binary_vapor)`
- `check_species_mass_balance(inlet, outlet_a, outlet_b)`
- `run_binary_flash(flash_tank, inlet_stream)`
- `run_binary_distillation(column_or_split_model, inlet_stream)`

The helper functions should make the binary assumption obvious and testable.

## Required output

The optimizer should print or return:

- best objective value
- cost basis used
- number of reactors
- reactor conversions and inlet temperatures
- flash temperatures and pressures
- distillation settings used by the chosen binary model
- fresh feed rates, if optimized
- distillate total flowrate
- propene flowrate in the distillate
- propene mole fraction in the distillate
- species mass-balance error for each separator
- cost breakdown by equipment type
- constraint violations or penalties, if any
- notes about which parts of the model were simplified

The final report should contain enough information to reproduce and inspect the best candidate.

## Validation checks

Add lightweight checks for the binary separation helpers:

- propane is conserved across reconstructed separator outlets
- propene is conserved across reconstructed separator outlets
- hydrogen is routed only to the vapor outlet
- water is routed only to the liquid outlet
- total flow is conserved across each reconstructed separator
- zero-flow propane/propene proxy streams are handled cleanly

These checks can be implemented as simple assertions, small test functions, or a script-level validation mode.

## Acceptance criteria

The task is done when:

- a separate binary-separation optimizer script exists
- the script runs without unhandled exceptions for a small optimizer configuration
- binary proxy streams are used for separation calculations
- reconstructed separator outlets conserve all four species
- hydrogen is routed to vapor outlets and water is routed to liquid outlets after binary separation
- the optimizer reports the best parameter set and cost breakdown
- the result states which separation assumptions were simplified
- infeasible candidates are handled cleanly by the objective function

## Non-goals for this task

This task does not require a physically rigorous four-component separation model.

This task does not require replacing the full optimizer from Task 1. The binary-separation optimizer is a simplified comparison and debugging tool.

Automatic placement of arbitrary heat exchangers, pumps, and compressors remains out of scope unless those units are already required by the selected simplified flowsheet.
