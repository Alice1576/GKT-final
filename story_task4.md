# Task 4: Rewrite `optimizeCodexBinary.py` with saturation-based flash preparation

Rewrite the binary optimizer so flash temperature and flash pressure are no longer independent decision variables. Each flash should be prepared by compressing to the selected pressure, moving the stream to the relevant saturation temperature from the Antoine equation, and condensing the component that defines that saturation temperature before the flash separation is applied.

The rest of the process should remain as close as possible to the current `optimizeCodexBinary.py` workflow.

## Current references

Use these files as references:

- `optimizeCodexBinary.py`
- `classes/Compressor.py`
- `classes/HeatExchanger.py`
- `classes/FlashTank.py`
- `classes/Stream.py`
- `thermo.py`

Keep the existing optimizer structure where possible:

- one function that builds optimizer bounds
- one function that unpacks optimizer parameters
- one function that simulates the plant
- one function that evaluates the objective
- one function that reports the final result

## Flash preparation model

Before every flash, apply this sequence:

1. Raise the stream pressure to the target flash pressure using the `Compressor` class when compression is required.
2. Cool or heat the stream to the pure-component saturation temperature at that flash pressure using the Antoine equation.
3. Add a condenser modeled as a heat exchanger with `U = 1000 W/(m^2 K)` that removes the heat of vaporization for the component used to define the saturation temperature. The heat of vaporization can be calculated using the Watson correlation in the 'Stream' class
4. Send the prepared stream into the flash separation step.

For flash 1:

- use the saturation temperature of water, `H2O`
- condense all incoming water
- route all condensed water to the liquid outlet
- the flash 1 liquid outlet should be 100 mol% water
- no propane, propene, or hydrogen should leave with the water outlet except for negligible numerical tolerance

For flash 2:

- use the saturation temperature of propene
- condense the propane and propene together into the liquid outlet
- route all hydrogen to the vapor outlet
- the flash 2 vapor outlet should be 100 mol% hydrogen
- no propane, propene, or water should leave with the hydrogen outlet except for negligible numerical tolerance

The saturation temperature should be calculated by inverting the existing Antoine relation in `thermo.py`, where `find_saturation_pressure(T, A, B, C)` returns pressure in mmHg. Convert the selected flash pressure from bar to mmHg before solving for temperature.
The 'run-method' should be used to calculate cost when necessary, but its stream outputs should NOT be used as they are based on rigorous models not appropriate for this task. Instead, streams can be constructed as necessary according to the assumptions given.

## Recycle and reheating

The water removed in flash 1 should be reheated to the same temperature as the feed entering the first reactor.

Document whether that reheated water stream is:

- recycled back to the reactor feed,
- reported as a separated process stream, or
- handled in another way that matches the current flowsheet.

The choice must be explicit in the code comments and final result output.

## Decision variables

Keep pressure as the flash decision variable:

- flash 1 pressure
- flash 2 pressure

Remove flash temperatures from the optimizer decision variables. They should be derived from:

- flash 1 pressure and water saturation temperature
- flash 2 pressure and propene saturation temperature

Keep the other optimizer variables from `optimizeCodexBinary.py` where they remain meaningful:

- number of reactors, `n`
- reactor inlet temperatures
- per-reactor propane conversions, with the final reactor conversion calculated to meet the total target conversion
- distillation feed temperature, if still used after flash 2
- distillation pressure
- distillation reflux ratio
- distillate fraction or distillate flowrate
- distillation stage count and feed stage
- fresh propane feed rate, if feed optimization is enabled
- fresh water feed rate or fixed water-to-propane ratio, matching the current optimizer configuration

If any variable is removed or made fixed by the new flash model, document that in the script comments and result output.

## Helper functions

Add small helper functions so the new flash logic is easy to inspect and test. Suggested functions:

- `saturation_temperature_from_antoine(species, pressure_bar)`
- `prepare_stream_for_saturation_flash(stream, pressure_bar, saturation_species, label)`
- `condense_component(stream, component, heat_exchanger_u, label)`
- `run_water_removal_flash(inlet_stream, pressure_bar)`
- `run_hydrogen_removal_flash(inlet_stream, pressure_bar)`
- `reheat_water_stream(water_stream, target_temperature)`

These helpers should return enough metadata to report compressor costs, heat exchanger costs, heat duties, saturation temperatures, and mass-balance errors.

## Cost model

Use the same cost basis as the current binary optimizer, preferably SEK/s of operation.

Include these costs where the classes support them:

- reactor CAPEX
- process oven CAPEX and fuel cost
- compressor CAPEX and utility costs for pressure increases before flashes
- flash tank CAPEX using `FlashTank.calculate_economics`
- condenser or heat exchanger CAPEX and utility costs for saturation cooling/heating and component condensation
- distillation column costs, if the current downstream binary distillation model remains in use

If the existing `HeatExchanger` class cannot directly represent latent-heat removal, add a clearly documented wrapper or simplified cost calculation rather than hiding the cost.

## Process constraints

A candidate solution is feasible only if:

- reactor calculations do not exceed equilibrium conversion
- all stream temperatures and pressures remain physically meaningful
- flash 1 saturation temperature is calculated from water at the selected flash 1 pressure
- flash 2 saturation temperature is calculated from propene at the selected flash 2 pressure
- flash 1 removes all water as a 100 mol% liquid water stream within tolerance
- flash 2 removes all hydrogen as a 100 mol% vapor hydrogen stream within tolerance
- propane and propene from flash 2 leave together as liquid within tolerance
- species mass balances close across both flash sections
- the distillate contains approximately 82 mol/s propene within the existing documented tolerance
- the distillate propene purity is at least 99.5 mol%

Infeasible candidates should return a large penalty or be rejected cleanly. The optimizer should not crash because a unit operation receives an infeasible input.

## Required output

The optimizer should print or return:

- best objective value
- cost basis used
- number of reactors
- reactor conversions and inlet temperatures
- flash pressures
- calculated flash saturation temperatures
- flash 1 water outlet flowrate and purity
- flash 2 hydrogen outlet flowrate and purity
- flash 2 propane/propene liquid outlet composition
- compressor duties, costs, and pressure changes
- condenser or heat exchanger duties, areas, and costs
- reheated water stream temperature and flowrate
- distillation settings, if distillation remains in the flowsheet
- fresh feed rates, if optimized
- distillate total flowrate
- propene flowrate in the distillate
- propene mole fraction in the distillate
- species mass-balance error for each flash section
- cost breakdown by equipment type
- constraint violations or penalties, if any
- notes about which flash temperatures were derived rather than optimized

The final report should contain enough information to reproduce and inspect the best candidate.

## Validation checks

Add lightweight checks for the new flash preparation helpers:

- Antoine inversion returns a saturation temperature whose saturation pressure matches the requested pressure within tolerance
- flash 1 water outlet contains all inlet water and no non-water species within tolerance
- flash 2 hydrogen vapor outlet contains all inlet hydrogen and no non-hydrogen species within tolerance
- flash 2 liquid outlet contains all inlet propane and propene within tolerance
- total species mass balance closes across each flash section
- no compressor is used when the inlet stream pressure is already at or above the target pressure
- compressor use on non-vapor streams is rejected or handled explicitly
- zero-flow component cases do not crash

These checks can be implemented as simple assertions, small test functions, or a script-level validation mode.

## Acceptance criteria

The task is done when:

- `optimizeCodexBinary.py` no longer optimizes flash temperatures
- flash temperatures are derived from Antoine saturation temperatures at the selected flash pressures
- flash 1 removes water as a 100 mol% liquid water stream within tolerance
- flash 2 removes hydrogen as a 100 mol% vapor hydrogen stream within tolerance
- propane and propene from flash 2 leave together as liquid within tolerance
- the water removed in flash 1 is reheated to the first reactor feed temperature
- compressor and condenser or heat exchanger costs are included or explicitly documented if excluded
- the optimizer runs without unhandled exceptions for a small optimizer configuration
- infeasible candidates are handled cleanly by the objective function
- the result report includes the derived flash temperatures, outlet purities, mass-balance errors, and cost breakdown

## Non-goals for this task

This task does not require a rigorous multi-component flash calculation after the forced condensation assumptions are applied.

This task does not require replacing the downstream binary distillation model unless the new flash outputs make a small interface adjustment necessary.

Automatic placement of arbitrary heat exchangers, pumps, and compressors outside the required flash preparation steps remains out of scope.
