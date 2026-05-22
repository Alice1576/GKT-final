# Task 6: Use `HeatExchanger` for flash heat-exchanger area and cost calculations

Rewrite the heat-exchanger costing in `optimizeCodexBinary.py` so area, CAPEX, and OPEX are calculated through `classes/HeatExchanger.py` rather than through a local shortcut formula in the optimizer.

The stream transformations may still be handled manually in `optimizeCodexBinary.py` where needed. This is especially important for latent heat removal, because condensation is currently represented by forced species routing rather than by `HeatExchanger.run`.

## Required behavior

Use the `HeatExchanger` class for:

- flash 1 sensible cooling or heating to the water saturation temperature
- flash 1 water condensation duty
- reheating the flash 1 water stream to the first reactor feed temperature
- flash 2 sensible cooling or heating to the propene saturation temperature
- flash 2 propene condensation duty

Condensers must not use an arbitrary temperature driving force. Condenser area should be based on cooling water entering at 14 deg C and leaving at 20 deg C. Use the condensing temperature from the relevant saturation calculation and calculate the condenser LMTD from those two cooling-water terminal temperatures.

If the condensing temperature is at or below the cooling-water outlet temperature, the candidate should be rejected or penalized cleanly because the specified cooling water cannot provide the required condensation.

For latent heat duties, add or use a duty-based method on `HeatExchanger` so the class can calculate:

- heat duty
- area
- CAPEX
- annual OPEX
- cost converted to SEK/s in the optimizer

The optimizer may still keep the forced outlet streams created in Task 4. The key requirement is that heat-exchanger area and cost metadata come from the heat-exchanger class.

## Acceptance criteria

The task is done when:

- `optimizeCodexBinary.py` imports and uses `HeatExchanger`
- the old local heat-exchanger area/CAPEX/OPEX shortcut is removed or replaced by calls to `HeatExchanger`
- latent duties can be costed without relying on `HeatExchanger.run` to change stream composition
- condensation heat exchangers use 14 deg C to 20 deg C cooling water instead of an arbitrary fixed temperature difference
- result metadata reports that heat-exchanger calculations came from `HeatExchanger`
- validation and a small optimizer run complete without unhandled exceptions
