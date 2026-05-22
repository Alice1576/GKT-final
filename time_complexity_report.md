# Time Complexity Report: Optimizer

## Scope

This report analyzes the time complexity and measured runtime behavior of the optimizer described in Story 2. The current best reference implementation is `optimizeCodex.py`, with older attempts in `optimize.py` and `optimizetest.py`.

The optimizer uses `scipy.optimize.differential_evolution` for each fixed plant structure. It searches continuous operating parameters for combinations of:

- number of reactors, `R`
- number of distillation stages, `T`
- feed-stage fraction

For each candidate, the objective function calls `simulate_plant`, which constructs and evaluates the plant, then returns cost plus penalties.

## Complexity Variables

The analysis uses these variables:

- `N`: total number of objective-function evaluations
- `R`: number of reactors
- `S`: number of chemical species; currently `S = 4`
- `F`: number of flash tanks; currently `F = 2`
- `T`: number of distillation stages/trays
- `D`: number of optimizer decision variables
- `P`: differential-evolution population multiplier, named `popsize` in SciPy
- `G`: differential-evolution generation count, named `maxiter` in SciPy
- `H`: number of fixed structural cases tested outside the optimizer
- `I_r`: number of reactor integration steps per reactor
- `I_f`: number of flash beta root-solver iterations
- `I_c`: number of nonlinear column-solver iterations
- `M`: number of distillation column unknowns

For the current `optimizeCodex.py` parameterization with feed optimization enabled:

```text
D = (R - 1) + R + 2 + 2 + 1 + 1 + 1 + 1 + 2
D = 2R + 9
```

The distillation model has:

```text
M = liquid compositions + vapor compositions + liquid flows + vapor flows + temperatures + duties
M = S*T + S*T + T + T + T + 2
M = (2S + 3)T + 2
```

With `S = 4`, this is:

```text
M = 11T + 2
```

## Optimizer Structure

The main optimizer is structured as:

1. Loop over fixed reactor counts.
2. Loop over fixed stage counts.
3. Loop over fixed feed-stage fractions.
4. Build bounds for the continuous variables.
5. Run differential evolution.
6. Re-simulate the best candidate for reporting.
7. Select the lowest objective over all structural cases.

The objective function itself performs:

1. Parameter unpacking and conversion calculation.
2. Fresh-feed stream construction.
3. Per-reactor oven heating and reactor simulation.
4. Flash 1 simulation and economics.
5. Flash 2 simulation and economics.
6. Distillation column nonlinear solve.
7. Product constraint checks.
8. Cost and penalty calculation.

## Objective-Function Complexity

### Parameter unpacking

Parameter unpacking copies slices from the parameter vector and computes the final reactor conversion.

```text
O(D + R)
```

Since `D = 2R + 9`, this is effectively linear in reactor count.

### Plant construction

Fresh feed and unit-operation objects are constructed inside every objective call.

```text
O(R + F + 1)
```

The constant `1` is the distillation column object. This cost is small compared with the column solve.

### Reactor simulation

Each reactor runs `Reactor.run2`, which uses a fixed conversion step size of `0.0025` until the target conversion is reached.

```text
O(R * I_r * S)
```

Because conversion is bounded below 1 and the step size is fixed, `I_r` is bounded by roughly `1 / 0.0025 = 400` steps per reactor. With current bounds, reactor cost is practically linear in `R`.

### Process oven calculations

The oven calculates stream enthalpy for each reactor inlet. Stream enthalpy loops over species.

```text
O(R * S)
```

### Flash tank calculations

Each flash tank computes K-values for all species, solves vapor fraction beta, and builds liquid and vapor streams.

```text
O(F * S * I_f)
```

With `F = 2`, `S = 4`, and a bounded scalar root solve, flash runtime is effectively constant for this project.

### Distillation calculations

The distillation column is the dominant cost. One residual evaluation is vectorized across stages and species:

```text
O(T * S)
```

However, the nonlinear solvers repeatedly call the residual. When `root` fails, the implementation falls back to `least_squares`, which estimates a dense numerical Jacobian. Since the column has `M = (2S + 3)T + 2` unknowns, dense finite-difference Jacobian estimation requires approximately `M + 1` residual calls per Jacobian update.

Practical solver cost:

```text
O(I_c * M * T * S)
```

Since `M = O(S*T)`, this becomes:

```text
O(I_c * S^2 * T^2)
```

Dense trust-region linear algebra can add a worse theoretical term:

```text
O(I_c * M^3) = O(I_c * S^3 * T^3)
```

In the measured runs, residual and numerical-derivative work dominated more than dense linear algebra, so the observed behavior is closer to superlinear/quadratic in `T` for the tested range.

### Cost calculations

Cost calculations are simple scalar formulas and dictionary summations.

```text
O(1)
```

If equipment cost coverage expands to arbitrary placed heat exchangers, pumps, and compressors, this may become linear in the number of added auxiliary units.

### Constraint and penalty calculations

Constraint checks use a fixed number of scalar comparisons and a few stream mole fractions.

```text
O(S)
```

This is effectively constant for the current four-species system.

## Full Optimizer Complexity

For one fixed structure, SciPy differential evolution evaluates roughly:

```text
N_structure ~= (G + 1) * P * D
```

where `P` is the SciPy `popsize` multiplier, not the absolute population count. The actual population size is `P * D`.

For all fixed structural cases:

```text
N_total ~= sum over structures ((G + 1) * P * D(R))
```

With the default `OptimizerConfig`:

```text
R values = 1, 2, 3
T values = 20, 30, 40, 60
feed-stage fractions = 1/3, 1/2, 2/3
G = 300
P = 15
D(R) = 2R + 9
```

There are:

```text
H = 3 * 4 * 3 = 36 structural cases
```

The estimated objective evaluations are:

```text
12 * (301 * 15 * 11) for R = 1
+ 12 * (301 * 15 * 13) for R = 2
+ 12 * (301 * 15 * 15) for R = 3
= 2,113,020 objective evaluations
```

Full optimizer work is therefore:

```text
O(N_total * C_eval)
```

where:

```text
C_eval = O(R * I_r * S + F * S * I_f + I_c * S^2 * T^2)
```

Using the worst-case dense solver term:

```text
C_eval = O(R * I_r * S + F * S * I_f + I_c * S^3 * T^3)
```

Parallel execution with `workers=-1` can reduce wall-clock time, but it does not reduce total computational work.

## Profiling Method

A profiling script was added:

```text
time_complexity_profile.py
```

It measures:

- reactor-only probes
- flash-only probes
- distillation-only probes
- full objective evaluations
- early rejection
- cost arithmetic

The command used was:

```powershell
& 'C:\Users\naytt\AppData\Local\Programs\Python\Python313\python.exe' .\time_complexity_profile.py
```

The project `.venv` launcher could not be used in this environment because it failed to import Python's `encodings` module, so the base Python 3.13.11 installation was used.

## Profiling Results

Measured on small repeatable probes:

| Experiment | Changed variable | Tested value | Runtime [s] | Observed trend |
| --- | ---: | ---: | ---: | --- |
| `reactor_only_R1` | `R` | 1 | 0.002199 | Very small |
| `reactor_only_R2` | `R` | 2 | 0.002149 | Still very small |
| `flash_only_F2` | `F` | 2 | 0.000095 | Negligible |
| `distillation_only_T4` | `T` | 4 | 0.041965 | Dominates unit-operation probes |
| `distillation_only_T6` | `T` | 6 | 0.057682 | Increases with `T` |
| `full_objective_T4` | `T` | 4 | 0.897579 | Expensive full solve |
| `full_objective_T6` | `T` | 6 | 1.770842 | Roughly doubles from `T=4` |
| `full_objective_R1` | `R` | 1 | 0.894573 | Column dominates |
| `full_objective_R2` | `R` | 2 | 0.883849 | Reactor count hidden by column cost |
| `full_objective_R3` | `R` | 3 | 0.880444 | Reactor count hidden by column cost |
| `early_reject` | infeasible candidate | equilibrium failure | 0.003876 | Cheap if rejected before column |
| `cost_math_10000x` | cost formulas | 10,000 repeats | 0.005880 | Cost arithmetic is negligible |

The full objective timings used a lower conversion target (`X_target = 0.20`) so the candidate reached the distillation step. With the Story 1 default conversion target (`X_target = 0.80`), many candidates fail at reactor equilibrium and return early.

## cProfile Findings

The cProfile sample confirms the bottleneck:

| Function / area | Cumulative time role |
| --- | --- |
| `VectorizedDistillationColumnTest.run` | Dominant unit-operation call |
| `optimizeCodex.simulate_plant` | Dominated by column solve when candidate reaches distillation |
| `VectorizedDistillationColumnTest.residual_wrapper` and `residuals` | Called thousands of times |
| `scipy.optimize.least_squares` / `trf` | Major cost when `root` does not converge |
| `scipy.optimize._numdiff.approx_derivative` | Major cost from dense numerical Jacobian estimation |
| `_compute_enthalpies` | Major repeated thermodynamic calculation |
| `_compute_K_matrix` | Repeated but smaller than enthalpy calculations |
| `Reactor.run2` | Small compared with distillation |

In the profiled run, `residual_wrapper` was called 9,090 times, and `_compute_enthalpies` was called 99,990 times. This explains why repeated thermodynamic calculations dominate the practical runtime once a candidate reaches the column.

## Scaling Experiments

### Number of reactors

Holding the column at `T = 4`, full objective runtime stayed around 0.88-0.89 s for `R = 1..3`. This does not mean reactors are free; it means the distillation solve is much larger than the reactor work in the tested range.

Expected scaling:

```text
O(R)
```

Observed scaling:

```text
Hidden by column cost for R <= 3
```

### Number of distillation stages

Full objective runtime increased from 0.897579 s at `T = 4` to 1.770842 s at `T = 6`. This is a strong superlinear increase over a small range.

Expected scaling:

```text
Practical: O(I_c * S^2 * T^2)
Worst-case dense solver: O(I_c * S^3 * T^3)
```

Observed scaling:

```text
Superlinear in T
```

### Number of decision variables

Decision variables grow as:

```text
D = 2R + 9
```

Increasing `D` affects runtime mainly through optimizer search cost, because differential evolution population size is `P * D`. It does not strongly affect one plant simulation unless the extra variables change `R`, `T`, or force more candidates to reach the expensive column.

Expected optimizer scaling:

```text
O((G + 1) * P * D * C_eval)
```

### Feasible vs. infeasible candidates

Early reactor-equilibrium failures returned in about 0.003876 s. Full objective evaluations that reached the column took about 0.9-1.8 s in the small stage-count tests.

This means infeasible candidates are cheap only when rejected before distillation. Candidates that pass reactor and flash checks but later fail product purity or production constraints are expensive because they still solve the column.

## Can the Optimizer Handle the Required Problem Size?

The default optimizer settings are too expensive for routine development runs if many candidates reach the distillation column.

Using the measured full objective range:

```text
2,113,020 evaluations * 0.9 s/evaluation ~= 22 days serial
2,113,020 evaluations * 1.77 s/evaluation ~= 43 days serial
```

Parallel workers reduce wall time, but the total work remains very high. The current default configuration may only be practical if most candidates are rejected early or if the run is distributed across many CPU cores.

For development, the optimizer should use much smaller `maxiter`, `popsize`, and structural grids. Full production optimization should only be attempted after the column bottleneck is reduced.

## Bottlenecks

The main bottleneck is the distillation column solve in `VectorizedDistillationColumnTest.run`.

Specific causes:

- the column has `M = 11T + 2` unknowns for four species
- `root` often falls back to `least_squares`
- `least_squares` uses dense numerical derivatives
- every Jacobian update requires many residual evaluations
- residual evaluations repeatedly call `_compute_enthalpies`
- the column is solved even for candidates that may obviously fail purity or production after separation

Secondary bottlenecks:

- `Reactor.run2` repeatedly calls `quad` through `heat_of_reaction`, but it is still much smaller than the column solve in current tests
- repeated object construction happens in every objective call, but it is not the dominant cost

Non-bottlenecks:

- flash calculations
- scalar cost calculations
- parameter unpacking
- result formatting

## Recommended Improvements

1. Reject infeasible candidates before distillation whenever possible.

   Add cheap pre-column checks for feed flow, flash split quality, approximate propene availability, and impossible distillate flow specifications. This is the highest-value improvement because early rejection is about 200-450 times faster than a full column solve in the measured probes.

2. Avoid dense numerical Jacobian work in the column solver.

   Provide an analytical Jacobian, sparse Jacobian structure, or a solver strategy that avoids dense finite-difference derivatives. cProfile shows `approx_derivative` and residual calls dominate the fallback path.

3. Cache repeated thermodynamic calculations inside column residual evaluation.

   `_compute_enthalpies` is called many times with repeated temperatures and species data. Move constant coefficient arrays out of the function and cache reusable temperature-polynomial terms where possible.

4. Reduce column solve frequency during optimization.

   Use a cheaper shortcut separation model during early global search, then run the full `VectorizedDistillationColumnTest` only for promising candidates.

5. Use smaller development optimizer settings.

   For development runs, prefer small values such as `maxiter <= 5`, `popsize <= 3`, and a single `(R, T, feed_stage)` structure. The default `maxiter=300`, `popsize=15`, and 36 structural cases imply about 2.1 million objective evaluations.

6. Narrow parameter bounds after initial exploration.

   Smaller bounds improve differential evolution convergence and reduce the fraction of candidates that waste time in physically poor regions.

7. Reuse or warm-start column solutions.

   Adjacent optimizer candidates often have similar column conditions. Reusing previous column solutions as initial guesses could reduce nonlinear solver iterations, but this requires careful handling with parallel workers.

## Final Complexity Summary

One objective evaluation:

```text
C_eval = O(D + R * I_r * S + F * S * I_f + I_c * S^2 * T^2)
```

Worst-case dense column solve:

```text
C_eval = O(D + R * I_r * S + F * S * I_f + I_c * S^3 * T^3)
```

One full optimizer run:

```text
C_optimizer = O(N_total * C_eval)
```

With differential evolution:

```text
N_total ~= sum over structures ((G + 1) * P * D)
```

For the current default configuration:

```text
N_total ~= 2,113,020 objective evaluations
```

The optimizer is therefore dominated by the number of full column solves, not by reactor count, flash calculations, cost calculations, or parameter unpacking.
