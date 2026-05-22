# Task 2: Analyze optimizer time complexity

Analyze the time complexity and runtime behavior of the optimizer from Story 1 in `story.md`.

The goal is to understand how the optimizer scales as the plant model and search space grow, and to identify which parts of the optimization workflow dominate runtime.

## Current references

Use the current optimizer implementation and previous attempts as references:

- `optimize.py`
- `optimizeCodex.py`
- `optimizetest.py`
- `profile_me.py`

Treat these files as starting points for analysis, not as final proof of complexity.

## Complexity variables

Define the variables that affect runtime before analyzing the optimizer.

At minimum, describe runtime in terms of:

- number of candidate evaluations, `N`
- number of reactors, `R`
- number of chemical species, `S`
- number of flash tanks, `F`
- number of distillation stages or trays, `T`
- number of optimizer decision variables, `D`
- number of objective-function iterations or generations, depending on optimizer type

If the optimizer uses a population-based method, also include:

- population size, `P`
- number of generations, `G`

If the optimizer uses a local numerical method, also include:

- number of function evaluations required by the solver
- cost of constraint checks
- cost of numerical derivative estimation, if used

## Work breakdown

Break the optimizer into measurable parts:

- parameter unpacking
- plant construction
- reactor simulation
- process oven calculations
- flash tank calculations
- distillation column calculations
- cost calculations
- constraint and penalty calculations
- optimizer search overhead
- result reporting

For each part, estimate the expected Big O cost and explain which input variable controls it.

## Expected analysis

The analysis should answer:

- how many times the objective function is called during one optimizer run
- which unit operation is the main runtime bottleneck
- whether runtime grows linearly, polynomially, or worse with the number of reactors and distillation stages
- whether increasing the number of decision variables mainly increases simulation cost or optimizer search cost
- whether infeasible candidates are cheap or expensive to reject
- whether repeated thermodynamic calculations dominate runtime
- whether any calculations can be cached or vectorized

## Profiling plan

Create or update a profiling script that measures the runtime of:

- one full objective-function evaluation
- reactor calculations only
- flash tank calculations only
- distillation calculations only
- cost calculations only
- one complete optimizer run with fixed settings

Use repeatable inputs so results can be compared after code changes.

The profiler should report:

- total runtime
- average objective evaluation time
- number of objective evaluations
- time spent per major unit operation, if measurable
- best objective value found during the profiled run

## Scaling experiments

Run timing experiments where one problem-size variable changes at a time.

Recommended experiments:

- vary number of reactors, `R`
- vary number of distillation stages, `T`
- vary number of optimizer decision variables, `D`
- vary optimizer population size or iteration limit
- compare feasible candidates with infeasible candidates

Keep all other settings fixed while measuring each variable.

Record results in a small table with:

- experiment name
- changed variable
- tested values
- objective evaluations
- total runtime
- average runtime per evaluation
- observed scaling trend

## Required output

The final time complexity report should include:

- a short explanation of the optimizer structure
- the selected complexity variables
- Big O estimate for one objective-function evaluation
- Big O estimate for one full optimizer run
- profiling results from the current implementation
- scaling experiment results
- identified runtime bottlenecks
- recommended improvements

## Recommended improvements to evaluate

Consider whether the optimizer can be improved by:

- rejecting infeasible candidates earlier
- caching repeated thermodynamic calculations
- vectorizing repeated calculations
- reducing unnecessary object construction inside the objective function
- simplifying cost calculations for clearly infeasible candidates
- limiting expensive distillation calculations until reactor and flash constraints pass
- using narrower parameter bounds
- reducing population size or iteration count during development runs

Only implement an improvement if the profiling results show that it targets a real bottleneck.

## Acceptance criteria

The task is done when:

- the optimizer runtime is expressed using clear complexity variables
- one objective-function evaluation has a documented Big O estimate
- one full optimizer run has a documented Big O estimate
- profiling data identifies the slowest parts of the optimizer
- scaling experiments show how runtime changes with larger problem sizes
- the final report explains whether the optimizer can handle the required problem size within a reasonable time
- at least one concrete optimization opportunity is identified and justified by profiling data

## Non-goals for this task

This task does not require rewriting the optimizer from scratch.

Large algorithmic changes, such as replacing the optimizer method entirely, should only be done after the current implementation has been measured and the bottleneck is understood.
