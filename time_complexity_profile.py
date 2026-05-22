from __future__ import annotations

import contextlib
import os
import time
from statistics import mean

import optimizeCodex as opt
from classes import FlashTank, ProcessOven, Reactor, Stream
from testmodul2 import VectorizedDistillationColumnTest


FULL_EVAL_PARAMS = [850, 450, 330, 10, 20, 330, 5, 2, 0.5, 110, 1000]
FULL_EVAL_PARAMS_R2 = [0.10, 850, 850, 450, 330, 10, 20, 330, 5, 2, 0.5, 110, 1000]
FULL_EVAL_PARAMS_R3 = [0.07, 0.07, 850, 850, 850, 450, 330, 10, 20, 330, 5, 2, 0.5, 110, 1000]
X_TARGET_FOR_FULL_EVAL = 0.20


def time_call(label: str, func, repeats: int = 1) -> tuple[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return label, mean(samples)


def reactor_probe(n: int = 1, conversion: float = 0.20) -> None:
    stream = Stream(
        temperature=298.15,
        pressure=1.01325,
        phase="vapor",
        flowrates={"propane": 110.0, "propene": 0.0, "H2": 0.0, "H2O": 1000.0},
    )
    oven = ProcessOven()
    for _ in range(n):
        stream = oven.run(stream, 850.0)
        reactor = Reactor(1.10325, 1120.0, catalyst_mass=None, conversion=conversion)
        stream = reactor.run2(stream)


def flash_probe() -> None:
    stream = Stream(
        temperature=600.0,
        pressure=1.10325,
        phase="vapor",
        flowrates={"propane": 88.0, "propene": 22.0, "H2": 22.0, "H2O": 1000.0},
    )
    flash1 = FlashTank(450.0, 10.0)
    liquid1, vapor1 = flash1.run(stream)
    flash1.calculate_economics(liquid1)
    flash2 = FlashTank(330.0, 20.0)
    liquid2, vapor2 = flash2.run(vapor1)
    flash2.calculate_economics(liquid2)


def column_probe(stages: int) -> None:
    feed = Stream(
        temperature=330.0,
        pressure=5.0,
        phase="liquid",
        flowrates={"propane": 50.0, "propene": 40.0, "H2": 1.0, "H2O": 1.0},
    )
    column = VectorizedDistillationColumnTest(
        pressure=5.0,
        stages=stages,
        feed_stage=opt.feed_stage_from_fraction(stages, 0.5),
        reflux_ratio=2.0,
        distillate_flowrate=0.5 * feed.total_flowrate(),
    )
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            column.run(feed)


def full_objective_probe(stages: int) -> None:
    opt.simulate_plant(
        params=FULL_EVAL_PARAMS,
        n=1,
        n_stages=stages,
        feed_stage=opt.feed_stage_from_fraction(stages, 0.5),
        include_feed_optimization=True,
        x_target=X_TARGET_FOR_FULL_EVAL,
        suppress_column_output=True,
    )


def full_objective_reactor_probe(n: int) -> None:
    params = {
        1: FULL_EVAL_PARAMS,
        2: FULL_EVAL_PARAMS_R2,
        3: FULL_EVAL_PARAMS_R3,
    }[n]
    opt.simulate_plant(
        params=params,
        n=n,
        n_stages=4,
        feed_stage=2,
        include_feed_optimization=True,
        x_target=X_TARGET_FOR_FULL_EVAL,
        suppress_column_output=True,
    )


def early_reject_probe() -> None:
    opt.simulate_plant(
        params=[850, 450, 330, 10, 20, 330, 5, 2, 0.5, 110, 1000],
        n=1,
        n_stages=6,
        feed_stage=3,
        include_feed_optimization=True,
        x_target=0.80,
        suppress_column_output=True,
    )


def cost_probe() -> None:
    for _ in range(10000):
        reactor_capex = opt.annualized_capex_sek_per_s(opt.usd_2010_to_installed_sek(250000.0))
        oven_capex = opt.annualized_capex_sek_per_s(opt.usd_2010_to_installed_sek(1000000.0))
        steam = opt.utility_cost_sek_per_s(2500.0, "steam")
        cooling = opt.utility_cost_sek_per_s(2000.0, "cooling_water")
        _ = reactor_capex + oven_capex + steam + cooling


def main() -> None:
    rows = [
        time_call("reactor_only_R1", lambda: reactor_probe(1), repeats=3),
        time_call("reactor_only_R2", lambda: reactor_probe(2, conversion=0.10), repeats=3),
        time_call("flash_only_F2", flash_probe, repeats=5),
        time_call("distillation_only_T4", lambda: column_probe(4)),
        time_call("distillation_only_T6", lambda: column_probe(6)),
        time_call("full_objective_T4", lambda: full_objective_probe(4)),
        time_call("full_objective_T6", lambda: full_objective_probe(6)),
        time_call("full_objective_R1", lambda: full_objective_reactor_probe(1)),
        time_call("full_objective_R2", lambda: full_objective_reactor_probe(2)),
        time_call("full_objective_R3", lambda: full_objective_reactor_probe(3)),
        time_call("early_reject", early_reject_probe, repeats=5),
        time_call("cost_math_10000x", cost_probe),
    ]

    print("experiment,seconds")
    for label, seconds in rows:
        print(f"{label},{seconds:.6f}")


if __name__ == "__main__":
    main()
