from classes import *
import numpy as np
from scipy.optimize import differential_evolution
from testmodul2 import *

from thermo import find_bubble_temperature

CEPCI2 = 800  # hittade endast värden för 2024, så använde det
CEPCI1 = 532.9
exchange = 9.7  # En dollar är värd ca. 9.7 kr.
lifetime = 10 * 365 * 24 * 3600

PROPENE_TARGET = 82.0  # mol/s propen i destillatet


def cost(params, n, N_stages, feed_stage, X_target, fresh_feed: Stream):
    penalty = 0.0

    X = params[:n - 1]
    T = params[n - 1:2 * n - 1]
    Tf1, Tf2 = params[2 * n - 1:2 * n + 1]
    Pf1, Pf2 = params[2 * n + 1:2 * n + 3]
    Tdest = params[2 * n + 3]
    Pdest = params[2 * n + 4]
    reflux_ratio = params[2 * n + 5]
    # BUG FIX: andelen av feeden som går ut i destillatet var i bounds men
    # packades aldrig upp – nu används den som distillate_fraction nedan.
    distillate_fraction = params[2 * n + 6]

    if n > 1:
        Xn = 1 - (1 - X_target) / np.prod([1 - x for x in X])
        X_all = list(X) + [Xn]
    else:
        X_all = [X_target]

    if any(x <= 0 or x >= 0.999 for x in X_all):
        penalty += 10

    stream = fresh_feed
    oven = ProcessOven()
    total_reactor_cost = 0

    flashtank1 = FlashTank(temperature=Tf1, pressure=Pf1)
    flashtank2 = FlashTank(temperature=Tf2, pressure=Pf2)

    for i in range(n):
        stream = oven.run(stream, T[i])
        reactor = Reactor(1.1035, 1120, None, conversion=X_all[i])

        try:
            stream = reactor.run2(stream)
        except ValueError:
            return 1e6

        total_reactor_cost += reactor.cost

    liquid_flash1, vapor_flash1 = flashtank1.run(stream)

    h2o_frac = liquid_flash1.mole_fraction()["H2O"]
    if h2o_frac < 0.9:
        penalty += max(0, 0.9 - h2o_frac) ** 2 * 1e6

    liquid_flash2, vapor_flash2 = flashtank2.run(vapor_flash1)

    h2_frac = vapor_flash2.mole_fraction()["H2"]
    if h2_frac < 0.9:
        penalty += max(0, 0.9 - h2_frac) ** 2 * 1e6

    liquid_flash2.temperature = Tdest

    # BUG FIX: beräkna destillatflödet från distillate_fraction i stället
    # för det hårdkodade värdet 100.
    distillate_flowrate = distillate_fraction * liquid_flash2.total_flowrate()

    column = VectorizedDistillationColumnTest(
        pressure=Pdest,
        reflux_ratio=reflux_ratio,
        stages=N_stages,
        feed_stage=feed_stage,
        distillate_flowrate=distillate_flowrate,
    )

    distillate_stream, _ = column.run(liquid_flash2)

    # --- Renhetskrav: ≥ 95 mol% propen i destillatet ---
    dist_pur = distillate_stream.flowrates["propene"] / distillate_stream.total_flowrate()
    if dist_pur < 0.95:
        penalty += 10 * 1 / dist_pur

    # --- NYTT: Flödeskrav: 82 mol/s propen i destillatet ---
    # Kvadratisk penalty skalad så att 1 mol/s avvikelse ger en kännbar kostnad.
    propene_flow = distillate_stream.flowrates["propene"]
    propene_error = propene_flow - PROPENE_TARGET
    penalty += 1e4 * propene_error ** 2

    return ((total_reactor_cost + oven.cost) * (CEPCI2 / CEPCI1) * exchange) / lifetime + oven.gas_cost + penalty


def main():
    fresh_feed = Stream(temperature=298.15, pressure=1.01325,
                        flowrates={"propane": 108.569, "propene": 0, "H2": 0, "H2O": 1085.69}, phase="vapor")
    X_target = 0.8

    N_stages = 75
    feed_stage = 42

    results = {}

    for n in range(1, 4):
        bounds = (
                [(0.2, 0.9)] * (n - 1) +  # omsättning på reaktorerna
                [(700, 950)] * n +          # temperatur in till reaktorerna
                [(400, 550)] +              # temperatur på flash 1
                [(273, 400)] +              # temperatur på flash 2
                [(5, 50)] +                 # tryck på flash 1
                [(1, 100)] +                # tryck på flash 2
                [(280, 400)] +              # temperatur på feeden in till dest.
                [(1, 20)] +                 # tryck på destillation
                [(1, 10)] +                 # reflux ratio
                [(0.4, 0.99)]               # andelen av feeden som går ut i destillatet
        )

        result = differential_evolution(
            cost,
            bounds,
            args=(n, N_stages, feed_stage, X_target, fresh_feed),
            maxiter=1000,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=7,
            workers=-1,
            updating="deferred",
            polish=False,
            disp=True,
        )

        results[result.fun] = {
            "n": n,
            "params": unpack_params(result.x, n, N_stages, feed_stage),
        }

    optimal_cost = min(results.keys())
    opt = results[optimal_cost]

    print(f"\n=== RESULTAT ===")
    print(f"Lägst kostnad: {optimal_cost:.2f}")
    print(f"Antal reaktorer: {opt['n']}")

    p = opt["params"]

    print(f"\nOmsättningsgrader (X): {p['X']}")
    print(f"Temperaturer i reaktorer (T): {p['T']}")

    print(f"\nFlash 1: T = {p['Tf'][0]:.2f} K, P = {p['Pf'][0]:.2f}")
    print(f"Flash 2: T = {p['Tf'][1]:.2f} K, P = {p['Pf'][1]:.2f}")

    print(f"\nDestillation:")
    print(f"  T_feed      = {p['Tdest']:.2f} K")
    print(f"  P           = {p['Pdest']:.2f} bar")
    print(f"  Steg        = {p['N_stages']}")
    print(f"  Feed-steg   = {p['feed_stage']}")
    print(f"  Reflux ratio= {p['reflux_ratio']:.2f}")
    print(f"  Dest.-andel = {p['distillate_fraction']:.3f}")


def unpack_params(params, n, N_stages, feed_stage):
    i = 0

    X = list(params[i:i + (n - 1)])
    i += (n - 1)

    T = list(params[i:i + n])
    i += n

    Tf1, Tf2 = params[i], params[i + 1]
    i += 2

    Pf1, Pf2 = params[i], params[i + 1]
    i += 2

    Tdest = params[i];       i += 1
    Pdest = params[i];       i += 1
    reflux_ratio = params[i]; i += 1
    distillate_fraction = params[i]; i += 1

    return {
        "X": X,
        "T": T,
        "Tf": (Tf1, Tf2),
        "Pf": (Pf1, Pf2),
        "Tdest": Tdest,
        "Pdest": Pdest,
        "N_stages": N_stages,
        "feed_stage": feed_stage,
        "reflux_ratio": reflux_ratio,
        "distillate_fraction": distillate_fraction,
    }




if __name__ == "__main__":
    main()