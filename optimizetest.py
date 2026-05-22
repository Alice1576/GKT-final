from classes import *
from testmodul2 import VectorizedDistillationColumnTest
import numpy as np
from scipy.optimize import differential_evolution

LIFETIME      = 10 * 365 * 24 * 3600
CEPCI_2010    = 532.9
CEPCI_2024    = 800.0
USD_TO_SEK    = 9.3
species_list  = ["propane", "propene", "H2", "H2O"]
PROPENE_TARGET = 82.52  # mol/s


# ──────────────────────────────────────────────────────────────────────────────
# Plant simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_plant(params, feed: Stream, n: int, N_stages: int, feed_stage: int, X_target: float):
    """
    Simulate the plant for a given parameter vector and feed stream.
    Returns (distillate, raw_cost, penalty).
    Returns (None, 0.0, 1e6) for any physically infeasible configuration.
    """
    penalty = 0.0

    # Unpack parameter vector
    X          = params[:n - 1]
    T          = params[n - 1 : 2 * n - 1]
    Tf1, Tf2   = params[2 * n - 1], params[2 * n]
    Pf1, Pf2   = params[2 * n + 1], params[2 * n + 2]
    T_dest     = params[2 * n + 3]
    P_dest     = params[2 * n + 4]
    reflux_ratio = params[2 * n + 5]
    dist_frac  = params[2 * n + 6]

    # Per-reactor conversions: last reactor absorbs remainder to hit X_target
    if n > 1:
        Xn    = 1.0 - (1.0 - X_target) / np.prod([1.0 - x for x in X])
        X_all = list(X) + [Xn]
    else:
        X_all = [X_target]

    if any(x <= 0 or x >= 0.999 for x in X_all):
        penalty += 10.0

    # ── Reactors ─────────────────────────────────────────────────────────────
    stream              = feed
    oven                = ProcessOven()
    total_reactor_cost  = 0.0

    for i in range(n):
        stream  = oven.run(stream, T[i])
        reactor = Reactor(1.10325, 1120, None, conversion=X_all[i])
        try:
            stream = reactor.run2(stream)
        except ValueError:
            return None, 0.0, 1e6
        total_reactor_cost += reactor.cost

    # ── Flash tanks ──────────────────────────────────────────────────────────
    flashtank1 = FlashTank(Tf1, Pf1)
    flashtank2 = FlashTank(Tf2, Pf2)

    liquid_flash1, vapor_flash1 = flashtank1.run(stream)

    if liquid_flash1 is None:
        return None, 0.0, 1e6

    # Flash 1: liquid should be water-rich (≥ 90 mol%)
    H2O_frac = liquid_flash1.mole_fraction()["H2O"]
    if H2O_frac < 0.9:
        penalty += max(0.0, 0.9 - H2O_frac) ** 2 * 1e6

    liquid_flash2, vapor_flash2 = flashtank2.run(vapor_flash1)

    if liquid_flash2 is None:
        return None, 0.0, 1e6

    # Flash 2: vapour should be H2-rich (≥ 95 mol%)
    H2_frac = vapor_flash2.mole_fraction()["H2"]
    if H2_frac < 0.95:
        penalty += max(0.0, 0.95 - H2_frac) ** 2 * 1e6

    # ── Distillation column ───────────────────────────────────────────────────
    liquid_flash2.temperature = T_dest

    F_tot               = liquid_flash2.total_flowrate()
    distillate_flowrate = dist_frac * F_tot

    column = VectorizedDistillationColumnTest(
        pressure=P_dest,
        stages=N_stages,
        feed_stage=feed_stage,
        reflux_ratio=reflux_ratio,
        distillate_flowrate=distillate_flowrate,
    )

    distillate, _ = column.run(liquid_flash2)

    if distillate is None:
        return None, 0.0, 1e6

    # Distillate purity: propene ≥ 99.5 mol%
    dist_pur = distillate.flowrates["propene"] / distillate.total_flowrate()
    if dist_pur < 0.995:
        penalty += max(0.0, 0.995 - dist_pur) ** 2 * 1e6

    # ── Cost ─────────────────────────────────────────────────────────────────
    raw_cost = (
        (total_reactor_cost + oven.cost) * (CEPCI_2024 / CEPCI_2010) * USD_TO_SEK
    ) / LIFETIME + oven.gas_cost

    return distillate, raw_cost, penalty


# ──────────────────────────────────────────────────────────────────────────────
# DE objective
# ──────────────────────────────────────────────────────────────────────────────

def cost(params, n: int, N_stages: int, feed_stage: int, X_target: float) -> float:
    """
    Objective function for differential_evolution.
    Builds the feed stream from the optimisation variables, runs the plant,
    and returns total cost + penalties.
    """
    F_propane = params[-2]
    F_water   = params[-1]

    feed = Stream(
        temperature=298.15,
        pressure=1.01325,
        flowrates={
            "propane": F_propane,
            "propene": 0.0,
            "H2":      0.0,
            "H2O":     F_water,
        },
        phase="vapor",
    )

    # plant_params = everything except F_propane and F_water
    plant_params = params[:-2]

    distillate, raw_cost, penalty = run_plant(
        plant_params, feed, n, N_stages, feed_stage, X_target
    )

    if distillate is None:
        return 1e6

    propene_out   = distillate.flowrates["propene"]
    prod_penalty  = (propene_out - PROPENE_TARGET) ** 2 * 1e6

    return raw_cost + penalty + prod_penalty


# ──────────────────────────────────────────────────────────────────────────────
# Result unpacking
# ──────────────────────────────────────────────────────────────────────────────

def unpack_params(params: np.ndarray, n: int, N_stages: int, feed_stage: int) -> dict:
    i = 0
    X = list(params[i : i + (n - 1)]); i += n - 1
    T = list(params[i : i + n]);       i += n
    Tf1, Tf2 = params[i], params[i + 1]; i += 2
    Pf1, Pf2 = params[i], params[i + 1]; i += 2
    Tdest = params[i];  i += 1
    Pdest = params[i];  i += 1
    reflux_ratio = params[i]; i += 1
    dist_frac    = params[i]; i += 1
    F_propane    = params[i]; i += 1
    F_water      = params[i]

    return {
        "X":           X,
        "T":           T,
        "Tf":          (Tf1, Tf2),
        "Pf":          (Pf1, Pf2),
        "Tdest":       Tdest,
        "Pdest":       Pdest,
        "N_stages":    N_stages,
        "feed_stage":  feed_stage,
        "reflux_ratio": reflux_ratio,
        "dist_frac":   dist_frac,
        "F_propane":   F_propane,
        "F_water":     F_water,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    X_target = 0.8
    results  = {}

    # Coarse structural grid — adjust as time allows
    n_values         = [1, 2, 3]
    N_stages_values  = [30, 40, 50, 60]
    # Feed stage at column third-point and mid-point
    feed_stage_fracs = [1/3, 1/2, 2/3]

    for n in n_values:
        for N_stages in N_stages_values:
            for frac in feed_stage_fracs:
                feed_stage = max(2, min(int(frac * N_stages), N_stages - 2))

                bounds = (
                    [(0.2, 0.9)] * (n - 1)   # per-reactor conversion (empty for n=1)
                    + [(700, 950)] * n         # reactor inlet temperatures [K]
                    + [(400, 550)]             # flash 1 temperature [K]
                    + [(273, 400)]             # flash 2 temperature [K]
                    + [(5, 50)]                # flash 1 pressure [bar]
                    + [(1, 100)]               # flash 2 pressure [bar]
                    + [(300, 900)]             # distillation feed temperature [K]
                    + [(1, 40)]                # distillation pressure [bar]
                    + [(1, 10)]                # reflux ratio
                    + [(0.01, 0.99)]           # dist_frac — fraction of column feed to distillate
                    + [(10, 300)]              # F_propane [mol/s]
                    + [(10, 3000)]             # F_water   [mol/s]
                )

                print(f"\n{'='*60}")
                print(f"n={n}, N_stages={N_stages}, feed_stage={feed_stage}")
                print(f"{'='*60}")

                result = differential_evolution(
                    cost,
                    bounds,
                    args=(n, N_stages, feed_stage, X_target),
                    maxiter=300,
                    popsize=12,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    seed=7,
                    workers=-1,          # use all CPU cores
                    updating="deferred",
                    polish=False,
                    disp=True,
                )

                key = (n, N_stages, feed_stage)
                results[key] = {
                    "cost":   result.fun,
                    "params": unpack_params(result.x, n, N_stages, feed_stage),
                }

                print(f"Best cost this run: {result.fun:.6f} SEK/s")

    # ── Find global optimum ───────────────────────────────────────────────────
    best_key  = min(results, key=lambda k: results[k]["cost"])
    best      = results[best_key]
    p         = best["params"]
    opt_cost  = best["cost"]

    print("\n" + "=" * 60)
    print("=== OPTIMALT RESULTAT ===")
    print("=" * 60)
    print(f"Lägst kostnad:        {opt_cost:.6f} SEK/s")
    print(f"Antal reaktorer:      {best_key[0]}")
    print(f"Destillationssteg:    {p['N_stages']}")
    print(f"Feedsteg:             {p['feed_stage']}")

    print(f"\nFärsk propanmatning:  {p['F_propane']:.2f} mol/s")
    print(f"Färsk ångmatning:     {p['F_water']:.2f} mol/s")

    print(f"\nOmsättningsgrader (X):          {p['X']}")
    print(f"Reaktortemperaturer (T) [K]:    {p['T']}")

    print(f"\nFlash 1: T = {p['Tf'][0]:.2f} K,  P = {p['Pf'][0]:.2f} bar")
    print(f"Flash 2: T = {p['Tf'][1]:.2f} K,  P = {p['Pf'][1]:.2f} bar")

    print(f"\nDestillation:")
    print(f"  Matningstemperatur  = {p['Tdest']:.2f} K")
    print(f"  Tryck               = {p['Pdest']:.2f} bar")
    print(f"  Antal steg          = {p['N_stages']}")
    print(f"  Feedsteg            = {p['feed_stage']}")
    print(f"  Refluxkvot          = {p['reflux_ratio']:.3f}")
    print(f"  Destillatandel      = {p['dist_frac']:.3f}")


if __name__ == "__main__":
    main()