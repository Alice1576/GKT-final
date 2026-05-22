from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import os
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple

import numpy as np
from scipy.optimize import differential_evolution

import ekonomi as eko
from classes import Compressor, FlashTank, HeatExchanger, ProcessOven, Reactor, Stream
from classes.VectorizedColumn import VectorizedDistillationColumn
from iterate_distillation_guess import GuessIterator


SPECIES = ("propane", "propene", "H2", "H2O")

PROPENE_TARGET_MOL_S = 82.0
PROPENE_TARGET_TOL_MOL_S = 0.5
PROPENE_PURITY_MIN = 0.995
PROPANE_PURCHASE_SEK_PER_TON = 4982.01
PROPENE_SALES_SEK_PER_TON = 10058.0
PROPANE_PURCHASE_SEK_PER_MOL = 0.2197
PROPENE_SALES_SEK_PER_MOL = 0.42324064
HYDROGEN_SALES_SEK_PER_MOL = 0.08627
WATER_RICH_LIQUID_MIN = 0.90
H2_RICH_VAPOR_MIN = 0.95

DEFAULT_X_TARGET = 0.80
REACTOR_PRESSURE_BAR = 1.10325
CATALYST_DENSITY_KG_M3 = 1120.0
DEFAULT_COMPRESSOR_EFFICIENCY = 0.75
PRESSURE_TOL_BAR = 1e-9
HEAT_EXCHANGER_U_W_M2K = 1000.0
HEAT_EXCHANGER_DEFAULT_DT_K = 10.0
COMPRESSOR_EQUIPMENT_KEY = "centrifugalkompressor"

COST_BASIS = "SEK/s of operation"
LANGFACTOR = 4.0
LIFETIME_YEARS = 10
OPERATING_SECONDS_PER_YEAR = eko.DRIFTTID_H * 3600
LIFETIME_OPERATING_SECONDS = LIFETIME_YEARS * OPERATING_SECONDS_PER_YEAR

HARD_PENALTY = 1e12
FAILURE_PENALTY_SPREAD = 1e9
PRODUCTION_PENALTY = 1e7
PURITY_PENALTY = 1e10
SEPARATION_PENALTY = 1e8


@dataclass
class SimulationResult:
    objective: float
    raw_cost: float
    penalty: float
    feasible: bool
    parameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    cost_breakdown: dict = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    excluded_costs: list[str] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    n_values: tuple[int, ...] = (1, 2, 3)
    stage_values: tuple[int, ...] = (20, 30, 40, 60)
    feed_stage_fracs: tuple[float, ...] = (1 / 3, 1 / 2, 2 / 3)
    maxiter: int = 300
    popsize: int = 15
    seed: int = 7
    workers: int = -1
    polish: bool = False
    include_feed_optimization: bool = True
    x_target: float = DEFAULT_X_TARGET
    tol: float = 1e-9
    atol: float = 0.0
    guess_iterator_steps: int = 20
    disp: bool = False


class OptimizerCase(NamedTuple):
    n: int
    n_stages: int
    feed_stage_fraction: float
    feed_stage: int
    seed: int


def resolve_worker_count(workers: int) -> int:
    if workers == -1:
        return os.cpu_count() or 1
    if workers < -1 or workers == 0:
        raise ValueError("workers must be -1 or a positive integer.")
    return workers


def usd_2010_to_installed_sek(cost_usd_2010: float) -> float:
    ensure_langfactor()
    return cost_usd_2010 * (eko.CEPCI_2024 / eko.CEPCI_2010) * eko.USD_TO_SEK * eko.LANGFAKTOR


def ensure_langfactor() -> None:
    eko.LANGFAKTOR = LANGFACTOR


def ensure_compressor_cost_key() -> None:
    ensure_langfactor()
    if COMPRESSOR_EQUIPMENT_KEY not in eko.EQUIPMENT_DATA:
        eko.EQUIPMENT_DATA[COMPRESSOR_EQUIPMENT_KEY] = (49000, 16800, 0.6, 0.0, 100000.0)
    if "el" not in eko.UTILITY_PRICES:
        eko.UTILITY_PRICES["el"] = 0.70


def annualized_capex_sek_per_s(capex_sek: float) -> float:
    return capex_sek / LIFETIME_OPERATING_SECONDS


def cost_sek_per_s_to_lifetime_sek(cost_sek_per_s: float) -> float:
    return float(cost_sek_per_s) * LIFETIME_OPERATING_SECONDS


def lifetime_cost_breakdown(cost_breakdown_sek_per_s: dict[str, float]) -> dict[str, float]:
    return {
        name: cost_sek_per_s_to_lifetime_sek(value)
        for name, value in cost_breakdown_sek_per_s.items()
    }


def material_economics_sek_per_s(
    *,
    fresh_propane_mol_s: float,
    propene_product_mol_s: float,
    hydrogen_product_mol_s: float,
    operating_cost_sek_per_s: float,
) -> dict[str, float]:
    propane_purchase_cost = fresh_propane_mol_s * PROPANE_PURCHASE_SEK_PER_MOL
    propene_sales_revenue = propene_product_mol_s * PROPENE_SALES_SEK_PER_MOL
    hydrogen_sales_revenue = hydrogen_product_mol_s * HYDROGEN_SALES_SEK_PER_MOL
    net_win = propene_sales_revenue + hydrogen_sales_revenue - propane_purchase_cost - operating_cost_sek_per_s
    return {
        "propane_purchase_cost_sek_per_s": propane_purchase_cost,
        "propene_sales_revenue_sek_per_s": propene_sales_revenue,
        "hydrogen_sales_revenue_sek_per_s": hydrogen_sales_revenue,
        "operating_cost_sek_per_s": operating_cost_sek_per_s,
        "net_win_sek_per_s": net_win,
    }


def net_win_sek_per_s(result: SimulationResult) -> float:
    metrics = result.metrics
    return material_economics_sek_per_s(
        fresh_propane_mol_s=metrics.get("fresh_propane_mol_s", 0.0),
        propene_product_mol_s=metrics.get("propene_flow_mol_s", 0.0),
        hydrogen_product_mol_s=metrics.get("flash2_hydrogen_flow_mol_s", 0.0),
        operating_cost_sek_per_s=result.raw_cost,
    )["net_win_sek_per_s"]


def net_win_lifetime_sek(result: SimulationResult) -> float:
    return cost_sek_per_s_to_lifetime_sek(net_win_sek_per_s(result))


def utility_cost_sek_per_s(q_kw: float, utility_type: str) -> float:
    return max(0.0, q_kw) * eko.UTILITY_PRICES[utility_type] / 3600.0


def vectorized_column_capex_sek_per_s(column: VectorizedDistillationColumn) -> float:
    ensure_langfactor()
    column.calculate_cost()
    column_cost = float(column.cost)
    if not np.isfinite(column_cost) or column_cost < 0.0:
        raise ValueError("Vectorized distillation column cost calculation returned an invalid cost.")
    return annualized_capex_sek_per_s(usd_2010_to_installed_sek(column_cost))


def stream_is_valid(stream: Stream | None, allow_zero: bool = False) -> bool:
    if stream is None:
        return False
    values = list(stream.flowrates.values())
    if not values or not all(np.isfinite(v) for v in values):
        return False
    if any(v < -1e-8 for v in values):
        return False
    total = stream.total_flowrate()
    if not np.isfinite(total):
        return False
    return total >= 0.0 if allow_zero else total > 1e-8


def mole_fraction(stream: Stream, species: str) -> float:
    total = stream.total_flowrate()
    if total <= 0:
        return 0.0
    return stream.flowrates.get(species, 0.0) / total


def normalized_flowrates(flowrates: dict[str, float]) -> dict[str, float]:
    return {species: float(flowrates.get(species, 0.0)) for species in SPECIES}


def copy_stream(
    stream: Stream,
    *,
    temperature: float | None = None,
    pressure: float | None = None,
    phase: str | None = None,
    flowrates: dict[str, float] | None = None,
) -> Stream:
    return Stream(
        temperature=stream.temperature if temperature is None else temperature,
        pressure=stream.pressure if pressure is None else pressure,
        phase=stream.phase if phase is None else phase,
        flowrates=normalized_flowrates(stream.flowrates if flowrates is None else flowrates),
    )


def compressor_cost_sek_per_s(compressor: Compressor) -> float:
    return annualized_capex_sek_per_s(compressor.capex) + compressor.annual_opex / OPERATING_SECONDS_PER_YEAR


def pressure_raise_with_compressor(
    stream: Stream,
    target_pressure: float,
    label: str,
) -> tuple[Stream, dict | None]:
    if target_pressure <= stream.pressure + PRESSURE_TOL_BAR:
        return copy_stream(stream, pressure=target_pressure), None
    if stream.phase != "vapor":
        return copy_stream(stream, pressure=target_pressure), {
            "label": label,
            "method": "direct pressure adjustment for non-vapor stream",
            "inlet_pressure_bar": stream.pressure,
            "outlet_pressure_bar": target_pressure,
            "cost_sek_per_s": 0.0,
        }

    compressor = Compressor(
        outlet_pressure=target_pressure,
        isentropic_efficiency=DEFAULT_COMPRESSOR_EFFICIENCY,
    )
    ensure_compressor_cost_key()
    outlet = compressor.run(stream)
    cost = compressor_cost_sek_per_s(compressor)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"{label} compressor returned an invalid cost.")

    return outlet, {
        "label": label,
        "inlet_pressure_bar": stream.pressure,
        "outlet_pressure_bar": outlet.pressure,
        "outlet_temperature_k": outlet.temperature,
        "energy_demand_kw": compressor.total_energy_demand_kW,
        "cooling_demand_kw": compressor.total_cooling_demand_kW,
        "cooler_area_m2": compressor.total_cooler_area_m2,
        "capex_sek": compressor.capex,
        "annual_opex_sek_per_year": compressor.annual_opex,
        "cost_sek_per_s": cost,
    }


def heat_exchanger_condition_stream(
    stream: Stream,
    target_temperature: float,
    label: str,
    target_pressure: float | None = None,
) -> tuple[Stream, dict | None]:
    pressure = stream.pressure if target_pressure is None else target_pressure
    if abs(target_temperature - stream.temperature) <= 1e-9:
        return copy_stream(stream, temperature=target_temperature, pressure=pressure), None

    outlet = copy_stream(stream, temperature=target_temperature, pressure=pressure)
    duty = outlet.enthalpy - stream.enthalpy
    utility_type = "steam" if duty > 0.0 else "cooling_water"
    exchanger = HeatExchanger(
        outlet_temperature=target_temperature,
        U=HEAT_EXCHANGER_U_W_M2K,
        hot_temperature=max(stream.temperature, target_temperature) + HEAT_EXCHANGER_DEFAULT_DT_K,
        mass_flow_hot=1e9,
        heat_capacity_hot=1.0,
        utility_type=utility_type,
    )
    ensure_langfactor()
    exchanger.calculate_from_duty(stream, abs(duty), delta_t=HEAT_EXCHANGER_DEFAULT_DT_K)
    cost = annualized_capex_sek_per_s(exchanger.capex) + exchanger.annual_opex / OPERATING_SECONDS_PER_YEAR
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(f"{label} heat exchanger returned an invalid cost.")

    return outlet, {
        "label": label,
        "inlet_temperature_k": stream.temperature,
        "outlet_temperature_k": target_temperature,
        "pressure_bar": pressure,
        "duty_w": duty,
        "utility_type": utility_type,
        "area_m2": exchanger.area,
        "capex_sek": exchanger.capex,
        "annual_opex_sek_per_year": exchanger.annual_opex,
        "cost_sek_per_s": cost,
        "calculation_class": "HeatExchanger",
    }


def condition_stream_to_temperature_pressure(
    stream: Stream,
    target_temperature: float,
    target_pressure: float,
    label: str,
) -> tuple[Stream, dict]:
    pressure_adjusted, compressor_meta = pressure_raise_with_compressor(stream, target_pressure, label)
    if target_pressure < stream.pressure - PRESSURE_TOL_BAR:
        pressure_adjusted = copy_stream(pressure_adjusted, pressure=target_pressure)
    conditioned, heat_exchanger_meta = heat_exchanger_condition_stream(
        pressure_adjusted,
        target_temperature,
        label,
        target_pressure=target_pressure,
    )
    return conditioned, {
        "label": label,
        "compressor": compressor_meta,
        "heat_exchanger": heat_exchanger_meta,
        "target_temperature_k": target_temperature,
        "target_pressure_bar": target_pressure,
    }


def run_vectorized_distillation_with_retry(
    column: VectorizedDistillationColumn,
    feed: Stream,
    *,
    suppress_column_output: bool = True,
    guess_iterator_steps: int = 20,
) -> tuple[Stream, Stream, dict]:
    metadata = {
        "class": "classes.VectorizedColumn.VectorizedDistillationColumn",
        "initial_success": False,
        "used_guess_iterator": False,
        "retry_success": None,
        "guess_iterator_steps": guess_iterator_steps,
        "solver_message": None,
    }

    def run_column(x0=None):
        if suppress_column_output:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    return column.run(feed, x0=x0)
        return column.run(feed, x0=x0)

    distillate, bottoms = run_column()
    metadata["initial_success"] = bool(column.sol is not None and column.sol.success)
    metadata["solver_message"] = None if column.sol is None else str(column.sol.message)
    if metadata["initial_success"]:
        return distillate, bottoms, metadata

    metadata["used_guess_iterator"] = True
    iterator = GuessIterator(feed=feed, column=column)
    if suppress_column_output:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                retry_guess = iterator.run(step=guess_iterator_steps)
    else:
        retry_guess = iterator.run(step=guess_iterator_steps)

    metadata["guess_iterator_returned_guess"] = retry_guess is not None
    if retry_guess is not None:
        distillate, bottoms = run_column(x0=retry_guess)

    metadata["retry_success"] = bool(column.sol is not None and column.sol.success)
    metadata["solver_message"] = None if column.sol is None else str(column.sol.message)
    if not metadata["retry_success"]:
        raise ValueError(
            f"VectorizedDistillationColumn failed to converge after GuessIterator retry: "
            f"{metadata['solver_message']}"
        )

    return distillate, bottoms, metadata


def build_bounds(
    n: int,
    include_feed_optimization: bool = True,
    x_target: float = DEFAULT_X_TARGET,
) -> list[tuple[float, float]]:
    """Bounds are intentionally explicit so the search space can be reviewed."""
    bounds: list[tuple[float, float]] = []

    if n > 1:
        max_free_conversion = 0.95 * (1.0 - (1.0 - x_target) ** (1.0 / (n - 1)))
        max_free_conversion = max(0.10, min(0.85, max_free_conversion))
        bounds.extend([(0.05, max_free_conversion)] * (n - 1))

    bounds.extend([(700.0, 950.0)] * n)  # reactor inlet temperatures [K]
    bounds.extend([(350.0, 550.0)])  # flash 1 temperature [K]
    bounds.extend([(250.0, 400.0)])  # flash 2 temperature [K]
    bounds.extend([(1.0, 50.0)])  # flash 1 pressure [bar]
    bounds.extend([(1.0, 100.0)])  # flash 2 pressure [bar]
    bounds.extend([(280.0, 450.0)])  # distillation feed temperature [K]
    bounds.extend([(1.0, 40.0)])  # distillation pressure [bar]
    bounds.extend([(0.5, 15.0)])  # reflux ratio
    bounds.extend([(0.01, 0.99)])  # fraction of column feed leaving as distillate

    if include_feed_optimization:
        bounds.extend([(80.0, 180.0)])  # fresh propane [mol/s]
        bounds.extend([(200.0, 2500.0)])  # fresh water [mol/s]

    return bounds


def compute_reactor_conversions(
    free_conversions: Iterable[float],
    n: int,
    x_target: float,
) -> tuple[list[float], list[str]]:
    free = list(float(x) for x in free_conversions)
    if n == 1:
        conversions = [x_target]
    else:
        remaining_factor = float(np.prod([1.0 - x for x in free]))
        if remaining_factor <= 0:
            return [], ["Invalid conversion vector: remaining propane factor is non-positive."]
        final_conversion = 1.0 - (1.0 - x_target) / remaining_factor
        conversions = free + [final_conversion]

    violations = []
    for i, conversion in enumerate(conversions, start=1):
        if not np.isfinite(conversion) or conversion <= 0.0 or conversion >= 0.999:
            violations.append(
                f"Reactor {i} conversion {conversion:.6g} is outside the valid interval."
            )
    return conversions, violations


def unpack_params(
    params: Iterable[float],
    n: int,
    n_stages: int,
    feed_stage: int,
    include_feed_optimization: bool = True,
    x_target: float = DEFAULT_X_TARGET,
) -> dict:
    params = list(float(x) for x in params)
    i = 0

    free_conversions = params[i : i + n - 1]
    i += n - 1

    reactor_temperatures = params[i : i + n]
    i += n

    flash1_temperature, flash2_temperature = params[i], params[i + 1]
    i += 2

    flash1_pressure, flash2_pressure = params[i], params[i + 1]
    i += 2

    distillation_feed_temperature = params[i]
    i += 1

    distillation_pressure = params[i]
    i += 1

    reflux_ratio = params[i]
    i += 1

    distillate_fraction = params[i]
    i += 1

    if include_feed_optimization:
        fresh_propane = params[i]
        i += 1
        fresh_water = params[i]
    else:
        fresh_propane = 108.569
        fresh_water = 1085.69

    conversions, conversion_violations = compute_reactor_conversions(
        free_conversions, n, x_target
    )

    return {
        "n": n,
        "N_stages": n_stages,
        "feed_stage": feed_stage,
        "X_target": x_target,
        "X_free": list(free_conversions),
        "X": conversions,
        "conversion_violations": conversion_violations,
        "T_reactors": list(reactor_temperatures),
        "flash_temperatures": (flash1_temperature, flash2_temperature),
        "flash_pressures": (flash1_pressure, flash2_pressure),
        "T_distillation_feed": distillation_feed_temperature,
        "P_distillation": distillation_pressure,
        "reflux_ratio": reflux_ratio,
        "distillate_fraction": distillate_fraction,
        "F_propane": fresh_propane,
        "F_water": fresh_water,
    }


def infeasible_penalty(parameters: dict | None = None) -> float:
    if not parameters:
        return HARD_PENALTY

    values = [
        *parameters.get("X_free", []),
        *parameters.get("T_reactors", []),
        *parameters.get("flash_temperatures", ()),
        *parameters.get("flash_pressures", ()),
        parameters.get("T_distillation_feed", 0.0),
        parameters.get("P_distillation", 0.0),
        parameters.get("reflux_ratio", 0.0),
        parameters.get("distillate_fraction", 0.0),
        parameters.get("F_propane", 0.0),
        parameters.get("F_water", 0.0),
    ]
    finite_values = [abs(float(value)) for value in values if np.isfinite(value)]
    if not finite_values:
        return HARD_PENALTY

    scaled = sum((index + 1) * value for index, value in enumerate(finite_values))
    fractional = scaled - np.floor(scaled)
    smooth_size_penalty = 1e5 * sum(finite_values)
    return HARD_PENALTY + smooth_size_penalty + FAILURE_PENALTY_SPREAD * fractional


def infeasible_result(reason: str, parameters: dict | None = None) -> SimulationResult:
    penalty = infeasible_penalty(parameters)
    return SimulationResult(
        objective=penalty,
        raw_cost=0.0,
        penalty=penalty,
        feasible=False,
        parameters=parameters or {},
        violations=[reason],
        excluded_costs=excluded_cost_notes(),
    )


def excluded_cost_notes() -> list[str]:
    return [
        "Pump costs are excluded because classes/Pump.py is currently empty.",
        "Pressure decreases are treated as direct pressure letdown with no recovery credit.",
    ]


def simulate_plant(
    params: Iterable[float],
    n: int,
    n_stages: int,
    feed_stage: int,
    include_feed_optimization: bool = True,
    x_target: float = DEFAULT_X_TARGET,
    suppress_column_output: bool = True,
    guess_iterator_steps: int = 20,
) -> SimulationResult:
    ensure_langfactor()
    p = unpack_params(
        params=params,
        n=n,
        n_stages=n_stages,
        feed_stage=feed_stage,
        include_feed_optimization=include_feed_optimization,
        x_target=x_target,
    )

    if p["conversion_violations"]:
        return infeasible_result("; ".join(p["conversion_violations"]), p)

    feed = Stream(
        temperature=298.15,
        pressure=1.01325,
        phase="vapor",
        flowrates={
            "propane": p["F_propane"],
            "propene": 0.0,
            "H2": 0.0,
            "H2O": p["F_water"],
        },
    )

    if not stream_is_valid(feed):
        return infeasible_result("Fresh feed stream is invalid.", p)

    violations: list[str] = []
    penalty = 0.0
    equipment_metadata: dict[str, dict] = {}

    try:
        stream = feed
        oven = ProcessOven()
        reactor_capex_usd_2010 = 0.0

        for reactor_index, (temperature, conversion) in enumerate(
            zip(p["T_reactors"], p["X"]), start=1
        ):
            if temperature < stream.temperature:
                violations.append(
                    f"Reactor {reactor_index} inlet temperature requires cooling before the process oven."
                )
                penalty += SEPARATION_PENALTY * ((stream.temperature - temperature) / 100.0) ** 2

            stream = oven.run(stream, temperature)
            reactor = Reactor(
                REACTOR_PRESSURE_BAR,
                CATALYST_DENSITY_KG_M3,
                catalyst_mass=None,
                conversion=conversion,
            )
            stream = reactor.run2(stream)

            if not stream_is_valid(stream) or reactor.cost is None:
                return infeasible_result(f"Reactor {reactor_index} produced an invalid state.", p)

            reactor_capex_usd_2010 += reactor.cost

        flash1_feed, flash1_conditioning = condition_stream_to_temperature_pressure(
            stream,
            p["flash_temperatures"][0],
            p["flash_pressures"][0],
            "Flash 1 feed conditioning",
        )
        equipment_metadata["flash1_conditioning"] = flash1_conditioning

        flash1 = FlashTank(
            temperature=p["flash_temperatures"][0],
            pressure=p["flash_pressures"][0],
        )
        liquid_flash1, vapor_flash1 = flash1.run(flash1_feed)
        if not stream_is_valid(liquid_flash1, allow_zero=True) or not stream_is_valid(vapor_flash1, allow_zero=True):
            return infeasible_result("Flash 1 produced an invalid stream.", p)
        flash1.calculate_economics(liquid_flash1)

        h2o_fraction = mole_fraction(liquid_flash1, "H2O")
        if h2o_fraction < WATER_RICH_LIQUID_MIN:
            shortfall = WATER_RICH_LIQUID_MIN - h2o_fraction
            violations.append(f"Flash 1 liquid water mole fraction is {h2o_fraction:.6f}.")
            penalty += SEPARATION_PENALTY * shortfall**2

        flash2_feed, flash2_conditioning = condition_stream_to_temperature_pressure(
            vapor_flash1,
            p["flash_temperatures"][1],
            p["flash_pressures"][1],
            "Flash 2 feed conditioning",
        )
        equipment_metadata["flash2_conditioning"] = flash2_conditioning

        flash2 = FlashTank(
            temperature=p["flash_temperatures"][1],
            pressure=p["flash_pressures"][1],
        )
        liquid_flash2, vapor_flash2 = flash2.run(flash2_feed)
        if not stream_is_valid(liquid_flash2, allow_zero=True) or not stream_is_valid(vapor_flash2, allow_zero=True):
            return infeasible_result("Flash 2 produced an invalid stream.", p)
        flash2.calculate_economics(liquid_flash2)

        h2_fraction = mole_fraction(vapor_flash2, "H2")
        if h2_fraction < H2_RICH_VAPOR_MIN:
            shortfall = H2_RICH_VAPOR_MIN - h2_fraction
            violations.append(f"Flash 2 vapor hydrogen mole fraction is {h2_fraction:.6f}.")
            penalty += SEPARATION_PENALTY * shortfall**2

        column_feed, column_feed_conditioning = condition_stream_to_temperature_pressure(
            liquid_flash2,
            p["T_distillation_feed"],
            p["P_distillation"],
            "Distillation feed conditioning",
        )
        equipment_metadata["distillation_feed_conditioning"] = column_feed_conditioning
        if not stream_is_valid(column_feed):
            return infeasible_result("Distillation feed stream is empty or invalid.", p)

        distillate_flowrate = p["distillate_fraction"] * column_feed.total_flowrate()
        if distillate_flowrate <= 0.0 or distillate_flowrate >= column_feed.total_flowrate():
            return infeasible_result("Distillate flowrate is outside the column feed flowrate.", p)

        column = VectorizedDistillationColumn(
            pressure=p["P_distillation"],
            stages=n_stages,
            feed_stage=feed_stage,
            reflux_ratio=p["reflux_ratio"],
            distillate_flowrate=distillate_flowrate,
        )

        distillate, bottoms, distillation_metadata = run_vectorized_distillation_with_retry(
            column,
            column_feed,
            suppress_column_output=suppress_column_output,
            guess_iterator_steps=guess_iterator_steps,
        )
        equipment_metadata["distillation_solver"] = distillation_metadata

        if not stream_is_valid(distillate) or not stream_is_valid(bottoms, allow_zero=True):
            return infeasible_result("Distillation column produced an invalid stream.", p)

        propene_flow = distillate.flowrates.get("propene", 0.0)
        distillate_total = distillate.total_flowrate()
        propene_purity = mole_fraction(distillate, "propene")

        production_shortfall = max(0.0, PROPENE_TARGET_MOL_S - propene_flow)
        penalty += PRODUCTION_PENALTY * production_shortfall**2
        if production_shortfall > PROPENE_TARGET_TOL_MOL_S:
            violations.append(
                f"Propene production is {propene_flow:.6f} mol/s, minimum target is {PROPENE_TARGET_MOL_S:.6f} mol/s."
            )

        if propene_purity < PROPENE_PURITY_MIN:
            shortfall = PROPENE_PURITY_MIN - propene_purity
            violations.append(f"Distillate propene purity is {propene_purity:.6f}.")
            penalty += PURITY_PENALTY * shortfall**2

        reactor_capex_sek = usd_2010_to_installed_sek(reactor_capex_usd_2010)
        oven_capex_sek = usd_2010_to_installed_sek(oven.cost)
        flash_capex_sek = flash1.cost + flash2.cost

        reactor_capex_sek_per_s = annualized_capex_sek_per_s(reactor_capex_sek)
        oven_capex_sek_per_s = annualized_capex_sek_per_s(oven_capex_sek)
        flash_capex_sek_per_s = annualized_capex_sek_per_s(flash_capex_sek)
        column_capex_sek_per_s = vectorized_column_capex_sek_per_s(column)

        compressor_costs_sek_per_s = 0.0
        heat_exchanger_costs_sek_per_s = 0.0
        for metadata in equipment_metadata.values():
            compressor = metadata.get("compressor")
            if compressor:
                compressor_costs_sek_per_s += compressor.get("cost_sek_per_s", 0.0)
            exchanger = metadata.get("heat_exchanger")
            if exchanger:
                heat_exchanger_costs_sek_per_s += exchanger.get("cost_sek_per_s", 0.0)

        oven_fuel_sek_per_s = max(0.0, oven.gas_cost)
        reboiler_utility_sek_per_s = utility_cost_sek_per_s(
            max(0.0, column.reboiler_duty) / 1000.0, "steam"
        )
        condenser_utility_sek_per_s = utility_cost_sek_per_s(
            abs(column.condenser_duty) / 1000.0, "cooling_water"
        )

        cost_breakdown = {
            "reactor_capex": reactor_capex_sek_per_s,
            "process_oven_capex": oven_capex_sek_per_s,
            "process_oven_fuel": oven_fuel_sek_per_s,
            "flash_tank_capex": flash_capex_sek_per_s,
            "distillation_reboiler_utility": reboiler_utility_sek_per_s,
            "distillation_condenser_utility": condenser_utility_sek_per_s,
            "distillation_column_capex": column_capex_sek_per_s,
            "stream_conditioning_heat_exchanger_costs": heat_exchanger_costs_sek_per_s,
            "stream_conditioning_compressor_costs": compressor_costs_sek_per_s,
            "pump_costs": 0.0,
        }

        raw_cost = float(sum(cost_breakdown.values()))
        objective = raw_cost + penalty
        lifetime_costs = lifetime_cost_breakdown(cost_breakdown)
        material_economics = material_economics_sek_per_s(
            fresh_propane_mol_s=p["F_propane"],
            propene_product_mol_s=propene_flow,
            hydrogen_product_mol_s=vapor_flash2.flowrates.get("H2", 0.0),
            operating_cost_sek_per_s=raw_cost,
        )
        material_economics_lifetime = {
            name.replace("_sek_per_s", "_lifetime_sek"): cost_sek_per_s_to_lifetime_sek(value)
            for name, value in material_economics.items()
        }

        metrics = {
            "feed_total_mol_s": feed.total_flowrate(),
            "equipment_metadata": equipment_metadata,
            "lifetime_years": LIFETIME_YEARS,
            "operating_hours_per_year": eko.DRIFTTID_H,
            "lifetime_operating_seconds": LIFETIME_OPERATING_SECONDS,
            "total_operating_cost_lifetime_sek": cost_sek_per_s_to_lifetime_sek(raw_cost),
            "cost_breakdown_lifetime_sek": lifetime_costs,
            "fresh_propane_mol_s": p["F_propane"],
            "propane_purchase_sek_per_ton": PROPANE_PURCHASE_SEK_PER_TON,
            "propene_sales_sek_per_ton": PROPENE_SALES_SEK_PER_TON,
            "propane_purchase_sek_per_mol": PROPANE_PURCHASE_SEK_PER_MOL,
            "propene_sales_sek_per_mol": PROPENE_SALES_SEK_PER_MOL,
            "hydrogen_sales_sek_per_mol": HYDROGEN_SALES_SEK_PER_MOL,
            "material_economics_sek_per_s": material_economics,
            "material_economics_lifetime_sek": material_economics_lifetime,
            "net_win_sek_per_s": material_economics["net_win_sek_per_s"],
            "net_win_lifetime_sek": material_economics_lifetime["net_win_lifetime_sek"],
            "distillate_total_mol_s": distillate_total,
            "distillate_flowrate_spec_mol_s": distillate_flowrate,
            "propene_flow_mol_s": propene_flow,
            "propene_minimum_target_mol_s": PROPENE_TARGET_MOL_S,
            "propene_production_shortfall_mol_s": production_shortfall,
            "propene_purity": propene_purity,
            "bottoms_propane_purity_mol_percent": 100.0 * mole_fraction(bottoms, "propane"),
            "flash1_liquid_h2o_fraction": h2o_fraction,
            "flash1_liquid_total_mol_s": liquid_flash1.total_flowrate(),
            "flash2_vapor_h2_fraction": h2_fraction,
            "flash2_hydrogen_flow_mol_s": vapor_flash2.flowrates.get("H2", 0.0),
            "flash2_liquid_total_mol_s": liquid_flash2.total_flowrate(),
            "column_reboiler_duty_w": column.reboiler_duty,
            "column_condenser_duty_w": column.condenser_duty,
            "column_diameter_m": column.diameter,
            "column_height_m": column.height,
            "column_tray_cost_base": column.tray_cost,
            "column_shell_cost_base": column.shell_cost,
            "column_total_cost_base": column.cost,
        }

        feasible = (
            production_shortfall <= PROPENE_TARGET_TOL_MOL_S
            and propene_purity >= PROPENE_PURITY_MIN
            and h2o_fraction >= WATER_RICH_LIQUID_MIN
            and h2_fraction >= H2_RICH_VAPOR_MIN
            and not any("requires cooling before the process oven" in v for v in violations)
        )

        return SimulationResult(
            objective=float(objective),
            raw_cost=raw_cost,
            penalty=float(penalty),
            feasible=feasible,
            parameters=p,
            metrics=metrics,
            cost_breakdown=cost_breakdown,
            violations=violations,
            excluded_costs=excluded_cost_notes(),
        )

    except Exception as exc:
        return infeasible_result(f"Simulation failed: {exc}", p)


def objective_function(
    params: Iterable[float],
    n: int,
    n_stages: int,
    feed_stage: int,
    include_feed_optimization: bool,
    x_target: float,
    guess_iterator_steps: int,
) -> float:
    result = simulate_plant(
        params=params,
        n=n,
        n_stages=n_stages,
        feed_stage=feed_stage,
        include_feed_optimization=include_feed_optimization,
        x_target=x_target,
        suppress_column_output=True,
        guess_iterator_steps=guess_iterator_steps,
    )
    if not np.isfinite(result.objective):
        return HARD_PENALTY
    return result.objective


def feed_stage_from_fraction(n_stages: int, fraction: float) -> int:
    return max(2, min(int(round(fraction * n_stages)), n_stages - 2))


def optimizer_cases(config: OptimizerConfig) -> list[OptimizerCase]:
    cases = []
    case_index = 0
    for n in config.n_values:
        for n_stages in config.stage_values:
            for fraction in config.feed_stage_fracs:
                cases.append(
                    OptimizerCase(
                        n=n,
                        n_stages=n_stages,
                        feed_stage_fraction=fraction,
                        feed_stage=feed_stage_from_fraction(n_stages, fraction),
                        seed=config.seed + case_index,
                    )
                )
                case_index += 1
    return cases


def run_optimizer_case(
    case: OptimizerCase,
    config: OptimizerConfig,
    de_workers: int,
) -> SimulationResult:
    bounds = build_bounds(
        case.n,
        config.include_feed_optimization,
        config.x_target,
    )
    updating = "deferred" if de_workers != 1 else "immediate"

    de_result = differential_evolution(
        objective_function,
        bounds,
        args=(
            case.n,
            case.n_stages,
            case.feed_stage,
            config.include_feed_optimization,
            config.x_target,
            config.guess_iterator_steps,
        ),
        maxiter=config.maxiter,
        popsize=config.popsize,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=case.seed,
        workers=de_workers,
        updating=updating,
        tol=config.tol,
        atol=config.atol,
        polish=config.polish,
        disp=config.disp,
    )

    simulation = simulate_plant(
        params=de_result.x,
        n=case.n,
        n_stages=case.n_stages,
        feed_stage=case.feed_stage,
        include_feed_optimization=config.include_feed_optimization,
        x_target=config.x_target,
        suppress_column_output=True,
        guess_iterator_steps=config.guess_iterator_steps,
    )
    simulation.metrics["optimizer_workers_requested"] = config.workers
    simulation.metrics["optimizer_case_seed"] = case.seed
    simulation.metrics["optimizer_feed_stage_fraction"] = case.feed_stage_fraction
    simulation.metrics["differential_evolution_workers"] = de_workers
    return simulation


def run_optimizer(config: OptimizerConfig) -> tuple[SimulationResult, list[SimulationResult]]:
    results: list[SimulationResult] = []
    cases = optimizer_cases(config)
    requested_worker_count = resolve_worker_count(config.workers)
    if config.workers == 1:
        effective_workers = 1
    elif len(cases) == 1:
        effective_workers = requested_worker_count
    else:
        effective_workers = min(requested_worker_count, len(cases))

    print(
        f"Optimizer workers: requested={config.workers}, effective={effective_workers}, "
        f"cases={len(cases)}"
    )

    if effective_workers == 1 or len(cases) == 1:
        de_workers = effective_workers if len(cases) == 1 else 1
        for case in cases:
            simulation = run_optimizer_case(case, config, de_workers)
            results.append(simulation)
            print(
                f"Completed full case n={case.n}, stages={case.n_stages}, "
                f"feed_stage={case.feed_stage}: objective={simulation.objective:.6g}, "
                f"feasible={simulation.feasible}"
            )
    else:
        parallel_config = OptimizerConfig(
            n_values=config.n_values,
            stage_values=config.stage_values,
            feed_stage_fracs=config.feed_stage_fracs,
            maxiter=config.maxiter,
            popsize=config.popsize,
            seed=config.seed,
            workers=config.workers,
            polish=config.polish,
            include_feed_optimization=config.include_feed_optimization,
            x_target=config.x_target,
            tol=config.tol,
            atol=config.atol,
            guess_iterator_steps=config.guess_iterator_steps,
            disp=False,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_case = {
                executor.submit(run_optimizer_case, case, parallel_config, 1): case
                for case in cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                simulation = future.result()
                results.append(simulation)
                print(
                    f"Completed full case n={case.n}, stages={case.n_stages}, "
                    f"feed_stage={case.feed_stage}: objective={simulation.objective:.6g}, "
                    f"feasible={simulation.feasible}"
                )

    best = min(results, key=lambda item: item.objective)
    best.metrics["optimizer_effective_workers"] = effective_workers
    best.metrics["optimizer_case_count"] = len(cases)
    best.metrics["optimizer_parallel_level"] = "cases" if effective_workers > 1 and len(cases) > 1 else "differential_evolution"
    return best, results


def format_result(result: SimulationResult) -> str:
    p = result.parameters
    m = result.metrics
    lines = [
        "",
        "=== FULL-SEPARATION OPTIMIZATION RESULT ===",
        f"Objective: {result.objective:.6g} {COST_BASIS}",
        f"Raw cost:  {result.raw_cost:.6g} {COST_BASIS}",
        f"Penalty:   {result.penalty:.6g}",
        f"Feasible:  {result.feasible}",
        f"Workers requested: {m.get('optimizer_workers_requested')}",
        f"Workers effective: {m.get('optimizer_effective_workers')}",
        f"Parallel level: {m.get('optimizer_parallel_level')}",
        "",
        "Plant parameters:",
        f"  Reactors: {p.get('n')}",
        f"  Reactor conversions: {p.get('X')}",
        f"  Reactor inlet temperatures [K]: {p.get('T_reactors')}",
        f"  Flash temperatures [K]: {p.get('flash_temperatures')}",
        f"  Flash pressures [bar]: {p.get('flash_pressures')}",
        f"  Distillation feed temperature [K]: {p.get('T_distillation_feed')}",
        f"  Distillation pressure [bar]: {p.get('P_distillation')}",
        f"  Distillation stages: {p.get('N_stages')}",
        f"  Feed stage: {p.get('feed_stage')}",
        f"  Reflux ratio: {p.get('reflux_ratio')}",
        f"  Distillate fraction: {p.get('distillate_fraction')}",
        f"  Fresh propane [mol/s]: {p.get('F_propane')}",
        f"  Fresh water [mol/s]: {p.get('F_water')}",
        "",
        "Product metrics:",
        f"  Distillate total flow [mol/s]: {m.get('distillate_total_mol_s')}",
        f"  Propene in distillate [mol/s]: {m.get('propene_flow_mol_s')}",
        f"  Minimum propene target [mol/s]: {m.get('propene_minimum_target_mol_s')}",
        f"  Propene shortfall [mol/s]: {m.get('propene_production_shortfall_mol_s')}",
        f"  Propene purity [mol fraction]: {m.get('propene_purity')}",
        f"  Bottoms propane purity [mol%]: {m.get('bottoms_propane_purity_mol_percent')}",
        f"  Flash 1 liquid H2O fraction: {m.get('flash1_liquid_h2o_fraction')}",
        f"  Flash 1 liquid total [mol/s]: {m.get('flash1_liquid_total_mol_s')}",
        f"  Flash 2 vapor H2 fraction: {m.get('flash2_vapor_h2_fraction')}",
        f"  Flash 2 hydrogen flow [mol/s]: {m.get('flash2_hydrogen_flow_mol_s')}",
        f"  Flash 2 liquid total [mol/s]: {m.get('flash2_liquid_total_mol_s')}",
        f"  Column diameter [m]: {m.get('column_diameter_m')}",
        f"  Column height [m]: {m.get('column_height_m')}",
        "",
        f"Cost breakdown [{COST_BASIS}]:",
    ]

    for name, value in result.cost_breakdown.items():
        lines.append(f"  {name}: {value:.6g}")

    if m.get("equipment_metadata"):
        lines.append("")
        lines.append("Stream conditioning equipment:")
        for label, metadata in m["equipment_metadata"].items():
            if label == "distillation_solver":
                continue
            lines.append(
                f"  {label}: target T={metadata.get('target_temperature_k')} K, "
                f"target P={metadata.get('target_pressure_bar')} bar"
            )
            compressor = metadata.get("compressor")
            if compressor:
                lines.append(
                    f"    pressure adjustment: {compressor.get('inlet_pressure_bar')} -> "
                    f"{compressor.get('outlet_pressure_bar')} bar, "
                    f"method={compressor.get('method', 'Compressor')}, "
                    f"cost={compressor.get('cost_sek_per_s')} {COST_BASIS}"
                )
            else:
                lines.append("    pressure adjustment: not needed")
            exchanger = metadata.get("heat_exchanger")
            if exchanger:
                lines.append(
                    f"    heat exchanger: duty={exchanger.get('duty_w')} W, "
                    f"area={exchanger.get('area_m2')} m2, "
                    f"utility={exchanger.get('utility_type')}, "
                    f"cost={exchanger.get('cost_sek_per_s')} {COST_BASIS}"
                )
            else:
                lines.append("    heat exchanger: not needed")

        solver_metadata = m["equipment_metadata"].get("distillation_solver")
        if solver_metadata:
            lines.append("")
            lines.append("Distillation solver:")
            lines.append(f"  Class: {solver_metadata.get('class')}")
            lines.append(f"  Initial success: {solver_metadata.get('initial_success')}")
            lines.append(f"  Used GuessIterator: {solver_metadata.get('used_guess_iterator')}")
            lines.append(f"  Retry success: {solver_metadata.get('retry_success')}")
            lines.append(f"  GuessIterator steps: {solver_metadata.get('guess_iterator_steps')}")
            lines.append(f"  Solver message: {solver_metadata.get('solver_message')}")

    lines.append("")
    lines.append(
        f"Total operating cost over {m.get('lifetime_years', LIFETIME_YEARS)} years "
        f"({m.get('operating_hours_per_year', eko.DRIFTTID_H)} h/year): "
        f"{m.get('total_operating_cost_lifetime_sek', cost_sek_per_s_to_lifetime_sek(result.raw_cost)):.6g} SEK"
    )
    if m.get("cost_breakdown_lifetime_sek"):
        lines.append("Lifetime cost breakdown [SEK]:")
        for name, value in m["cost_breakdown_lifetime_sek"].items():
            lines.append(f"  {name}: {value:.6g}")

    lines.append("")
    lines.append("Material economics and net win:")
    lines.append(
        f"  Propane purchase price: {m.get('propane_purchase_sek_per_ton', PROPANE_PURCHASE_SEK_PER_TON)} "
        f"SEK/ton ({m.get('propane_purchase_sek_per_mol', PROPANE_PURCHASE_SEK_PER_MOL)} SEK/mol)"
    )
    lines.append(
        f"  Propene sales price: {m.get('propene_sales_sek_per_ton', PROPENE_SALES_SEK_PER_TON)} "
        f"SEK/ton ({m.get('propene_sales_sek_per_mol', PROPENE_SALES_SEK_PER_MOL)} SEK/mol)"
    )
    lines.append(
        f"  Hydrogen sales price: {m.get('hydrogen_sales_sek_per_mol', HYDROGEN_SALES_SEK_PER_MOL)} SEK/mol"
    )
    for name, value in m.get("material_economics_sek_per_s", {}).items():
        lines.append(f"  {name}: {value:.6g} SEK/s")
    lines.append(f"  net_win_lifetime_sek: {net_win_lifetime_sek(result):.6g} SEK")
    if m.get("material_economics_lifetime_sek"):
        lines.append("Material economics over lifetime [SEK]:")
        for name, value in m["material_economics_lifetime_sek"].items():
            lines.append(f"  {name}: {value:.6g}")

    if result.violations:
        lines.append("")
        lines.append("Constraint violations or penalties:")
        for violation in result.violations:
            lines.append(f"  - {violation}")

    if result.excluded_costs:
        lines.append("")
        lines.append("Excluded costs:")
        for note in result.excluded_costs:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the GKT propane-to-propene plant.")
    parser.add_argument("--n-values", default="3", help="Comma-separated reactor counts.")
    parser.add_argument("--stage-values", default="60, 70, 80, 85", help="Comma-separated distillation stage counts.")
    parser.add_argument("--feed-stage-fracs", default="0.3333333333,0.5,0.6666666667", help="Comma-separated feed-stage fractions.")
    parser.add_argument("--maxiter", type=int, default=300, help="Differential evolution iterations per structure.")
    parser.add_argument("--popsize", type=int, default=15, help="Differential evolution population multiplier.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-9, help="Relative convergence tolerance for differential evolution.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute convergence tolerance for differential evolution.")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--fixed-feed", action="store_true", help="Use the original fixed feed instead of optimizing feed.")
    parser.add_argument("--x-target", type=float, default=DEFAULT_X_TARGET)
    parser.add_argument("--guess-iterator-steps", type=int, default=20, help="Continuation steps for GuessIterator when distillation fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OptimizerConfig(
        n_values=csv_ints(args.n_values),
        stage_values=csv_ints(args.stage_values),
        feed_stage_fracs=csv_floats(args.feed_stage_fracs),
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=args.workers,
        polish=args.polish,
        include_feed_optimization=not args.fixed_feed,
        x_target=args.x_target,
        tol=args.tol,
        atol=args.atol,
        guess_iterator_steps=args.guess_iterator_steps,
        disp=args.disp,
    )
    best, _ = run_optimizer(config)
    print(format_result(best))


if __name__ == "__main__":
    main()
