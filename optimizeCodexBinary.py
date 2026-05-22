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
from classes import VectorizedDistillationColumn
from thermo import antoine, find_saturation_pressure


SPECIES = ("propane", "propene", "H2", "H2O")
BINARY_SPECIES = ("propane", "propene")

PROPENE_TARGET_MOL_S = 82.0
PROPENE_TARGET_TOL_MOL_S = 0.5
PROPENE_PURITY_MIN = 0.995
PROPANE_PURCHASE_SEK_PER_TON = 4982.01
PROPENE_SALES_SEK_PER_TON = 10058.0
PROPANE_PURCHASE_SEK_PER_MOL = 0.2197
PROPENE_SALES_SEK_PER_MOL = 0.42324064
HYDROGEN_SALES_SEK_PER_MOL = 0.08627

DEFAULT_X_TARGET = 0.80
REACTOR_PRESSURE_BAR = 1.10325
CATALYST_DENSITY_KG_M3 = 1120.0
FEED_WATER_TO_PROPANE_RATIO = 10.0
DEFAULT_COMPRESSOR_EFFICIENCY = 0.75
PRESSURE_TOL_BAR = 1e-9
MMHG_PER_BAR = 750.06156130264
FLASH_CONDENSER_U_W_M2K = 1000.0
COLUMN_HEAT_EXCHANGER_U_W_M2K = 1000.0
HEAT_EXCHANGER_DEFAULT_DT_K = 10.0
COMPRESSOR_EQUIPMENT_KEY = "centrifugalkompressor"
COOLING_WATER_INLET_TEMPERATURE_K = 14.0 + 273.15
COOLING_WATER_OUTLET_TEMPERATURE_K = 20.0 + 273.15
REBOILER_HOT_INLET_TEMPERATURES_K = (180.0 + 273.15, 210.0 + 273.15)
REBOILER_HOT_OUTLET_MIN_TEMPERATURE_K = 20.0 + 273.15
REBOILER_MIN_APPROACH_K = 1e-6

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
MASS_BALANCE_TOL = 1e-6


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
    simplifications: list[str] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    n_values: tuple[int, ...] = (1, 2, 3)
    stage_values: tuple[int, ...] = (4, 6)
    feed_stage_fracs: tuple[float, ...] = (0.5,)
    maxiter: int = 5
    popsize: int = 4
    seed: int = 7
    workers: int = -1
    polish: bool = False
    include_feed_optimization: bool = True
    x_target: float = DEFAULT_X_TARGET
    tol: float = 1e-6
    atol: float = 0.0
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


def capex_langfactor_notes() -> list[str]:
    return [
        f"LANGFAKTOR is forced to {LANGFACTOR} before CAPEX calculations.",
        "Reactor, process oven, and vectorized distillation column base costs are converted with usd_2010_to_installed_sek(...), which includes LANGFAKTOR.",
        "Flash tank, heat exchanger, and compressor CAPEX use ekonomi.calculate_capex(...), which includes LANGFAKTOR.",
    ]


def annualized_capex_sek_per_s(capex_sek: float) -> float:
    return capex_sek / LIFETIME_OPERATING_SECONDS


def cost_sek_per_s_to_lifetime_sek(cost_sek_per_s: float) -> float:
    return float(cost_sek_per_s) * LIFETIME_OPERATING_SECONDS


def lifetime_cost_breakdown(cost_breakdown_sek_per_s: dict[str, float]) -> dict[str, float]:
    return {
        name: cost_sek_per_s_to_lifetime_sek(value)
        for name, value in cost_breakdown_sek_per_s.items()
    }


def total_operating_cost_per_lifetime(result: SimulationResult) -> float:
    return cost_sek_per_s_to_lifetime_sek(result.raw_cost)


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
    net_win = (
        propene_sales_revenue
        + hydrogen_sales_revenue
        - propane_purchase_cost
        - operating_cost_sek_per_s
    )
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


def lmtd_from_terminal_differences(delta_t_a: float, delta_t_b: float) -> float:
    if delta_t_a <= 0.0 or delta_t_b <= 0.0:
        raise ValueError("Heat exchanger terminal temperature differences must be positive.")
    if abs(delta_t_a - delta_t_b) < 1e-9:
        return float(delta_t_a)
    return float((delta_t_a - delta_t_b) / np.log(delta_t_a / delta_t_b))


def vectorized_column_capex_sek_per_s(column: VectorizedDistillationColumn) -> float:
    ensure_langfactor()
    column.calculate_cost()
    column_cost = float(column.cost)
    if not np.isfinite(column_cost) or column_cost < 0.0:
        raise ValueError("Vectorized distillation column cost calculation returned an invalid cost.")
    return annualized_capex_sek_per_s(usd_2010_to_installed_sek(column_cost))


def compressor_cost_sek_per_s(compressor: Compressor) -> float:
    return annualized_capex_sek_per_s(compressor.capex) + compressor.annual_opex / OPERATING_SECONDS_PER_YEAR


def ensure_compressor_cost_key() -> None:
    ensure_langfactor()
    if COMPRESSOR_EQUIPMENT_KEY not in eko.EQUIPMENT_DATA:
        # Local fallback for this optimizer. S is compressor power in kW.
        eko.EQUIPMENT_DATA[COMPRESSOR_EQUIPMENT_KEY] = (49000, 16800, 0.6, 1.0, 100000.0)
    if "el" not in eko.UTILITY_PRICES:
        eko.UTILITY_PRICES["el"] = 0.70


def heat_exchanger_cost_metadata(
    *,
    stream: Stream,
    duty_w: float,
    utility_type: str,
    label: str,
    u_w_m2k: float = FLASH_CONDENSER_U_W_M2K,
    delta_t_k: float = HEAT_EXCHANGER_DEFAULT_DT_K,
    condensation_temperature_k: float | None = None,
) -> dict:
    ensure_langfactor()
    abs_duty_w = abs(float(duty_w))
    exchanger = HeatExchanger(
        outlet_temperature=stream.temperature,
        U=u_w_m2k,
        hot_temperature=stream.temperature + max(delta_t_k, 1e-6),
        mass_flow_hot=1e9,
        heat_capacity_hot=1.0,
        utility_type=utility_type,
    )
    if condensation_temperature_k is None:
        exchanger.calculate_from_duty(stream, abs_duty_w, delta_t=max(delta_t_k, 1e-6))
        lmtd = None
        cooling_water_inlet = None
        cooling_water_outlet = None
    else:
        exchanger.calculate_condensation_with_cooling_water(
            stream,
            abs_duty_w,
            condensing_temperature=condensation_temperature_k,
            cooling_water_inlet_temperature=COOLING_WATER_INLET_TEMPERATURE_K,
            cooling_water_outlet_temperature=COOLING_WATER_OUTLET_TEMPERATURE_K,
        )
        lmtd = exchanger.lmtd
        cooling_water_inlet = COOLING_WATER_INLET_TEMPERATURE_K
        cooling_water_outlet = COOLING_WATER_OUTLET_TEMPERATURE_K
    capex_sek = float(exchanger.capex)
    annual_opex = float(exchanger.annual_opex)
    capex_cost_sek_per_s = annualized_capex_sek_per_s(capex_sek)
    utility_cost = annual_opex / OPERATING_SECONDS_PER_YEAR
    return {
        "label": label,
        "duty_w": duty_w,
        "utility_type": utility_type,
        "u_w_m2k": u_w_m2k,
        "area_m2": float(exchanger.area),
        "area_for_cost_m2": max(10.0, min(float(exchanger.area), 1000.0)),
        "capex_sek": capex_sek,
        "annual_opex_sek_per_year": annual_opex,
        "capex_cost_sek_per_s": capex_cost_sek_per_s,
        "utility_cost_sek_per_s": utility_cost,
        "cost_sek_per_s": capex_cost_sek_per_s + utility_cost,
        "calculation_class": "HeatExchanger",
        "condensation_temperature_k": condensation_temperature_k,
        "cooling_water_inlet_temperature_k": cooling_water_inlet,
        "cooling_water_outlet_temperature_k": cooling_water_outlet,
        "lmtd_k": lmtd,
    }


def distillation_condenser_heat_exchanger_metadata(
    top_stream: Stream,
    condenser_duty_w: float,
) -> dict:
    return heat_exchanger_cost_metadata(
        stream=top_stream,
        duty_w=condenser_duty_w,
        utility_type="cooling_water",
        label="Binary distillation top condenser",
        u_w_m2k=COLUMN_HEAT_EXCHANGER_U_W_M2K,
        condensation_temperature_k=top_stream.temperature,
    )


def reboiler_heat_exchanger_metadata_for_hot_inlet(
    bottom_stream: Stream,
    reboiler_duty_w: float,
    hot_inlet_temperature_k: float,
) -> dict:
    ensure_langfactor()
    duty_w = max(0.0, float(reboiler_duty_w))
    process_temperature_k = float(bottom_stream.temperature)
    hot_outlet_temperature_k = max(
        REBOILER_HOT_OUTLET_MIN_TEMPERATURE_K,
        process_temperature_k + REBOILER_MIN_APPROACH_K,
    )
    if hot_inlet_temperature_k <= hot_outlet_temperature_k:
        raise ValueError(
            "Reboiler hot utility inlet temperature must be above the hot utility outlet temperature."
        )

    delta_t_hot_in = hot_inlet_temperature_k - process_temperature_k
    delta_t_hot_out = hot_outlet_temperature_k - process_temperature_k
    lmtd = lmtd_from_terminal_differences(delta_t_hot_in, delta_t_hot_out)

    exchanger = HeatExchanger(
        outlet_temperature=bottom_stream.temperature,
        U=COLUMN_HEAT_EXCHANGER_U_W_M2K,
        hot_temperature=hot_inlet_temperature_k,
        mass_flow_hot=1e9,
        heat_capacity_hot=1.0,
        utility_type="steam",
    )
    exchanger.calculate_from_duty(bottom_stream, duty_w, delta_t=lmtd)

    capex_sek = float(exchanger.capex)
    annual_opex = float(exchanger.annual_opex)
    capex_cost_sek_per_s = annualized_capex_sek_per_s(capex_sek)
    utility_cost = annual_opex / OPERATING_SECONDS_PER_YEAR
    return {
        "label": "Binary distillation bottom reboiler",
        "duty_w": duty_w,
        "utility_type": "steam",
        "u_w_m2k": COLUMN_HEAT_EXCHANGER_U_W_M2K,
        "area_m2": float(exchanger.area),
        "area_for_cost_m2": max(10.0, min(float(exchanger.area), 1000.0)),
        "capex_sek": capex_sek,
        "annual_opex_sek_per_year": annual_opex,
        "capex_cost_sek_per_s": capex_cost_sek_per_s,
        "utility_cost_sek_per_s": utility_cost,
        "cost_sek_per_s": capex_cost_sek_per_s + utility_cost,
        "calculation_class": "HeatExchanger",
        "process_temperature_k": process_temperature_k,
        "hot_inlet_temperature_k": hot_inlet_temperature_k,
        "hot_outlet_temperature_k": hot_outlet_temperature_k,
        "minimum_hot_outlet_temperature_k": REBOILER_HOT_OUTLET_MIN_TEMPERATURE_K,
        "delta_t_hot_in_k": delta_t_hot_in,
        "delta_t_hot_out_k": delta_t_hot_out,
        "lmtd_k": lmtd,
    }


def cheapest_reboiler_heat_exchanger_metadata(
    bottom_stream: Stream,
    reboiler_duty_w: float,
) -> dict:
    candidates = []
    rejected = []
    for hot_inlet_temperature_k in REBOILER_HOT_INLET_TEMPERATURES_K:
        try:
            candidates.append(
                reboiler_heat_exchanger_metadata_for_hot_inlet(
                    bottom_stream,
                    reboiler_duty_w,
                    hot_inlet_temperature_k,
                )
            )
        except ValueError as exc:
            rejected.append(
                {
                    "hot_inlet_temperature_k": hot_inlet_temperature_k,
                    "reason": str(exc),
                }
            )

    if not candidates:
        raise ValueError("No valid reboiler heat exchanger utility temperature was available.")

    chosen = min(candidates, key=lambda candidate: candidate["cost_sek_per_s"])
    chosen["candidate_hot_inlet_temperatures_k"] = REBOILER_HOT_INLET_TEMPERATURES_K
    chosen["candidate_costs_sek_per_s"] = {
        candidate["hot_inlet_temperature_k"]: candidate["cost_sek_per_s"]
        for candidate in candidates
    }
    chosen["rejected_candidates"] = rejected
    return chosen


def pressure_raise_with_compressor(
    stream: Stream,
    target_pressure: float,
    label: str,
) -> tuple[Stream, dict | None]:
    if target_pressure <= stream.pressure + PRESSURE_TOL_BAR:
        return stream, None
    if stream.phase != "vapor":
        raise ValueError(
            f"{label} requires a pressure increase from {stream.pressure:.6g} to "
            f"{target_pressure:.6g} bar, but the stream is {stream.phase}; a compressor is only used for vapor streams."
        )

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


def saturation_temperature_from_antoine(species: str, pressure_bar: float) -> float:
    if species not in antoine:
        raise ValueError(f"No Antoine constants are available for {species}.")
    if pressure_bar <= 0.0 or not np.isfinite(pressure_bar):
        raise ValueError(f"Invalid saturation pressure for {species}: {pressure_bar} bar.")

    pressure_mmhg = pressure_bar * MMHG_PER_BAR
    A, B, C = antoine[species]
    denominator = A - np.log(pressure_mmhg)
    if abs(denominator) <= 1e-12:
        raise ValueError(f"Antoine inversion is singular for {species} at {pressure_bar} bar.")

    temperature = B / denominator - C
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Invalid saturation temperature for {species} at {pressure_bar} bar.")

    return float(temperature)


def latent_heat_flow_w(
    stream: Stream,
    component: str,
    temperature: float,
    flow_mol_s: float | None = None,
) -> float:
    flow = float(stream.flowrates.get(component, 0.0) if flow_mol_s is None else flow_mol_s)
    if flow <= MASS_BALANCE_TOL:
        return 0.0

    std_h_vap = stream.standard_enthalpy_of_vaporization[component]
    critical_temperature = stream.critical_temperature[component]
    t_ref = 298.15
    ratio = (1.0 - temperature / critical_temperature) / (1.0 - t_ref / critical_temperature)
    if ratio <= 0.0:
        raise ValueError(
            f"{component} saturation temperature {temperature:.6g} K is at or above the critical region."
        )
    return flow * std_h_vap * (ratio ** 0.38)


def prepare_stream_for_saturation_flash(
    stream: Stream,
    pressure_bar: float,
    saturation_species: str,
    label: str,
) -> tuple[Stream, dict]:
    pressure_adjusted, compressor_meta = pressure_raise_with_compressor(stream, pressure_bar, label)
    saturation_temperature = saturation_temperature_from_antoine(saturation_species, pressure_bar)

    prepared = copy_stream(
        pressure_adjusted,
        temperature=saturation_temperature,
        pressure=pressure_bar,
    )
    sensible_duty = prepared.enthalpy - pressure_adjusted.enthalpy
    utility_type = "steam" if sensible_duty > 0.0 else "cooling_water"
    sensible_hx_meta = heat_exchanger_cost_metadata(
        stream=pressure_adjusted,
        duty_w=sensible_duty,
        utility_type=utility_type,
        label=f"{label} sensible heat exchanger",
    )

    return prepared, {
        "label": label,
        "saturation_species": saturation_species,
        "saturation_temperature_k": saturation_temperature,
        "target_pressure_bar": pressure_bar,
        "compressor": compressor_meta,
        "sensible_heat_exchanger": sensible_hx_meta,
    }


def condense_component(
    stream: Stream,
    component: str,
    heat_exchanger_u: float,
    label: str,
) -> dict:
    latent_duty = -latent_heat_flow_w(stream, component, stream.temperature)
    return heat_exchanger_cost_metadata(
        stream=stream,
        duty_w=latent_duty,
        utility_type="cooling_water",
        label=label,
        u_w_m2k=heat_exchanger_u,
        condensation_temperature_k=stream.temperature,
    )


def condense_equivalent_heavy_components(
    stream: Stream,
    reference_component: str,
    condensed_components: tuple[str, ...],
    heat_exchanger_u: float,
    label: str,
) -> dict:
    condensed_flow_mol_s = sum(float(stream.flowrates.get(component, 0.0)) for component in condensed_components)
    latent_duty = -latent_heat_flow_w(
        stream,
        reference_component,
        stream.temperature,
        flow_mol_s=condensed_flow_mol_s,
    )
    metadata = heat_exchanger_cost_metadata(
        stream=stream,
        duty_w=latent_duty,
        utility_type="cooling_water",
        label=label,
        u_w_m2k=heat_exchanger_u,
        condensation_temperature_k=stream.temperature,
    )
    metadata["reference_latent_heat_component"] = reference_component
    metadata["condensed_components"] = condensed_components
    metadata["condensed_flow_mol_s"] = condensed_flow_mol_s
    return metadata


def run_water_removal_flash(inlet_stream: Stream, pressure_bar: float) -> tuple[Stream, Stream, dict[str, float], dict]:
    prepared, metadata = prepare_stream_for_saturation_flash(
        inlet_stream,
        pressure_bar,
        "H2O",
        "Flash 1",
    )
    metadata["latent_condenser"] = condense_component(
        prepared,
        "H2O",
        FLASH_CONDENSER_U_W_M2K,
        "Flash 1 water condenser",
    )

    liquid = copy_stream(
        prepared,
        phase="liquid",
        flowrates={
            "propane": 0.0,
            "propene": 0.0,
            "H2": 0.0,
            "H2O": prepared.flowrates.get("H2O", 0.0),
        },
    )
    vapor = copy_stream(
        prepared,
        phase="vapor",
        flowrates={
            "propane": prepared.flowrates.get("propane", 0.0),
            "propene": prepared.flowrates.get("propene", 0.0),
            "H2": prepared.flowrates.get("H2", 0.0),
            "H2O": 0.0,
        },
    )
    errors = check_species_mass_balance(prepared, liquid, vapor)
    return liquid, vapor, errors, metadata


def run_hydrogen_removal_flash(inlet_stream: Stream, pressure_bar: float) -> tuple[Stream, Stream, dict[str, float], dict]:
    prepared, metadata = prepare_stream_for_saturation_flash(
        inlet_stream,
        pressure_bar,
        "propene",
        "Flash 2",
    )
    metadata["latent_condenser"] = condense_equivalent_heavy_components(
        prepared,
        "propene",
        ("propane", "propene"),
        FLASH_CONDENSER_U_W_M2K,
        "Flash 2 propane/propene condenser",
    )

    liquid = copy_stream(
        prepared,
        phase="liquid",
        flowrates={
            "propane": prepared.flowrates.get("propane", 0.0),
            "propene": prepared.flowrates.get("propene", 0.0),
            "H2": 0.0,
            "H2O": prepared.flowrates.get("H2O", 0.0),
        },
    )
    vapor = copy_stream(
        prepared,
        phase="vapor",
        flowrates={
            "propane": 0.0,
            "propene": 0.0,
            "H2": prepared.flowrates.get("H2", 0.0),
            "H2O": 0.0,
        },
    )
    errors = check_species_mass_balance(prepared, liquid, vapor)
    return liquid, vapor, errors, metadata


def reheat_water_stream_with_oven(
    water_stream: Stream,
    target_temperature: float,
    oven: ProcessOven,
) -> tuple[Stream, dict]:
    inlet_enthalpy = water_stream.enthalpy
    reheated = oven.run(water_stream, target_temperature)
    duty = reheated.enthalpy - inlet_enthalpy
    return reheated, {
        "label": "Flash 1 separated water oven reheater",
        "duty_w": duty,
        "outlet_temperature_k": target_temperature,
        "calculation_class": "ProcessOven",
        "note": "Separated flash 1 water is reheated by the process oven, not by a heat exchanger.",
    }


def heat_first_reactor_propane_feed_only(
    feed: Stream,
    oven: ProcessOven,
    reactor_temperature: float,
) -> tuple[Stream, dict]:
    dry_feed_flows = normalized_flowrates(feed.flowrates)
    preheated_water_flow = dry_feed_flows["H2O"]
    dry_feed_flows["H2O"] = 0.0

    dry_feed = copy_stream(feed, flowrates=dry_feed_flows)
    heated_dry_feed = oven.run(dry_feed, reactor_temperature)
    reactor_feed_flows = normalized_flowrates(heated_dry_feed.flowrates)
    reactor_feed_flows["H2O"] = preheated_water_flow

    reactor_feed = copy_stream(
        heated_dry_feed,
        temperature=reactor_temperature,
        flowrates=reactor_feed_flows,
    )
    water_at_reactor_temperature = copy_stream(
        feed,
        temperature=reactor_temperature,
        flowrates={
            "propane": 0.0,
            "propene": 0.0,
            "H2": 0.0,
            "H2O": preheated_water_flow,
        },
    )

    return reactor_feed, {
        "fresh_water_mol_s": preheated_water_flow,
        "water_temperature_k": reactor_temperature,
        "water_heated_in_process_oven": False,
        "note": (
            "Fresh/reactor-feed water is assumed to have been heated after flash 1, "
            "so the first process oven heats only the propane-containing feed."
        ),
        "water_enthalpy_at_reactor_temperature_w": water_at_reactor_temperature.enthalpy,
    }


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
    return total >= -1e-8 if allow_zero else total > 1e-8


def mole_fraction(stream: Stream, species: str) -> float:
    total = stream.total_flowrate()
    if total <= 0:
        return 0.0
    return stream.flowrates.get(species, 0.0) / total


def make_binary_proxy_stream(stream: Stream, phase: str | None = None) -> Stream:
    """Keep propane/propene only; force hydrogen and water to zero."""
    return Stream(
        temperature=stream.temperature,
        pressure=stream.pressure,
        phase=stream.phase if phase is None else phase,
        flowrates={
            "propane": float(stream.flowrates.get("propane", 0.0)),
            "propene": float(stream.flowrates.get("propene", 0.0)),
            "H2": 0.0,
            "H2O": 0.0,
        },
    )


def zero_binary_outlets(proxy: Stream) -> tuple[Stream, Stream]:
    liquid = copy_stream(proxy, phase="liquid", flowrates={species: 0.0 for species in SPECIES})
    vapor = copy_stream(proxy, phase="vapor", flowrates={species: 0.0 for species in SPECIES})
    return liquid, vapor


def reconstruct_binary_separator_outlets(
    original_stream: Stream,
    binary_liquid: Stream,
    binary_vapor: Stream,
) -> tuple[Stream, Stream]:
    """Route H2 to vapor and H2O to liquid after binary propane/propene separation."""
    liquid = Stream(
        temperature=binary_liquid.temperature,
        pressure=binary_liquid.pressure,
        phase="liquid",
        flowrates={
            "propane": float(binary_liquid.flowrates.get("propane", 0.0)),
            "propene": float(binary_liquid.flowrates.get("propene", 0.0)),
            "H2": 0.0,
            "H2O": float(original_stream.flowrates.get("H2O", 0.0)),
        },
    )
    vapor = Stream(
        temperature=binary_vapor.temperature,
        pressure=binary_vapor.pressure,
        phase="vapor",
        flowrates={
            "propane": float(binary_vapor.flowrates.get("propane", 0.0)),
            "propene": float(binary_vapor.flowrates.get("propene", 0.0)),
            "H2": float(original_stream.flowrates.get("H2", 0.0)),
            "H2O": 0.0,
        },
    )
    return liquid, vapor


def check_species_mass_balance(
    inlet: Stream,
    outlet_a: Stream,
    outlet_b: Stream,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for species in SPECIES:
        inlet_flow = float(inlet.flowrates.get(species, 0.0))
        outlet_flow = float(outlet_a.flowrates.get(species, 0.0)) + float(outlet_b.flowrates.get(species, 0.0))
        errors[species] = outlet_flow - inlet_flow
    return errors


def max_abs_balance_error(errors: dict[str, float]) -> float:
    if not errors:
        return 0.0
    return max(abs(value) for value in errors.values())


def binary_routing_violations(liquid: Stream, vapor: Stream, label: str) -> list[str]:
    violations = []
    if abs(liquid.flowrates.get("H2", 0.0)) > MASS_BALANCE_TOL:
        violations.append(f"{label} liquid contains hydrogen after binary reconstruction.")
    if abs(vapor.flowrates.get("H2O", 0.0)) > MASS_BALANCE_TOL:
        violations.append(f"{label} vapor contains water after binary reconstruction.")
    return violations


def run_binary_flash(flash_tank: FlashTank, inlet_stream: Stream) -> tuple[Stream, Stream, dict[str, float]]:
    proxy = make_binary_proxy_stream(inlet_stream)
    if proxy.flowrates["propane"] + proxy.flowrates["propene"] <= MASS_BALANCE_TOL:
        binary_liquid, binary_vapor = zero_binary_outlets(proxy)
    else:
        binary_liquid, binary_vapor = flash_tank.run(proxy)

    liquid, vapor = reconstruct_binary_separator_outlets(
        original_stream=inlet_stream,
        binary_liquid=binary_liquid,
        binary_vapor=binary_vapor,
    )
    errors = check_species_mass_balance(inlet_stream, liquid, vapor)
    return liquid, vapor, errors


def run_binary_distillation(
    column: VectorizedDistillationColumn,
    inlet_stream: Stream,
    suppress_column_output: bool = True,
) -> tuple[Stream, Stream, dict[str, float]]:
    proxy = make_binary_proxy_stream(inlet_stream, phase="liquid")
    proxy_total = proxy.flowrates["propane"] + proxy.flowrates["propene"]
    if proxy_total <= MASS_BALANCE_TOL:
        binary_bottoms, binary_distillate = zero_binary_outlets(proxy)
    elif suppress_column_output:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                binary_distillate, binary_bottoms = column.run(proxy)
    else:
        binary_distillate, binary_bottoms = column.run(proxy)

    bottoms, distillate = reconstruct_binary_separator_outlets(
        original_stream=inlet_stream,
        binary_liquid=binary_bottoms,
        binary_vapor=binary_distillate,
    )
    errors = check_species_mass_balance(inlet_stream, distillate, bottoms)
    return distillate, bottoms, errors


def build_bounds(
    n: int,
    include_feed_optimization: bool = True,
    x_target: float = DEFAULT_X_TARGET,
) -> list[tuple[float, float]]:
    """Bounds are intentionally explicit so the binary optimizer is reviewable."""
    bounds: list[tuple[float, float]] = []

    if n > 1:
        max_free_conversion = 0.95 * (1.0 - (1.0 - x_target) ** (1.0 / (n - 1)))
        max_free_conversion = max(0.10, min(0.85, max_free_conversion))
        bounds.extend([(0.05, max_free_conversion)] * (n - 1))

    bounds.extend([(700.0, 950.0)] * n)  # reactor inlet temperatures [K]
    bounds.extend([(1.0, 50.0)])  # flash 1 pressure [bar]
    bounds.extend([(10.5, 45.0)])  # flash 2 pressure [bar], compatible with 14-20 C cooling water and below propene critical pressure
    bounds.extend([(280.0, 450.0)])  # distillation feed temperature [K]
    bounds.extend([(1.0, 40.0)])  # distillation pressure [bar]
    bounds.extend([(0.5, 15.0)])  # reflux ratio
    bounds.extend([(0.01, 0.99)])  # fraction of binary column feed leaving as distillate

    if include_feed_optimization:
        bounds.extend([(80.0, 180.0)])  # fresh propane [mol/s]

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
    else:
        fresh_propane = 108.569
    fresh_water = FEED_WATER_TO_PROPANE_RATIO * fresh_propane

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
        "flash_pressures": (flash1_pressure, flash2_pressure),
        "flash_temperature_basis": (
            "Flash 1 temperature is derived from water saturation at flash 1 pressure.",
            "Flash 2 temperature is derived from propene saturation at flash 2 pressure.",
        ),
        "T_distillation_feed": distillation_feed_temperature,
        "P_distillation": distillation_pressure,
        "reflux_ratio": reflux_ratio,
        "distillate_fraction": distillate_fraction,
        "F_propane": fresh_propane,
        "F_water": fresh_water,
        "fresh_feed_water_to_propane_ratio": FEED_WATER_TO_PROPANE_RATIO,
    }


def infeasible_penalty(parameters: dict | None = None) -> float:
    if not parameters:
        return HARD_PENALTY

    values = [
        *parameters.get("X_free", []),
        *parameters.get("T_reactors", []),
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


def simplification_notes() -> list[str]:
    return [
        "Flash temperatures are derived from Antoine saturation temperatures and are not optimized.",
        "Flash 1 forcibly condenses and removes all water as a pure liquid stream.",
        "Flash 2 forcibly routes all hydrogen to vapor and all propane/propene to liquid.",
        "The water entering the first reactor is assumed preheated after flash 1 and is not heated again in the first process oven.",
        "Distillation uses VectorizedDistillationColumn on the binary proxy feed.",
        "Distillation column shell and tray CAPEX use VectorizedDistillationColumn.calculate_cost().",
        "Distillation condenser and reboiler area and CAPEX use classes/HeatExchanger.py with U = 1000 W/(m2 K).",
    ]


def excluded_cost_notes() -> list[str]:
    return [
        "Flash and distillation heat exchanger stream routing is handled in the optimizer, but area, CAPEX, and OPEX metadata are calculated with classes/HeatExchanger.py.",
        "Separated flash 1 water reheating is charged to the process oven, not to a heat exchanger.",
        "Pump costs are excluded because classes/Pump.py is currently empty.",
    ]


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
        simplifications=simplification_notes(),
    )


def simulate_plant(
    params: Iterable[float],
    n: int,
    n_stages: int,
    feed_stage: int,
    include_feed_optimization: bool = True,
    x_target: float = DEFAULT_X_TARGET,
    suppress_column_output: bool = True,
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
    mass_balance_errors: dict[str, dict[str, float]] = {}
    flash_metadata: dict[str, dict] = {}
    feed_heating_metadata: dict[str, dict] = {}

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

            if reactor_index == 1:
                stream, feed_heating_metadata["first_reactor_feed"] = heat_first_reactor_propane_feed_only(
                    stream,
                    oven,
                    temperature,
                )
            else:
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

            stream = copy_stream(stream)
            reactor_capex_usd_2010 += reactor.cost

        liquid_flash1, vapor_flash1, flash1_errors, flash1_meta = run_water_removal_flash(
            stream,
            p["flash_pressures"][0],
        )
        flash_metadata["flash1"] = flash1_meta
        mass_balance_errors["flash1"] = flash1_errors
        if not stream_is_valid(liquid_flash1, allow_zero=True) or not stream_is_valid(vapor_flash1, allow_zero=True):
            return infeasible_result("Water-removal flash 1 produced an invalid stream.", p)
        flash1 = FlashTank(
            temperature=flash1_meta["saturation_temperature_k"],
            pressure=p["flash_pressures"][0],
        )
        flash1.calculate_economics(liquid_flash1)

        flash1_balance_error = max_abs_balance_error(flash1_errors)
        if flash1_balance_error > MASS_BALANCE_TOL:
            violations.append(f"Flash 1 mass balance error is {flash1_balance_error:.6g} mol/s.")
            penalty += SEPARATION_PENALTY * flash1_balance_error**2
        violations.extend(binary_routing_violations(liquid_flash1, vapor_flash1, "Flash 1"))

        flash1_water_purity = mole_fraction(liquid_flash1, "H2O")
        if liquid_flash1.total_flowrate() > MASS_BALANCE_TOL and flash1_water_purity < 1.0 - MASS_BALANCE_TOL:
            violations.append(f"Flash 1 liquid water purity is {flash1_water_purity:.6f}.")
            penalty += PURITY_PENALTY * (1.0 - flash1_water_purity) ** 2

        reheated_water, water_reheater_meta = reheat_water_stream_with_oven(
            liquid_flash1,
            p["T_reactors"][0],
            oven,
        )
        flash_metadata["flash1"]["water_oven_reheater"] = water_reheater_meta
        flash_metadata["flash1"]["water_handling"] = (
            "Water entering the first reactor is assumed to have been heated after flash 1. "
            "It is reheated by the process oven and is not charged as a heat-exchanger duty."
        )

        liquid_flash2, vapor_flash2, flash2_errors, flash2_meta = run_hydrogen_removal_flash(
            vapor_flash1,
            p["flash_pressures"][1],
        )
        flash_metadata["flash2"] = flash2_meta
        mass_balance_errors["flash2"] = flash2_errors
        if not stream_is_valid(liquid_flash2, allow_zero=True) or not stream_is_valid(vapor_flash2, allow_zero=True):
            return infeasible_result("Hydrogen-removal flash 2 produced an invalid stream.", p)
        flash2 = FlashTank(
            temperature=flash2_meta["saturation_temperature_k"],
            pressure=p["flash_pressures"][1],
        )
        flash2.calculate_economics(liquid_flash2)

        flash2_balance_error = max_abs_balance_error(flash2_errors)
        if flash2_balance_error > MASS_BALANCE_TOL:
            violations.append(f"Flash 2 mass balance error is {flash2_balance_error:.6g} mol/s.")
            penalty += SEPARATION_PENALTY * flash2_balance_error**2
        violations.extend(binary_routing_violations(liquid_flash2, vapor_flash2, "Flash 2"))

        flash2_hydrogen_purity = mole_fraction(vapor_flash2, "H2")
        if vapor_flash2.total_flowrate() > MASS_BALANCE_TOL and flash2_hydrogen_purity < 1.0 - MASS_BALANCE_TOL:
            violations.append(f"Flash 2 vapor hydrogen purity is {flash2_hydrogen_purity:.6f}.")
            penalty += PURITY_PENALTY * (1.0 - flash2_hydrogen_purity) ** 2

        if (
            liquid_flash2.flowrates.get("propane", 0.0) + liquid_flash2.flowrates.get("propene", 0.0)
            <= MASS_BALANCE_TOL
        ):
            return infeasible_result("Flash 2 liquid contains no propane or propene for distillation.", p)

        p["flash_temperatures"] = (
            flash1_meta["saturation_temperature_k"],
            flash2_meta["saturation_temperature_k"],
        )

        column_feed = Stream(
            temperature=p["T_distillation_feed"],
            pressure=p["P_distillation"],
            phase="liquid",
            flowrates=liquid_flash2.flowrates.copy(),
        )
        if not stream_is_valid(column_feed):
            return infeasible_result("Binary distillation feed stream is empty or invalid.", p)

        binary_column_feed = make_binary_proxy_stream(column_feed, phase="liquid")
        binary_column_feed_total = binary_column_feed.total_flowrate()
        if binary_column_feed_total <= MASS_BALANCE_TOL:
            return infeasible_result("Binary distillation feed contains no propane or propene.", p)

        distillate_flowrate = p["distillate_fraction"] * binary_column_feed_total
        if distillate_flowrate <= 0.0 or distillate_flowrate >= binary_column_feed_total:
            return infeasible_result("Distillate flowrate is outside the binary column feed flowrate.", p)

        column = VectorizedDistillationColumn(
            pressure=p["P_distillation"],
            stages=n_stages,
            feed_stage=feed_stage,
            reflux_ratio=p["reflux_ratio"],
            distillate_flowrate=distillate_flowrate,
        )

        distillate, bottoms, column_errors = run_binary_distillation(
            column=column,
            inlet_stream=column_feed,
            suppress_column_output=suppress_column_output,
        )
        mass_balance_errors["distillation"] = column_errors

        if not stream_is_valid(distillate) or not stream_is_valid(bottoms, allow_zero=True):
            return infeasible_result("Binary distillation produced an invalid stream.", p)

        column_balance_error = max_abs_balance_error(column_errors)
        if column_balance_error > MASS_BALANCE_TOL:
            violations.append(f"Distillation mass balance error is {column_balance_error:.6g} mol/s.")
            penalty += SEPARATION_PENALTY * column_balance_error**2
        violations.extend(binary_routing_violations(bottoms, distillate, "Distillation"))

        propene_flow = distillate.flowrates.get("propene", 0.0)
        distillate_total = distillate.total_flowrate()
        propene_purity = mole_fraction(distillate, "propene")
        bottoms_propane_purity_mol_percent = 100.0 * mole_fraction(bottoms, "propane")

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
        column_condenser_heat_exchanger = distillation_condenser_heat_exchanger_metadata(
            distillate,
            column.condenser_duty,
        )
        column_reboiler_heat_exchanger = cheapest_reboiler_heat_exchanger_metadata(
            bottoms,
            column.reboiler_duty,
        )
        column_heat_exchanger_capex_sek_per_s = (
            column_condenser_heat_exchanger["capex_cost_sek_per_s"]
            + column_reboiler_heat_exchanger["capex_cost_sek_per_s"]
        )

        compressor_costs_sek_per_s = 0.0
        heat_exchanger_costs_sek_per_s = 0.0
        for metadata in flash_metadata.values():
            compressor = metadata.get("compressor")
            if compressor:
                compressor_costs_sek_per_s += compressor["cost_sek_per_s"]
            for key in ("sensible_heat_exchanger", "latent_condenser"):
                exchanger = metadata.get(key)
                if exchanger:
                    heat_exchanger_costs_sek_per_s += exchanger["cost_sek_per_s"]

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
            "binary_distillation_reboiler_utility": reboiler_utility_sek_per_s,
            "binary_distillation_condenser_utility": condenser_utility_sek_per_s,
            "binary_distillation_column_capex": column_capex_sek_per_s,
            "binary_distillation_heat_exchanger_capex": column_heat_exchanger_capex_sek_per_s,
            "flash_heat_exchanger_and_condenser_costs": heat_exchanger_costs_sek_per_s,
            "flash_compressor_costs": compressor_costs_sek_per_s,
            "pump_costs": 0.0,
        }
        capex_langfactor_applied = {
            "reactor_capex": LANGFACTOR,
            "process_oven_capex": LANGFACTOR,
            "flash_tank_capex": LANGFACTOR,
            "binary_distillation_column_capex": LANGFACTOR,
            "binary_distillation_heat_exchanger_capex": LANGFACTOR,
            "flash_heat_exchanger_and_condenser_costs": LANGFACTOR,
            "flash_compressor_costs": LANGFACTOR,
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

        mass_balance_max = {
            label: max_abs_balance_error(errors)
            for label, errors in mass_balance_errors.items()
        }
        metrics = {
            "feed_total_mol_s": feed.total_flowrate(),
            "feed_heating_metadata": feed_heating_metadata,
            "first_reactor_water_heated_in_process_oven": feed_heating_metadata.get(
                "first_reactor_feed", {}
            ).get("water_heated_in_process_oven"),
            "lifetime_years": LIFETIME_YEARS,
            "langfactor": eko.LANGFAKTOR,
            "capex_langfactor_applied": capex_langfactor_applied,
            "capex_langfactor_notes": capex_langfactor_notes(),
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
            "binary_column_feed_total_mol_s": binary_column_feed_total,
            "distillate_total_mol_s": distillate_total,
            "distillate_flowrate_spec_mol_s": distillate_flowrate,
            "propene_flow_mol_s": propene_flow,
            "propene_minimum_target_mol_s": PROPENE_TARGET_MOL_S,
            "propene_production_shortfall_mol_s": production_shortfall,
            "propene_purity": propene_purity,
            "bottoms_propane_purity_mol_percent": bottoms_propane_purity_mol_percent,
            "flash_saturation_temperatures_k": p["flash_temperatures"],
            "flash_metadata": flash_metadata,
            "flash1_water_flow_mol_s": liquid_flash1.flowrates.get("H2O", 0.0),
            "flash1_liquid_h2o_fraction": mole_fraction(liquid_flash1, "H2O"),
            "flash2_hydrogen_flow_mol_s": vapor_flash2.flowrates.get("H2", 0.0),
            "flash2_vapor_h2_fraction": mole_fraction(vapor_flash2, "H2"),
            "flash2_liquid_propane_mol_s": liquid_flash2.flowrates.get("propane", 0.0),
            "flash2_liquid_propene_mol_s": liquid_flash2.flowrates.get("propene", 0.0),
            "reheated_water_temperature_k": reheated_water.temperature,
            "reheated_water_flow_mol_s": reheated_water.total_flowrate(),
            "water_handling": flash_metadata["flash1"]["water_handling"],
            "mass_balance_errors": mass_balance_errors,
            "mass_balance_max_abs": mass_balance_max,
            "column_reboiler_duty_w": column.reboiler_duty,
            "column_condenser_duty_w": column.condenser_duty,
            "column_diameter_m": column.diameter,
            "column_height_m": column.height,
            "column_tray_cost_base": column.tray_cost,
            "column_shell_cost_base": column.shell_cost,
            "column_total_cost_base": column.cost,
            "column_heat_cost": column.heat_cost,
            "column_heat_exchangers": {
                "condenser": column_condenser_heat_exchanger,
                "reboiler": column_reboiler_heat_exchanger,
            },
        }

        feasible = (
            production_shortfall <= PROPENE_TARGET_TOL_MOL_S
            and propene_purity >= PROPENE_PURITY_MIN
            and all(error <= MASS_BALANCE_TOL for error in mass_balance_max.values())
            and not any("requires cooling before the process oven" in v for v in violations)
            and not any("contains hydrogen" in v or "contains water" in v for v in violations)
            and not any("purity" in v for v in violations)
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
            simplifications=simplification_notes(),
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
) -> float:
    result = simulate_plant(
        params=params,
        n=n,
        n_stages=n_stages,
        feed_stage=feed_stage,
        include_feed_optimization=include_feed_optimization,
        x_target=x_target,
        suppress_column_output=True,
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
                f"Completed binary case n={case.n}, stages={case.n_stages}, "
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
                    f"Completed binary case n={case.n}, stages={case.n_stages}, "
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
        "=== BINARY-SEPARATION OPTIMIZATION RESULT ===",
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
        f"  Flash pressures [bar]: {p.get('flash_pressures')}",
        f"  Derived flash saturation temperatures [K]: {p.get('flash_temperatures')}",
        f"  Flash temperature basis: {p.get('flash_temperature_basis')}",
        f"  Binary distillation feed temperature [K]: {p.get('T_distillation_feed')}",
        f"  Binary distillation pressure [bar]: {p.get('P_distillation')}",
        f"  Binary distillation stages: {p.get('N_stages')}",
        f"  Feed stage: {p.get('feed_stage')}",
        f"  Reflux ratio: {p.get('reflux_ratio')}",
        f"  Distillate fraction of binary feed: {p.get('distillate_fraction')}",
        f"  Fresh propane [mol/s]: {p.get('F_propane')}",
        f"  Fresh water [mol/s]: {p.get('F_water')}",
        f"  Fresh feed water:propane ratio: {p.get('fresh_feed_water_to_propane_ratio')}:1",
        f"  First reactor water heated in process oven: {m.get('first_reactor_water_heated_in_process_oven')}",
        "",
        "Product metrics:",
        f"  Distillate total flow [mol/s]: {m.get('distillate_total_mol_s')}",
        f"  Propene in distillate [mol/s]: {m.get('propene_flow_mol_s')}",
        f"  Minimum propene target [mol/s]: {m.get('propene_minimum_target_mol_s')}",
        f"  Propene shortfall [mol/s]: {m.get('propene_production_shortfall_mol_s')}",
        f"  Propene purity [mol fraction]: {m.get('propene_purity')}",
        f"  Bottoms propane purity [mol%]: {m.get('bottoms_propane_purity_mol_percent')}",
        f"  Binary column feed total [mol/s]: {m.get('binary_column_feed_total_mol_s')}",
        f"  Flash 1 water outlet flow [mol/s]: {m.get('flash1_water_flow_mol_s')}",
        f"  Flash 1 liquid H2O fraction: {m.get('flash1_liquid_h2o_fraction')}",
        f"  Flash 2 hydrogen outlet flow [mol/s]: {m.get('flash2_hydrogen_flow_mol_s')}",
        f"  Flash 2 vapor H2 fraction: {m.get('flash2_vapor_h2_fraction')}",
        f"  Flash 2 liquid propane [mol/s]: {m.get('flash2_liquid_propane_mol_s')}",
        f"  Flash 2 liquid propene [mol/s]: {m.get('flash2_liquid_propene_mol_s')}",
        f"  Reheated water temperature [K]: {m.get('reheated_water_temperature_k')}",
        f"  Reheated water flow [mol/s]: {m.get('reheated_water_flow_mol_s')}",
        f"  Water handling: {m.get('water_handling')}",
        f"  Column diameter [m]: {m.get('column_diameter_m')}",
        f"  Column height [m]: {m.get('column_height_m')}",
        f"  Column tray cost [base currency]: {m.get('column_tray_cost_base')}",
        f"  Column shell cost [base currency]: {m.get('column_shell_cost_base')}",
        "",
        "Species mass-balance max errors [mol/s]:",
    ]

    for label, value in m.get("mass_balance_max_abs", {}).items():
        lines.append(f"  {label}: {value:.6g}")

    if m.get("feed_heating_metadata"):
        lines.append("")
        lines.append("Feed heating metadata:")
        for label, metadata in m["feed_heating_metadata"].items():
            lines.append(
                f"  {label}: fresh_water={metadata.get('fresh_water_mol_s')} mol/s, "
                f"water_temperature={metadata.get('water_temperature_k')} K, "
                f"heated_in_process_oven={metadata.get('water_heated_in_process_oven')}"
            )
            lines.append(f"    {metadata.get('note')}")

    if m.get("flash_metadata"):
        lines.append("")
        lines.append("Flash preparation metadata:")
        for label, metadata in m["flash_metadata"].items():
            lines.append(
                f"  {label}: species={metadata.get('saturation_species')}, "
                f"T_sat={metadata.get('saturation_temperature_k')}, "
                f"P={metadata.get('target_pressure_bar')}"
            )
            compressor = metadata.get("compressor")
            if compressor:
                lines.append(
                    f"    compressor: {compressor.get('inlet_pressure_bar')} -> "
                    f"{compressor.get('outlet_pressure_bar')} bar, "
                    f"{compressor.get('energy_demand_kw')} kW, "
                    f"{compressor.get('cost_sek_per_s')} {COST_BASIS}"
                )
            else:
                lines.append("    compressor: not used")
            for key in ("sensible_heat_exchanger", "latent_condenser"):
                exchanger = metadata.get(key)
                if exchanger:
                    lines.append(
                        f"    {key}: duty={exchanger.get('duty_w')} W, "
                        f"area={exchanger.get('area_m2')} m2, "
                        f"cost={exchanger.get('cost_sek_per_s')} {COST_BASIS}, "
                        f"class={exchanger.get('calculation_class')}"
                    )
                    if exchanger.get("lmtd_k") is not None:
                        lines.append(
                            f"      cooling water: {exchanger.get('cooling_water_inlet_temperature_k')} -> "
                            f"{exchanger.get('cooling_water_outlet_temperature_k')} K, "
                            f"LMTD={exchanger.get('lmtd_k')} K"
                        )
                    if exchanger.get("condensed_components"):
                        lines.append(
                            f"      latent heat basis: components={exchanger.get('condensed_components')}, "
                            f"flow={exchanger.get('condensed_flow_mol_s')} mol/s, "
                            f"reference={exchanger.get('reference_latent_heat_component')}"
                        )
            water_oven_reheater = metadata.get("water_oven_reheater")
            if water_oven_reheater:
                lines.append(
                    f"    water_oven_reheater: duty={water_oven_reheater.get('duty_w')} W, "
                    f"class={water_oven_reheater.get('calculation_class')}"
                )
                lines.append(f"      {water_oven_reheater.get('note')}")

    if m.get("column_heat_exchangers"):
        lines.append("")
        lines.append("Distillation heat exchanger metadata:")
        for label, exchanger in m["column_heat_exchangers"].items():
            lines.append(
                f"  {label}: duty={exchanger.get('duty_w')} W, "
                f"area={exchanger.get('area_m2')} m2, "
                f"capex_cost={exchanger.get('capex_cost_sek_per_s')} {COST_BASIS}, "
                f"utility_cost={exchanger.get('utility_cost_sek_per_s')} {COST_BASIS}, "
                f"class={exchanger.get('calculation_class')}"
            )
            if exchanger.get("lmtd_k") is not None:
                lines.append(f"    LMTD={exchanger.get('lmtd_k')} K")
            if label == "condenser":
                lines.append(
                    f"    cooling water: {exchanger.get('cooling_water_inlet_temperature_k')} -> "
                    f"{exchanger.get('cooling_water_outlet_temperature_k')} K, "
                    f"condensing_temperature={exchanger.get('condensation_temperature_k')} K"
                )
            if label == "reboiler":
                lines.append(
                    f"    selected hot stream: {exchanger.get('hot_inlet_temperature_k')} -> "
                    f"{exchanger.get('hot_outlet_temperature_k')} K, "
                    f"process_temperature={exchanger.get('process_temperature_k')} K"
                )
                lines.append(
                    f"    candidate costs: {exchanger.get('candidate_costs_sek_per_s')}"
                )

    lines.append("")
    lines.append(f"Cost breakdown [{COST_BASIS}]:")
    for name, value in result.cost_breakdown.items():
        lines.append(f"  {name}: {value:.6g}")

    if m.get("capex_langfactor_notes"):
        lines.append("")
        lines.append("CAPEX Lang factor handling:")
        lines.append(f"  LANGFAKTOR: {m.get('langfactor')}")
        for note in m["capex_langfactor_notes"]:
            lines.append(f"  - {note}")
        for name, value in m.get("capex_langfactor_applied", {}).items():
            lines.append(f"  {name}: LANGFAKTOR {value}")

    lines.append("")
    lines.append(
        f"Total operating cost over {m.get('lifetime_years', LIFETIME_YEARS)} years "
        f"({m.get('operating_hours_per_year', eko.DRIFTTID_H)} h/year): "
        f"{total_operating_cost_per_lifetime(result):.6g} SEK"
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

    if result.simplifications:
        lines.append("")
        lines.append("Binary separation simplifications:")
        for note in result.simplifications:
            lines.append(f"  - {note}")

    if result.excluded_costs:
        lines.append("")
        lines.append("Excluded costs:")
        for note in result.excluded_costs:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def validate_binary_helpers() -> None:
    pressure_bar = 2.0
    saturation_temperature = saturation_temperature_from_antoine("H2O", pressure_bar)
    saturation_pressure_bar = find_saturation_pressure(
        saturation_temperature,
        *antoine["H2O"],
    ) / MMHG_PER_BAR
    assert abs(saturation_pressure_bar - pressure_bar) <= 1e-8

    inlet = Stream(
        temperature=350.0,
        pressure=5.0,
        phase="vapor",
        flowrates={"propane": 10.0, "propene": 5.0, "H2": 3.0, "H2O": 7.0},
    )
    proxy = make_binary_proxy_stream(inlet)
    assert proxy.flowrates == {"propane": 10.0, "propene": 5.0, "H2": 0.0, "H2O": 0.0}

    binary_liquid = Stream(
        temperature=330.0,
        pressure=5.0,
        phase="liquid",
        flowrates={"propane": 6.0, "propene": 1.0, "H2": 0.0, "H2O": 0.0},
    )
    binary_vapor = Stream(
        temperature=330.0,
        pressure=5.0,
        phase="vapor",
        flowrates={"propane": 4.0, "propene": 4.0, "H2": 0.0, "H2O": 0.0},
    )
    liquid, vapor = reconstruct_binary_separator_outlets(inlet, binary_liquid, binary_vapor)
    errors = check_species_mass_balance(inlet, liquid, vapor)
    assert max_abs_balance_error(errors) <= MASS_BALANCE_TOL
    assert liquid.flowrates["H2"] == 0.0
    assert vapor.flowrates["H2O"] == 0.0
    assert liquid.flowrates["H2O"] == inlet.flowrates["H2O"]
    assert vapor.flowrates["H2"] == inlet.flowrates["H2"]

    zero_inlet = Stream(
        temperature=350.0,
        pressure=5.0,
        phase="vapor",
        flowrates={"propane": 0.0, "propene": 0.0, "H2": 2.0, "H2O": 4.0},
    )
    zero_proxy = make_binary_proxy_stream(zero_inlet)
    zero_liq, zero_vap = zero_binary_outlets(zero_proxy)
    liquid, vapor = reconstruct_binary_separator_outlets(zero_inlet, zero_liq, zero_vap)
    errors = check_species_mass_balance(zero_inlet, liquid, vapor)
    assert max_abs_balance_error(errors) <= MASS_BALANCE_TOL
    assert liquid.flowrates["H2O"] == 4.0
    assert vapor.flowrates["H2"] == 2.0

    water_liquid, water_vapor, water_errors, water_metadata = run_water_removal_flash(
        inlet,
        inlet.pressure,
    )
    assert water_metadata["compressor"] is None
    assert max_abs_balance_error(water_errors) <= MASS_BALANCE_TOL
    assert water_liquid.flowrates["H2O"] == inlet.flowrates["H2O"]
    assert water_liquid.flowrates["propane"] == 0.0
    assert water_liquid.flowrates["propene"] == 0.0
    assert water_liquid.flowrates["H2"] == 0.0
    assert water_vapor.flowrates["H2O"] == 0.0

    hydrogen_liquid, hydrogen_vapor, hydrogen_errors, hydrogen_metadata = run_hydrogen_removal_flash(
        water_vapor,
        12.0,
    )
    assert max_abs_balance_error(hydrogen_errors) <= MASS_BALANCE_TOL
    assert hydrogen_vapor.flowrates["H2"] == water_vapor.flowrates["H2"]
    assert hydrogen_vapor.flowrates["propane"] == 0.0
    assert hydrogen_vapor.flowrates["propene"] == 0.0
    assert hydrogen_vapor.flowrates["H2O"] == 0.0
    assert hydrogen_liquid.flowrates["propane"] == water_vapor.flowrates["propane"]
    assert hydrogen_liquid.flowrates["propene"] == water_vapor.flowrates["propene"]
    assert hydrogen_metadata["latent_condenser"]["reference_latent_heat_component"] == "propene"
    assert hydrogen_metadata["latent_condenser"]["condensed_components"] == ("propane", "propene")
    assert hydrogen_metadata["latent_condenser"]["condensed_flow_mol_s"] == (
        water_vapor.flowrates["propane"] + water_vapor.flowrates["propene"]
    )

    validation_oven = ProcessOven()
    reheated, reheater_meta = reheat_water_stream_with_oven(water_liquid, 700.0, validation_oven)
    assert reheated.temperature == 700.0
    assert reheated.total_flowrate() == water_liquid.total_flowrate()
    assert reheater_meta["calculation_class"] == "ProcessOven"

    validation_bottoms = Stream(
        temperature=320.0,
        pressure=12.0,
        flowrates={"propane": 10.0, "propene": 1.0, "H2": 0.0, "H2O": 0.0},
        phase="liquid",
    )
    validation_distillate = Stream(
        temperature=305.0,
        pressure=12.0,
        flowrates={"propane": 0.1, "propene": 10.0, "H2": 0.0, "H2O": 0.0},
        phase="vapor",
    )
    reboiler_metadata = cheapest_reboiler_heat_exchanger_metadata(validation_bottoms, 1.0e6)
    condenser_metadata = distillation_condenser_heat_exchanger_metadata(validation_distillate, -1.0e6)
    assert reboiler_metadata["calculation_class"] == "HeatExchanger"
    assert reboiler_metadata["hot_inlet_temperature_k"] == max(REBOILER_HOT_INLET_TEMPERATURES_K)
    assert condenser_metadata["calculation_class"] == "HeatExchanger"
    assert condenser_metadata["cooling_water_inlet_temperature_k"] == COOLING_WATER_INLET_TEMPERATURE_K


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the GKT plant with binary separation assumptions.")
    parser.add_argument("--n-values", default="3", help="Comma-separated reactor counts.")
    parser.add_argument("--stage-values", default="89, 90, 91, 92, 93, 94, 95, 96", help="Comma-separated binary distillation stage counts.")
    parser.add_argument("--feed-stage-fracs", default="0.5, 0.55, 0.6, 0.7, 0.72, 0.74, 0.76, 0.8, 0.82, 0.83, 0.85, 0.9", help="Comma-separated feed-stage fractions.")
    parser.add_argument("--maxiter", type=int, default=5, help="Differential evolution iterations per structure.")
    parser.add_argument("--popsize", type=int, default=4, help="Differential evolution population multiplier.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative convergence tolerance for differential evolution.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute convergence tolerance for differential evolution.")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--disp", action="store_true")
    parser.add_argument("--fixed-feed", action="store_true", help="Use the original fixed feed instead of optimizing feed.")
    parser.add_argument("--x-target", type=float, default=DEFAULT_X_TARGET)
    parser.add_argument("--validate", action="store_true", help="Run binary helper validation checks and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate:
        validate_binary_helpers()
        print("Binary separation helper validation passed.")
        return

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
        disp=args.disp,
    )
    best, _ = run_optimizer(config)
    print(format_result(best))


if __name__ == "__main__":
    main()
