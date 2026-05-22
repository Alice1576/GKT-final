from __future__ import annotations

import contextlib
import os
import re
from ast import literal_eval
from dataclasses import dataclass

from classes import Compressor, FlashTank, ProcessOven, Reactor, Stream, VectorizedDistillationColumn
from optimizeCodexBinary import (
    CATALYST_DENSITY_KG_M3,
    DEFAULT_COMPRESSOR_EFFICIENCY,
    FRESH_FEED_TEMPERATURE_K,
    MASS_BALANCE_TOL,
    REACTOR_PRESSURE_BAR,
    SPECIES,
    check_species_mass_balance,
    copy_stream,
    ensure_compressor_cost_key,
    make_binary_proxy_stream,
    max_abs_balance_error,
    normalized_flowrates,
    reconstruct_binary_separator_outlets,
    saturation_temperature_from_antoine,
)


DATA_FILE = "data_slutrapport.txt"
PRINT_INTERNAL_COLUMN_STREAMS = True
OUTPUT_FILE = "plant_stream_inventory.txt"
COLUMN_STREAMS_LATEX_FILE = "column_internal_streams_table.tex"


@dataclass
class PlantCase:
    reactor_temperatures_k: list[float]
    reactor_conversions: list[float]
    flash_pressures_bar: list[float]
    distillation_feed_temperature_k: float
    distillation_pressure_bar: float
    distillation_stages: int
    distillation_feed_stage: int
    reflux_ratio: float
    distillate_fraction: float
    fresh_propane_mol_s: float
    fresh_water_mol_s: float


@dataclass
class StreamRecord:
    name: str
    stream: Stream
    source: str
    internal_column_stream: bool = False


def add_stream(
    records: list[StreamRecord],
    name: str,
    stream: Stream,
    source: str,
    internal_column_stream: bool = False,
) -> Stream:
    records.append(
        StreamRecord(
            name=name,
            stream=copy_stream(stream),
            source=source,
            internal_column_stream=internal_column_stream,
        )
    )
    return stream


def fmt(value: float) -> str:
    return f"{value:.12g}"


def fmt_latex(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6g}"


def read_report_value(report_text: str, label: str) -> str:
    pattern = rf"^\s*{re.escape(label)}:\s*(.+?)\s*$"
    match = re.search(pattern, report_text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find '{label}' in {DATA_FILE}.")
    return match.group(1)


def load_plant_case(path: str = DATA_FILE) -> PlantCase:
    with open(path, "r", encoding="utf-8") as file:
        report_text = file.read()

    return PlantCase(
        reactor_temperatures_k=list(literal_eval(read_report_value(report_text, "Reactor inlet temperatures [K]"))),
        reactor_conversions=list(literal_eval(read_report_value(report_text, "Reactor conversions"))),
        flash_pressures_bar=list(literal_eval(read_report_value(report_text, "Flash pressures [bar]"))),
        distillation_feed_temperature_k=float(read_report_value(report_text, "Binary distillation feed temperature [K]")),
        distillation_pressure_bar=float(read_report_value(report_text, "Binary distillation pressure [bar]")),
        distillation_stages=int(read_report_value(report_text, "Binary distillation stages")),
        distillation_feed_stage=int(read_report_value(report_text, "Feed stage")),
        reflux_ratio=float(read_report_value(report_text, "Reflux ratio")),
        distillate_fraction=float(read_report_value(report_text, "Distillate fraction of binary feed")),
        fresh_propane_mol_s=float(read_report_value(report_text, "Fresh propane [mol/s]")),
        fresh_water_mol_s=float(read_report_value(report_text, "Fresh water [mol/s]")),
    )


def format_stream(record: StreamRecord, index: int) -> str:
    stream = record.stream
    mole_fractions = stream.mole_fraction()
    flow_lines = [
        f"    {species}: {fmt(stream.flowrates.get(species, 0.0))} mol/s"
        f"  x={fmt(mole_fractions.get(species, 0.0))}"
        for species in SPECIES
    ]
    return "\n".join(
        [
            f"[{index:03d}] {record.name}",
            f"  source: {record.source}",
            f"  phase: {stream.phase}",
            f"  temperature: {fmt(stream.temperature)} K ({fmt(stream.temperature - 273.15)} deg C)",
            f"  pressure: {fmt(stream.pressure)} bar",
            f"  total flow: {fmt(stream.total_flowrate())} mol/s",
            f"  enthalpy: {fmt(stream.enthalpy)} W",
            "  component flows and mole fractions:",
            *flow_lines,
        ]
    )


def heat_first_reactor_feed(
    records: list[StreamRecord],
    feed: Stream,
    oven: ProcessOven,
    reactor_temperature: float,
) -> Stream:
    feed_flows = normalized_flowrates(feed.flowrates)
    water_flow = feed_flows["H2O"]

    water_side_stream = copy_stream(
        feed,
        temperature=reactor_temperature,
        flowrates={"propane": 0.0, "propene": 0.0, "H2": 0.0, "H2O": water_flow},
    )
    add_stream(
        records,
        "First reactor preheated water side stream",
        water_side_stream,
        "Optimizer assumption: water is available at the first reactor temperature after flash-1 reheating.",
    )

    feed_flows["H2O"] = 0.0
    dry_feed = copy_stream(feed, flowrates=feed_flows)
    add_stream(
        records,
        "First process oven propane-only inlet",
        dry_feed,
        "Dry propane-containing part of the fresh feed before ProcessOven.run.",
    )

    heated_dry_feed = oven.run(dry_feed, reactor_temperature)
    add_stream(
        records,
        "First process oven propane-only outlet",
        heated_dry_feed,
        "ProcessOven.run on the propane-containing feed only.",
    )

    reactor_feed_flows = normalized_flowrates(heated_dry_feed.flowrates)
    reactor_feed_flows["H2O"] = water_flow
    reactor_feed = copy_stream(
        heated_dry_feed,
        temperature=reactor_temperature,
        flowrates=reactor_feed_flows,
    )
    add_stream(
        records,
        "Reactor 1 inlet after recombining preheated water",
        reactor_feed,
        "Heated propane feed plus the preheated water side stream.",
    )
    return reactor_feed


def pressure_raise_with_record(
    records: list[StreamRecord],
    stream: Stream,
    target_pressure: float,
    label: str,
) -> Stream:
    if target_pressure <= stream.pressure:
        return stream
    ensure_compressor_cost_key()
    compressor = Compressor(
        outlet_pressure=target_pressure,
        isentropic_efficiency=DEFAULT_COMPRESSOR_EFFICIENCY,
    )
    outlet = compressor.run(stream)
    add_stream(
        records,
        f"{label} compressor outlet",
        outlet,
        (
            "Compressor.run; "
            f"power={fmt(compressor.total_energy_demand_kW)} kW, "
            f"intercooling={fmt(compressor.total_cooling_demand_kW)} kW."
        ),
    )
    return outlet


def prepare_flash_feed(
    records: list[StreamRecord],
    stream: Stream,
    pressure: float,
    saturation_species: str,
    label: str,
) -> Stream:
    pressure_adjusted = pressure_raise_with_record(records, stream, pressure, label)
    saturation_temperature = saturation_temperature_from_antoine(saturation_species, pressure)
    prepared = copy_stream(
        pressure_adjusted,
        temperature=saturation_temperature,
        pressure=pressure,
    )
    add_stream(
        records,
        f"{label} saturated feed",
        prepared,
        (
            "Sensible heat-exchanger routing from optimizeCodexBinary.py; "
            f"T is {saturation_species} saturation at {fmt(pressure)} bar."
        ),
    )
    return prepared


def split_flash1(records: list[StreamRecord], inlet: Stream) -> tuple[Stream, Stream, FlashTank]:
    flash = FlashTank(temperature=inlet.temperature, pressure=inlet.pressure)
    liquid = copy_stream(
        inlet,
        phase="liquid",
        flowrates={"propane": 0.0, "propene": 0.0, "H2": 0.0, "H2O": inlet.flowrates["H2O"]},
    )
    vapor = copy_stream(
        inlet,
        phase="vapor",
        flowrates={
            "propane": inlet.flowrates["propane"],
            "propene": inlet.flowrates["propene"],
            "H2": inlet.flowrates["H2"],
            "H2O": 0.0,
        },
    )
    flash.calculate_economics(liquid)
    add_stream(records, "Flash 1 liquid outlet", liquid, "FlashTank water-removal split: all H2O to liquid.")
    add_stream(records, "Flash 1 vapor outlet", vapor, "FlashTank water-removal split: propane/propene/H2 to vapor.")
    return liquid, vapor, flash


def split_flash2(records: list[StreamRecord], inlet: Stream) -> tuple[Stream, Stream, FlashTank]:
    flash = FlashTank(temperature=inlet.temperature, pressure=inlet.pressure)
    liquid = copy_stream(
        inlet,
        phase="liquid",
        flowrates={
            "propane": inlet.flowrates["propane"],
            "propene": inlet.flowrates["propene"],
            "H2": 0.0,
            "H2O": inlet.flowrates["H2O"],
        },
    )
    vapor = copy_stream(
        inlet,
        phase="vapor",
        flowrates={"propane": 0.0, "propene": 0.0, "H2": inlet.flowrates["H2"], "H2O": 0.0},
    )
    flash.calculate_economics(liquid)
    add_stream(records, "Flash 2 liquid outlet", liquid, "FlashTank hydrogen-removal split: propane/propene to liquid.")
    add_stream(records, "Flash 2 vapor outlet", vapor, "FlashTank hydrogen-removal split: all H2 to vapor.")
    return liquid, vapor, flash


def run_reactor_section(records: list[StreamRecord], case: PlantCase) -> tuple[Stream, ProcessOven]:
    oven = ProcessOven()
    stream = Stream(
        temperature=FRESH_FEED_TEMPERATURE_K,
        pressure=1.01325,
        phase="vapor",
        flowrates={
            "propane": case.fresh_propane_mol_s,
            "propene": 0.0,
            "H2": 0.0,
            "H2O": case.fresh_water_mol_s,
        },
    )
    add_stream(records, "Fresh feed", stream, "Fresh propane and water feed before plant conditioning.")

    for index, (temperature, conversion) in enumerate(
        zip(case.reactor_temperatures_k, case.reactor_conversions),
        start=1,
    ):
        if index == 1:
            stream = heat_first_reactor_feed(records, stream, oven, temperature)
        else:
            stream = oven.run(stream, temperature)
            add_stream(
                records,
                f"Process oven {index} outlet / reactor {index} inlet",
                stream,
                f"ProcessOven.run to reactor {index} inlet temperature.",
            )

        reactor = Reactor(
            pressure=REACTOR_PRESSURE_BAR,
            catalyst_density=CATALYST_DENSITY_KG_M3,
            catalyst_mass=None,
            conversion=conversion,
        )
        stream = reactor.run2(stream)
        add_stream(
            records,
            f"Reactor {index} outlet",
            stream,
            (
                "Reactor.run2; "
                f"conversion={fmt(conversion)}, "
                f"catalyst_mass={fmt(reactor.catalyst_mass)} kg, "
                f"volume={fmt(reactor.volume)} m3."
            ),
        )

    return stream, oven


def run_distillation_section(
    records: list[StreamRecord],
    flash2_liquid: Stream,
    case: PlantCase,
) -> tuple[Stream, Stream, VectorizedDistillationColumn]:
    column_feed = Stream(
        temperature=case.distillation_feed_temperature_k,
        pressure=case.distillation_pressure_bar,
        phase="liquid",
        flowrates=normalized_flowrates(flash2_liquid.flowrates),
    )
    add_stream(
        records,
        "Binary distillation feed",
        column_feed,
        "Liquid flash-2 outlet after distillation feed temperature/pressure conditioning.",
    )

    proxy_feed = make_binary_proxy_stream(column_feed, phase="liquid")
    add_stream(
        records,
        "Binary distillation model proxy feed",
        proxy_feed,
        "Model stream used by VectorizedDistillationColumn.run; H2 and H2O are forced to zero.",
    )

    distillate_flowrate = case.distillate_fraction * proxy_feed.total_flowrate()
    column = VectorizedDistillationColumn(
        pressure=case.distillation_pressure_bar,
        stages=case.distillation_stages,
        feed_stage=case.distillation_feed_stage,
        reflux_ratio=case.reflux_ratio,
        distillate_flowrate=distillate_flowrate,
    )

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            binary_distillate, binary_bottoms = column.run(proxy_feed)

    bottoms, distillate = reconstruct_binary_separator_outlets(
        original_stream=column_feed,
        binary_liquid=binary_bottoms,
        binary_vapor=binary_distillate,
    )
    add_stream(records, "Distillation distillate product", distillate, "VectorizedDistillationColumn.run vapor product.")
    add_stream(records, "Distillation bottoms product", bottoms, "VectorizedDistillationColumn.run liquid product.")

    if PRINT_INTERNAL_COLUMN_STREAMS:
        for stage, (liquid, vapor) in sorted(column.streams.items()):
            add_stream(
                records,
                f"Column stage {stage + 1:02d} internal liquid stream",
                liquid,
                "Internal liquid stream generated by VectorizedDistillationColumn.run.",
                internal_column_stream=True,
            )
            add_stream(
                records,
                f"Column stage {stage + 1:02d} internal vapor stream",
                vapor,
                "Internal vapor stream generated by VectorizedDistillationColumn.run.",
                internal_column_stream=True,
            )

    return distillate, bottoms, column


def print_mass_balance(label: str, inlet: Stream, outlet_a: Stream, outlet_b: Stream) -> None:
    errors = check_species_mass_balance(inlet, outlet_a, outlet_b)
    print(f"{label} max absolute species balance error: {fmt(max_abs_balance_error(errors))} mol/s")
    for species in SPECIES:
        print(f"  {species}: {fmt(errors[species])} mol/s")


def format_mass_balance(label: str, inlet: Stream, outlet_a: Stream, outlet_b: Stream) -> list[str]:
    errors = check_species_mass_balance(inlet, outlet_a, outlet_b)
    lines = [f"{label} max absolute species balance error: {fmt(max_abs_balance_error(errors))} mol/s"]
    for species in SPECIES:
        lines.append(f"  {species}: {fmt(errors[species])} mol/s")
    return lines


def format_column_streams_latex(records: list[StreamRecord]) -> str:
    internal_records = [record for record in records if record.internal_column_stream]
    lines = [
        "% Generated by testmodul3.py.",
        "% Requires \\usepackage{longtable}.",
        "\\begin{longtable}{r l r r r r r r r}",
        "\\hline",
        "Stage & Phase & $T$ [K] & $P$ [bar] & Total [mol/s] & Propane [mol/s] & Propene [mol/s] & H$_2$ [mol/s] & H$_2$O [mol/s] \\\\",
        "\\hline",
        "\\endfirsthead",
        "\\hline",
        "Stage & Phase & $T$ [K] & $P$ [bar] & Total [mol/s] & Propane [mol/s] & Propene [mol/s] & H$_2$ [mol/s] & H$_2$O [mol/s] \\\\",
        "\\hline",
        "\\endhead",
    ]

    for record in internal_records:
        stream = record.stream
        parts = record.name.split()
        stage = parts[2] if len(parts) >= 3 else ""
        phase = stream.phase
        flows = stream.flowrates
        lines.append(
            " & ".join(
                [
                    str(int(stage)),
                    phase,
                    fmt_latex(stream.temperature),
                    fmt_latex(stream.pressure),
                    fmt_latex(stream.total_flowrate()),
                    fmt_latex(flows.get("propane", 0.0)),
                    fmt_latex(flows.get("propene", 0.0)),
                    fmt_latex(flows.get("H2", 0.0)),
                    fmt_latex(flows.get("H2O", 0.0)),
                ]
            )
            + r" \\"
        )

    lines.extend(["\\hline", "\\end{longtable}", ""])
    return "\n".join(lines)


def build_plant_streams(case: PlantCase) -> tuple[list[StreamRecord], dict[str, Stream], VectorizedDistillationColumn]:
    records: list[StreamRecord] = []

    reactor_outlet, oven = run_reactor_section(records, case)

    flash1_feed = prepare_flash_feed(records, reactor_outlet, case.flash_pressures_bar[0], "H2O", "Flash 1")
    flash1_liquid, flash1_vapor, _ = split_flash1(records, flash1_feed)

    reheated_water = oven.run(flash1_liquid, case.reactor_temperatures_k[0])
    add_stream(
        records,
        "Flash 1 separated water after process oven reheating",
        reheated_water,
        "ProcessOven.run on separated flash-1 water for recycle/feed water heating.",
    )

    flash2_feed = prepare_flash_feed(records, flash1_vapor, case.flash_pressures_bar[1], "propene", "Flash 2")
    flash2_liquid, flash2_vapor, _ = split_flash2(records, flash2_feed)

    distillate, bottoms, column = run_distillation_section(records, flash2_liquid, case)

    named_streams = {
        "flash1_feed": flash1_feed,
        "flash1_liquid": flash1_liquid,
        "flash1_vapor": flash1_vapor,
        "flash2_feed": flash2_feed,
        "flash2_liquid": flash2_liquid,
        "flash2_vapor": flash2_vapor,
        "distillate": distillate,
        "bottoms": bottoms,
    }
    return records, named_streams, column


def main() -> None:
    case = load_plant_case()
    records, streams, column = build_plant_streams(case)
    plant_records = [record for record in records if not record.internal_column_stream]
    internal_column_records = [record for record in records if record.internal_column_stream]

    lines = [
        "=== PLANT STREAM INVENTORY FROM optimizeCodexBinary.py BEST CASE ===",
        f"Source report file: {DATA_FILE}",
        f"Number of printed plant streams: {len(plant_records)}",
        f"Internal column streams written separately: {len(internal_column_records)}",
        f"Internal column stream LaTeX file: {COLUMN_STREAMS_LATEX_FILE}",
        "",
    ]

    for index, record in enumerate(plant_records, start=1):
        lines.append(format_stream(record, index))
        lines.append("")

    lines.extend(
        [
            "=== STREAM MASS BALANCE CHECKS ===",
            *format_mass_balance("Flash 1", streams["flash1_feed"], streams["flash1_liquid"], streams["flash1_vapor"]),
            *format_mass_balance("Flash 2", streams["flash2_feed"], streams["flash2_liquid"], streams["flash2_vapor"]),
            *format_mass_balance("Distillation", streams["flash2_liquid"], streams["distillate"], streams["bottoms"]),
            "",
            "=== COLUMN DUTIES ===",
            f"Condenser duty: {fmt(column.condenser_duty)} W",
            f"Reboiler duty: {fmt(column.reboiler_duty)} W",
        ]
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")

    with open(COLUMN_STREAMS_LATEX_FILE, "w", encoding="utf-8") as file:
        file.write(format_column_streams_latex(records))

    if max_abs_balance_error(
        check_species_mass_balance(streams["flash2_liquid"], streams["distillate"], streams["bottoms"])
    ) > MASS_BALANCE_TOL:
        raise RuntimeError("Distillation mass balance exceeds tolerance.")

    print(f"Wrote {len(plant_records)} plant stream records to {OUTPUT_FILE}")
    print(f"Wrote {len(internal_column_records)} internal column stream rows to {COLUMN_STREAMS_LATEX_FILE}")
    print(f"Condenser duty: {fmt(column.condenser_duty)} W")
    print(f"Reboiler duty: {fmt(column.reboiler_duty)} W")


if __name__ == "__main__":
    main()
