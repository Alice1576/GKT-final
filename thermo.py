from scipy.optimize import fsolve, root_scalar
import numpy as np
from classes import Stream

antoine = {"H2": (13.6333, 164.90, 3.19),  # för T i Kelvin och P i mmHg
               "propane": (15.7260, 1872.46, -25.26),
               "propene": (15.7027, 1807.53, -26.15),
               "H2O": (18.3036, 3816.44, -46.13)
               }

def find_bubble_temperature(pressure, mole_fractions: dict):

    temp = fsolve(lambda temperature: bubble_residual(temperature, pressure, mole_fractions), 500)[0]

    return temp

def bubble_residual(temperature, pressure, mole_fractions):
    sum = 0.0
    for species in mole_fractions:
        A, B, C = antoine[species]
        saturation_pressure_mmHg = find_saturation_pressure(temperature, A, B, C)
        saturation_pressure_bar = saturation_pressure_mmHg / 750.06156130264
        K = saturation_pressure_bar / pressure

        sum += mole_fractions[species] * K

    return sum - 1




def find_dew_temperature(pressure, mole_fractions: dict):

    temp = fsolve(lambda temperature: dew_residual(temperature, pressure, mole_fractions), 500)[0]

    return temp


def dew_residual(temperature, pressure, mole_fractions) -> float:

    sum = 0.0
    for species in mole_fractions:
        A,B,C = antoine[species]
        saturation_pressure_mmHg = find_saturation_pressure(temperature, A, B, C)
        saturation_pressure_bar = saturation_pressure_mmHg / 750.06156130264
        K = saturation_pressure_bar / pressure

        sum += (mole_fractions[species] / K)


    return sum - 1


def find_saturation_pressure(T, A, B, C):

    exponent = np.exp( (A - B / (T + C)))

    return exponent



def find_flashtank_equilibrium_composition(temperature: float, pressure: float, inlet_stream: Stream):

    liquid = {}
    vapor = {}

    species_data = {}

    for species in inlet_stream.flowrates.keys():
        A, B, C = antoine[species]
        arg = A - B / (temperature + C)
        arg = np.clip(arg, -100.0, 100.0)  # prevent Inf K values
        P_sat_bar = np.exp(arg) / 750.06156130264
        K = P_sat_bar / pressure

        total_mole_fraction = inlet_stream.mole_fraction()[species]
        species_data[species] = (total_mole_fraction, K)

    beta_sol = solve_flash_beta(species_data)

    if not 0 <= beta_sol <= 1:
        raise ValueError("beta_sol must be between 0 and 1")

    total_vapor_flowrate = beta_sol * inlet_stream.total_flowrate()
    total_liquid_flowrate = inlet_stream.total_flowrate() - total_vapor_flowrate

    for species in inlet_stream.flowrates.keys():
        total_mole_fraction = species_data[species][0]
        k = species_data[species][1]

        liquid_mole_fraction = total_mole_fraction / (1 + beta_sol * (k - 1))
        vapor_mole_fraction = k * liquid_mole_fraction

        liquid_flowrate = liquid_mole_fraction * total_liquid_flowrate
        vapor_flowrate = vapor_mole_fraction * total_vapor_flowrate

        liquid[species] = liquid_flowrate
        vapor[species] = vapor_flowrate

    liquid_stream = Stream(temperature=temperature, pressure=pressure, flowrates=liquid, phase="liquid")
    vapor_stream = Stream(temperature=temperature, pressure=pressure, flowrates=vapor, phase="vapor")

    return liquid_stream, vapor_stream


def solve_flash_beta(species_data: dict, tol: float = 1e-12) -> float:
    F0 = find_beta(0.0, species_data)
    F1 = find_beta(1.0, species_data)

    if abs(F0) < tol:
        return 0.0

    if abs(F1) < tol:
        return 1.0

    if F0 < 0.0 and F1 < 0.0:
        return 0.0

    if F0 > 0.0 and F1 > 0.0:
        return 1.0

    if F0 * F1 > 0.0:
        raise ValueError(f"Flash beta is not bracketed: F(0)={F0}, F(1)={F1}")

    sol = root_scalar(
        lambda beta: find_beta(beta, species_data),
        bracket=[0.0, 1.0],
        method="brentq",
        xtol=tol,
        rtol=tol,
    )

    if not sol.converged:
        raise ValueError(f"Flash beta solve failed: F(0)={F0}, F(1)={F1}")

    return float(np.clip(sol.root, 0.0, 1.0))


def find_beta(beta: float, species_data: dict):
    total = 0

    for species in species_data.keys():
        z = species_data[species][0]
        k = species_data[species][1]

        total += (z * (k - 1)) / (1 + beta * (k - 1))

    return total


def find_compressor_outlet_temperature(heat_capacities: dict, inlet_temperature: float, outlet_temperature: float,
                                       inlet_pressure: float, outlet_pressure: float) -> float:
    T1 = inlet_temperature
    T2 = outlet_temperature
    P1 = inlet_pressure
    P2 = outlet_pressure

    sum = 0
    for species in heat_capacities:
        A, B, C, D = heat_capacities[species]

        sum += A * np.log(T2) + B * T2 + 1 / 2 * C * (T2 ** 2) + 1 / 3 * D * (T2 ** 3) - (
                A * np.log(T1) + B * T1 + 1 / 2 * C * (T1 ** 2) + 1 / 3 * D * (T1 ** 3))

    return sum - 8.314 * np.log(P2 / P1)


def compressor_outlet_temperature(heat_capacities: dict, inlet_pressure: float, outlet_pressure: float,
                                  inlet_temperature: float) -> float:
    temperature_sol = fsolve(
        lambda outlet_temperature: find_compressor_outlet_temperature(heat_capacities=heat_capacities,
                                                                      inlet_temperature=inlet_temperature,
                                                                      outlet_temperature=outlet_temperature,
                                                                      inlet_pressure=inlet_pressure,
                                                                      outlet_pressure=outlet_pressure), x0=500)[0]

    return temperature_sol
