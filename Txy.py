import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

antoine = {
    "propene": (15.7027, 1807.53, -26.15),
    "propane": (15.7260, 1872.46, -25.26)
}


def Txy_diagram(P: float):
    """
    Skapar ett T-x-y diagram givet ett tryck P
    """
    x1_vals = np.linspace(0, 1, 50)

    T_vals = []
    y1_vals = []

    species_list = ["propane", "propene"]

    for x1 in x1_vals:
        x = {"propane": 1 - x1,
             "propene": x1}

        Tb = find_bubble_temperature(P, x)
        T_vals.append(Tb)

        y = {}

        for sp in species_list:
            A, B, C = antoine[sp]
            Psat = find_saturation_pressure(Tb, A, B, C) / 750.06156130264
            K = Psat / P

            y[sp] = x[sp] * K

        y1_vals.append(y["propene"])

    plt.figure(figsize=(8, 6))

    plt.plot(x1_vals, T_vals, label="Bubbellinje", )
    plt.plot(y1_vals, T_vals, label="Dagglinje")
    plt.xlabel("Molfraktion propen")
    plt.ylabel("Temperatur (K)")
    plt.legend()
    plt.grid(True)
    plt.show()


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
        A, B, C = antoine[species]
        saturation_pressure_mmHg = find_saturation_pressure(temperature, A, B, C)
        saturation_pressure_bar = saturation_pressure_mmHg / 750.06156130264
        K = saturation_pressure_bar / pressure

        sum += (mole_fractions[species] / K)

    return sum - 1


def find_saturation_pressure(T, A, B, C):
    exponent = np.exp((A - B / (T + C)))

    return exponent
