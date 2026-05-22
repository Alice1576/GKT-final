from classes import Stream
from scipy.optimize import fsolve
from thermo import find_saturation_pressure, find_beta


class Tray:

    def __init__(self, pressure: float):
        self.pressure = pressure
        self.temperature = None

    def run(self, *streams: Stream) -> tuple[Stream, Stream]:

        inlet_enthalpy = 0
        total_flowrates = {"propane": 0, "propene": 0, "H2": 0, "H2O": 0}
        T_guess = None

        for stream in streams:
            inlet_enthalpy += stream.enthalpy

            for species in total_flowrates:
                total_flowrates[species] += stream.flowrates[species]

            if T_guess is None:
                T_guess = stream.temperature

        if T_guess is None:
            raise ValueError("At least one stream must be provided")

        self.temperature = \
        fsolve(lambda temperature: self.enthalpy_balance(temperature, inlet_enthalpy, total_flowrates), T_guess)[0]

        liquid_stream, vapor_stream = self.flash_at_T(self.temperature, self.pressure, total_flowrates)

        return liquid_stream, vapor_stream

    def enthalpy_balance(self, temperature: float, inlet_enthalpy: float, total_flowrates: dict) -> float:

        liquid_stream, vapor_stream = self.flash_at_T(temperature=temperature, pressure=self.pressure,
                                                      total_flowrates=total_flowrates)
        new_enthalpy = liquid_stream.enthalpy + vapor_stream.enthalpy

        return new_enthalpy - inlet_enthalpy

    @staticmethod
    def flash_at_T(temperature: float, pressure: float, total_flowrates: dict) -> tuple[Stream, Stream]:

        antoine = {"H2": (13.6333, 164.90, 3.19),  # för T i Kelvin och P i mmHg
                   "propane": (15.7260, 1872.46, -25.26),
                   "propene": (15.7027, 1807.53, -26.15),
                   "H2O": (18.3036, 3816.44, -46.13)
                   }

        total_flow = sum(total_flowrates.values())
        z = {k: v / total_flow for k, v in total_flowrates.items()}
        species_data = {}

        for species in total_flowrates.keys():
            A, B, C = antoine[species]
            saturation_pressure_mmHg = find_saturation_pressure(temperature, A, B, C)
            saturation_pressure_bar = saturation_pressure_mmHg / 750.06156130264
            K = saturation_pressure_bar / pressure

            species_data[species] = (z[species], K)

        F0 = find_beta(0, species_data)
        F1 = find_beta(1, species_data)

        if F0 < 0 and F1 < 0:
            beta_sol = 0.0

        elif F0 > 0 and F1 > 0:
            beta_sol = 1.0

        elif F0 > 0 > F1:
            beta_sol = fsolve(lambda beta: find_beta(beta, species_data), 0.5)[0]

        x = {}
        y = {}

        for species, (z_i, K_i) in species_data.items():
            x_i = z_i / (1 + beta_sol * (K_i - 1))
            y_i = K_i * x_i

            x[species] = x_i
            y[species] = y_i

        liquid_flows = {k: x[k] * (1 - beta_sol) * total_flow for k in x}
        vapor_flows = {k: y[k] * beta_sol * total_flow for k in y}

        liquid_stream = Stream(temperature=temperature, pressure=pressure, flowrates=liquid_flows, phase="liquid")
        vapor_stream = Stream(temperature=temperature, pressure=pressure, flowrates=vapor_flows, phase="vapor")

        return liquid_stream, vapor_stream
