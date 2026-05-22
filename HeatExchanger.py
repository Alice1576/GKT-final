import numpy as np
from classes.Stream import Stream
from scipy.optimize import fsolve
import ekonomi as eko


class HeatExchanger:
    def __init__(
        self,
        outlet_temperature: float,
        U: float,
        hot_temperature: float,
        mass_flow_hot: float,
        heat_capacity_hot: float,
        utility_type: str,
    ):
        self.outlet_temperature = outlet_temperature
        self.U = U

        self.utility_type = utility_type

        self.heat_demand = None
        self.area = None
        self.cost = None

        self.capex = 0
        self.annual_opex = 0

        self.hot_temperature = hot_temperature
        self.mass_flow_hot = mass_flow_hot
        self.heat_capacity_hot = heat_capacity_hot

    def efficiency(self, Q, C_min, Th_in, Tc_in):
        return Q / (C_min * abs(Th_in - Tc_in))

    def epsilon_NTU_relation(self, NTU, C_r):
        return (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))

    def NTU_solver(self, epsilon, C_r):
        def equation(NTU):
            return self.epsilon_NTU_relation(NTU, C_r) - epsilon

        sol = fsolve(equation, 0.1)[0]

        return sol

    def find_area(self, stream: Stream, Th_in, mh, cph):
        Tc_in = stream.temperature
        Cc = 0

        for species, flowrate in stream.flowrates.items():
            A, B, C, D = stream.heat_capacities[species]
            cp_molar = A + B * Tc_in + C * (Tc_in ** 2) + D * (Tc_in ** 3)
            Cc += flowrate * cp_molar

        Ch = mh * cph

        C_min = min(Cc, Ch)
        C_max = max(Cc, Ch)
        if C_min <= 0.0 or C_max <= 0.0:
            return 0.0
        C_r = C_min / C_max

        Q = self.heat_demand

        if abs(Th_in - Tc_in) < 1e-6 or Q < 1e-6:
            return 0.0

        epsilon = self.efficiency(Q, C_min, Th_in, Tc_in)
        if epsilon >= 1.0:
            print(
                f"Varning: Epsilon >= 1 (epsilon = {epsilon:.2f}). "
                "Varmevaxling ar fysiskt omojlig med angivet flode/temperatur pa mediet."
            )
            epsilon = 0.999
        NTU = self.NTU_solver(epsilon, C_r)

        A = NTU * C_min / self.U

        return A

    def run(self, stream: Stream) -> Stream:
        outlet_stream = Stream(
            temperature=self.outlet_temperature,
            pressure=stream.pressure,
            flowrates=stream.flowrates,
            phase=stream.phase,
        )

        self.calculate_from_duty(stream, abs(outlet_stream.enthalpy - stream.enthalpy))

        return outlet_stream

    def calculate_from_duty(self, stream: Stream, heat_demand: float, delta_t: float | None = None):
        """Calculate area and cost for an externally known heat duty."""
        self.heat_demand = abs(heat_demand)

        if delta_t is None:
            self.area = self.find_area(
                stream,
                self.hot_temperature,
                self.mass_flow_hot,
                self.heat_capacity_hot,
            )
        elif self.heat_demand <= 1e-9:
            self.area = 0.0
        else:
            self.area = self.heat_demand / (self.U * max(abs(delta_t), 1e-6))

        area_for_cost = max(10.0, min(self.area, 1000.0))
        self.capex = eko.calculate_capex("heat_exchanger", area_for_cost)
        self.cost = self.capex

        Q_kW = self.heat_demand / 1000
        self.annual_opex = eko.calculate_opex(self.utility_type, Q_kW)

        return self

    def calculate_condensation_with_cooling_water(
        self,
        stream: Stream,
        heat_demand: float,
        condensing_temperature: float,
        cooling_water_inlet_temperature: float = 14.0 + 273.15,
        cooling_water_outlet_temperature: float = 20.0 + 273.15,
    ):
        """Calculate condenser area and cost using cooling water from 14 C to 20 C."""
        self.heat_demand = abs(heat_demand)

        delta_t_in = condensing_temperature - cooling_water_outlet_temperature
        delta_t_out = condensing_temperature - cooling_water_inlet_temperature
        if delta_t_in <= 0.0 or delta_t_out <= 0.0:
            raise ValueError(
                "Condensing temperature must be above both cooling-water temperatures."
            )

        if abs(delta_t_out - delta_t_in) < 1e-9:
            lmtd = delta_t_in
        else:
            lmtd = (delta_t_out - delta_t_in) / np.log(delta_t_out / delta_t_in)

        self.area = 0.0 if self.heat_demand <= 1e-9 else self.heat_demand / (self.U * lmtd)

        area_for_cost = max(10.0, min(self.area, 1000.0))
        self.capex = eko.calculate_capex("heat_exchanger", area_for_cost)
        self.cost = self.capex

        Q_kW = self.heat_demand / 1000
        self.annual_opex = eko.calculate_opex(self.utility_type, Q_kW)
        self.cooling_water_inlet_temperature = cooling_water_inlet_temperature
        self.cooling_water_outlet_temperature = cooling_water_outlet_temperature
        self.lmtd = lmtd

        return self
