from .Stream import *
from scipy.optimize import least_squares
import numpy as np
from thermo import find_saturation_pressure

mm_Hg_to_bar = 1 / 750.06156130264

species_list = ["propane", "propene", "H2", "H2O"]
antoine = {"H2": (13.6333, 164.90, 3.19),  # för T i Kelvin och P i mmHg
           "propane": (15.7260, 1872.46, -25.26),
           "propene": (15.7027, 1807.53, -26.15),
           "H2O": (18.3036, 3816.44, -46.13)
           }


class DistillationColumn:
    def __init__(self, pressure, stages, feed_stage, reflux_ratio: float, tray_efficiency: float, tray_distance: float):
        self.pressure = pressure
        self.stages = stages
        self.feed_stage = feed_stage
        self.reflux_ratio = reflux_ratio
        self.tray_efficiency = tray_efficiency
        self.tray_distance = tray_distance

        self.streams = {}
        self.vapor_streams = None
        self.liquid_streams = None

        self.diameter = None
        self.height = None
        self.reboiler_duty = 0
        self.condenser_duty = 0

        self.tray_cost = None
        self.condenser_cost = None
        self.reboiler_cost = None
        self.shell_cost = None
        self.total_cost = None

    def run(self, feed: Stream):

        P = self.pressure
        N = self.stages

        liq_comp_guess, liq_flows_guess, vap_flows_guess, temperatures_guess, reboiler_guess = self.initial_guess(feed)
        x0 = self.pack(liq_comp=liq_comp_guess, liq_flows=liq_flows_guess, vap_flows=vap_flows_guess,
                       temperatures=temperatures_guess, reboiler_duty=reboiler_guess)

        bounds = self.get_bounds()

        sol = least_squares(
            fun=lambda x: self.residual_wrapper(feed_stream=feed, x=x),
            x0=x0,
            bounds=bounds,
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=50000,
            verbose=2
        )

        res_arr = sol.fun
        max_idx = np.argmax(np.abs(res_arr))
        print("Largest residual at index", max_idx, "value:", res_arr[max_idx])
        if not sol.success:
            print(f"Solver did not converge: {sol.message}")

        liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty = self.unpack(P, N, sol.x)

        print("liq_flows:", liq_flows)
        print("vap_flows:", vap_flows)

        self.reboiler_duty = reboiler_duty

        distillate_flowrates = {}
        distillate_temperature = temperatures[0]

        for sp in species_list:
            distillate_flowrates[sp] = vap_flows[1] * vap_comp[1][sp] - liq_flows[0] * liq_comp[0][sp]

        distillate = Stream(pressure=P, temperature=distillate_temperature, flowrates=distillate_flowrates,
                            phase="liquid")

        for i, liq_flow in enumerate(liq_flows):
            T = temperatures[i]

            liq_stream = Stream(temperature=T, pressure=self.pressure,
                                flowrates={sp: liq_flow * liq_comp[i][sp] for sp in species_list}, phase="liquid")

            vap_stream = Stream(temperature=T, pressure=self.pressure,
                                flowrates={sp: vap_flows[i] * vap_comp[i][sp] for sp in species_list}, phase="vapor")

            self.streams[i] = (liq_stream, vap_stream)

        self.condenser_duty = (distillate.enthalpy + self.streams[0][0].enthalpy) - self.streams[1][1].enthalpy

        bottoms_flowrates = {}
        for sp in species_list:
            bottoms_flowrates[sp] = liq_comp[N][sp] * liq_flows[N]

        bottoms_temperature = temperatures[N]

        bottoms = Stream(pressure=P, temperature=bottoms_temperature, flowrates=bottoms_flowrates, phase="liquid")

        return distillate, bottoms

    def residual_wrapper(self, feed_stream: Stream, x: list):

        P = self.pressure
        N = self.stages
        liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty = self.unpack(P, N, x)

        return self.residuals(feed_stream, liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty)

    def residuals(self, feed: Stream, liq_comp: list[dict], vap_comp: list[dict], liq_flows: list, vap_flows: list,
                  temperatures: list, reboiler_duty: float):

        F_tot = feed.total_flowrate()
        bubble_res = []
        mass_res = []
        enthalpy_res = []
        condenser_mass_res = []
        condenser_comp_res = []
        summation_res = []

        N = self.stages
        P = self.pressure
        R = self.reflux_ratio
        Q = reboiler_duty

        for i in range(1, N + 1):
            T = temperatures[i]
            Kx_sum = 0

            for sp in species_list:
                P_sat = find_saturation_pressure(T, *antoine[sp]) * mm_Hg_to_bar
                K = P_sat / P

                Kx_sum += liq_comp[i][sp] * K

            res = Kx_sum - 1
            bubble_res.append(res)

        for i in range(1, N + 1):

            T_above = temperatures[i - 1]
            T = temperatures[i]
            T_below = temperatures[i + 1] if i < N else 300

            liq_flowrates_in = {}
            liq_flowrates_out = {}
            vap_flowrates_in = {}
            vap_flowrates_out = {}

            for sp in species_list:
                liq_flowrates_in[sp] = liq_comp[i - 1][sp] * liq_flows[i - 1]
                liq_flowrates_out[sp] = liq_comp[i][sp] * liq_flows[i]

                if i < N:
                    vap_flowrates_in[sp] = vap_comp[i + 1][sp] * vap_flows[i + 1]
                else:
                    vap_flowrates_in[sp] = 0.0

                vap_flowrates_out[sp] = vap_comp[i][sp] * vap_flows[i]

            enthalpy_in = self._stream_enthalpy(T_above, liq_flowrates_in, "liquid") + self._stream_enthalpy(T_below,
                                                                                                             vap_flowrates_in,
                                                                                                             "vapor")
            enthalpy_out = self._stream_enthalpy(T, liq_flowrates_out, "liquid") + self._stream_enthalpy(T,
                                                                                                         vap_flowrates_out,
                                                                                                         "vapor")

            if i == self.feed_stage:
                enthalpy_in += feed.enthalpy

            if i == N:
                enthalpy_in += Q

            res = enthalpy_in - enthalpy_out

            enthalpy_res.append(res)

        for i in range(1, N + 1):
            flowrates_in = {}
            flowrates_out = {}

            for sp in species_list:
                if i < N:
                    flowrates_in[sp] = liq_comp[i - 1][sp] * liq_flows[i - 1] + vap_comp[i + 1][sp] * vap_flows[i + 1]

                else:
                    flowrates_in[sp] = liq_comp[i - 1][sp] * liq_flows[i - 1]

                if i == self.feed_stage:
                    flowrates_in[sp] += feed.flowrates[sp]

                flowrates_out[sp] = liq_comp[i][sp] * liq_flows[i] + vap_comp[i][sp] * vap_flows[i]

            for sp in species_list:
                res = flowrates_in[sp] - flowrates_out[sp]

                mass_res.append(res)

        for i in range(N + 1):
            mole_fraction_sum = 0
            for sp in species_list:
                mole_fraction_sum += liq_comp[i][sp]

            res = mole_fraction_sum - 1
            summation_res.append(res)

        L0 = liq_flows[0]
        V1 = vap_flows[1]
        T0 = temperatures[0]

        res = V1 - (1 + 1 / R) * L0
        condenser_mass_res.append(res)

        Kx_sum = 0
        for sp in species_list:
            P_sat = find_saturation_pressure(T0, *antoine[sp]) * mm_Hg_to_bar
            K = P_sat / P

            Kx_sum += vap_comp[1][sp] * K

        res = Kx_sum - 1
        condenser_comp_res.append(res)

        for sp in species_list[:3]:
            res = liq_comp[0][sp] - vap_comp[1][sp]
            condenser_comp_res.append(res)

        enthalpy_res = [r / 1e6 for r in enthalpy_res]

        return bubble_res + enthalpy_res + condenser_mass_res + condenser_comp_res + mass_res + summation_res

    @staticmethod
    def pack(liq_comp: list[dict], liq_flows: list, vap_flows: list, temperatures: list, reboiler_duty: float):
        x = []

        for dictionary in liq_comp:
            for sp in species_list:
                x.append(dictionary[sp])

        for flow in liq_flows:
            x.append(flow)

        for flow in vap_flows:
            x.append(flow)

        for temperature in temperatures:
            x.append(temperature)

        x.append(reboiler_duty)

        return x

    @staticmethod
    def unpack(P: float, N: int, x: list[float]):
        liq_comp = []
        vap_comp = []
        liq_flows = []
        vap_flows = []
        temperatures = []

        k = 0

        for i in range(N + 1):
            mole_fractions = {}

            for sp in species_list:
                mole_fractions[sp] = x[k]
                k += 1

            liq_comp.append(mole_fractions)

        for i in range(N + 1):
            liq_flows.append(x[k])
            k += 1

        for i in range(N + 1):
            if i == 0:
                vap_flows.append(0.0)
            else:
                vap_flows.append(x[k])
                k += 1

        for i in range(N + 1):
            T = x[k]
            temperatures.append(T)
            k += 1

        reboiler_duty = x[k]

        for i in range(N + 1):
            mole_fractions = {}
            T = temperatures[i]
            for sp in species_list:
                P_sat = find_saturation_pressure(T, *antoine[sp]) * mm_Hg_to_bar
                K = P_sat / P

                y = K * liq_comp[i][sp]
                mole_fractions[sp] = y

            vap_comp.append(mole_fractions)

        return liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty

    def initial_guess(self, feed: Stream):

        N = self.stages

        liq_comp = []
        liq_flows = []
        vap_flows = []
        temperatures = []

        F = feed.flowrates
        F_tot = feed.total_flowrate()

        for i in range(N + 1):
            mole_fractions = {sp: F[sp] / F_tot for sp in species_list}
            liq_comp.append(mole_fractions)

        for i in range(N + 1):
            liq_flows.append(F_tot / N)

        for i in range(1, N + 1):
            vap_flows.append(F_tot / N)

        temperatures.append(300)
        for i in range(1, N + 1):
            temperatures.append(320 + (80 / N) * i)

        reboiler_duty = 1e6

        return liq_comp, liq_flows, vap_flows, temperatures, reboiler_duty

    def get_bounds(self):
        N = self.stages
        lb = []
        ub = []

        for _ in range(N + 1):
            for sp in species_list:
                lb.append(0.0)
                ub.append(1.0)

        for _ in range(N + 1):
            lb.append(0.0)
            ub.append(np.inf)

        for _ in range(N):
            lb.append(0.0)
            ub.append(np.inf)

        for _ in range(N + 1):
            lb.append(200.0)
            ub.append(950.0)

        ub.append(np.inf)
        lb.append(0.0)

        return (lb, ub)

    @staticmethod
    def _stream_enthalpy(T, flowrates, phase):

        heat_capacities = {"propane": (-4.224, 0.3063, -1.588 * 10 ** (-4), 3.215 * 10 ** (-8)),  # (A,B,C,D)
                           "propene": (3.71, 0.2345, -1.160 * 10 ** (-4), 2.205 * 10 ** (-8)),
                           "H2": (27.14, 0.009274, -1.381 * 10 ** (-5), 7.645 * 10 ** (-9)),
                           "H2O": (32.24, 0.001924, 1.055 * 10 ** (-5), -3.596 * 10 ** (-9))
                           }

        standard_enthalpy_of_vaporization = {"propane": 16.25 * 1000,
                                             "propene": 16.04 * 1000,
                                             "H2": 15.30 * 1000,
                                             "H2O": 44 * 1000
                                             }

        critical_temperature = {"propane": 369.8,
                                "propene": 365.57,
                                "H2": 33.19,
                                "H2O": 647
                                }

        T_ref = 298.15
        enthalpy = 0

        for species, flow in flowrates.items():
            A = heat_capacities[species][0]
            B = heat_capacities[species][1]
            C = heat_capacities[species][2]
            D = heat_capacities[species][3]

            enthalpy += flow * ((
                                        A * T + 0.5 * B * T ** 2 + 1 / 3 * C * T ** 3 + 0.25 * D * T ** 4) - (
                                        A * T_ref + 0.5 * B * T_ref ** 2 + 1 / 3 * C * T_ref ** 3 + 0.25 * D * T_ref ** 4))  # tog bort enthalpy of formation för att undvika inkonsistens med vätskefas

            if phase == "liquid":
                std_H_vap = standard_enthalpy_of_vaporization[species]
                Tc = critical_temperature[species]

                ratio = (1 - T / Tc) / (1 - T_ref / Tc)

                if ratio <= 0:
                    H_vap = 0.0
                else:
                    H_vap = flow * std_H_vap * (ratio ** 0.38)  # Watsons korrelation

                enthalpy -= H_vap

        return enthalpy  # J/s

    def calculate_cost(self, trays: int, pressure: float, temperature: float, liquid_flow: float, vapor_flow: float,
                       molar_masses: dict, density: dict, vap_comp: dict):

        R = 8.314

        molar_mass = sum(vap_comp[sp] * molar_masses[sp] for sp in species_list)

        vap_density = pressure * molar_mass / (R * temperature)
        liq_density = molar_mass / (sum(vap_comp[sp] * molar_masses[sp] / density[sp] for sp in species_list))

        liq_flow_kg = (liquid_flow * molar_mass * liquid_flow / 1000) / liq_density
        vap_flow_kg = (vapor_flow * molar_mass * vapor_flow / 1000) / vap_density

        F_LV = (liq_flow_kg / vap_flow_kg) * np.sqrt(vap_density / liq_density)
        C_F = 0

        F_st = 0.007  # N/m, värde taget från random hemsida
        C_F = 10  # Tar random värde, har inget än

        C = F_st * C_F

        U_f = C * ((liq_density - vap_density) / vap_density)

        U = 0.7 * U_f

        A_aktiv = vapor_flow / U

        A_tot = A_aktiv / 0.8

        diameter = 2 * np.sqrt(A_tot / np.pi)

        cost = 0
        if 0.5 <= diameter <= 5:
            cost = 210 + 400 * diameter ** (1.9)

        elif diameter < 0.5:
            cost = 210 + 400 * 0.5 ** (1.9)

        elif diameter > 5:
            cost = 210 + 400 * 5 ** (1.9)

        self.diameter = diameter
        self.tray_cost = trays / self.tray_efficiency * cost

        self.height = self.tray_distance * trays

        column_thickness = (1.1 * self.pressure * self.diameter) / (2 * 74.5 - 1.32 * self.pressure)

        V_shell = np.pi * (self.diameter / 2 + column_thickness) * self.height - np.pi * (
                self.diameter / 2) * self.height

        shell_mass = V_shell * 8000

        if 120 <= shell_mass <= 250000:
            self.shell_cost = 17400 + 79 * shell_mass ** (0.85)

        elif shell_mass < 120:
            self.shell_cost = 17400 + 79 * 120 ** (0.85)

        elif shell_mass > 250000:
            self.shell_cost = 17400 + 79 * 250000 ** (0.85)

        condenser_duty_kW = self.condenser_duty / 1000
        cost_per_kWh = 0.05  # kr/kWh
        cost_per_second = condenser_duty_kW * cost_per_kWh / 3600

        self.condenser_cost = cost_per_second

        reboiler_duty_kW = self.reboiler_duty / 1000
        cost_per_kWh = 0.16  # kr/kWh
        cost_per_second = reboiler_duty_kW * cost_per_kWh / 3600

        self.reboiler_cost = cost_per_second
