from classes.Stream import Stream
from scipy.optimize import fsolve, root, least_squares
from thermo import find_dew_temperature, find_bubble_temperature, find_saturation_pressure
import numpy as np


class OldDistillationColumn:
    def __init__(self, pressure: float, reflux_ratio: float, stages: int,
                 tray_distance: float, tray_efficiency: float, feed_stage: int, reboiler_duty: float):

        self.pressure = pressure
        self.stages = stages
        self.reflux_ratio = reflux_ratio
        self.tray_distance = tray_distance
        self.tray_efficiency = tray_efficiency
        self.reboiler_duty = reboiler_duty

        self.streams = {}
        self.feed_stage = feed_stage
        self.vapor_streams = None
        self.liquid_streams = None

        self.condenser_cost = 0
        self.reboiler_cost = 0
        self.cost = 0

    def run(self, stream: Stream) -> tuple[Stream, Stream]:

        species_list = ["propane", "propene", "H2", "H2O"]

        P = self.pressure
        N = self.stages
        L_guess, V_guess, T_guess = self.initial_guess(stream)
        x0 = self.pack(L=L_guess, V=V_guess, T=T_guess)

        nL = (N + 1) * len(species_list)
        nV = N * len(species_list)
        nT = N + 1

        lb = np.zeros(nL + nV + nT)
        ub = np.full(nL + nV + nT, np.inf)

        # Set temperature lower bound (e.g. 200 K)
        for i in range(nL + nV, nL + nV + nT):
            lb[i] = 200

        sol = least_squares(lambda x: self.residual_wrapper(stream, x), x0=x0, method="trf", max_nfev=20000,
                            bounds=(lb, ub), ftol=1e-8, gtol=1e-8, xtol=1e-8)

        L, V, T = self.unpack(self.stages, sol.x)

        D = {sp: V[0][sp] - L[0][sp] for sp in species_list}
        T_D = T[0]
        distillate_stream = Stream(temperature=T_D, pressure=P, flowrates=D, phase="liquid")

        for i in range(N):
            liq_stream = Stream(temperature=T[i + 1], pressure=P, flowrates=L[i + 1], phase="liquid")
            vap_stream = Stream(temperature=T[i + 1], pressure=P, flowrates=V[i], phase="vapor")
            self.streams[i + 1] = (liq_stream, vap_stream)

        bottoms_stream = self.streams[N][0]

        return distillate_stream, bottoms_stream, sol

    def residual_wrapper(self, feed_stream: Stream, x: list):

        N = self.stages
        L, V, T = self.unpack(N, x)

        return self.residuals(feed_stream, L, V, T)

    @staticmethod
    def pack(L: list[dict], V: list[dict], T: list[float]):
        x = []

        for flow in L:
            for sp in flow:
                x.append(flow[sp])

        for flow in V:
            for sp in flow:
                x.append(flow[sp])

        for temp in T:
            x.append(temp)

        return x

    @staticmethod
    def unpack(N: int, x: list[float]):
        species_list = ["propane", "propene", "H2", "H2O"]
        L = []
        V = []
        T = []

        k = 0

        for i in range(N + 1):
            flowrates = {}
            for sp in species_list:
                flowrates[sp] = x[k]
                k += 1

            L.append(flowrates)

        for i in range(N):
            flowrates = {}
            for sp in species_list:
                flowrates[sp] = x[k]
                k += 1

            V.append(flowrates)

        for i in range(N + 1):
            T.append(x[k])
            k += 1

        return L, V, T

    def residuals(self, feed: Stream, liquid_flowrates: list[dict], vapor_flowrates: list[dict],
                  stream_temperatures: list[float]) -> list[float]:
        species_list = ["propane", "propene", "H2", "H2O"]

        antoine = {"H2": (13.6333, 164.90, 3.19),  # för T i Kelvin och P i mmHg
                   "propane": (15.7260, 1872.46, -25.26),
                   "propene": (15.7027, 1807.53, -26.15),
                   "H2O": (18.3036, 3816.44, -46.13)
                   }

        feed_stage = self.feed_stage
        V = vapor_flowrates
        L = liquid_flowrates
        R = self.reflux_ratio
        P = self.pressure
        N = self.stages
        Q = self.reboiler_duty

        tray_comp_res = []
        enthalpy_res = []
        top_tray_res = []
        tray_mass_res = []

        T_stage = stream_temperatures

        for i in range(N):
            for sp in species_list:
                V_in = V[i + 1][sp] if i < N - 1 else 0.0
                V_out = V[i][sp]

                L_in = L[i][sp]
                L_out = L[i + 1][sp]

                res = L_in + V_in - (V_out + L_out)

                if i + 1 == feed_stage:
                    res += feed.flowrates[sp]

                tray_mass_res.append(res)

        for i in range(N):
            total_liq_flowrate = sum(L[i + 1][sp] for sp in species_list)
            liq_comp = {sp: L[i + 1][sp] / total_liq_flowrate for sp in species_list}
            temp = T_stage[i + 1]

            total_vap_flowrate = sum(V[i][sp] for sp in species_list)
            vap_comp = {sp: V[i][sp] / total_vap_flowrate for sp in species_list}

            for sp in species_list:
                Psat_bar = find_saturation_pressure(temp, *antoine[sp]) / 750.06
                K = Psat_bar / P
                y = K * liq_comp[sp]
                tray_comp_res.append(y - vap_comp[sp])

        for i in range(N):
            T_above = T_stage[i]
            T_below = T_stage[i + 2] if i < N - 1 else T_stage[i]
            T_curr = T_stage[i + 1]

            if i < N - 1:
                vap_stream_in = Stream(temperature=T_below, pressure=P, flowrates=V[i + 1], phase="vapor")

            else:
                vap_stream_in = Stream(temperature=T_below, pressure=P, flowrates={sp: 0.0 for sp in species_list},
                                       phase="vapor")

            vap_stream_out = Stream(temperature=T_curr, pressure=P, flowrates=V[i], phase="vapor")
            liq_stream_in = Stream(temperature=T_above, pressure=P, flowrates=L[i], phase="liquid")
            liq_stream_out = Stream(temperature=T_curr, pressure=P, flowrates=L[i + 1], phase="liquid")

            enthalpy_in = liq_stream_in.enthalpy + vap_stream_in.enthalpy

            if i == N - 1:
                enthalpy_in += Q

            if i + 1 == feed_stage:
                enthalpy_in += feed.enthalpy

            enthalpy_out = liq_stream_out.enthalpy + vap_stream_out.enthalpy
            enthalpy_res.append(enthalpy_in - enthalpy_out)

        L0_tot = sum(L[0][sp] for sp in species_list)
        V1_tot = sum(V[0][sp] for sp in species_list)

        res = V1_tot - L0_tot * (1 + 1 / R)
        top_tray_res.append(res)

        for sp in species_list[:3]:
            x_reflux = L[0][sp] / L0_tot
            y_top_tray = V[0][sp] / V1_tot
            top_tray_res.append(x_reflux - y_top_tray)

        bubble_res_condenser = 0
        for sp in species_list:
            Psat_bar = find_saturation_pressure(T_stage[0], *antoine[sp]) / 750.06
            K = Psat_bar / P
            bubble_res_condenser += (L[0][sp] / L0_tot) * K
            res = bubble_res_condenser - 1
        top_tray_res.append(res)

        enthalpy_res = [r / 1e8 for r in enthalpy_res]
        tray_mass_res = [r / 100 for r in tray_mass_res]


        return tray_comp_res + top_tray_res + tray_mass_res + enthalpy_res




    def initial_guess(self, feed: Stream):
        species_list = ["propane", "propene", "H2", "H2O"]
        N = self.stages

        L = []
        V = []
        T = []

        F = feed.flowrates

        for i in range(N):
            flows = {}
            for sp in species_list:
                flows[sp] = F[sp] / N
            V.append(flows)

        for i in range(N + 1):
            flows = {}
            for sp in species_list:
                flows[sp] = F[sp] / N
            L.append(flows)

        T.append(300)
        for i in range(1, N + 1):
            T.append(320 + (80 / N) * i)

        return L, V, T
