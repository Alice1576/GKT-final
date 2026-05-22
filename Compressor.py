import numpy as np
from classes.Stream import Stream
from scipy.optimize import fsolve
import ekonomi as eko

class Compressor:
    def __init__(self, outlet_pressure: float, isentropic_efficiency: float, num_stages: int = 3):
        self.outlet_pressure = outlet_pressure # Slutgiltigt tryck
        self.eta_is = isentropic_efficiency
        self.num_stages = num_stages # Antal steg, default 3 enligt MATLAB-skriptet

        # Kostnadsattribut som anropas i optimize.py
        self.capex = 0
        self.annual_opex = 0
        self.cost = 0 # Behålls för bakåtkompatibilitet

        # För att spara design-parametrar
        self.total_energy_demand_kW = 0
        self.total_cooling_demand_kW = 0
        self.total_cooler_area_m2 = 0

    def find_isentropic_outlet_temperature(self, stream: Stream, inlet_T: float, P_in: float, P_out: float) -> float:
        """Beräknar ut-temperaturen för en ISENTROPISK kompression mha entropibalans (S_in = S_out)."""
        def entropy_balance(T2):
            sum_entropy = 0
            # S2 - S1 = integral(Cp/T)dT - R*ln(P2/P1)
            for species, flowrate in stream.flowrates.items():
                A, B, C, D = stream.heat_capacities[species]
                # integral av Cp/T från T1 till T2
                integral = A * np.log(T2/inlet_T) + B * (T2 - inlet_T) + 0.5 * C * (T2**2 - inlet_T**2) + (1/3) * D * (T2**3 - inlet_T**3)
                sum_entropy += flowrate * integral

            # Molärt R = 8.314 J/(mol*K)
            total_mol_flow = stream.total_flowrate()
            return sum_entropy - (total_mol_flow * 8.314 * np.log(P_out / P_in))

        # fsolve hittar den isentropiska temperaturen. Startgissning = inlet_T + 50
        T_is_out = fsolve(entropy_balance, inlet_T + 50)[0]
        return T_is_out

    def calculate_actual_outlet(self, stream: Stream, inlet_T: float, P_in: float, P_out: float):
        """Beräknar verklig ut-temperatur och arbete (W) för ETT kompressorsteg."""
        # 1. Hitta isentropisk ut-temperatur
        T_is_out = self.find_isentropic_outlet_temperature(stream, inlet_T, P_in, P_out)

        # 2. Skapa isentropisk ström för att hitta delta_H_isentropic
        stream_is = Stream(temperature=T_is_out, pressure=P_out, flowrates=stream.flowrates, phase="vapor")
        delta_H_is = stream_is.enthalpy - stream.enthalpy

        # 3. Verkligt arbete = Isentropiskt arbete / Isentropverkningsgrad
        W_actual = delta_H_is / self.eta_is

        # 4. Hitta verklig ut-temperatur genom att leta upp den T_out som motsvarar stream.enthalpy + W_actual
        target_enthalpy = stream.enthalpy + W_actual

        def enthalpy_balance(T_actual):
            temp_stream = Stream(temperature=T_actual, pressure=P_out, flowrates=stream.flowrates, phase="vapor")
            return temp_stream.enthalpy - target_enthalpy

        T_out_actual = fsolve(enthalpy_balance, T_is_out + 10)[0]

        return T_out_actual, W_actual

    def calculate_cooler_area(self, Q_kyl_W: float, T_in_hot: float, T_out_hot: float) -> float:
        """Beräknar värmeväxlararean för EN mellankylare baserat på MATLAB-koden."""
        Tkv = 14 + 273.15 # Inkommande kylvatten
        Tkvmax = 20 + 273.15 # Utgående kylvatten

        if Q_kyl_W <= 0 or abs(T_in_hot - T_out_hot) < 1e-6:
            return 0.0

        # LMTD: Notera att kylvattnet är motströms. T_in_hot möter Tkvmax, T_out_hot möter Tkv.
        delta_T_1 = T_in_hot - Tkvmax
        delta_T_2 = T_out_hot - Tkv

        # Undvik nolldivision eller log(0)
        if delta_T_1 <= 0 or delta_T_2 <= 0 or abs(delta_T_1 - delta_T_2) < 1e-6:
            # Förenkling om LMTD kraschar: Aritmetisk medeltemp-differens
            delta_T_lm = ((T_in_hot + T_out_hot)/2) - ((Tkv + Tkvmax)/2)
        else:
            delta_T_lm = (delta_T_1 - delta_T_2) / np.log(delta_T_1 / delta_T_2)

        U_kyl = 200 # W/(m2K) för gas-vätska
        A_kyl = Q_kyl_W / (U_kyl * delta_T_lm) #
        return A_kyl

    def run(self, stream: Stream) -> Stream:
        # Om tryckökning ej krävs
        if self.outlet_pressure <= stream.pressure:
            return Stream(temperature=stream.temperature, pressure=stream.pressure, flowrates=stream.flowrates, phase=stream.phase)

        inlet_T_global = stream.temperature
        P_in = stream.pressure
        P_out = self.outlet_pressure

        # Tryckförhållande per steg (P_step)
        P_ratio_per_stage = (P_out / P_in) ** (1 / self.num_stages)

        current_stream = stream
        total_W_W = 0
        total_A_m2 = 0
        total_Q_kyl_W = 0

        for stage in range(self.num_stages):
            P_stage_out = current_stream.pressure * P_ratio_per_stage

            # 1. Kompression
            T_out_actual, W_stage = self.calculate_actual_outlet(current_stream, current_stream.temperature, current_stream.pressure, P_stage_out)
            total_W_W += W_stage

            # Skapa stream ut från kompressorn
            comp_out_stream = Stream(temperature=T_out_actual, pressure=P_stage_out, flowrates=stream.flowrates, phase="vapor")

            # 2. Mellankylning (Alla steg utom det allra sista ska kylas!)
            if stage < self.num_stages - 1:
                # Kyler alltid ner till den ursprungliga temperaturen (inlet_T_global)
                cooler_out_stream = Stream(temperature=inlet_T_global, pressure=P_stage_out, flowrates=stream.flowrates, phase="vapor")

                # Kylbehov i Watt
                Q_kyl = comp_out_stream.enthalpy - cooler_out_stream.enthalpy
                total_Q_kyl_W += Q_kyl

                # Area
                A_stage = self.calculate_cooler_area(Q_kyl, T_out_actual, inlet_T_global)
                total_A_m2 += A_stage

                # Sätt nästa in-ström till kylarens ut-ström
                current_stream = cooler_out_stream
            else:
                # Sista kompressorsteget: Ingen kylning görs efteråt i denna enhet
                current_stream = comp_out_stream

        # SPARAR DESIGN-VÄRDEN
        self.total_energy_demand_kW = total_W_W / 1000
        self.total_cooling_demand_kW = total_Q_kyl_W / 1000
        self.total_cooler_area_m2 = total_A_m2
        self.outlet_temperature = current_stream.temperature

        # ekonomi
        #  Kostnad för kompressorn (centrifugalkompressor) via modulen
        capex_comp = eko.calculate_capex("centrifugalkompressor", self.total_energy_demand_kW)

        # Kostnad för mellankylarna (värmeväxlare)
        # Eftersom vi har (num_stages - 1) kylare, räknar vi ut kostnaden per kylare och multiplicerar upp.
        capex_coolers = 0
        if self.num_stages > 1:
            area_per_cooler = self.total_cooler_area_m2 / (self.num_stages - 1)
            # Tvinga in arean i tillåtet intervall för prisuppskattning (10-1000)
            area_for_cost = max(10.0, min(area_per_cooler, 1000.0))
            capex_per_cooler = eko.calculate_capex("heat_exchanger", area_for_cost)
            capex_coolers = capex_per_cooler * (self.num_stages - 1)

        self.capex = capex_comp + capex_coolers
        self.cost = self.capex # För bakåtkompatibilitet med ert optimize.py

        #  El till kompressorn
        opex_el = eko.calculate_opex("el", self.total_energy_demand_kW)

        # Kylvatten till mellankylarna
        opex_kyl = eko.calculate_opex("cooling_water", self.total_cooling_demand_kW)

        self.annual_opex = opex_el + opex_kyl

        return current_stream