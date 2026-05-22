from classes.Stream import Stream
from scipy.optimize import fsolve
import ekonomi as eko

class Pump:
    def __init__(self, outlet_pressure: float, efficiency: float = 0.75):
        """
        Skapar en pump. Standardverkningsgraden för centrifugalpumpar
        brukar antas vara ca 75 % (0.75) om inget annat anges.
        """
        self.outlet_pressure = outlet_pressure
        self.efficiency = efficiency

        # Kostnadsattribut
        self.capex = 0
        self.annual_opex = 0
        self.cost = 0
        self.energy_demand_kW = 0

    def run(self, stream: Stream) -> Stream:
        # Om tryckökning ej krävs
        if self.outlet_pressure <= stream.pressure:
            return Stream(temperature=stream.temperature, pressure=self.outlet_pressure,
                          flowrates=stream.flowrates, phase="liquid")

        P_in_Pa = stream.pressure * 100000
        P_out_Pa = self.outlet_pressure * 100000
        delta_P = P_out_Pa - P_in_Pa

        # Beräkna pumpens arbete
        # För vätskor antar vi inkompressibilitet: W_ideal = V_tot * delta_P
        # Vi gör en förenklad uppskattning av den molära volymen för vätskan (ca 50 cm^3/mol för kolväten/vatten).
        molar_volume_m3_per_mol = 50e-6
        total_mol_flow = stream.total_flowrate()
        volumetric_flow_m3_s = total_mol_flow * molar_volume_m3_per_mol

        ideal_work_W = volumetric_flow_m3_s * delta_P
        actual_work_W = ideal_work_W / self.efficiency

        # Beräkna temperaturökningen
        # Det verkliga arbetet tillför energi till vätskan, vilket ökar dess entalpi marginellt.
        target_enthalpy = stream.enthalpy + actual_work_W

        def enthalpy_balance(T_actual):
            temp_stream = Stream(temperature=T_actual, pressure=self.outlet_pressure,
                                 flowrates=stream.flowrates, phase="liquid")
            return temp_stream.enthalpy - target_enthalpy

        # Temperaturökningen i en pump är oftast väldigt liten (någon tiondels grad)
        T_out_actual = fsolve(enthalpy_balance, stream.temperature)[0]

        # Uppdatera ekonomin
        self.energy_demand_kW = actual_work_W / 1000

        self.capex = eko.calculate_capex("pump", self.energy_demand_kW)
        self.cost = self.capex

        self.annual_opex = eko.calculate_opex("el", self.energy_demand_kW)

        # Returnera den trycksatta vätskeströmmen
        return Stream(temperature=T_out_actual, pressure=self.outlet_pressure,
                      flowrates=stream.flowrates, phase="liquid")