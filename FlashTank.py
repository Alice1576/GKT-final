from classes.Stream import Stream
from thermo import *
import ekonomi as eko

class FlashTank:
    """

    """
    def __init__(self, temperature: float, pressure: float):
        self.temperature = temperature
        self.pressure = pressure
        self.diameter = None
        self.height = None

        # Ekonomiska variabler
        self.capex = 0
        self.annual_opex = 0  # En vanlig flash-tank kräver ingen värme/kyla
        self.cost = 0

    def run(self, stream: Stream) -> tuple[Stream,Stream]:

        liquid_stream, vapor_stream = find_flashtank_equilibrium_composition(temperature = self.temperature, pressure = self.pressure, inlet_stream = stream)
        return liquid_stream, vapor_stream

    def calculate_economics(self, liquid_stream: Stream):
        """
        Grov dimensionering av Flash-tanken baserat på vätskeflödet.
        Vi antar en uppehållstid för vätskan på 5 minuter (300 sekunder).
        """
        # Hämta molflödet av vätska (mol/s)
        F_liq = liquid_stream.total_flowrate()

        # Om vi av misstag bara har ånga sätter vi ett litet dummy-flöde för att slippa kraschar
        if F_liq <= 0:
            F_liq = 0.1

        # Grov uppskattning: Antar en genomsnittlig molär volym för kondensatet (vatten)
        # ca 18 ml/mol = 0.000018 m3/mol.
        volumetric_flow_liq = F_liq * 0.000018  # m3/s

        # Vätskevolym i tanken (5 min uppehållstid)
        liquid_volume = volumetric_flow_liq * 300

        # Flash-tanken är normalt fylld till 50% med vätska, så total volym är dubbelt
        total_volume = liquid_volume * 2
        total_volume = max(total_volume, 0.5)  # Sätter en minsta volym på 0.5 m3

        # Anta standardförhållande Höjd / Diameter = 3
        # Volym = pi * (D/2)^2 * 3D = (3*pi/4) * D^3
        self.diameter = ((4 * total_volume) / (3 * np.pi)) ** (1 / 3)
        self.height = 3 * self.diameter

        # --- CAPEX-BERÄKNING (Samma formel som kolonnen) ---
        P_design_Pa = self.pressure * 100000 * 1.10
        tillaten_spanning = 74.5 * (10 ** 6)  # Pa för Kolstål
        t = (P_design_Pa * self.diameter) / (2 * tillaten_spanning - P_design_Pa) + 0.003

        # Skalmassa (kg)
        volume_steel = np.pi * self.diameter * self.height * t
        mass_kg = volume_steel * 7800

        # Begränsa för kostnadskurvan
        mass_for_cost = max(160.0, min(mass_kg, 250000.0))

        # Räkna ut priset
        self.capex = eko.calculate_capex("vessel_vertical_cs", mass_for_cost)
        self.cost = self.capex  # Spara till optimize.py