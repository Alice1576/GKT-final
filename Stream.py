class Stream:
    def __init__(self, temperature: float, pressure: float, phase: str, flowrates: dict):
        self.temperature = temperature
        self.pressure = pressure
        self.flowrates = flowrates  # {species: float}, t.ex. self.flowrates[propan] = 0.1
        self.phase = phase  # "liquid" eller "vapor". Vi antar att strömmar av olika faser inte kan blandas.

        self.heat_capacities = {"propane": (-4.224, 0.3063, -1.588 * 10 ** (-4), 3.215 * 10 ** (-8)),  # (A,B,C,D)
                                "propene": (3.71, 0.2345, -1.160 * 10 ** (-4), 2.205 * 10 ** (-8)),
                                "H2": (27.14, 0.009274, -1.381 * 10 ** (-5), 7.645 * 10 ** (-9)),
                                "H2O": (32.24, 0.001924, 1.055 * 10 ** (-5), -3.596 * 10 ** (-9))
                                }

        self.enthalpy_of_formation = {"propane": -105.2 * 1000,
                                      "propene": 20.4 * 1000,
                                      "H2": 0.0 * 1000,
                                      "H2O": -285.83 * 1000
                                      }

        self.standard_enthalpy_of_vaporization = { "propane": 16.25 * 1000,
                                                   "propene": 16.04 * 1000,
                                                   "H2": 15.30 * 1000,
                                                   "H2O": 44 * 1000
                                                 }

        self.critical_temperature = { "propane": 369.8,
                                      "propene": 365.57,
                                      "H2": 33.19,
                                      "H2O": 647
                                     }

    def total_flowrate(self) -> float:
        return sum(self.flowrates.values())

    def mole_fraction(self) -> dict:
        total = self.total_flowrate()
        if total != 0:
            return {species: frac / total for species, frac in self.flowrates.items()}
        else:
            return {species: 0 for species in self.flowrates.keys()}

    # värmekapaciteter anges i J/molK. Vi använder modellen Cp = A+BT+CT^2+DT^3. Temperaturen är i K.
    # bildningsentalpin anges vid 1 bar i J/mol.
    @property
    def enthalpy(self):

        T_ref = 298.15
        T = self.temperature
        enthalpy = 0

        for species, flow in self.flowrates.items():
            A = self.heat_capacities[species][0]
            B = self.heat_capacities[species][1]
            C = self.heat_capacities[species][2]
            D = self.heat_capacities[species][3]


            enthalpy += flow * ((
                    A * T + 0.5 * B * T ** 2 + 1 / 3 * C * T ** 3 + 0.25 * D * T ** 4) - (
                                        A * T_ref + 0.5 * B * T_ref ** 2 + 1 / 3 * C * T_ref ** 3 + 0.25 * D * T_ref ** 4)) #tog bort enthalpy of formation för att undvika inkonsistens med vätskefas

            if self.phase == "liquid":
                std_H_vap = self.standard_enthalpy_of_vaporization[species]
                Tc = self.critical_temperature[species]

                ratio = (1 - T / Tc) / (1 - T_ref / Tc)

                if ratio <= 0:
                    H_vap = 0.0
                else:
                    H_vap = flow * std_H_vap * (ratio ** 0.38) #Watsons korrelation

                enthalpy -= H_vap

        return enthalpy #J/s

    def __add__(self, other: "Stream") -> "Stream":

        new_flows = self.flowrates.copy()
        for species, flow in other.flowrates.items():
            new_flows[species] = flow + new_flows.get(species, 0)

        return Stream(temperature=self.temperature, pressure=self.pressure, flowrates=new_flows, phase=self.phase)

    def __mul__(self, a: float) -> "Stream":
        new_flows = self.flowrates.copy()
        for species in new_flows.keys():
            new_flows[species] *= a

        return Stream(temperature=self.temperature, pressure=self.pressure, flowrates=new_flows, phase=self.phase)

    def __repr__(self):
        return f"(Phase: {self.phase}, T = {self.temperature} K, P={self.pressure} bar, {self.flowrates})"
