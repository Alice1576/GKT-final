"""
Här ör klassen för processugnen. Man anger tillsammans med sin ingående ström en utgående temperatur i run-metoden
och heat_demand beräknas som entalpiskillnaden mellan in- och utströmmarna (konstant tryck).
self.heat_demand är effektbehovet och uppdateras kontinuerligt då man kör strömmar genom ugnen och self.gas_cost är
kostnaden på gasen och uppdateras också kontinuerligt baserat på effektbehovet. Attributet cost är helt enkelt kostnaden
för ugnen och anges i USD (2010).
"""


from classes.Stream import Stream


class ProcessOven:
    def __init__(self):
        self.gas_cost = 0
        self.heat_demand = 0 #kr/s
        self.cost = 0 #USD (2010)

    def run(self, stream: Stream, outlet_temperature) -> Stream:
        outlet_stream = Stream(temperature=outlet_temperature, pressure=stream.pressure,
                               flowrates=stream.flowrates, phase=stream.phase)
        outlet_stream.temperature = outlet_temperature
        heat_demand = outlet_stream.enthalpy - stream.enthalpy  # J/s = W

        heat_demand_in_kW = (heat_demand / 1000)
        cost_per_kWh = 0.20
        cost_per_second = heat_demand_in_kW * cost_per_kWh / 3600

        self.gas_cost += cost_per_second

        self.heat_demand += heat_demand_in_kW
        total_heat_demand_in_MW = self.heat_demand / 1000

        if 30 <= total_heat_demand_in_MW <= 120:
            self.cost = 43000 + 111000 * (total_heat_demand_in_MW ** 0.8)

        elif total_heat_demand_in_MW < 30:
            self.cost = 43000 + 111000 * (30 ** 0.8)

        elif total_heat_demand_in_MW > 120:
            self.cost = 43000 + 111000 * (120 ** 0.8)

        return outlet_stream
