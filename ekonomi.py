#!/usr/bin/env python3

CEPCI_2010 = 532.9 # Basår för kostnadskurvor
CEPCI_2024 = 800.0 # Nya indexet
USD_TO_SEK = 9.3   #
LANGFAKTOR = 4.0   #

# Utrustningsdata,  'namn': a, b, n, S_min, S_max,  från tabellen
EQUIPMENT_DATA = {
    "heat_exchanger": (32000, 70, 1.2, 10, 1000),      # S = area (m2)
    "vessel_vertical_cs": (11600, 34, 0.85, 160, 250000), # S = skalmassa (kg)
    "sieve_trays": (130, 440, 1.8, 0.5, 5.0)           # S = diameter (m)
}

def calculate_capex(equipment_type: str, S: float) -> float:
    """Beräknar total installerad kostnad i SEK."""
    if equipment_type not in EQUIPMENT_DATA:
        raise ValueError(f"Utrustningstyp {equipment_type} saknas i databasen.")

    a, b, n, S_min, S_max = EQUIPMENT_DATA[equipment_type]

    # Kollar om S är utanför min/max-gränserna
    if not (S_min <= S <= S_max):
        print(f"Varning: {equipment_type} med S={S} är utanför giltigt intervall ({S_min}-{S_max}).")

    # Bas-kostnad i USD (2010)
    C_2010_USD = a + b * (S ** n)

    # Uppräkning till 2024 och konvertering till installerad SEK
    C_nutid_USD = C_2010_USD * (CEPCI_2024 / CEPCI_2010)
    total_cost_SEK = C_nutid_USD * USD_TO_SEK * LANGFAKTOR
    return total_cost_SEK


DRIFTTID_H = 8000 # timmar/år

UTILITY_PRICES = {
    "steam": 0.16,        # kr/kWh
    "cooling_water": 0.05,# kr/kWh
    "cooling_med": 1.00   # under 14 grader, kr/kWh
}

def calculate_opex(utility_type: str, Q_kW: float) -> float:
    """Beräknar årlig driftskostnad i SEK för ett visst kyl/värmebehov."""
    if utility_type not in UTILITY_PRICES:
        raise ValueError("Okänd förnödenhet.")

    pris_per_kWh = UTILITY_PRICES[utility_type]

    # Kostnad = Effekt (kW) * Drifttid (h/år) * Pris (kr/kWh)
    return Q_kW * DRIFTTID_H * pris_per_kWh