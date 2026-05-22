from .Stream import Stream
from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt

heat_capacities = {"propane": (-4.224, 0.3063, -1.588 * 10 ** (-4), 3.215 * 10 ** (-8)),  # (A,B,C,D)
                   "propene": (3.71, 0.2345, -1.160 * 10 ** (-4), 2.205 * 10 ** (-8)),
                   "H2": (27.14, 0.009274, -1.381 * 10 ** (-5), 7.645 * 10 ** (-9)),
                   "H2O": (32.24, 0.001924, 1.055 * 10 ** (-5), -3.596 * 10 ** (-9))
                   }


class Reactor:
    """
    Man kan välja huruvida man vill fixera reaktorns conversion eller katalysatormassa själv när man instansierar
    ett reaktorobjekt. Om man önskar bestämma katalysatormassa sätter man conversion=None, och om man önskar
    bestämma omsättningsgrad av propan över reaktorn sätter man catalyst_mass = None.
    Använd run2 om du fixerar omsättningsgraden, annars run1.

    Pyplot används i run1. Det är väldigt stora datormängder så det kan ta ganska lång tid. Du kan kommentera ut det
    om du bara vill köra metoden utan grafer. Egentligen är det dålig stil att lägga plotfunktionen direkt i metoden
    men jag har lite ont om tid.

    Tryck anges i bar, temperatur i Kelvin, catalyst_mass i kg, diameter i kg. Volymen beräknas i m^3 och cost
    i USD (2010).

    """

    def __init__(self, pressure: float, catalyst_density: float, catalyst_mass, conversion):
        self.pressure = pressure
        self.density = catalyst_density
        self.catalyst_mass = catalyst_mass
        self.volume = None
        self.diameter = None
        self.cost = None  # USD (2010)
        self.conversion = conversion

    def Cp(self, species, temperature: float) -> float:
        A, B, C, D = heat_capacities[species]
        return A + B * temperature + C * temperature ** 2 + D * temperature ** 3

    def heat_of_reaction(self, temperature) -> float:
        T_ref = 298.15  # K
        delta_H_ref = 124 * (10 ** 3)  # J/mol

        def delta_Cp(temperature):
            return -self.Cp("propane", temperature) + self.Cp("propene", temperature) + self.Cp("H2", temperature)

        integral, _ = quad(delta_Cp, T_ref, temperature)

        return delta_H_ref + integral

    @staticmethod
    def flows(F_A0, F_B0, F_C0, F_D0, X):
        F_A = F_A0 * (1 - X)
        F_B = F_A0 * X + F_B0
        F_C = F_A0 * X + F_C0
        F_D = F_D0
        F_T = F_A + F_B + F_C + F_D
        return F_A, F_B, F_C, F_D, F_T

    def partial_pressures(self, F_A0, F_B0, F_C0, F_D0, X):
        F_A, F_B, F_C, F_D, F_T = self.flows(F_A0, F_B0, F_C0, F_D0, X)

        P_A = F_A / F_T * self.pressure
        P_B = F_B / F_T * self.pressure
        P_C = F_C / F_T * self.pressure
        P_D = F_D / F_T * self.pressure

        return P_A, P_B, P_C, P_D

    def ODE_solver1(self, initial_conditions: list, W_max: float, step: float, F_A0, F_B0, F_C0, F_D0):
        X, T = initial_conditions
        W = 0.0
        W_list = [W]
        X_list = [X]
        T_list = [T]
        FA_list = [F_A0]
        FB_list = [F_B0]
        FC_list = [F_C0]

        while W < W_max:
            F_A, F_B, F_C, F_D, _ = self.flows(F_A0, F_B0, F_C0, F_D0, X)
            Cp_total = F_A * self.Cp("propane", T) + F_B * self.Cp("propene", T) + F_C * self.Cp("H2",
                                                                                                 T) + F_D * self.Cp(
                "H2O", T)
            r = self.rate(T, X, F_A0, F_B0, F_C0, F_D0)

            X += (-r / F_A0) * step
            T += step * (- r * self.heat_of_reaction(T)) / (-Cp_total)
            W += step

            W_list.append(W)
            X_list.append(X)
            T_list.append(T)
            FA_list.append(F_A)
            FB_list.append(F_B)
            FC_list.append(F_C)

        return W_list, X_list, T_list, FA_list, FB_list, FC_list

    def ODE_solver2(self, initial_conditions: list, X_max: float, step: float, F_A0, F_B0, F_C0, F_D0):

        V, T = initial_conditions
        X = 0.0

        X_list = [X]
        V_list = [V]
        T_list = [T]

        while X < X_max:
            F_A, F_B, F_C, F_D, _ = self.flows(F_A0, F_B0, F_C0, F_D0, X)
            Cp_total = F_A * self.Cp("propane", T) + F_B * self.Cp("propene", T) + F_C * self.Cp("H2",
                                                                                                 T) + F_D * self.Cp(
                "H2O", T)

            r = self.rate(T, X, F_A0, F_B0, F_C0, F_D0)

            T += (F_A0 * self.heat_of_reaction(T) / (-Cp_total)) * step
            V += (-F_A0 / (self.density * r)) * step
            X += step

            X_list.append(X)
            V_list.append(V)
            T_list.append(T)

        return X_list, V_list, T_list

    def rate(self, temperature: float, conversion: float, F_A0: float, F_B0: float, F_C0: float, F_D0: float) -> float:
        X = conversion
        T = temperature
        R = 8.314  # J/molK

        k = 4.622 * 10 ** 4 * np.exp(-(35.5 * 10 ** 3) / (R * T))  # mol /(kg cat. s bar)
        K_e = 1.44 * 10 ** 7 * np.exp(-(128.2 * 10 ** 3) / (R * T))  # bar
        K_1 = 5.042 * 10 ** 4  # /bar

        P_A, P_B, P_C, _ = self.partial_pressures(F_A0, F_B0, F_C0, F_D0, X)

        if P_A - P_B * P_C / K_e < 10 ** (-6):
            raise ValueError("Equilibrium conversion exceeded")

        r = - k * (P_A - (P_B * P_C / K_e)) / (1 + K_1 * P_B)

        return r

    def run1(self, stream: Stream) -> Stream:
        F_A0 = stream.flowrates["propane"]
        F_B0 = stream.flowrates["propene"]
        F_C0 = stream.flowrates["H2"]
        F_D0 = stream.flowrates["H2O"]
        T0 = stream.temperature

        W_list, X_list, T_list, FA_list, FB_list, FC_list = self.ODE_solver1([0, T0], self.catalyst_mass, 0.0025, F_A0,
                                                                             F_B0, F_C0, F_D0)

        X_out = X_list[-1]
        T_out = T_list[-1]
        self.volume = self.catalyst_mass / self.density
        self.conversion = X_out

        F_A = F_A0 * (1 - X_out)
        F_B = F_A0 * X_out + F_B0
        F_C = F_A0 * X_out + F_C0
        outlet_flows = {"propane": F_A, "propene": F_B, "H2": F_C, "H2O": F_D0}

        t = (self.pressure * 100000 * 1.10 * self.diameter) / (
                2 * 74.5 * (10 ** 6) - 1.2 * self.pressure * 100000 * 1.10)

        shell_volume = ((t + self.diameter / 2) ** 2) * (4 * self.volume / self.diameter - self.volume)
        shell_mass = 8000 * shell_volume

        if 120 <= shell_mass <= 50000:
            self.cost = 12800 + 73 * (shell_mass ** 0.85)

        elif shell_mass < 120:
            self.cost = 12800 + 73 * (120 ** 0.85)

        elif shell_mass > 50000:
            self.cost = 12800 + 73 * (50000 ** 0.85)

        plt.plot(W_list, X_list)
        plt.xlabel("Massa katalysator (kg)", fontsize=12)
        plt.ylabel("Omsättningsgrad av propan", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        plt.plot(W_list, T_list)
        plt.xlabel("Massa katalysator (kg)", fontsize=12)
        plt.ylabel("Temperatur (K)", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        plt.plot(W_list, FA_list, label="Propan")
        plt.plot(W_list, FB_list, label="Propen")
        plt.plot(W_list, FC_list, label="Vätgas")
        plt.xlabel("Massa katalysator (kg)", fontsize=12)
        plt.ylabel("Molära flöden (mol/s)", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()

        return Stream(temperature=T_out, pressure=self.pressure, flowrates=outlet_flows, phase=stream.phase)

    def run2(self, stream: Stream) -> Stream:
        F_A0 = stream.flowrates["propane"]
        F_B0 = stream.flowrates["propene"]
        F_C0 = stream.flowrates["H2"]
        F_D0 = stream.flowrates["H2O"]
        T0 = stream.temperature

        X_list, V_list, T_list = self.ODE_solver2([0, T0], self.conversion, 0.0025, F_A0, F_B0, F_C0, F_D0)

        T_out = T_list[-1]
        self.volume = V_list[-1]
        self.catalyst_mass = self.volume * self.density

        F_A = F_A0 * (1 - self.conversion)
        F_B = F_A0 * self.conversion + F_B0
        F_C = F_A0 * self.conversion + F_C0
        outlet_flows = {"propane": F_A, "propene": F_B, "H2": F_C, "H2O": F_D0}

        self.diameter = ((2 * self.volume) / np.pi) ** (1 / 3)

        t = (self.pressure * 100000 * 1.10 * self.diameter) / (
                2 * 74500000 * (10 ** 6) - 1.2 * self.pressure * 100000 * 1.10)

        shell_volume = 4 * ((t + self.diameter / 2) ** 2) * self.volume / (self.diameter ** 2) - self.volume
        shell_mass = 8000 * shell_volume

        if 120 <= shell_mass <= 50000:
            self.cost = 1.5 * (12800 + 73 * (shell_mass ** 0.85))

        elif shell_mass < 120:
            self.cost = 1.5 * (12800 + 73 * (120 ** 0.85))

        elif shell_mass > 50000:
            self.cost = 1.5 * (12800 + 73 * (50000 ** 0.85))

        return Stream(temperature=T_out, pressure=self.pressure, flowrates=outlet_flows, phase=stream.phase)
