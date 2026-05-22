from classes import Stream
from scipy.optimize import least_squares, root
import numpy as np

mm_Hg_to_bar = 1 / 750.06156130264
species_list = ["propane", "propene", "H2", "H2O"]


class VectorizedDistillationColumnTest:
    def __init__(self, pressure, stages, feed_stage, reflux_ratio: float, distillate_flowrate: float):
        self.pressure = pressure
        self.stages = stages
        self.feed_stage = feed_stage
        self.reflux_ratio = reflux_ratio
        self.distillate_flowrate = distillate_flowrate

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

        liq_comp_guess, vap_comp_guess, liq_flows_guess, vap_flows_guess, temperatures_guess, reboiler_guess, condenser_guess = self.initial_guess(
            feed)
        x0 = self.pack(liq_comp=liq_comp_guess, vap_comp=vap_comp_guess, liq_flows=liq_flows_guess,
                       vap_flows=vap_flows_guess,
                       temperatures=temperatures_guess, reboiler_duty=reboiler_guess, condenser_duty=condenser_guess)

        sol = root(
            fun=lambda x: self.residual_wrapper(feed_stream=feed, x=x),
            x0=x0,
            method="hybr"
        )

        if not sol.success:
            print(f"Solver did not converge: {sol.message}, testing least_squares with 'trf' instead...")

            bounds = self.get_bounds()

            sol = least_squares(
                fun=lambda x: self.residual_wrapper(feed_stream=feed, x=x),
                x0=x0,
                bounds=bounds,
                method="trf",
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=20000,
                verbose=2
            )


        print(sol.fun)

        res_arr = sol.fun
        max_idx = np.argmax(np.abs(res_arr))
        print("Largest residual at index", max_idx, "value:", res_arr[max_idx])

        liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty, condenser_duty = self.unpack(N, sol.x)

        self.reboiler_duty = float(reboiler_duty)
        self.condenser_duty = float(condenser_duty)

        distillate_flowrates = {}
        distillate_temperature = float(temperatures[0, 0])

        sp_idx = {sp: i for i, sp in enumerate(species_list)}

        for sp in species_list:
            idx = sp_idx[sp]
            flow = float(vap_flows[0, 0]) * float(vap_comp[0, idx])
            distillate_flowrates[sp] = float(flow)

        distillate = Stream(pressure=P, temperature=distillate_temperature, flowrates=distillate_flowrates,
                            phase="vapor")

        for i in range(len(liq_flows)):
            T = float(temperatures[i, 0])
            L = float(liq_flows[i, 0])
            V = float(vap_flows[i, 0])

            liq_stream = Stream(temperature=T, pressure=self.pressure,
                                flowrates={sp: L * float(liq_comp[i, sp_idx[sp]]) for sp in species_list},
                                phase="liquid")

            vap_stream = Stream(temperature=T, pressure=self.pressure,
                                flowrates={sp: V * float(vap_comp[i, sp_idx[sp]]) for sp in species_list},
                                phase="vapor")

            self.streams[i] = (liq_stream, vap_stream)

        bottoms_flowrates = {sp: float(liq_comp[N - 1, sp_idx[sp]]) * float(liq_flows[N - 1, 0]) for sp in species_list}
        bottoms_temperature = float(temperatures[N - 1, 0])

        bottoms = Stream(pressure=P, temperature=bottoms_temperature, flowrates=bottoms_flowrates, phase="liquid")

        return distillate, bottoms

    def residual_wrapper(self, feed_stream: Stream, x: list):

        N = self.stages
        liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty, condenser_duty = self.unpack(N, x)

        return self.residuals(feed_stream, liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty,
                              condenser_duty)

    def residuals(self, feed: Stream, liq_comp, vap_comp, liq_flows, vap_flows,
                  temperatures, reboiler_duty, condenser_duty):

        F_tot = feed.total_flowrate()
        F_comp = np.zeros(4)

        for i, sp in enumerate(species_list):
            F_comp[i] = feed.flowrates[sp] / F_tot

        R = self.reflux_ratio
        Q = reboiler_duty
        C = condenser_duty

        # Massbalans över bottnarna. Exkluderar återkokaren och kondensorn.
        vap_in = vap_flows[2:] * vap_comp[2:]
        liq_in = liq_flows[:-2] * liq_comp[:-2]

        vap_out = vap_flows[1:-1] * vap_comp[1:-1]
        liq_out = liq_flows[1:-1] * liq_comp[1:-1]

        mass_res = vap_in + liq_in - vap_out - liq_out

        # Massbalans över partialåterkokaren
        liq_in_reboiler = liq_flows[-2] * liq_comp[-2]
        liq_out_reboiler = liq_flows[-1] * liq_comp[-1]
        vap_out_reboiler = vap_flows[-1] * vap_comp[-1]
        reboiler_mass_res = liq_in_reboiler - liq_out_reboiler - vap_out_reboiler

        # Massbalans över partialkondensorn
        liq_out_condenser = liq_flows[0] * liq_comp[0]
        vap_out_condenser = vap_flows[0] * vap_comp[0]
        vap_in_condenser = vap_flows[1] * vap_comp[1]
        condenser_mass_res = vap_in_condenser - vap_out_condenser - liq_out_condenser

        # Massbidraget från feedtray
        feed_mass_term = np.zeros_like(mass_res)
        feed_mass_term[self.feed_stage - 1, :] = F_tot * F_comp
        mass_res += feed_mass_term

        # Entalpibalns över bottnarna. Exkluderar återkokaren och kondensorn.
        liq_enthalpy_out = self._compute_enthalpies(temperatures[1: -1], liq_flows[1: -1] * liq_comp[1: -1], "liquid")
        vap_enthalpy_out = self._compute_enthalpies(temperatures[1: -1], vap_flows[1: -1] * vap_comp[1: -1], "vapor")

        liq_enthalpy_in = self._compute_enthalpies(temperatures[: -2], liq_flows[: -2] * liq_comp[: -2], "liquid")
        vap_enthalpy_in = self._compute_enthalpies(temperatures[2:], vap_flows[2:] * vap_comp[2:], "vapor")

        enthalpy_res = vap_enthalpy_in + liq_enthalpy_in - vap_enthalpy_out - liq_enthalpy_out

        feed_enthalpy_term = np.zeros_like(enthalpy_res)
        feed_temperature = np.array([feed.temperature])

        feed_enthalpy_term[self.feed_stage - 1] = self._compute_enthalpies(feed_temperature, F_tot * F_comp, feed.phase)

        enthalpy_res += feed_enthalpy_term

        # Entalpibalans över återkokaren
        reboiler_liq_enthalpy_in = self._compute_enthalpies(temperatures[-2], liq_flows[-2] * liq_comp[-2], "liquid")
        reboiler_vap_enthalpy_out = self._compute_enthalpies(temperatures[-1], vap_flows[-1] * vap_comp[-1], "vapor")
        reboiler_liq_enthalpy_out = self._compute_enthalpies(temperatures[-1], liq_flows[-1] * liq_comp[-1], "liquid")

        reboiler_enthalpy_res = reboiler_liq_enthalpy_in + Q - reboiler_vap_enthalpy_out - reboiler_liq_enthalpy_out

        # Entalpibalans över kondensorn
        condenser_vap_enthalpy_in = self._compute_enthalpies(temperatures[1], vap_flows[1] * vap_comp[1], "vapor")
        condenser_vap_enthalpy_out = self._compute_enthalpies(temperatures[0], vap_flows[0] * vap_comp[0], "vapor")
        condenser_liq_enthalpy_out = self._compute_enthalpies(temperatures[0], liq_flows[0] * liq_comp[0], "liquid")

        condenser_enthalpy_res = condenser_vap_enthalpy_in + C - condenser_vap_enthalpy_out - condenser_liq_enthalpy_out

        # Molbråksbalans. Ser till att molbråken i vätskefas summerar till 1 och inte något annat.
        summation_x_res = np.sum(liq_comp, axis=1) - 1.0
        summation_y_res = np.sum(vap_comp, axis=1) - 1.0

        # Bubbelpunktsbalans över allt, inklusive kondensorn och återkokaren.
        # Vid mättnadstemperaturen i varje jämviktssteg måste Σ(K_i*x_i) = 1

        K_matrix = self._compute_K_matrix(temperatures)

        equilibrium_res = vap_comp - liq_comp * K_matrix

        # Total flödesbalans och kompositionskontroll över kondensorn. Vätskan som lämnar kondensorn som reflux måste ha
        # samma komposition som ånga som flödar upp till den.

        condenser_reflux_res = liq_flows[0] - R * vap_flows[0]
        condenser_top_spec_res = vap_flows[0] - self.distillate_flowrate

        enthalpy_res /= 1e6
        condenser_enthalpy_res /= 1e6
        reboiler_enthalpy_res /= 1e6

        return np.concatenate((
            equilibrium_res.flatten(),
            enthalpy_res.flatten(),
            mass_res.flatten(),
            summation_x_res.flatten(),
            summation_y_res.flatten(),
            reboiler_mass_res.flatten(),
            reboiler_enthalpy_res.flatten(),
            condenser_enthalpy_res.flatten(),
            condenser_mass_res.flatten(),
            condenser_reflux_res.flatten(),
            condenser_top_spec_res.flatten()
        ))

    @staticmethod
    def pack(liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty, condenser_duty):
        """
        :param liq_comp:
        :param vap_comp:
        :param liq_flows:
        :param vap_flows:
        :param temperatures:
        :param reboiler_duty:
        :param condenser_duty:

        Denna metod packar ihop alla arrayer tillbaka till en 1D-array. Temperaturerna logaritmeras så att solvern
        inte kan testa negativa temperaturer (ln(a) endast definierad för a > 0). I unpack exponentieras
        temperaturarrayen igen med np.exp()

        """

        x = np.concatenate((
            liq_comp.flatten(),
            vap_comp.flatten(),
            liq_flows.flatten(),
            vap_flows.flatten(),
            np.log(temperatures).flatten(),
            reboiler_duty.flatten(),
            condenser_duty.flatten()
        ))

        return x

    @staticmethod
    def unpack(N: int, x):
        """
        :param N:
        :param x:

        Denna metod packar packar upp en 1D-array som innehåller alla gissningar och returnerar arrays med
        dimension (N+1, 4)

        """

        liq_comp = x[0: 4 * N].reshape((N, 4))
        idx = 4 * N

        vap_comp = x[idx: idx + 4 * N].reshape((N, 4))
        idx += 4 * N

        liq_flows = x[idx: idx + N].reshape((N, 1))
        idx += N

        vap_flows = x[idx: idx + N].reshape((N, 1))
        idx += N

        temperatures = np.exp(x[idx: idx + N].reshape((N, 1)))

        reboiler_duty = x[-2]
        condenser_duty = x[-1]

        return liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty, condenser_duty

    def initial_guess(self, feed: Stream):

        R = self.reflux_ratio
        N = self.stages
        D = self.distillate_flowrate
        F_tot = feed.total_flowrate()
        B = F_tot - D

        temperatures = np.linspace(300, 320, N)

        liq_comp = np.zeros((N, 4))

        for i, sp in enumerate(species_list):
            liq_comp[:, i] = feed.flowrates[sp] / F_tot

        K_matrix = self._compute_K_matrix(temperatures.reshape(-1, 1))
        vap_comp = liq_comp * K_matrix
        vap_comp /= np.sum(vap_comp, axis=1, keepdims=True)

        # Vi antar CMO i vår initialgissning av de interna flödena:
        liq_flows = np.zeros((N, 1))
        vap_flows = np.zeros((N, 1))

        q = 0.0
        if feed.phase == "vapor":
            q = 0.0
        elif feed.phase == "liquid":
            q = 1.0

        L_rect = R * D
        V_rect = (1 + R) * D

        L_strip = L_rect + q * F_tot
        V_strip = V_rect - (1 - q) * F_tot

        #Vätskeflödena
        liq_flows[0, 0] = L_rect
        liq_flows[1 : self.feed_stage, 0] = L_rect
        liq_flows[self.feed_stage : N - 1, 0] = L_strip
        liq_flows[-1, 0] = B

        #Ångflödena
        vap_flows[0, 0] = D
        vap_flows[1 : self.feed_stage, 0] = V_rect
        vap_flows[self.feed_stage :, 0] = V_strip

        reboiler_duty = np.array([1e6])
        condenser_duty = np.array([-1e6])

        return liq_comp, vap_comp, liq_flows, vap_flows, temperatures, reboiler_duty, condenser_duty

    def get_bounds(self):
        N = self.stages
        lb = []
        ub = []

        for _ in range(N):
            for sp in species_list:
                lb.append(0.0)
                ub.append(1.0)

        for _ in range(N):
            for sp in species_list:
                lb.append(0.0)
                ub.append(1.0)

        for _ in range(N):
            lb.append(0.0)
            ub.append(np.inf)

        for _ in range(N):
            lb.append(0.0)
            ub.append(np.inf)

        # Gränserna för temperaturerna
        for _ in range(N):
            lb.append(np.log(200.0))
            ub.append(np.log(700))

        # Gränserna för återkokarvärmet
        ub.append(np.inf)
        lb.append(0.0)

        # Gränserna för kondensorvärmet
        ub.append(0.0)
        lb.append(-np.inf)

        return (lb, ub)

    @staticmethod
    def _compute_enthalpies(T, flowrates, phase):
        """
        :param T: Array med dimemsion (N+1, 1)
        :param flowrates: Array med dimension (N+1, 4).
        :param phase: "liquid" eller "vapor"
        """

        # Raderna är i ordning propan, propen, H2, H2O och kolonnerna i ordning (A,B,C,D)
        heat_capacities = np.array([
            [-4.224, 0.3063, -1.586e-4, 3.215e-8],
            [3.71, 0.2345, -1.160e-4, 2.205e-8],
            [27.14, 0.009274, -1.381e-5, 7.645e-9],
            [32.24, 0.001924, 1.055e-5, -3.596e-9]
        ])

        std_enthalpy_vap = np.array([16.25e3, 16.04e3, 15.30e3, 44e3])

        critical_temperature = np.array([369.8, 365.57, 33.19, 647])

        T_ref = 298.15

        A, B, C, D = heat_capacities[:, 0], heat_capacities[:, 1], heat_capacities[:, 2], heat_capacities[:, 3]

        def get_h_integral(temperatures):
            return A * temperatures + 0.5 * B * temperatures ** 2 + (
                    1 / 3) * C * temperatures ** 3 + 0.25 * D * temperatures ** 4

        molar_enthalpy = get_h_integral(T) - get_h_integral(T_ref)

        if phase == "liquid":
            ratio = (1 - T / critical_temperature) / (1 - T_ref / critical_temperature)

            ratio = np.maximum(ratio, 0)

            molar_vap_enthalpy = std_enthalpy_vap * (ratio ** 0.38)  # Watsons korrelation
            molar_enthalpy -= molar_vap_enthalpy

        enthalpy = np.sum(flowrates * molar_enthalpy, axis=-1)

        return enthalpy  # J/s

    def _compute_K_matrix(self, temperatures):
        A_coeffs = np.array([15.7260, 15.7027, 13.6333, 18.3036])
        B_coeffs = np.array([1872.46, 1807.53, 164.90, 3816.44])
        C_coeffs = np.array([-25.26, -26.15, 3.19, -46.13])

        arg = A_coeffs - B_coeffs / (temperatures + C_coeffs)

        # Klipper så att vi inte får overflowfel om solvern testar ofysikaliska temperaturer
        arg = np.clip(arg, -100.0, 100.0)

        P_sat = np.exp(arg) * mm_Hg_to_bar

        K_matrix = P_sat / self.pressure

        return K_matrix
