import numpy as np


class McCabeColumn:
    """
    Ett generiskt McCabe-Thiele script för att beräkna antalet ideala bottnar, feedsteget m.m.
    """

    def __init__(self, feed, xF, xD, xB, q, R, alpha):

        self.feed = feed
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.q = q
        self.R = R
        self.alpha = alpha

    def equilibrium_x(self, y):

        return y / (self.alpha - y * (self.alpha - 1))

    def rect_line(self, x):

        return (self.R / (self.R + 1)) * x + (self.xD / (self.R + 1))

    def q_intersect(self):

        if self.q == 1:
            x_int = self.xF
            y_int = self.rect_line(x_int)

        else:
            m_q = self.q / (self.q - 1)
            b_q = -self.xF / (self.q - 1)

            m_r = self.R / (self.R + 1)
            b_r = self.xD / (self.R + 1)

            x_int = (b_r - b_q) / (m_q - m_r)
            y_int = m_q * x_int + b_q

        return x_int, y_int

    def strip_line(self, x, x_int, y_int):
        m_s = (y_int - self.xB) / (x_int - self.xB)
        b_s = self.xB - m_s * self.xB

        return m_s * x + b_s

    def equilibrium_y(self, x):

        return (self.alpha * x) / (1 + (self.alpha - 1) * x)

    def R_min(self):

        y = self.equilibrium_y(self.xF)
        m = (y - self.xD) / (self.xF - self.xD)

        return m / (1 - m)

    def flow_rates(self, F):

        D = F * (self.xF - self.xB) / (self.xD - self.xB)

        B = F - D

        return D, B

    def run(self):
        x_int, y_int = self.q_intersect()
        stages = 0
        feed_stage = 0
        x_curr = self.xD
        y_curr = self.xD

        D, B = self.flow_rates(self.feed)

        while x_curr > self.xB:
            stages += 1

            x_eq = self.equilibrium_x(y_curr)
            x_curr = x_eq

            if x_curr < x_int and feed_stage == 0:
                feed_stage = stages

            if x_curr > x_int:
                y_next = self.rect_line(x_curr)

            else:
                y_next = self.strip_line(x_curr, x_int, y_int)

            y_curr = y_next

            if stages > 300:
                R_min = self.R_min()
                if self.R < R_min:
                    raise ValueError("Konvergerade inte för att R är för lågt!")
                else:
                    raise ValueError("Konvergerade inte!")

        real_stages = stages / 0.7

        print(f"Antal teoretiska steg: {stages}")
        print(f"Verkliga steg: {real_stages}")
        print(f"Optimalt feedsteg: {feed_stage}")
        print(f"Destillatflöde: {D}")
        print(f"Bottenflöde: {B}")

        return stages, feed_stage