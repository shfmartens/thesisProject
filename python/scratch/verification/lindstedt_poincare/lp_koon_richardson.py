import math
import numpy as np
import pandas as pd
from sympy import Symbol
from sympy.solvers import nsolve


class RichardsonApproximation:
    def __init__(self):

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        mu_Earth_Moon = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
        mu_Sun_Earth = (EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER) / (
        EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER + SUN_GRAVITATIONAL_PARAMETER)

        P = 27.321661 * 24 * 3600
        # self.mu = mu_Earth_Moon
        # self.d = 384400
        # self.n = 2 * math.pi / P


        # self.mu = 3.041036e-6
        # self.d = 1.49598e8
        # self.n = 1.99099e-7
        self.mu = mu_Sun_Earth

        # x, y, z = self.compute_coefficients('Horizontal', 1)
        x, y, z = self.compute_coefficients('Halo', 1)
        # x, y, z = self.compute_coefficients('Halo', 2)

        pass

    def compute_coefficients(self, type, lagrange_point_nr):

        mu = self.mu

        gammaL = Symbol('gammaL')
        if lagrange_point_nr == 1:
            gammaL = nsolve(gammaL ** 5 - (3 - mu) * gammaL ** 4 + (3 - 2 * mu) * gammaL ** 3 - mu * gammaL ** 2 + 2 * mu * gammaL - mu, gammaL, 1)
            c2 = 1 / gammaL ** 3 * ((1) ** 2 * mu + (-1) ** 2 * (1 - mu) * gammaL ** (2 + 1) / (1 - gammaL) ** (2 + 1))
            c3 = 1 / gammaL ** 3 * ((1) ** 3 * mu + (-1) ** 3 * (1 - mu) * gammaL ** (3 + 1) / (1 - gammaL) ** (3 + 1))
            c4 = 1 / gammaL ** 3 * ((1) ** 4 * mu + (-1) ** 4 * (1 - mu) * gammaL ** (4 + 1) / (1 - gammaL) ** (4 + 1))

        if lagrange_point_nr == 2:
            gammaL = nsolve(gammaL ** 5 + (3 - mu) * gammaL ** 4 + (3 - 2 * mu) * gammaL ** 3 - mu * gammaL ** 2 - 2 * mu * gammaL - mu, gammaL, 1)

            c2 = 1 / gammaL ** 3 * ((-1) ** 2 * mu + (-1) ** 2 * (1 - mu) * gammaL ** (2 + 1) / (1 + gammaL) ** (2 + 1))
            c3 = 1 / gammaL ** 3 * ((-1) ** 3 * mu + (-1) ** 3 * (1 - mu) * gammaL ** (3 + 1) / (1 + gammaL) ** (3 + 1))
            c4 = 1 / gammaL ** 3 * ((-1) ** 4 * mu + (-1) ** 4 * (1 - mu) * gammaL ** (4 + 1) / (1 + gammaL) ** (4 + 1))

        l = Symbol('l')
        l = nsolve(l ** 4 + (c2 - 2) * l ** 2 - (c2 - 1) * (1 + 2 * c2), l, 1)

        k = 2 * l / (l ** 2 + 1 - c2)
        delta = l ** 2 - c2

        d1 = 3 * l ** 2 / k * (k * (6 * l ** 2 - 1) - 2 * l)
        d2 = 8 * l ** 2 / k * (k * (11 * l ** 2 - 1) - 2 * l)

        a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2))
        a22 = 3 * c3 / (4 * (1 + 2 * c2))
        a23 = -3 * c3 * l / (4 * k * d1) * (3 * k ** 3 * l - 6 * k * (k - l) + 4)
        a24 = -3 * c3 * l / (4 * k * d1) * (2 + 3 * k * l)

        b21 = -3 * c3 * l / (2 * d1) * (3 * k * l - 4)
        b22 = 3 * c3 * l / d1

        d21 = -c3 / (2 * l ** 2)

        a31 = -9 * l / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)) + (9 * l ** 2 + 1 - c2) / (
        2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2))
        a32 = - 1 / d2 * (9 * l / 4 * (4 * c3 * (k * a24 - b22) + k * c4) + 3 / 2 * (9 * l ** 2 + 1 - c2) * (
        c3 * (k * b22 + d21 - 2 * a24) - c4))

        b31 = 3 / (8 * d2) * (
        8 * l * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2)) + (9 * l ** 2 + 1 + 2 * c2) * (
        4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)))
        b32 = 1 / d2 * (9 * l * (c3 * (k * b22 + d21 - 2 * a24) - c4) + 3 / 8 * (9 * l ** 2 + 1 + 2 * c2) * (
        4 * c3 * (k * a24 - b22) + k * c4))

        d31 = 3 / (64 * l ** 2) * (4 * c3 * a24 + c4)
        d32 = 3 / (64 * l ** 2) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))

        a1 = -3 / 2 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k ** 2)
        a2 = 3 / 2 * c3 * (a24 - 2 * a22) + 9 / 8 * c4

        s1 = 1 / (2 * l * (l * (1 + k ** 2) - 2 * k)) * (
        3 / 2 * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21) - 3 / 8 * c4 * (
        3 * k ** 4 - 8 * k ** 2 + 8))
        s2 = 1 / (2 * l * (l * (1 + k ** 2) - 2 * k)) * (
        3 / 2 * c3 * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + 5 * d21) + 3 / 8 * c4 * (12 - k ** 2))

        l1 = a1 + 2 * l ** 2 * s1
        l2 = a2 + 2 * l ** 2 * s2

        column_name = 'L' + str(lagrange_point_nr) + ' Orbits'

        df = pd.DataFrame({column_name: [gammaL, l, k, delta, c2, c3, c4, s1, s2, l1, l2, a1, a2, d1, d2, a21, a22, a23, a24, a31, a32, b21, b22, b31, b32, d21, d31, d32]},
                          index=['gammaL', 'l', 'k', 'delta', 'c2', 'c3', 'c4', 's1', 's2', 'l1', 'l2', 'a1', 'a2', 'd1', 'd2', 'a21', 'a22', 'a23', 'a24', 'a31', 'a32', 'b21', 'b22', 'b31', 'b32', 'd21', 'd31', 'd32'])

        if type == 'Horizontal':
            Ax = 10e-3*gammaL
            Az = 0
            print(Ax)
            pass
        if type == 'Vertical':
            Ax = 0
            Az = 10e-3*gammaL*20
            # Az = 57000 / (self.d*gammaL)
            pass
        if type == 'Halo':
            # Az = 10e-3 #*gammaL
            # Az = 125000/self.d #*gammaL

            Az = 110000/(149.6e6*gammaL)
            Ax = np.sqrt(float((-delta - l2*Az**2) / l1))
            print(Ax)
            print(Az)
            pass

        tau1 = 0
        deltan = 2-3

        x = a21 * Ax**2 + a22 * Az**2 - Ax * math.cos(tau1) + (a23 * Ax**2 - a24 * Az**2) * math.cos(2 * tau1) + (a31 * Ax**3 - a32 * Ax * Az**2) * math.cos(3 * tau1)
        y = k * Ax * math.sin(tau1) + (b21 * Ax**2 - b22 * Az**2) * math.sin(2 * tau1) + (b31 * Ax**3 - b32 * Ax * Az**2) * math.sin(3 * tau1)
        z = deltan * Az * math.cos(tau1) + deltan * d21 * Ax * Az * (math.cos(2 * tau1) - 3) + deltan * (d32 * Az * Ax**2 - d31 * Az**3) * math.cos(3 * tau1)
        xdot = l*Ax*math.sin(tau1) - 2 * l * (a23 * Ax**2 - a24 * Az**2) * math.sin(2 * tau1) - 3 * l * (a31 * Ax**3 - a32 * Ax * Az**2) * math.sin(3 * tau1)
        ydot = l * (k * Ax * math.cos(tau1) + 2 * (b21 * Ax**2 - b22 * Az**2) * math.cos(2 * tau1) + 3*(b31 * Ax**3 - b32 * Ax * Az**2) * math.cos(3 * tau1))
        zdot = -l * deltan * Az * math.sin(tau1) - 2 * l * deltan * d21 * Ax * Az * math.sin(2 * tau1) - 3 * l * deltan * (d32 * Az * Ax**2 - d31 * Az**3) * math.sin(3 * tau1)

        print('x: ' + str(x))
        print('y: ' + str(y))
        print('z: ' + str(z))
        print('xdot: ' + str(xdot))
        print('ydot: ' + str(ydot))
        print('zdot: ' + str(zdot) + '\n')

        if lagrange_point_nr == 1:
            print('X: ' + str((x - 1) * gammaL + 1 - mu))
            pass
        if lagrange_point_nr == 2:
            print('X: ' + str((x + 1) * gammaL + 1 - mu))
            pass

        print('Y: ' + str(y * gammaL))
        print('Z: ' + str(z * gammaL))
        print('Xdot: ' + str(xdot * gammaL))
        print('Ydot: ' + str(ydot * gammaL))
        print('Zdot: ' + str(zdot * gammaL))

        omega1 = 0
        omega2 = s1*Ax**2 + s2*Az**2
        omega = 1 + omega1 + omega2

        T = 2 * math.pi / (l * omega)
        # print('T: ' + str(T/self.n/86400))
        # print(df)
        return x, y, z


if __name__ == "__main__":
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    mu_Earth_Moon = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
    mu_Sun_Earth = (EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER) / (EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER + SUN_GRAVITATIONAL_PARAMETER)

    P = 27.321661*24*3600
    n = 2*math.pi/P

    # L1 Horizontal Lyapunov
    richardson_approximation = RichardsonApproximation()
