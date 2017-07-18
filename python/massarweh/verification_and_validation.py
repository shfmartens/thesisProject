import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
# import seaborn as sns

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, cr3bp_velocity, load_initial_conditions_incl_M


class DisplayPeriodicityValidation:

    def __init__(self, orbit_type, lagrange_point_nr):
        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr

        initial_conditions_file_path = '../../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_initial_conditions.txt'
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

        self.C, self.T, self.x, self.eigenvalues, self.D, self.orderOfLinearInstability = [], [], [], [], [], []
        self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, self.lambda6 = [], [], [], [], [], []
        self.v1, self.v2, self.v3 = [], [], []

        for row in initial_conditions_incl_m_df.iterrows():
            self.C.append(row[1][1])
            self.T.append(row[1][2])
            self.x.append(np.array(row[1][3]))
            # self.X.append(np.array(row[1][3:9]))
            M = np.matrix([list(row[1][9:15]), list(row[1][15:21]), list(row[1][21:27]), list(row[1][27:33]), list(row[1][33:39]), list(row[1][39:45])])

            eigenvalue = np.linalg.eigvals(M)

            # Sorting eigenvalues from largest to smallest norm
            sorting_index = abs(eigenvalue).argsort()[::-1]

            self.eigenvalues.append(eigenvalue[sorting_index])
            self.lambda1.append(eigenvalue[sorting_index[0]])
            self.lambda2.append(eigenvalue[sorting_index[1]])
            self.lambda3.append(eigenvalue[sorting_index[2]])
            self.lambda4.append(eigenvalue[sorting_index[3]])
            self.lambda5.append(eigenvalue[sorting_index[4]])
            self.lambda6.append(eigenvalue[sorting_index[5]])


            reduction = 0
            for i in range(3):
                if (abs(eigenvalue[sorting_index[i]]) - 1.0) < 1e-2:
                    reduction += 1

            self.orderOfLinearInstability.append(3-reduction)

            self.v1.append(abs(eigenvalue[sorting_index[0]] + eigenvalue[sorting_index[5]]) / 2)
            self.v2.append(abs(eigenvalue[sorting_index[1]] + eigenvalue[sorting_index[4]]) / 2)
            self.v3.append(abs(eigenvalue[sorting_index[2]] + eigenvalue[sorting_index[3]]) / 2)
            self.D.append(np.linalg.det(M))

        self.figSize = (20, 20)
        self.suptitleSize = 20
        self.xlim = [min(self.x), max(self.x)]
        pass

    def plot_family(self):
        plt.figure()

        for i in range(0, len(self.C), 50):
            df = load_orbit('../data/raw/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_' + str(i) + '.txt')
            plt.plot(df['x'], df['y'])
        plt.xlabel('x [-]')
        plt.ylabel('y [-]')
        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': family', size=self.suptitleSize)
        plt.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_family.png')
        # plt.show()
        pass

    def plot_orbital_energy(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)

        arr[0, 0].plot(self.x, self.C)
        arr[0, 0].set_ylabel('C [-]')
        arr[0, 0].set_title('Orbital energy')

        arr[1, 0].set_xlabel('x [-]')

        arr[0, 1].plot(self.T, self.C)
        arr[0, 1].set_title('T vs C')

        arr[1, 1].plot(self.T, self.x)
        arr[1, 1].set_title('Orbital period')
        arr[1, 1].set_xlabel('T [-]')
        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')
        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': orbital energy and period', size=self.suptitleSize)
        plt.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_orbital_energy.png')
        # plt.show()
        pass

    def plot_monodromy_analysis(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)

        arr[0, 0].scatter(self.x, self.orderOfLinearInstability)
        arr[0, 0].set_ylabel('Order of linear instability [-]')
        arr[0, 0].set_xlim(self.xlim)
        arr[0, 0].set_ylim([0, 3])

        l1 = [abs(entry) for entry in self.lambda1]
        l2 = [abs(entry) for entry in self.lambda2]
        l3 = [abs(entry) for entry in self.lambda3]
        l4 = [abs(entry) for entry in self.lambda4]
        l5 = [abs(entry) for entry in self.lambda5]
        l6 = [abs(entry) for entry in self.lambda6]

        arr[0, 1].semilogy(self.x, l1)
        arr[0, 1].semilogy(self.x, l2)
        arr[0, 1].semilogy(self.x, l3)
        arr[0, 1].semilogy(self.x, l4)
        arr[0, 1].semilogy(self.x, l5)
        arr[0, 1].semilogy(self.x, l6)
        arr[0, 1].set_xlim(self.xlim)
        arr[0, 1].set_ylim([1e-4, 1e4])
        arr[0, 1].set_title('$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        arr[0, 1].set_ylabel('Eigenvalues module [-]')

        d = [abs(entry-1) for entry in self.D]
        arr[1, 0].semilogy(self.x, d)
        arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([1e-14, 1e-6])
        arr[1, 0].set_ylabel('Error ||1-Det(M)||')

        l3zoom = [abs(entry-1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[1, 1].semilogy(self.x, l3zoom)
        arr[1, 1].semilogy(self.x, l4zoom)
        arr[1, 1].semilogy(self.xlim, [1e-3, 1e-3], '--')
        arr[1, 1].set_xlim(self.xlim)
        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1, 1].set_ylabel('ZOOM: $|||\lambda_i|-1|| \forall i=3,4$')
        arr[1, 1].set_xlabel('x-axis [-]')

        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')
        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': analysis Monodromy matrix', size=self.suptitleSize)
        plt.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_monodromy_analysis.png')
        # plt.show()
        pass

    def plot_stability(self):
        f, arr = plt.subplots(3, 3, figsize=self.figSize)

        arr[0, 0].scatter(np.real(self.lambda1), np.imag(self.lambda1))
        arr[0, 0].scatter(np.real(self.lambda6), np.imag(self.lambda6))
        arr[0, 0].set_xlim([0, 3000])
        arr[0, 0].set_ylim([-1000, 1000])
        arr[0, 0].set_title('$\lambda_1, 1/\lambda_1$')
        arr[0, 0].set_xlabel('Re')
        arr[0, 0].set_ylabel('Im')

        arr[0, 1].scatter(np.real(self.lambda2), np.imag(self.lambda2))
        arr[0, 1].scatter(np.real(self.lambda5), np.imag(self.lambda5))
        arr[0, 1].set_xlim([-8, 2])
        arr[0, 1].set_ylim([-4, 4])
        arr[0, 1].set_title('$\lambda_2, 1/\lambda_2$')
        arr[0, 1].set_xlabel('Re')

        arr[0, 2].scatter(np.real(self.lambda3), np.imag(self.lambda3))
        arr[0, 2].scatter(np.real(self.lambda4), np.imag(self.lambda4))
        arr[0, 2].set_xlim([-1.5, 1.5])
        arr[0, 2].set_ylim([-1, 1])
        arr[0, 2].set_title('$\lambda_3, 1/\lambda_3$')
        arr[0, 2].set_xlabel('Re')

        arr[1, 0].scatter(self.x, np.angle(self.lambda1, deg=True))
        arr[1, 0].scatter(self.x, np.angle(self.lambda6, deg=True))
        arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([-180, 180])
        arr[1, 0].set_ylabel('Phase [$^\deg$]')

        arr[1, 1].scatter(self.x, np.angle(self.lambda2, deg=True))
        arr[1, 1].scatter(self.x, np.angle(self.lambda5, deg=True))
        arr[1, 1].set_xlim(self.xlim)
        arr[1, 1].set_ylim([-180, 180])

        arr[1, 2].scatter(self.x, np.angle(self.lambda3, deg=True))
        arr[1, 2].scatter(self.x, np.angle(self.lambda4, deg=True))
        arr[1, 2].set_xlim(self.xlim)
        arr[1, 2].set_ylim([-180, 180])

        arr[2, 0].semilogy(self.x, self.v1)
        arr[2, 0].set_xlim(self.xlim)
        arr[2, 0].set_ylim([1e-1, 1e4])
        arr[2, 0].set_ylabel('Value index [-]')
        arr[2, 0].set_title('$v_1$')

        arr[2, 1].semilogy(self.x, self.v2)
        arr[2, 1].set_xlim(self.xlim)
        arr[2, 1].set_ylim([1e-1, 1e1])
        arr[2, 1].set_title('$v_2$')
        arr[2, 1].set_xlabel('x-axis [-]')

        arr[2, 2].semilogy(self.x, self.v3)
        arr[2, 2].set_xlim(self.xlim)
        arr[2, 2].set_ylim([1e-1, 1e1])
        arr[2, 2].set_title('$v_3$')

        for i in range(3):
            for j in range(3):
                arr[i, j].grid(True, which='both', ls=':')

        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': eigenvalues $\lambda_i$ & stability index $v_i$', size=self.suptitleSize)
        plt.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_stability.png')
        # plt.show()
        pass


if __name__ == '__main__':
    lagrange_points = [1, 2]

    for lagrange_point in lagrange_points:
        display_periodicity_validation = DisplayPeriodicityValidation('horizontal', lagrange_point)
        display_periodicity_validation.plot_family()
        display_periodicity_validation.plot_orbital_energy()
        display_periodicity_validation.plot_monodromy_analysis()
        display_periodicity_validation.plot_stability()
