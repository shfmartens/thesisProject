import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
import time

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, load_initial_conditions_incl_M


class DisplayPeriodicityValidation:

    def __init__(self, orbit_type, lagrange_point_nr):
        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr

        initial_conditions_file_path = '../../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_initial_conditions.txt'
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

        differential_correction_file_path = '../../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_differential_correction.txt'
        differential_correction_df = load_differential_corrections(differential_correction_file_path)

        self.C = []
        self.T = []
        self.x = []
        self.X = []
        self.delta_r = []
        self.delta_v = []

        self.numberOfIterations = []
        self.C_half_period = []
        self.T_half_period = []
        self.X_half_period = []

        self.eigenvalues = []
        self.D = []
        self.orderOfLinearInstability = []
        self.lambda1 = []
        self.lambda2 = []
        self.lambda3 = []
        self.lambda4 = []
        self.lambda5 = []
        self.lambda6 = []
        self.v1 = []
        self.v2 = []
        self.v3 = []
        
        for row in differential_correction_df.iterrows():
            self.numberOfIterations.append(row[1][0])
            self.C_half_period.append(row[1][1])
            self.T_half_period.append(row[1][2])
            self.X_half_period.append(np.array(row[1][3:9]))

        for row in initial_conditions_incl_m_df.iterrows():
            self.C.append(row[1][0])
            self.T.append(row[1][1])
            self.x.append(row[1][2])
            self.X.append(np.array(row[1][2:8]))

            # self.X.append(np.array(row[1][3:9]))
            M = np.matrix([list(row[1][8:14]), list(row[1][14:20]), list(row[1][20:26]), list(row[1][26:32]), list(row[1][32:38]), list(row[1][38:44])])

            eigenvalue = np.linalg.eigvals(M)
            for l in eigenvalue:
                if l.imag == 0.0:

                    pass
                print(l)
            # lambda1 =

            print(eigenvalue.real.max().index)

            time.sleep(5)

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

        for i in range(0, len(self.C)):
            df = load_orbit('../../data/raw/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_' + str(i) + '.txt')
            self.delta_r.append(np.sqrt((df.head(1)['x'].values - df.tail(1)['x'].values) ** 2 +
                                        (df.head(1)['y'].values - df.tail(1)['y'].values) ** 2 +
                                        (df.head(1)['z'].values - df.tail(1)['z'].values) ** 2))

            self.delta_v.append(np.sqrt((df.head(1)['xdot'].values - df.tail(1)['xdot'].values) ** 2 +
                                        (df.head(1)['ydot'].values - df.tail(1)['ydot'].values) ** 2 +
                                        (df.head(1)['zdot'].values - df.tail(1)['zdot'].values) ** 2))

        self.figSize = (20, 20)
        self.suptitleSize = 20
        self.xlim = [min(self.x), max(self.x)]
        pass

    def plot_family(self):
        colors = sns.color_palette("Blues", n_colors=len(range(0, len(self.C))))

        # Plot 1: 3d overview
        fig1 = plt.figure(figsize=self.figSize)
        ax1 = fig1.gca()

        # Plot 2: subplots
        fig2 = plt.figure(figsize=self.figSize)
        ax2 = fig2.add_subplot(2, 2, 1, projection='3d')
        ax3 = fig2.add_subplot(2, 2, 2)
        ax4 = fig2.add_subplot(2, 2, 3)
        ax5 = fig2.add_subplot(2, 2, 4)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='grey')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='grey')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'], color='grey')
            ax4.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='grey')
            ax5.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='grey')

        bodies_df = load_bodies_location()
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)

        x = bodies_df['Moon']['r'] * np.outer(np.cos(phi), np.sin(theta)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies_df['Moon']['y']
        z = bodies_df['Moon']['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies_df['Moon']['z']

        ax1.scatter(x, y, color='black')
        ax2.plot_surface(x, y, z, color='black')
        ax3.scatter(x, z, color='black')
        ax4.scatter(y, z, color='black')
        ax5.scatter(x, y, color='black')

        for i in range(0, len(self.C), 50):
            df = load_orbit('../../data/raw/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_' + str(i) + '.txt')
            ax1.plot(df['x'], df['y'], color=colors[i], alpha=0.5)
            ax2.plot(df['x'], df['y'], df['z'], color=colors[i], alpha=0.5)
            ax3.plot(df['x'], df['z'], color=colors[i], alpha=0.5)
            ax4.plot(df['y'], df['z'], color=colors[i], alpha=0.5)
            ax5.plot(df['x'], df['y'], color=colors[i], alpha=0.5)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.set_zlabel('z [-]')
        ax2.set_zlim([-0.25, 0.25])
        ax2.grid(True, which='both', ls=':')
        ax2.view_init(30, -120)

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('z [-]')
        ax3.set_ylim([-0.25, 0.25])
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('y [-]')
        ax4.set_ylabel('z [-]')
        ax4.set_ylim([-0.25, 0.25])
        ax4.grid(True, which='both', ls=':')

        ax5.set_xlabel('x [-]')
        ax5.set_ylabel('y [-]')
        ax5.grid(True, which='both', ls=':')

        fig1.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': family', size=self.suptitleSize)
        fig1.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_family.png')
        fig2.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_family_subplots.png')

        # plt.show()
        plt.close()
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
        plt.close()
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
        plt.close()
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
        plt.close()
        pass

    def plot_periodicity_validation(self):
        f, arr = plt.subplots(4, 2, figsize=self.figSize)

        delta_y_half_period = []
        delta_xdot_half_period = []
        delta_zdot_half_period = []
        for i in range(len(self.X)):
            delta_y_half_period.append(abs(self.X[i][1] - self.X_half_period[i][1]))
            delta_xdot_half_period.append(abs(self.X[i][3] - self.X_half_period[i][3]))
            delta_zdot_half_period.append(abs(self.X[i][5] - self.X_half_period[i][5]))

        delta_J = [abs(C0 - C1) for C0, C1 in zip(self.C, self.C_half_period)]
        delta_T = [abs(T/2 - t) for T, t in zip(self.T, self.T_half_period)]

        arr[0, 0].semilogy(self.x, self.delta_r)
        arr[0, 0].semilogy(self.x, 1e-10 * np.ones(len(self.x)), color='red')
        arr[0, 0].set_xlim(self.xlim)
        arr[0, 0].set_ylim([1e-16, 1e-10])
        arr[0, 0].set_title('$||r(0) - r(T)||$')

        arr[1, 0].semilogy(self.x, self.delta_v)
        arr[1, 0].semilogy(self.x, 1e-10 * np.ones(len(self.x)), color='red')
        arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([1e-16, 1e-10])
        arr[1, 0].set_title('$||v(0) - v(T)||$')

        arr[2, 0].plot(self.x, self.numberOfIterations)
        arr[2, 0].set_xlim(self.xlim)
        arr[2, 0].set_ylim([0, 20])
        arr[2, 0].set_title('Number of iterations $N^\circ$')

        arr[0, 1].semilogy(self.x, delta_J)
        arr[0, 1].semilogy(self.x, 1e-12 * np.ones(len(self.x)), color='red')
        arr[0, 1].set_xlim(self.xlim)
        arr[0, 1].set_ylim([1e-16, 1e-10])
        arr[0, 1].set_title('$||J(0) - J(T/2)||}$')

        arr[1, 1].semilogy(self.x, delta_xdot_half_period)
        arr[1, 1].semilogy(self.x, delta_zdot_half_period)
        arr[1, 1].semilogy(self.x, 1e-12 * np.ones(len(self.x)), color='red')
        arr[1, 1].set_xlim(self.xlim)
        arr[1, 1].set_ylim([1e-16, 1e-10])
        arr[1, 1].set_title('$||\dot{x}(0) - \dot{x}(T/2)||, ||\dot{z}(0) - \dot{z}(T/2)||$')

        arr[2, 1].semilogy(self.x, delta_y_half_period)
        arr[2, 1].semilogy(self.x, 1e-12*np.ones(len(self.x)), color='red')
        arr[2, 1].set_xlim(self.xlim)
        arr[2, 1].set_ylim([1e-16, 1e-10])
        arr[2, 1].set_title('$||y(0) - y(T/2)||$')

        arr[3, 1].semilogy(self.x, delta_T)
        # arr[3, 1].semilogy(self.x, 1e-12*np.ones(len(self.x)))
        arr[3, 1].set_xlim(self.xlim)
        arr[3, 1].set_ylim([1e-16, 1e-10])
        arr[3, 1].set_title('$||T/2 - t||$')

        for i in range(4):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitType + ': periodicity validation', size=self.suptitleSize)
        plt.savefig('../../data/figures/' + self.orbitType + '_L' + str(self.lagrangePointNr) + '_periodicity.png')
        # plt.show()
        plt.close()
        pass


if __name__ == '__main__':
    lagrange_points = [1, 2]
    orbit_types = ['horizontal', 'vertical', 'halo']

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:

            if orbit_type == 'horizontal' and lagrange_point == 2:
                continue

            display_periodicity_validation = DisplayPeriodicityValidation(orbit_type, lagrange_point)
            display_periodicity_validation.plot_family()
            display_periodicity_validation.plot_orbital_energy()
            display_periodicity_validation.plot_monodromy_analysis()
            display_periodicity_validation.plot_stability()
            display_periodicity_validation.plot_periodicity_validation()
