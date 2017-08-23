import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns

sns.set_style("whitegrid")
import time

plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy


class DisplayPeriodicityValidation:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id):
        print('=======================')
        print(str(orbit_type) + ' in L' + str(lagrange_point_nr))
        print('=======================')
        self.orbitType = orbit_type
        if self.orbitType == 'halo_n':
            self.orbitTypeForTitle = 'HaloN'
        else:
            self.orbitTypeForTitle = orbit_type.capitalize()
            if self.orbitTypeForTitle == ('Horizontal' or 'Vertical'):
                self.orbitTypeForTitle += ' Lyapunov'

        self.lagrangePointNr = lagrange_point_nr

        self.C = []
        self.T = []
        self.x = []
        self.X = []
        self.delta_r = []
        self.delta_v = []
        self.delta_x = []
        self.delta_y = []
        self.delta_z = []
        self.delta_x_dot = []
        self.delta_y_dot = []
        self.delta_z_dot = []

        self.numberOfIterations = []
        self.C_half_period = []
        self.T_half_period = []
        self.X_half_period = []

        self.eigenvalues = []
        self.D = []
        self.orderOfLinearInstability = []
        self.orbitIdBifurcations = []
        self.lambda1 = []
        self.lambda2 = []
        self.lambda3 = []
        self.lambda4 = []
        self.lambda5 = []
        self.lambda6 = []
        self.v1 = []
        self.v2 = []
        self.v3 = []

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.maxEigenvalueDeviation = 1.0e-3  # Changed from 1e-3

        self.orbitDf = load_orbit('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '.txt')

        self.eigenvectorDf_S = pd.read_table('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorDf_U = pd.read_table('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus.txt')
        self.W_S_min = load_manifold('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_min.txt')
        self.W_U_plus = load_manifold('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus.txt')
        self.W_U_min = load_manifold('../../data/raw/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_min.txt')

        self.numberOfOrbitsPerManifold = len(set(self.W_S_plus.index.get_level_values(0)))
        self.phase = []
        self.C_diff_start_W_S_plus = []
        self.C_diff_start_W_S_min = []
        self.C_diff_start_W_U_plus = []
        self.C_diff_start_W_U_min = []

        self.C_along_0_W_S_plus = []
        self.T_along_0_W_S_plus = []
        self.C_along_0_W_S_min = []
        self.T_along_0_W_S_min = []
        self.C_along_0_W_U_plus = []
        self.T_along_0_W_U_plus = []
        self.C_along_0_W_U_min = []
        self.T_along_0_W_U_min = []

        self.W_S_plus_dx = []
        self.W_S_plus_dy = []
        self.W_S_min_dx = []
        self.W_S_min_dy = []
        self.W_U_plus_dx = []
        self.W_U_plus_dy = []
        self.W_U_min_dx = []
        self.W_U_min_dy = []

        first_state_on_manifold = self.W_S_plus.xs(0).head(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_plus.xs(0).iterrows():
            self.T_along_0_W_S_plus.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_S_plus.append(abs(jacobi_on_manifold-first_jacobi_on_manifold))

        first_state_on_manifold = self.W_S_min.xs(0).head(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_min.xs(0).iterrows():
            self.T_along_0_W_S_min.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_S_min.append(abs(jacobi_on_manifold-first_jacobi_on_manifold))

        first_state_on_manifold = self.W_U_plus.xs(0).head(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_U_plus.xs(0).iterrows():
            self.T_along_0_W_U_plus.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_U_plus.append(abs(jacobi_on_manifold - first_jacobi_on_manifold))

        first_state_on_manifold = self.W_U_min.xs(0).head(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_U_min.xs(0).iterrows():
            self.T_along_0_W_U_min.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_U_min.append(abs(jacobi_on_manifold-first_jacobi_on_manifold))

        for i in range(self.numberOfOrbitsPerManifold):
            self.phase.append(i/self.numberOfOrbitsPerManifold)

            # On orbit
            state_on_orbit = self.eigenvectorLocationDf_S.xs(i).values
            jacobi_on_orbit = computeJacobiEnergy(state_on_orbit[0], state_on_orbit[1], state_on_orbit[2],
                                                  state_on_orbit[3], state_on_orbit[4], state_on_orbit[5])

            # W_S_plus
            state_on_manifold = self.W_S_plus.xs(i).head(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_S_plus.append(abs(jacobi_on_manifold-jacobi_on_orbit))
            state_on_manifold = self.W_S_plus.xs(i).tail(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1-self.massParameter)):
                self.W_S_plus_dx.append(0)
                self.W_S_plus_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_S_plus_dx.append(abs(state_on_manifold[0] - (1-self.massParameter)))
                self.W_S_plus_dy.append(0)

            # W_S_min
            state_on_manifold = self.W_S_min.xs(i).head(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_S_min.append(abs(jacobi_on_manifold - jacobi_on_orbit))
            state_on_manifold = self.W_S_min.xs(i).tail(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1-self.massParameter)):
                self.W_S_min_dx.append(0)
                self.W_S_min_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_S_min_dx.append(abs(state_on_manifold[0] - (1-self.massParameter)))
                self.W_S_min_dy.append(0)

            # W_U_plus
            state_on_manifold = self.W_U_plus.xs(i).head(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_U_plus.append(abs(jacobi_on_manifold - jacobi_on_orbit))
            state_on_manifold = self.W_U_plus.xs(i).tail(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1-self.massParameter)):
                self.W_U_plus_dx.append(0)
                self.W_U_plus_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_U_plus_dx.append(abs(state_on_manifold[0] - (1-self.massParameter)))
                self.W_U_plus_dy.append(0)

            # W_U_min
            state_on_manifold = self.W_U_min.xs(i).head(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_U_min.append(abs(jacobi_on_manifold - jacobi_on_orbit))
            state_on_manifold = self.W_U_min.xs(i).tail(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1-self.massParameter)):
                self.W_U_min_dx.append(0)
                self.W_U_min_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_U_min_dx.append(abs(state_on_manifold[0] - (1-self.massParameter)))
                self.W_U_min_dy.append(0)
            pass
        # for row in self.W_S_plus.iterrows():
        #     self.T.append(row[1][0])
        #     self.x.append(row[1][1])
        #     self.X.append(np.array(row[1][1:7]))
        #
        #     # self.X.append(np.array(row[1][3:9]))
        #     M = np.matrix(
        #         [list(row[1][7:13]), list(row[1][13:19]), list(row[1][19:25]), list(row[1][25:31]), list(row[1][31:37]),
        #          list(row[1][37:43])])
        #
        #     eigenvalue = np.linalg.eigvals(M)
        #     sorting_indices = abs(eigenvalue).argsort()[::-1]
        #     print(eigenvalue[sorting_indices])
        #     self.eigenvalues.append(eigenvalue[sorting_indices])
        #     self.lambda1.append(eigenvalue[sorting_indices[0]])
        #     self.lambda2.append(eigenvalue[sorting_indices[1]])
        #     self.lambda3.append(eigenvalue[sorting_indices[2]])
        #     self.lambda4.append(eigenvalue[sorting_indices[3]])
        #     self.lambda5.append(eigenvalue[sorting_indices[4]])
        #     self.lambda6.append(eigenvalue[sorting_indices[5]])
        #
        #     # Determine order of linear instability
        #     reduction = 0
        #     for i in range(6):
        #         if (abs(eigenvalue[i]) - 1.0) < 1e-2:
        #             reduction += 1
        #
        #     if len(self.orderOfLinearInstability) > 0:
        #         # Check for a bifurcation, when the order of linear instability changes
        #         if (6 - reduction) != self.orderOfLinearInstability[-1]:
        #             self.orbitIdBifurcations.append(row[0])
        #
        #     self.orderOfLinearInstability.append(6 - reduction)
        #     self.v1.append(abs(eigenvalue[sorting_indices[0]] + eigenvalue[sorting_indices[5]]) / 2)
        #     self.v2.append(abs(eigenvalue[sorting_indices[1]] + eigenvalue[sorting_indices[4]]) / 2)
        #     self.v3.append(abs(eigenvalue[sorting_indices[2]] + eigenvalue[sorting_indices[3]]) / 2)
        #     self.D.append(np.linalg.det(M))
        # print('Index for bifurcations: ')
        # print(self.orbitIdBifurcations)

        # Position/velocity differences at crossing
        # for i in range(0, len(self.C)):
        #     df = load_orbit(
        #         '../../data/raw/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(i) + '.txt')
        #     self.delta_r.append(np.sqrt((df.head(1)['x'].values - df.tail(1)['x'].values) ** 2 +
        #                                 (df.head(1)['y'].values - df.tail(1)['y'].values) ** 2 +
        #                                 (df.head(1)['z'].values - df.tail(1)['z'].values) ** 2))
        #
        #     self.delta_v.append(np.sqrt((df.head(1)['xdot'].values - df.tail(1)['xdot'].values) ** 2 +
        #                                 (df.head(1)['ydot'].values - df.tail(1)['ydot'].values) ** 2 +
        #                                 (df.head(1)['zdot'].values - df.tail(1)['zdot'].values) ** 2))
        #
        #     self.delta_x.append(abs(df.head(1)['x'].values - df.tail(1)['x'].values))
        #     self.delta_y.append(abs(df.head(1)['y'].values - df.tail(1)['y'].values))
        #     self.delta_z.append(abs(df.head(1)['z'].values - df.tail(1)['z'].values))
        #     self.delta_x_dot.append(abs(df.head(1)['xdot'].values - df.tail(1)['xdot'].values))
        #     self.delta_y_dot.append(abs(df.head(1)['ydot'].values - df.tail(1)['ydot'].values))
        #     self.delta_z_dot.append(abs(df.head(1)['zdot'].values - df.tail(1)['zdot'].values))

        # self.figSize = (20, 20)
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        self.colorPaletteStable = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        self.colorPaletteUnstable = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        self.plottingColors = {'lambda1': blues[40],
                               'lambda2': greens[50],
                               'lambda3': blues[90],
                               'lambda4': blues[90],
                               'lambda5': greens[70],
                               'lambda6': blues[60],
                               'singleLine': blues[80],
                               'doubleLine': [greens[50], blues[80]],
                               'tripleLine': [blues[40], greens[50], blues[80]],
                               'W_S_plus': self.colorPaletteStable[50],
                               'W_S_min': self.colorPaletteStable[80],
                               'W_U_plus': self.colorPaletteUnstable[50],
                               'W_U_min': self.colorPaletteUnstable[80],
                               'limit': 'black',
                               'orbit': 'navy'}
        self.suptitleSize = 20


        # self.xlim = [min(self.x), max(self.x)]
        pass

    def plot_manifolds(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1 = fig.add_subplot(2, 2, 4)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 2)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax0.plot_surface(x, y, z, color='black')
            ax1.contourf(x, z, y, colors='black')
            ax2.contourf(y, z, x, colors='black')
            ax3.contourf(x, y, z, colors='black')

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax0.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax1.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax2.plot(self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax1.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax2.plot(self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax1.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax2.plot(self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax1.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax2.plot(self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
        plot_alpha = 1
        line_width = 2
        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax3.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)

        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.set_zlabel('z [-]')
        ax0.set_zlim([-0.4, 0.4])
        ax0.grid(True, which='both', ls=':')
        ax0.view_init(30, -120)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('z [-]')
        # ax1.set_ylim([-0.4, 0.4])
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('y [-]')
        ax2.set_ylabel('z [-]')
        # ax2.set_ylim([-0.4, 0.4])
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview',
                     size=self.suptitleSize)

        fig.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_manifold_subplots.pdf')
        plt.close()
        pass

    def plot_eigenvectors(self):
        # Plot: subplots
        fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]*0.5))
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)


        # Determine color for plot
        plot_alpha = 1
        line_width = 1
        color = self.plottingColors['orbit']
        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], color=color, linewidth=line_width)
        ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=color, linewidth=line_width)
        ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=color, linewidth=line_width)

        eigenvector_offset = 5e3*1e-6
        line_width = 1.5

        for idx in range(self.numberOfOrbitsPerManifold):

            x_S = [self.eigenvectorLocationDf_S.xs(idx)[0] - eigenvector_offset * self.eigenvectorDf_S.xs(idx)[0],
                   self.eigenvectorLocationDf_S.xs(idx)[0] + eigenvector_offset * self.eigenvectorDf_S.xs(idx)[0]]
            y_S = [self.eigenvectorLocationDf_S.xs(idx)[1] - eigenvector_offset * self.eigenvectorDf_S.xs(idx)[1],
                   self.eigenvectorLocationDf_S.xs(idx)[1] + eigenvector_offset * self.eigenvectorDf_S.xs(idx)[1]]
            z_S = [self.eigenvectorLocationDf_S.xs(idx)[2] - eigenvector_offset * self.eigenvectorDf_S.xs(idx)[2],
                   self.eigenvectorLocationDf_S.xs(idx)[2] + eigenvector_offset * self.eigenvectorDf_S.xs(idx)[2]]

            x_U = [self.eigenvectorLocationDf_U.xs(idx)[0] - eigenvector_offset * self.eigenvectorDf_U.xs(idx)[0],
                   self.eigenvectorLocationDf_U.xs(idx)[0] + eigenvector_offset * self.eigenvectorDf_U.xs(idx)[0]]
            y_U = [self.eigenvectorLocationDf_U.xs(idx)[1] - eigenvector_offset * self.eigenvectorDf_U.xs(idx)[1],
                   self.eigenvectorLocationDf_U.xs(idx)[1] + eigenvector_offset * self.eigenvectorDf_U.xs(idx)[1]]
            z_U = [self.eigenvectorLocationDf_U.xs(idx)[2] - eigenvector_offset * self.eigenvectorDf_U.xs(idx)[2],
                   self.eigenvectorLocationDf_U.xs(idx)[2] + eigenvector_offset * self.eigenvectorDf_U.xs(idx)[2]]

            ax0.plot(x_S, y_S, color=self.colorPaletteStable[idx], linewidth=line_width)
            ax1.plot(x_S, z_S, color=self.colorPaletteStable[idx], linewidth=line_width)
            ax2.plot(y_S, z_S, color=self.colorPaletteStable[idx], linewidth=line_width)

            ax0.plot(x_U, y_U, color=self.colorPaletteUnstable[idx], linewidth=line_width)
            ax1.plot(x_U, z_U, color=self.colorPaletteUnstable[idx], linewidth=line_width)
            ax2.plot(y_U, z_U, color=self.colorPaletteUnstable[idx], linewidth=line_width)
            pass

        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.grid(True, which='both', ls=':')

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('z [-]')
        ax1.set_ylim([-0.15, 0.15])
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('y [-]')
        ax2.set_ylabel('z [-]')
        ax2.set_ylim([-0.15, 0.15])
        ax2.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' $\{ \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^S_i}{|\mathbf{v}^S_i|}, \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^U_i}{|\mathbf{v}^U_i|} \}$ - Spatial overview',
                     size=self.suptitleSize)

        fig.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_eigenvector_subplots.pdf')
        plt.close()
        pass

    def plot_orbital_energy(self):

        f, arr = plt.subplots(1, 2, figsize=(self.figSize[0], self.figSize[1] * 0.5))
        for i in range(2):
            arr[i].grid(True, which='both', ls=':')

        arr[0].plot(self.x, self.C, c=self.plottingColors['doubleLine'][0])
        arr[0].set_ylabel('C [-]')
        arr[0].set_title('Spatial dependance')
        arr[0].tick_params('y', colors=self.plottingColors['doubleLine'][0])
        arr[0].set_xlabel('x [-]')

        ax2 = arr[0].twinx()
        ax2.plot(self.x, self.T, c=self.plottingColors['doubleLine'][1], linestyle=':')
        ax2.tick_params('y', colors=self.plottingColors['doubleLine'][1])
        ax2.set_ylabel('T [-]')

        arr[1].plot(self.T, self.C, c=self.plottingColors['singleLine'])
        arr[1].set_title('Relative dependance')
        arr[1].set_xlabel('T [-]')
        arr[1].set_ylabel('C [-]')

        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.suptitle('L' + str(self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' - Orbital energy and period',
                     size=self.suptitleSize)
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_orbital_energy.png')
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_orbital_energy.pdf')
        # plt.close()
        pass

    def plot_monodromy_analysis(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        arr[1, 0].scatter(self.x, self.orderOfLinearInstability, s=size, c=self.plottingColors['singleLine'])
        arr[1, 0].set_ylabel('Order of linear instability [-]')
        arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([0, 3])

        l1 = [abs(entry) for entry in self.lambda1]
        l2 = [abs(entry) for entry in self.lambda2]
        l3 = [abs(entry) for entry in self.lambda3]
        l4 = [abs(entry) for entry in self.lambda4]
        l5 = [abs(entry) for entry in self.lambda5]
        l6 = [abs(entry) for entry in self.lambda6]

        arr[0, 0].semilogy(self.x, l1, c=self.plottingColors['lambda1'])
        arr[0, 0].semilogy(self.x, l2, c=self.plottingColors['lambda2'])
        arr[0, 0].semilogy(self.x, l3, c=self.plottingColors['lambda3'])
        arr[0, 0].semilogy(self.x, l4, c=self.plottingColors['lambda4'])
        arr[0, 0].semilogy(self.x, l5, c=self.plottingColors['lambda5'])
        arr[0, 0].semilogy(self.x, l6, c=self.plottingColors['lambda6'])
        arr[0, 0].set_xlim(self.xlim)
        arr[0, 0].set_ylim([1e-4, 1e4])
        arr[0, 0].set_title(
            '$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        arr[0, 0].set_ylabel('Eigenvalues module [-]')

        d = [abs(entry - 1) for entry in self.D]
        arr[0, 1].semilogy(self.x, d, c=self.plottingColors['singleLine'], linewidth=1)
        arr[0, 1].set_xlim(self.xlim)
        arr[0, 1].set_ylim([1e-14, 1e-6])
        arr[0, 1].set_ylabel('Error $| 1 - Det(M) |$')

        l3zoom = [abs(entry - 1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[1, 1].semilogy(self.x, l3zoom, c=self.plottingColors['doubleLine'][0], linewidth=1)
        arr[1, 1].semilogy(self.x, l4zoom, c=self.plottingColors['doubleLine'][1], linewidth=1, linestyle=':')
        arr[1, 1].semilogy(self.xlim, [1e-3, 1e-3], '--', c=self.plottingColors['limit'], linewidth=1)
        arr[1, 1].set_xlim(self.xlim)
        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1, 1].set_ylabel(' $|  | \lambda_i|-1  |  \\forall i=3,4$')
        arr[1, 1].set_xlabel('x-axis [-]')

        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')
        plt.suptitle(
            'L' + str(self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' - Eigensystem analysis monodromy matrix',
            size=self.suptitleSize)
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_monodromy_analysis.png')
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_monodromy_analysis.pdf')
        # plt.close()
        pass

    def plot_stability(self):
        unit_circle_1 = plt.Circle((0, 0), 1, color='grey', fill=False)
        unit_circle_2 = plt.Circle((0, 0), 1, color='grey', fill=False)

        size = 7

        f, arr = plt.subplots(3, 3, figsize=self.figSize)

        arr[0, 0].scatter(np.real(self.lambda1), np.imag(self.lambda1), c=self.plottingColors['lambda1'], s=size)
        arr[0, 0].scatter(np.real(self.lambda6), np.imag(self.lambda6), c=self.plottingColors['lambda6'], s=size)
        arr[0, 0].set_xlim([0, 3000])
        arr[0, 0].set_ylim([-1000, 1000])
        arr[0, 0].set_title('$\lambda_1, 1/\lambda_1$')
        arr[0, 0].set_xlabel('Re')
        arr[0, 0].set_ylabel('Im')

        arr[0, 1].scatter(np.real(self.lambda2), np.imag(self.lambda2), c=self.plottingColors['lambda2'], s=size)
        arr[0, 1].scatter(np.real(self.lambda5), np.imag(self.lambda5), c=self.plottingColors['lambda5'], s=size)
        arr[0, 1].set_xlim([-8, 2])
        arr[0, 1].set_ylim([-4, 4])
        arr[0, 1].set_title('$\lambda_2, 1/\lambda_2$')
        arr[0, 1].set_xlabel('Re')
        arr[0, 1].add_artist(unit_circle_1)

        arr[0, 2].scatter(np.real(self.lambda3), np.imag(self.lambda3), c=self.plottingColors['lambda3'], s=size)
        arr[0, 2].scatter(np.real(self.lambda4), np.imag(self.lambda4), c=self.plottingColors['lambda4'], s=size)
        arr[0, 2].set_xlim([-1.5, 1.5])
        arr[0, 2].set_ylim([-1, 1])
        arr[0, 2].set_title('$\lambda_3, 1/\lambda_3$')
        arr[0, 2].set_xlabel('Re')
        arr[0, 2].add_artist(unit_circle_2)

        arr[1, 0].scatter(self.x, np.angle(self.lambda1, deg=True), c=self.plottingColors['lambda1'], s=size)
        arr[1, 0].scatter(self.x, np.angle(self.lambda6, deg=True), c=self.plottingColors['lambda6'], s=size)
        arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([-180, 180])
        arr[1, 0].set_ylabel('Phase [$^\circ$]')

        arr[1, 1].scatter(self.x, np.angle(self.lambda2, deg=True), c=self.plottingColors['lambda2'], s=size)
        arr[1, 1].scatter(self.x, np.angle(self.lambda5, deg=True), c=self.plottingColors['lambda5'], s=size)
        arr[1, 1].set_xlim(self.xlim)
        arr[1, 1].set_ylim([-180, 180])

        arr[1, 2].scatter(self.x, np.angle(self.lambda3, deg=True), c=self.plottingColors['lambda3'], s=size)
        arr[1, 2].scatter(self.x, np.angle(self.lambda4, deg=True), c=self.plottingColors['lambda4'], s=size)
        arr[1, 2].set_xlim(self.xlim)
        arr[1, 2].set_ylim([-180, 180])

        arr[2, 0].semilogy(self.x, self.v1, c=self.plottingColors['lambda6'])
        arr[2, 0].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 0].set_xlim(self.xlim)
        arr[2, 0].set_ylim([1e-1, 1e4])
        arr[2, 0].set_ylabel('Value index [-]')
        arr[2, 0].set_title('$v_1$')

        arr[2, 1].semilogy(self.x, self.v2, c=self.plottingColors['lambda5'])
        arr[2, 1].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 1].set_xlim(self.xlim)
        arr[2, 1].set_ylim([1e-1, 1e1])
        arr[2, 1].set_title('$v_2$')
        arr[2, 1].set_xlabel('x-axis [-]')

        arr[2, 2].semilogy(self.x, self.v3, c=self.plottingColors['lambda4'])
        arr[2, 2].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 2].set_xlim(self.xlim)
        arr[2, 2].set_ylim([1e-1, 1e1])
        arr[2, 2].set_title('$v_3$')

        for i in range(3):
            for j in range(3):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('L' + str(
            self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' - Eigenvalues $\lambda_i$ \& stability index $v_i$',
                     size=self.suptitleSize)
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_stability.png')
        # plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_stability.pdf')
        # plt.close()
        pass

    def plot_periodicity_validation(self):
        f, arr = plt.subplots(3, 1, figsize=self.figSize)
        linewidth = 1
        scatter_size = 10
        ylim = [1e-16, 1e-9]

        arr[0].semilogy(self.phase, self.C_diff_start_W_S_plus, linewidth=linewidth, c=self.plottingColors['W_S_plus'], label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S+}$')
        arr[0].semilogy(self.phase, self.C_diff_start_W_S_min, linewidth=linewidth, c=self.plottingColors['W_S_min'], label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
        arr[0].semilogy(self.phase, self.C_diff_start_W_U_plus, linewidth=linewidth, c=self.plottingColors['W_U_plus'], label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U+}$')
        arr[0].semilogy(self.phase, self.C_diff_start_W_U_min, linewidth=linewidth, c=self.plottingColors['W_U_min'], label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')
        arr[0].legend(frameon=True, loc='upper right')
        # arr[0, 0].semilogy(self.phase, 1e-12 * np.ones(len(self.phase)), color=self.plottingColors['limit'], linewidth=1,
        #                    linestyle='--')
        arr[0].set_xlim([0, 1])
        arr[0].set_xlabel('$\\tau$ [-]')
        arr[0].set_ylim(ylim)
        arr[0].set_ylabel('$|C(\mathbf{X^i_0}) - C(\mathbf{X^p})|$ [-]')
        arr[0].set_title('Jacobi energy deviation between orbit and manifold')

        if self.lagrangePointNr == 1:
            arr[1].scatter(self.phase, self.W_S_plus_dx, c=self.plottingColors['W_S_plus'], label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$', s=scatter_size)
            arr[1].scatter(self.phase, self.W_S_min_dy, c=self.plottingColors['W_S_min'], label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', s=scatter_size, marker='v')
            arr[1].scatter(self.phase, self.W_U_plus_dx, c=self.plottingColors['W_U_plus'], label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$', s=scatter_size)
            arr[1].scatter(self.phase, self.W_U_min_dy, c=self.plottingColors['W_U_plus'], label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', s=scatter_size, marker='v')
        if self.lagrangePointNr == 2:
            arr[1].scatter(self.phase, self.W_S_plus_dy, c=self.plottingColors['W_S_plus'],
                           label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$', s=scatter_size)
            arr[1].scatter(self.phase, self.W_S_min_dx, c=self.plottingColors['W_S_min'],
                           label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', s=scatter_size,
                           marker='v')
            arr[1].scatter(self.phase, self.W_U_plus_dy, c=self.plottingColors['W_U_plus'],
                           label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$', s=scatter_size)
            arr[1].scatter(self.phase, self.W_U_min_dx, c=self.plottingColors['W_U_plus'],
                           label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', s=scatter_size,
                           marker='v')

        arr[1].set_ylabel('$|x^i_n - (1-\mu)|, \; |y^i_n| [-]$')  #  \; \\forall i =0, 1, \ldots m \in \mathcal{W}
        arr[1].set_yscale("log")
        arr[1].legend(frameon=True, loc='upper right')
        arr[1].set_xlim([0, 1])
        arr[1].set_xlabel('$\\tau$ [-]')
        arr[1].set_ylim(ylim)
        arr[1].set_title('Position deviation at $U_i \;  \\forall \; i = 1, \ldots, 4$')

        arr[2].scatter(self.T_along_0_W_S_plus, self.C_along_0_W_S_plus, c=self.plottingColors['W_S_plus'],
                       label='$\mathbf{X}^0_i \; \\forall \; i \in \mathcal{W}^{S+}$', s=scatter_size)
        arr[2].scatter(self.T_along_0_W_S_min, self.C_along_0_W_S_min, c=self.plottingColors['W_S_min'],
                       label='$\mathbf{X}^0_i \; \\forall \; i \in \mathcal{W}^{S-}$', s=scatter_size, marker='v')
        arr[2].scatter(self.T_along_0_W_U_plus, self.C_along_0_W_U_plus, c=self.plottingColors['W_U_plus'],
                       label='$\mathbf{X}^0_i \; \\forall \; i \in \mathcal{W}^{U+}$', s=scatter_size)
        arr[2].scatter(self.T_along_0_W_U_min, self.C_along_0_W_U_min, c=self.plottingColors['W_U_min'],
                       label='$\mathbf{X}^0_i \; \\forall \; i \in \mathcal{W}^{U-}$', s=scatter_size, marker='v')
        arr[2].legend(frameon=True, loc='upper right')
        arr[2].set_xlabel('$|t|$ [-]')
        arr[2].set_ylabel('$|C(\mathbf{X^0_i}) - C(\mathbf{X^0_0})|$ [-]')
        arr[2].set_ylim(ylim)
        arr[2].set_title('Energy deviation along the manifold ($\\tau=0$)')
        arr[2].set_yscale("log")

        for i in range(3):
            arr[i].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        plt.suptitle(
            'L' + str(self.lagrangePointNr) + ' ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Validation',
            size=self.suptitleSize)

        plt.savefig('../../data/figures/manifold/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_manifold_periodicity.pdf')
        plt.close()
        pass


if __name__ == '__main__':
    lagrange_points = [1, 2]
    orbit_types = ['horizontal', 'vertical', 'halo']
    # lagrange_points = [1]
    # orbit_types = ['halo']

    orbit_id = {1: {'horizontal': 577, 'halo': 836, 'vertical': 1159},
                2: {'horizontal': 760, 'halo': 651, 'vertical': 1275}}

    for lagrange_point in lagrange_points:
        for orbit_type in orbit_types:
            if orbit_type == 'horizontal' and lagrange_point == 2:
                continue
            display_periodicity_validation = DisplayPeriodicityValidation(orbit_type, lagrange_point, orbit_id[lagrange_point][orbit_type])
            display_periodicity_validation.plot_manifolds()
            display_periodicity_validation.plot_eigenvectors()
            # display_periodicity_validation.plot_orbital_energy()
            # display_periodicity_validation.plot_monodromy_analysis()
            # display_periodicity_validation.plot_stability()
            display_periodicity_validation.plot_periodicity_validation()
            # plt.show()
            del display_periodicity_validation
