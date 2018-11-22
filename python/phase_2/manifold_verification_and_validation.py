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
from scipy.interpolate import interp1d
sns.set_style("whitegrid")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, cr3bp_velocity


class DisplayPeriodicityValidation:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        print('========================')
        print(str(orbit_type) + ' in L' + str(lagrange_point_nr))
        print('========================')
        self.orbitType = orbit_type
        self.orbitId = orbit_id
        self.orbitTypeForTitle = orbit_type.capitalize()
        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.lagrangePointNr = lagrange_point_nr

        self.eigenvalues = []
        self.D = []
        self.T = []
        self.X = []
        self.x = []
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

        self.maxEigenvalueDeviation = 1.0e-3

        # initial_conditions_file_path = '../../data/raw/orbits/L' + str(lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
        # initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)
        # self.C = initial_conditions_incl_m_df.iloc[orbit_id][0]

        self.orbitDf = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '.txt')
        self.C = computeJacobiEnergy(self.orbitDf.iloc[0]['x'], self.orbitDf.iloc[0]['y'],
                                     self.orbitDf.iloc[0]['z'], self.orbitDf.iloc[0]['xdot'],
                                     self.orbitDf.iloc[0]['ydot'], self.orbitDf.iloc[0]['zdot'])

        self.eigenvectorDf_S = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorDf_U = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus.txt')
        self.W_S_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_min.txt')
        self.W_U_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus.txt')
        self.W_U_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_min.txt')

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

        first_state_on_manifold = self.W_S_plus.xs(0).tail(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_plus.xs(0).iloc[::-1].iterrows():
            self.T_along_0_W_S_plus.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_S_plus.append(abs(jacobi_on_manifold-first_jacobi_on_manifold))

        first_state_on_manifold = self.W_S_min.xs(0).tail(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_min.xs(0).iloc[::-1].iterrows():
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
            state_on_manifold = self.W_S_plus.xs(i).tail(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_S_plus.append(abs(jacobi_on_manifold-jacobi_on_orbit))
            state_on_manifold = self.W_S_plus.xs(i).head(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1-self.massParameter)):
                self.W_S_plus_dx.append(0)
                self.W_S_plus_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_S_plus_dx.append(abs(state_on_manifold[0] - (1-self.massParameter)))
                self.W_S_plus_dy.append(0)

            # W_S_min
            state_on_manifold = self.W_S_min.xs(i).tail(1).values[0]
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],
                                                     state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_diff_start_W_S_min.append(abs(jacobi_on_manifold - jacobi_on_orbit))
            state_on_manifold = self.W_S_min.xs(i).head(1).values[0]
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

        # W_S_plus_incl_STM = load_manifold_incl_stm(
        #     '../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
        #         orbit_id) + '_W_S_plus.txt')
        #
        # for row in W_S_plus_incl_STM.xs(0).iterrows():
        #
        #     self.T.append(row[0])
        #     self.x.append(row[1][1])
        #     self.X.append(np.array(row[1][0:6]))
        #     # self.X.append(np.array(row[1][3:9]))
        #     M = np.matrix(
        #         [list(row[1][6:12]), list(row[1][12:18]), list(row[1][18:24]), list(row[1][24:30]), list(row[1][30:36]),
        #          list(row[1][36:42])])
        #     eigenvalue = np.linalg.eigvals(M)
        #     sorting_indices = abs(eigenvalue).argsort()[::-1]
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

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        self.colorPaletteStable = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        self.colorPaletteUnstable = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'W_S_plus': self.colorPaletteStable[int(0.9*self.numberOfOrbitsPerManifold)],
                               'W_S_min': self.colorPaletteStable[int(0.4*self.numberOfOrbitsPerManifold)],
                               'W_U_plus': self.colorPaletteUnstable[int(0.9*self.numberOfOrbitsPerManifold)],
                               'W_U_min': self.colorPaletteUnstable[int(0.4*self.numberOfOrbitsPerManifold)],
                               'limit': 'black',
                               'orbit': 'navy'}
        self.suptitleSize = 20
        # self.xlim = [min(self.x), max(self.x)]
        pass

    def plot_manifolds(self):
        # Plot: subplots
        if self.orbitType == 'horizontal':
            fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]/2))
            ax0 = fig.add_subplot(1, 2, 1, projection='3d')
            ax3 = fig.add_subplot(1, 2, 2)
        else:
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
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')
            if self.orbitType != 'horizontal':
                ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
                ax2.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax0.plot_surface(x, y, z, color='black')
            ax3.contourf(x, y, z, colors='black')
            if self.orbitType != 'horizontal':
                ax1.contourf(x, z, y, colors='black')
                ax2.contourf(y, z, x, colors='black')

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax0.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
        plot_alpha = 1
        line_width = 2
        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax3.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        if self.orbitType != 'horizontal':
            ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
            ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)

        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.set_zlabel('z [-]')
        ax0.set_zlim([-0.4, 0.4])
        ax0.grid(True, which='both', ls=':')
        ax0.view_init(30, -120)

        if self.orbitType != 'horizontal':
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
        if self.orbitType != 'horizontal':
            fig.subplots_adjust(top=0.9)
        else:
            fig.subplots_adjust(top=0.8)

        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)

        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_zoom(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lagrangePointNr == 2:
            if self.orbitType == 'horizontal':
                ax.set_xlim([1-self.massParameter, 1.45])
                ax.set_ylim(-0.15, 0.15)
            else:
                ax.set_xlim([1 - self.massParameter, 1.25])
                ax.set_ylim(-0.05, 0.05)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L2']

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)

        ax.annotate('\\textbf{Unstable exterior} $\\mathbf{ \mathcal{W}^{U+}}$',
                    xy=(1.44, -0.1), xycoords='data',
                    xytext=(-100, 50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_U_plus'], linewidth=2))

        ax.annotate('\\textbf{Stable exterior} $\\mathbf{ \mathcal{W}^{S+}}$',
                    xy=(1.37, 0.025), xycoords='data',
                    xytext=(20, 50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_S_plus'], linewidth=2))

        ax.annotate('\\textbf{Unstable interior} $\\mathbf{ \mathcal{W}^{U-}}$',
                    xy=(1.01, 0.11), xycoords='data',
                    xytext=(50, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_U_min'], linewidth=2))
        ax.annotate('\\textbf{Stable interior} $\\mathbf{ \mathcal{W}^{S-}}$',
                    xy=(1.1, -0.11), xycoords='data',
                    xytext=(-150, -10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_S_min'], linewidth=2))

        # plt.show()
        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots_zoom.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_total(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        W_S_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_plus.txt')
        W_S_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_min.txt')
        W_U_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_plus.txt')
        W_U_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_min.txt')

        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax.plot(W_S_plus_L1.xs(manifold_orbit_number)['x'], W_S_plus_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_S_min_L1.xs(manifold_orbit_number)['x'], W_S_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_plus_L1.xs(manifold_orbit_number)['x'], W_U_plus_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_min_L1.xs(manifold_orbit_number)['x'], W_U_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        orbitDf_L1 = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '.txt')
        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax.plot(orbitDf_L1['x'], orbitDf_L1['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        # Plot zero velocity surface
        x_range = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 0.001)
        y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, 3.15)
        if z_mesh.min() < 0:
            plt.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        plt.suptitle('$L_1, L_2$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)
        line_width = 2
        plt.plot([-2.3, -3.8], [0, 0], 'k-', lw=line_width)
        plt.plot([-0.78, -0.25], [0, 0], 'k-', lw=line_width)
        plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, 0.11], 'k-', lw=line_width)
        plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, -0.11], 'k-', lw=line_width)

        size = 15
        ax.text(-2.1, 0, "$\mathbf{U_4}$", ha="center", va="center", size=size)
        ax.text(-1.0, 0, "$\mathbf{U_1}$", ha="center", va="center", size=size)
        ax.text(1.0 - self.massParameter, 0.25, "$\mathbf{U_3}$", ha="center", va="center", size=size)
        ax.text(1.0 - self.massParameter, -0.3, "$\mathbf{U_2}$", ha="center", va="center", size=size)

        ax.set_xlim([-6, 4])
        ax.set_ylim([-3, 3])

        # plt.show()
        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots_total.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_total_zoom(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        W_S_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_plus.txt')
        W_S_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_min.txt')
        W_U_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_plus.txt')
        W_U_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_min.txt')

        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax.plot(W_S_plus_L1.xs(manifold_orbit_number)['x'], W_S_plus_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_S_min_L1.xs(manifold_orbit_number)['x'], W_S_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_plus_L1.xs(manifold_orbit_number)['x'], W_U_plus_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_min_L1.xs(manifold_orbit_number)['x'], W_U_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        orbitDf_L1 = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '.txt')
        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax.plot(orbitDf_L1['x'], orbitDf_L1['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        # Plot zero velocity surface
        x_range = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 0.001)
        y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, 3.15)
        if z_mesh.min() < 0:
            plt.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        plt.suptitle('$L_1, L_2$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)
        line_width = 2
        plt.plot([-2.3, -3.8], [0, 0], 'k-', lw=line_width)
        plt.plot([-0.78, -0.25], [0, 0], 'k-', lw=line_width)
        plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, 0.11], 'k-', lw=line_width)
        plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, -0.11], 'k-', lw=line_width)

        size = 15
        ax.text(-2.1, 0, "$\mathbf{U_4}$", ha="center", va="center", size=size)
        ax.text(-1.0, 0, "$\mathbf{U_1}$", ha="center", va="center", size=size)
        ax.text(1.005, 0.11, "$\mathbf{U_3}$", ha="center", va="center", size=size)
        ax.text(1.005, -0.115, "$\mathbf{U_2}$", ha="center", va="center", size=size)

        ax.set_xlim([0.55, 1.45])
        ax.set_ylim([-0.2, 0.2])

        # plt.show()
        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots_total_zoom.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_total_zoom_2(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        W_S_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_plus.txt')
        W_S_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_min.txt')
        W_U_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_plus.txt')
        W_U_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_min.txt')

        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            # ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            # ax.plot(W_S_plus_L1.xs(manifold_orbit_number)['x'], W_S_plus_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_S_min_L1.xs(manifold_orbit_number)['x'], W_S_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(W_U_plus_L1.xs(manifold_orbit_number)['x'], W_U_plus_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_min_L1.xs(manifold_orbit_number)['x'], W_U_min_L1.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        # orbitDf_L1 = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '.txt')
        # ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        # ax.plot(orbitDf_L1['x'], orbitDf_L1['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
        #         linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        ax.set_xlim([-1.0, 0.0])
        ax.set_ylim([-0.3, 0.3])

        # Plot zero velocity surface
        x_range = np.arange(-6.0, 4.0, 0.001)
        y_range = np.arange(-3.0, 3.0, 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, 3.15)
        if z_mesh.min() < 0:
            plt.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        plt.suptitle('$L_1$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S-}, \mathcal{W}^{U-} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)
        line_width = 2
        # plt.plot([-2.3, -3.8], [0, 0], 'k-', lw=line_width)
        plt.plot([-0.78, -0.25], [0, 0], 'k-', lw=line_width)
        # plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, 0.11], 'k-', lw=line_width)
        # plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, -0.11], 'k-', lw=line_width)

        size = 15
        # ax.text(-2.1, 0, "$\mathbf{U_4}$", ha="center", va="center", size=size)
        ax.text(-0.2, 0, "$\mathbf{U_1}$", ha="center", va="center", size=size)
        # ax.text(1.005, 0.11, "$\mathbf{U_3}$", ha="center", va="center", size=size)
        # ax.text(1.005, -0.115, "$\mathbf{U_2}$", ha="center", va="center", size=size)

        # plt.show()
        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots_total_zoom_2.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_total_zoom_3(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        # W_S_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_plus.txt')
        # W_S_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_S_min.txt')
        # W_U_plus_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_plus.txt')
        # W_U_min_L1 = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '_W_U_min.txt')

        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            # ax.plot(W_S_plus_L1.xs(manifold_orbit_number)['x'], W_S_plus_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(W_S_min_L1.xs(manifold_orbit_number)['x'], W_S_min_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(W_U_plus_L1.xs(manifold_orbit_number)['x'], W_U_plus_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            # ax.plot(W_U_min_L1.xs(manifold_orbit_number)['x'], W_U_min_L1.xs(manifold_orbit_number)['y'],
            #         color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        # orbitDf_L1 = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(1) + '_' + self.orbitType + '_' + str(330) + '.txt')
        # ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        # ax.plot(orbitDf_L1['x'], orbitDf_L1['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
        #         linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        # Plot zero velocity surface
        x_range = np.arange(-6.0, 4.0, 0.001)
        y_range = np.arange(-3.0, 3.0, 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, 3.15)
        if z_mesh.min() < 0:
            plt.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        plt.suptitle('$L_2$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S+}, \mathcal{W}^{U+} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)
        line_width = 2
        plt.plot([-2.3, -3.8], [0, 0], 'k-', lw=line_width)
        # plt.plot([-0.78, -0.25], [0, 0], 'k-', lw=line_width)
        # plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, 0.11], 'k-', lw=line_width)
        # plt.plot([1 - self.massParameter, 1 - self.massParameter], [0.0, -0.11], 'k-', lw=line_width)

        size = 15
        ax.text(-2.1, 0, "$\mathbf{U_4}$", ha="center", va="center", size=size)
        # ax.text(-0.2, 0, "$\mathbf{U_1}$", ha="center", va="center", size=size)
        # ax.text(1.005, 0.11, "$\mathbf{U_3}$", ha="center", va="center", size=size)
        # ax.text(1.005, -0.115, "$\mathbf{U_2}$", ha="center", va="center", size=size)
        ax.set_xlim([-4.0, -0.5])
        ax.set_ylim([-1.0, 1.0])

        # plt.show()
        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots_total_zoom_3.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_eigenvectors(self):
        # Plot: subplots
        fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]*0.5))
        # fig = plt.figure(figsize=self.figSize)
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

        if self.orbitType == 'vertical':
            eigenvector_offset = 0.004
        else:
            eigenvector_offset = 0.02

        for idx in range(self.numberOfOrbitsPerManifold):
            if idx%4 != 0:
                continue

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

            # ax0.plot(x_S, y_S, color=self.plottingColors['W_S_min'], linewidth=line_width)
            ax0.annotate("", xy=(x_S[0], y_S[0]), xytext=(x_S[1], y_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax1.annotate("", xy=(x_S[0], z_S[0]), xytext=(x_S[1], z_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax2.annotate("", xy=(y_S[0], z_S[0]), xytext=(y_S[1], z_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            # ax1.plot(x_S, z_S, color=self.colorPaletteStable[idx], linewidth=line_width)
            # ax2.plot(y_S, z_S, color=self.colorPaletteStable[idx], linewidth=line_width)

            # ax0.annotate("", xy=(x_U[0], y_U[0]), xytext=(x_U[1], y_U[1]), arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0))
            ax0.annotate("", xy=(x_U[0], y_U[0]), xytext=(x_U[1], y_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            ax1.annotate("", xy=(x_U[0], z_U[0]), xytext=(x_U[1], z_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            ax2.annotate("", xy=(y_U[0], z_U[0]), xytext=(y_U[1], z_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            # ax1.plot(x_U, z_U, color=self.colorPaletteUnstable[idx], linewidth=line_width)
            # ax2.plot(y_U, z_U, color=self.colorPaletteUnstable[idx], linewidth=line_width)
            pass
        xlim = ax0.get_xlim()
        ax0.set_xlim(xlim[0]*0.975, xlim[1]*1.025)
        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.grid(True, which='both', ls=':')

        zlim = ax1.get_ylim()
        ax1.set_xlim(xlim[0] * 0.975, xlim[1] * 1.025)
        ax1.set_ylim(zlim[0] * 1.05, zlim[1] * 1.05)
        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('z [-]')
        ax1.grid(True, which='both', ls=':')

        xlim = ax2.get_xlim()
        ax2.set_ylim(zlim[0] * 1.05, zlim[1] * 1.05)
        ax2.set_xlim(xlim[0] * 1.05, xlim[1] * 1.05)
        ax2.set_xlabel('y [-]')
        ax2.set_ylabel('z [-]')
        ax2.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        fig.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^S_i}{|\mathbf{v}^S_i|}, \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^U_i}{|\mathbf{v}^U_i|} \}$ - Spatial overview  at C = ' + str(np.round(self.C, 3)),
                     size=self.suptitleSize)

        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector_subplots.pdf',
                    transparent=True)
#        fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector_subplots.png')
        plt.close()
        pass

    def plot_stm_analysis(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        arr[1, 0].scatter(self.T, self.orderOfLinearInstability, s=size, c=self.plottingColors['singleLine'])
        arr[1, 0].set_ylabel('Order of linear instability [-]')
        # arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([0, 3])
        arr[1, 0].set_xlabel('t [-]')

        l1 = [abs(entry) for entry in self.lambda1]
        l2 = [abs(entry) for entry in self.lambda2]
        l3 = [abs(entry) for entry in self.lambda3]
        l4 = [abs(entry) for entry in self.lambda4]
        l5 = [abs(entry) for entry in self.lambda5]
        l6 = [abs(entry) for entry in self.lambda6]

        arr[0, 0].semilogy(self.T, l1, c=self.plottingColors['lambda1'])
        arr[0, 0].semilogy(self.T, l2, c=self.plottingColors['lambda2'])
        arr[0, 0].semilogy(self.T, l3, c=self.plottingColors['lambda3'])
        arr[0, 0].semilogy(self.T, l4, c=self.plottingColors['lambda4'])
        arr[0, 0].semilogy(self.T, l5, c=self.plottingColors['lambda5'])
        arr[0, 0].semilogy(self.T, l6, c=self.plottingColors['lambda6'])
        # arr[0, 0].set_xlim(self.xlim)
        # arr[0, 0].set_ylim([1e-4, 1e4])
        arr[0, 0].set_title(
            '$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        arr[0, 0].set_ylabel('Eigenvalues module [-]')

        d = [abs(entry - 1) for entry in self.D]
        arr[0, 1].semilogy(self.T, d, c=self.plottingColors['singleLine'], linewidth=1)
        # arr[0, 1].set_xlim(self.xlim)
        # arr[0, 1].set_ylim([1e-14, 1e-6])
        arr[0, 1].set_ylabel('$| 1 - Det(M) |$')

        l3zoom = [abs(entry - 1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[1, 1].semilogy(self.T, l3zoom, c=self.plottingColors['doubleLine'][0], linewidth=1)
        arr[1, 1].semilogy(self.T, l4zoom, c=self.plottingColors['doubleLine'][1], linewidth=1, linestyle=':')
        # arr[1, 1].semilogy(self.xlim, [1e-3, 1e-3], '--', c=self.plottingColors['limit'], linewidth=1)
        # arr[1, 1].set_xlim(self.xlim)
        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1, 1].set_ylabel(' $|  | \lambda_i|-1  |  \\forall i=3,4$')
        arr[1, 1].set_xlabel('t [-]')

        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')
        plt.suptitle(
            '$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' - Eigensystem analysis STM  $(\\tau = 0 \in \mathcal{W}^{S+})$',
            size=self.suptitleSize)

        plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_stm_analysis.pdf',
                    transparent=True)
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

        arr[1, 0].scatter(self.T, np.angle(self.lambda1, deg=True), c=self.plottingColors['lambda1'], s=size)
        arr[1, 0].scatter(self.T, np.angle(self.lambda6, deg=True), c=self.plottingColors['lambda6'], s=size)
        # arr[1, 0].set_xlim(self.xlim)
        arr[1, 0].set_ylim([-180, 180])
        arr[1, 0].set_ylabel('Phase [$^\circ$]')

        arr[1, 1].scatter(self.T, np.angle(self.lambda2, deg=True), c=self.plottingColors['lambda2'], s=size)
        arr[1, 1].scatter(self.T, np.angle(self.lambda5, deg=True), c=self.plottingColors['lambda5'], s=size)
        # arr[1, 1].set_xlim(self.xlim)
        arr[1, 1].set_ylim([-180, 180])

        arr[1, 2].scatter(self.T, np.angle(self.lambda3, deg=True), c=self.plottingColors['lambda3'], s=size)
        arr[1, 2].scatter(self.T, np.angle(self.lambda4, deg=True), c=self.plottingColors['lambda4'], s=size)
        # arr[1, 2].set_xlim(self.xlim)
        arr[1, 2].set_ylim([-180, 180])

        arr[2, 0].semilogy(self.T, self.v1, c=self.plottingColors['lambda6'])
        arr[2, 0].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        # arr[2, 0].set_xlim(self.xlim)
        arr[2, 0].set_ylim([1e-1, 1e4])
        arr[2, 0].set_ylabel('Value index [-]')
        arr[2, 0].set_title('$v_1$')

        arr[2, 1].semilogy(self.T, self.v2, c=self.plottingColors['lambda5'])
        arr[2, 1].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        # arr[2, 1].set_xlim(self.xlim)
        arr[2, 1].set_ylim([1e-1, 1e1])
        arr[2, 1].set_title('$v_2$')
        arr[2, 1].set_xlabel('t [-]')

        arr[2, 2].semilogy(self.T, self.v3, c=self.plottingColors['lambda4'])
        arr[2, 2].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        # arr[2, 2].set_xlim(self.xlim)
        arr[2, 2].set_ylim([1e-1, 1e1])
        arr[2, 2].set_title('$v_3$')

        for i in range(3):
            for j in range(3):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('$L_' + str(
            self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' - Eigenvalues $\lambda_i$ \& stability index $v_i$ $(\\tau = 0 \in \mathcal{W}^{S+})$',
                     size=self.suptitleSize)
        plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_stability.pdf',
                    transparent=True)
        plt.close()
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

        arr[0].set_xlim([0, 1])
        arr[0].set_xlabel('$\\tau$ [-]')
        arr[0].set_ylim(ylim)
        arr[0].set_ylabel('$|C(\mathbf{X^i_0}) - C(\mathbf{X^p})|$ [-]')
        arr[0].set_title('Jacobi energy offset from orbit to manifold')

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

        arr[1].set_ylabel('$|x^i_n - (1-\mu)|, \; |y^i_n|$ [-]')  #  \; \\forall i =0, 1, \ldots m \in \mathcal{W}
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
            '$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Validation at C = ' + str(np.round(self.C, 3)),
            size=self.suptitleSize)

        plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_periodicity.pdf',
                    transparent=True)
        # plt.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_periodicity.png')
        plt.close()
        pass

    def plot_jacobi_validation(self):
        f, arr = plt.subplots(3, 2, figsize=self.figSize)

        highlight_alpha = 0.2
        ylim = [1e-16, 1e-9]
        t_min = 0
        step_size = 0.05

        arr[0, 0].semilogy(self.phase, self.C_diff_start_W_S_plus, c=self.plottingColors['W_S_plus'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S+}$')
        arr[0, 0].semilogy(self.phase, self.C_diff_start_W_S_min, c=self.plottingColors['W_S_min'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
        arr[0, 0].semilogy(self.phase, self.C_diff_start_W_U_plus, c=self.plottingColors['W_U_plus'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U+}$')
        arr[0, 0].semilogy(self.phase, self.C_diff_start_W_U_min, c=self.plottingColors['W_U_min'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')
        # arr[0, 0].legend(frameon=True, loc='upper right')

        arr[0, 0].set_xlim([0, 1])
        arr[0, 0].set_xlabel('$\\tau$ [-]')
        arr[0, 0].set_ylim(ylim)
        arr[0, 0].set_ylabel('$|C(\mathbf{X^i_0}) - C(\mathbf{X^p})|$ [-]')
        arr[0, 0].set_title('Jacobi energy deviation between orbit and manifold')

        # TODO decide whether to filter out trajectories intersecting Moon
        w_s_plus_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_S_plus_dx}).set_index('phase')
        w_s_plus_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_S_plus_dy}).set_index('phase')
        w_s_min_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_S_min_dx}).set_index('phase')
        w_s_min_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_S_min_dy}).set_index('phase')
        w_u_plus_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_U_plus_dx}).set_index('phase')
        w_u_plus_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_U_plus_dy}).set_index('phase')
        w_u_min_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_U_min_dx}).set_index('phase')
        w_u_min_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_U_min_dy}).set_index('phase')

        if self.lagrangePointNr == 1:
            arr[0, 1].semilogy(w_s_plus_dx[w_s_plus_dx['dx'] < 1e-10], c=self.plottingColors['W_S_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            arr[0, 1].semilogy(w_s_min_dy[w_s_min_dy['dy'] < 1e-10], c=self.plottingColors['W_S_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            arr[0, 1].semilogy(w_u_plus_dx[w_u_plus_dx['dx'] < 1e-10], c=self.plottingColors['W_U_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            arr[0, 1].semilogy(w_u_min_dy[w_u_min_dy['dy'] < 1e-10], c=self.plottingColors['W_U_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

            # arr[0, 1].semilogy(self.phase, self.W_S_plus_dx, c=self.plottingColors['W_S_plus'],
            #                label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            # arr[0, 1].semilogy(self.phase, self.W_S_min_dy, c=self.plottingColors['W_S_min'],
            #                label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            # arr[0, 1].semilogy(self.phase, self.W_U_plus_dx, c=self.plottingColors['W_U_plus'],
            #                label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            # arr[0, 1].semilogy(self.phase, self.W_U_min_dy, c=self.plottingColors['W_U_min'],
            #                label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')
        if self.lagrangePointNr == 2:
            arr[0, 1].semilogy(w_s_plus_dy[w_s_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_S_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            arr[0, 1].semilogy(w_s_min_dx[w_s_min_dx['dx'] < 1e-10], c=self.plottingColors['W_S_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            arr[0, 1].semilogy(w_u_plus_dy[w_u_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_U_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            arr[0, 1].semilogy(w_u_min_dx[w_u_min_dx['dx'] < 1e-10], c=self.plottingColors['W_U_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

        arr[0, 1].set_ylabel('$|x^i_{t_f} - (1-\mu)|, \; |y^i_{t_f}|$ [-]')  # \; \\forall i =0, 1, \ldots m \in \mathcal{W}
        arr[0, 1].legend(frameon=True, loc='center left',  bbox_to_anchor=(1, 0.5))
        arr[0, 1].set_xlim([0, 1])
        arr[0, 1].set_xlabel('$\\tau$ [-]')
        arr[0, 1].set_ylim(ylim)
        if self.lagrangePointNr == 1:
            arr[0, 1].set_title('Position deviation at $U_i \;  \\forall \; i = 1, 2, 3$')
        else:
            arr[0, 1].set_title('Position deviation at $U_i \;  \\forall \; i = 2, 3, 4$')

        w_s_plus_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_s_min_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_u_plus_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_u_min_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))

        for i in range(self.numberOfOrbitsPerManifold):
            w_s_plus_t = []
            w_s_min_t = []
            w_u_plus_t = []
            w_u_min_t = []
            w_s_plus_delta_j = []
            w_s_min_delta_j = []
            w_u_plus_delta_j = []
            w_u_min_delta_j = []
            w_s_plus_first_state = self.W_S_plus.xs(0).head(1).values[0]
            w_s_min_first_state = self.W_S_min.xs(0).head(1).values[0]
            w_u_plus_first_state = self.W_U_plus.xs(0).head(1).values[0]
            w_u_min_first_state = self.W_U_min.xs(0).head(1).values[0]
            w_s_plus_first_jacobi = computeJacobiEnergy(w_s_plus_first_state[0], w_s_plus_first_state[1],
                                                        w_s_plus_first_state[2], w_s_plus_first_state[3],
                                                        w_s_plus_first_state[4], w_s_plus_first_state[5])
            w_s_min_first_jacobi = computeJacobiEnergy(w_s_min_first_state[0], w_s_min_first_state[1],
                                                       w_s_min_first_state[2], w_s_min_first_state[3],
                                                       w_s_min_first_state[4], w_s_min_first_state[5])
            w_u_plus_first_jacobi = computeJacobiEnergy(w_u_plus_first_state[0], w_u_plus_first_state[1],
                                                        w_u_plus_first_state[2], w_u_plus_first_state[3],
                                                        w_u_plus_first_state[4], w_u_plus_first_state[5])
            w_u_min_first_jacobi = computeJacobiEnergy(w_u_min_first_state[0], w_u_min_first_state[1],
                                                       w_u_min_first_state[2], w_u_min_first_state[3],
                                                       w_u_min_first_state[4], w_u_min_first_state[5])
            for row in self.W_S_plus.xs(i).iterrows():
                w_s_plus_t.append(abs(row[0]))
                w_s_plus_state = row[1].values
                w_s_plus_jacobi = computeJacobiEnergy(w_s_plus_state[0], w_s_plus_state[1], w_s_plus_state[2],
                                                      w_s_plus_state[3], w_s_plus_state[4], w_s_plus_state[5])
                w_s_plus_delta_j.append(w_s_plus_jacobi - w_s_plus_first_jacobi)
            for row in self.W_S_min.xs(i).iterrows():
                w_s_min_t.append(abs(row[0]))
                w_s_min_state = row[1].values
                w_s_min_jacobi = computeJacobiEnergy(w_s_min_state[0], w_s_min_state[1], w_s_min_state[2],
                                                     w_s_min_state[3], w_s_min_state[4], w_s_min_state[5])
                w_s_min_delta_j.append(w_s_min_jacobi - w_s_min_first_jacobi)
            for row in self.W_U_plus.xs(i).iterrows():
                w_u_plus_t.append(abs(row[0]))
                w_u_plus_state = row[1].values
                w_u_plus_jacobi = computeJacobiEnergy(w_u_plus_state[0], w_u_plus_state[1], w_u_plus_state[2],
                                                      w_u_plus_state[3], w_u_plus_state[4], w_u_plus_state[5])
                w_u_plus_delta_j.append(w_u_plus_jacobi - w_u_plus_first_jacobi)
            for row in self.W_U_min.xs(i).iterrows():
                w_u_min_t.append(abs(row[0]))
                w_u_min_state = row[1].values
                w_u_min_jacobi = computeJacobiEnergy(w_u_min_state[0], w_u_min_state[1], w_u_min_state[2],
                                                     w_u_min_state[3], w_u_min_state[4], w_u_min_state[5])
                w_u_min_delta_j.append(w_u_min_jacobi - w_u_min_first_jacobi)

            w_s_plus_f = interp1d(w_s_plus_t, w_s_plus_delta_j)
            w_s_min_f = interp1d(w_s_min_t, w_s_min_delta_j)
            w_u_plus_f = interp1d(w_u_plus_t, w_u_plus_delta_j)
            w_u_min_f = interp1d(w_u_min_t, w_u_min_delta_j)

            w_s_plus_t_max = np.floor(max(w_s_plus_t) * 1/step_size) * step_size  # Round to nearest step-size
            w_s_min_t_max = np.floor(max(w_s_min_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_plus_t_max = np.floor(max(w_u_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_min_t_max = np.floor(max(w_u_min_t) * 1 / step_size) * step_size  # Round to nearest step-size

            w_s_plus_t_new = np.linspace(t_min, w_s_plus_t_max, np.round((w_s_plus_t_max - t_min) / step_size)+1)
            w_s_min_t_new = np.linspace(t_min, w_s_min_t_max, np.round((w_s_min_t_max - t_min) / step_size) + 1)
            w_u_plus_t_new = np.linspace(t_min, w_u_plus_t_max, np.round((w_u_plus_t_max - t_min) / step_size) + 1)
            w_u_min_t_new = np.linspace(t_min, w_u_min_t_max, np.round((w_u_min_t_max - t_min) / step_size) + 1)

            w_s_plus_df_temp = pd.DataFrame({i: w_s_plus_f(w_s_plus_t_new)}, index=w_s_plus_t_new)
            w_s_min_df_temp = pd.DataFrame({i: w_s_min_f(w_s_min_t_new)}, index=w_s_min_t_new)
            w_u_plus_df_temp = pd.DataFrame({i: w_u_plus_f(w_u_plus_t_new)}, index=w_u_plus_t_new)
            w_u_min_df_temp = pd.DataFrame({i: w_u_min_f(w_u_min_t_new)}, index=w_u_min_t_new)

            w_s_plus_df[i] = w_s_plus_df_temp[i]
            w_s_min_df[i] = w_s_min_df_temp[i]
            w_u_plus_df[i] = w_u_plus_df_temp[i]
            w_u_min_df[i] = w_u_min_df_temp[i]

            # Plot real data as check
            # arr[0, 0].plot(w_s_plus_t, w_s_plus_delta_j, 'o')
        w_s_plus_df = w_s_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_s_min_df = w_s_min_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_plus_df = w_u_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_min_df = w_u_min_df.dropna(axis=0, how='all').fillna(method='ffill')

        # Plot W^S+
        y1 = w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3*w_s_plus_df.std(axis=1)
        y2 = w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3*w_s_plus_df.std(axis=1)

        arr[1, 0].fill_between(w_s_plus_df.mean(axis=1).index,
                               y1=y1,
                               y2=y2, where=y1 >= y2,
                               facecolor=self.plottingColors['W_S_plus'], interpolate=True, alpha=highlight_alpha)
        l1, = arr[1, 0].plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3*w_s_plus_df.std(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{S+} \pm 3\sigma_t^{S+} $', color=self.plottingColors['W_S_plus'], linestyle=':')
        l2, = arr[1, 0].plot(w_s_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{S+}$', color=self.plottingColors['W_S_plus'])
        arr[1, 0].plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3*w_s_plus_df.std(axis=1).fillna(method='ffill'), color=self.plottingColors['W_S_plus'], linestyle=':')
        arr[1, 0].set_ylabel('$C(\mathbf{X^i_t}) - C(\mathbf{X^p})$ [-]')
        arr[1, 0].set_title('Energy deviation on manifold ($\\forall i \in \mathcal{W}^{S+}$)', loc='right')


        # Plot W^S-
        arr[1, 1].fill_between(w_s_min_df.mean(axis=1).index,
                               y1=w_s_min_df.mean(axis=1).fillna(method='ffill') + 3*w_s_min_df.std(axis=1),
                               y2=w_s_min_df.mean(axis=1).fillna(method='ffill') - 3*w_s_min_df.std(axis=1),
                               facecolor=self.plottingColors['W_S_min'], interpolate=True, alpha=highlight_alpha)
        l3, = arr[1, 1].plot(w_s_min_df.mean(axis=1).fillna(method='ffill') + 3*w_s_min_df.std(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{S-} \pm 3\sigma_t^{S-}$', color=self.plottingColors['W_S_min'], linestyle=':')
        l4, = arr[1, 1].plot(w_s_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{S-}$', color=self.plottingColors['W_S_min'])
        arr[1, 1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2, l3, l4])
        arr[1, 1].plot(w_s_min_df.mean(axis=1).fillna(method='ffill') - 3*w_s_min_df.std(axis=1).fillna(method='ffill'), color=self.plottingColors['W_S_min'], linestyle=':')
        arr[1, 1].set_ylabel('$C(\mathbf{X^i_t}) - C(\mathbf{X^p})$ [-]')

        arr[1, 1].set_title('Energy deviation on manifold ($\\forall i \in \mathcal{W}^{S-}$)', loc='right')


        # Plot W^U+
        arr[2, 0].fill_between(w_u_plus_df.mean(axis=1).index,
                               y1=w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3*w_u_plus_df.std(axis=1),
                               y2=w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3*w_u_plus_df.std(axis=1),
                               facecolor=self.plottingColors['W_U_plus'], interpolate=True, alpha=highlight_alpha)
        l5, = arr[2, 0].plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3*w_u_plus_df.std(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{U+} \pm 3\sigma_t^{U+}$', color=self.plottingColors['W_U_plus'], linestyle=':')
        l6, = arr[2, 0].plot(w_u_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{U+}$', color=self.plottingColors['W_U_plus'])
        arr[2, 0].plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3*w_u_plus_df.std(axis=1).fillna(method='ffill'), color=self.plottingColors['W_U_plus'], linestyle=':')
        arr[2, 0].set_ylabel('$C(\mathbf{X^i_t}) - C(\mathbf{X^p})$  [-]')
        arr[2, 0].set_title('Energy deviation on manifold ($\\forall i \in \mathcal{W}^{U+}$)', loc='right')


        # Plot W^U-
        arr[2, 1].fill_between(w_u_min_df.mean(axis=1).index,
                               y1=w_u_min_df.mean(axis=1).fillna(method='ffill') + 3*w_u_min_df.std(axis=1).fillna(method='ffill'),
                               y2=w_u_min_df.mean(axis=1).fillna(method='ffill') - 3*w_u_min_df.std(axis=1).fillna(method='ffill'),
                               facecolor=self.plottingColors['W_U_min'], interpolate=True, alpha=highlight_alpha)
        l7, = arr[2, 1].plot(w_u_min_df.mean(axis=1).fillna(method='ffill') + 3*w_u_min_df.std(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{U-} \pm 3\sigma_t^{U-}$', color=self.plottingColors['W_U_min'], linestyle=':')
        l8, = arr[2, 1].plot(w_u_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{C}_t^{U-}$', color=self.plottingColors['W_U_min'])
        arr[2, 1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l5, l6, l7, l8])
        arr[2, 1].plot(w_u_min_df.mean(axis=1).fillna(method='ffill') - 3*w_u_min_df.std(axis=1).fillna(method='ffill'), color=self.plottingColors['W_U_min'], linestyle=':')
        arr[2, 1].set_ylabel('$C(\mathbf{X^i_t}) - C(\mathbf{X^p})$  [-]')
        arr[2, 1].set_title('Energy deviation on manifold ($\\forall i \in \mathcal{W}^{U-}$)', loc='right')

        arr[2, 0].set_xlabel('$|t|$ [-]')
        arr[2, 1].set_xlabel('$|t|$  [-]')

        ylim = [min(arr[1, 0].get_ylim()[0], arr[1, 1].get_ylim()[0], arr[2, 0].get_ylim()[0], arr[2, 1].get_ylim()[0]),
                max(arr[1, 0].get_ylim()[1], arr[1, 1].get_ylim()[1], arr[2, 0].get_ylim()[1], arr[2, 1].get_ylim()[1])]

        for i in range(1, 3):
            for j in range(2):
                arr[i, j].set_ylim(ylim)

        for i in range(3):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        plt.suptitle(
            '$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Jacobi verification at C = ' + str(np.round(self.C, 3)),
            size=self.suptitleSize)
        # plt.show()
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_manifold_jacobi_validation.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_manifold_jacobi_validation.pdf',
                        transparent=True)
        # plt.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_periodicity.png')
        plt.close()
        pass

    def plot_orbit_offsets(self):
        f, arr = plt.subplots(3, 2, figsize=self.figSize, sharex=True)

        linewidth = 1
        scatter_size = 10
        highlight_alpha = 0.2
        ylim = [1e-16, 1e-9]
        legend_loc = 'lower left'
        for i in range(2):
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[0].values, label='x', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][0])
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[1].values, label='y', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][1])
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[2].values, label='z', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][2])
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[3].values, label='$\dot{x}$', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][0], linestyle='--')
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[4].values, label='$\dot{y}$', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][1], linestyle='--')
            arr[0, i].plot(self.phase, self.eigenvectorLocationDf_S[5].values, label='$\dot{z}$', linewidth=linewidth,
                               c=self.plottingColors['tripleLine'][2], linestyle='--')
        arr[0, 0].set_title('State on orbit')
        arr[0, 1].set_title('State on orbit')
        arr[0, 1].legend(frameon=True, bbox_to_anchor=(1.2, 0))

        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[0].values), label='x', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][0])
        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[1].values), label='y', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][1])
        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[2].values), label='z', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][2])
        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[3].values), label='$\dot{x}$', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][0], linestyle='--')
        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[4].values), label='$\dot{y}$', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][1], linestyle='--')
        arr[1, 0].plot(self.phase, abs(self.eigenvectorDf_S[5].values), label='$\dot{z}$', linewidth=linewidth,
                           c=self.plottingColors['tripleLine'][2], linestyle='--')
        # arr[1, 0].plot(self.phase, np.sqrt(self.eigenvectorDf_S[0].values**2 + self.eigenvectorDf_S[1].values**2 +
        #                                    self.eigenvectorDf_S[2].values**2), label='pos', linewidth=2, c='orange')
        # arr[1, 0].plot(self.phase, np.sqrt(self.eigenvectorDf_S[3].values ** 2 + self.eigenvectorDf_S[4].values ** 2 +
        #                                    self.eigenvectorDf_S[5].values ** 2), label='vel', linewidth=2, c='r')

        arr[1, 0].set_title('Absolute stable eigenvector offset from orbit')

        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[0].values), label='x', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][0])
        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[1].values), label='y', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][1])
        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[2].values), label='z', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][2])
        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[3].values), label='$\dot{x}$', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][0], linestyle='--')
        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[4].values), label='$\dot{y}$', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][1], linestyle='--')
        arr[1, 1].plot(self.phase, abs(self.eigenvectorDf_U[5].values), label='$\dot{z}$', linewidth=linewidth,
                       c=self.plottingColors['tripleLine'][2], linestyle='--')

        # arr[1, 1].plot(self.phase, np.sqrt(self.eigenvectorDf_U[0].values ** 2 + self.eigenvectorDf_U[1].values ** 2 +
        #                                    self.eigenvectorDf_U[2].values ** 2), label='pos', linewidth=2, c='orange')
        # arr[1, 1].plot(self.phase, np.sqrt(self.eigenvectorDf_U[3].values ** 2 + self.eigenvectorDf_U[4].values ** 2 +
        #                                    self.eigenvectorDf_U[5].values ** 2), label='vel', linewidth=2, c='r')
        arr[1, 1].set_title('Absolute unstable eigenvector offset from orbit')


        l1, = arr[2, 0].semilogy(self.phase, self.C_diff_start_W_S_plus, linewidth=linewidth, c=self.plottingColors['W_S_plus'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S+}$')
        l2, = arr[2, 0].semilogy(self.phase, self.C_diff_start_W_S_min, linewidth=linewidth, c=self.plottingColors['W_S_min'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
        l3, = arr[2, 1].semilogy(self.phase, self.C_diff_start_W_U_plus, linewidth=linewidth, c=self.plottingColors['W_U_plus'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U+}$')
        l4, = arr[2, 1].semilogy(self.phase, self.C_diff_start_W_U_min, linewidth=linewidth, c=self.plottingColors['W_U_min'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')
        arr[2, 1].legend(frameon=True, bbox_to_anchor=(1.4, 0.9), handles=[l1, l2, l3, l4])

        arr[2, 0].set_xlabel('$\\tau$ [-]')
        arr[2, 0].set_ylabel('$|C(\mathbf{X^i_0}) - C(\mathbf{X^p})|$ [-]')
        arr[2, 0].set_title('Jacobi energy deviation between orbit and manifold')
        arr[2, 1].set_title('Jacobi energy deviation between orbit and manifold')

        arr[0, 0].set_ylabel('x [-]')
        arr[1, 0].set_ylabel('x [-]')
        arr[0, 0].set_xlim([0, 1])
        arr[1, 0].set_ylim([0, 1])
        arr[1, 1].set_ylim([0, 1])
        for i in range(3):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        plt.suptitle(
            '$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Validation at C = ' + str(np.round(self.C, 3)),
            size=self.suptitleSize)
        plt.show()
        plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_orbit_offsets.pdf',
                    transparent=True)
        plt.close()
        pass


if __name__ == '__main__':
    low_dpi = False
    lagrange_points = [1, 2]
    orbit_types = ['horizontal', 'vertical', 'halo']
    c_levels = [3.05, 3.1, 3.15]

    # lagrange_points = [2]
    # orbit_types = ['vertical']
    c_levels = [3.15]

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                display_periodicity_validation = DisplayPeriodicityValidation(orbit_type, lagrange_point,
                                                                              orbit_ids[orbit_type][lagrange_point][c_level],
                                                                              low_dpi=low_dpi)
                # display_periodicity_validation.plot_manifolds()
                # display_periodicity_validation.plot_manifold_zoom()
                # display_periodicity_validation.plot_manifold_total()
                # display_periodicity_validation.plot_manifold_total_zoom()
                # display_periodicity_validation.plot_manifold_total_zoom_2()
                # display_periodicity_validation.plot_manifold_total_zoom_3()

                # display_periodicity_validation.plot_eigenvectors()
                # display_periodicity_validation.plot_stm_analysis()
                # display_periodicity_validation.plot_stability()

                # display_periodicity_validation.plot_periodicity_validation()
                display_periodicity_validation.plot_jacobi_validation()
                # display_periodicity_validation.plot_orbit_offsets()
                # plt.show()
                del display_periodicity_validation
