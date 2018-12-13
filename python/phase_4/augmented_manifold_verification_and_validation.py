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
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, \
    load_manifold_augmented, cr3bp_velocity, computeIntegralOfMotion, load_spacecraft_properties

class DisplayAugmentedValidation:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id, thrust_restriction, spacecraft_name, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        print('========================')
        print(str(orbit_type) + ' in L' + str(lagrange_point_nr))
        print('========================')
        self.orbitType = orbit_type
        self.orbitId = orbit_id
        self.orbitTypeForTitle = orbit_type.capitalize()
        self.thrustRestrictionForTitle = thrust_restriction.capitalize()
        self.spacecraftNameForTitle = spacecraft_name.capitalize()

        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.lagrangePointNr = lagrange_point_nr

        self.thrustRestriction = thrust_restriction
        self.spacecraftName = spacecraft_name

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
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (
                    MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.maxEigenvalueDeviation = 1.0e-3

        self.orbitDf = load_orbit(
            '../../data/raw/orbits/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
                orbit_id) + '.txt')
        self.C = computeJacobiEnergy(self.orbitDf.iloc[0]['x'], self.orbitDf.iloc[0]['y'],
                                     self.orbitDf.iloc[0]['z'], self.orbitDf.iloc[0]['xdot'],
                                     self.orbitDf.iloc[0]['ydot'], self.orbitDf.iloc[0]['zdot'])

        self.eigenvectorDf_S = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorDf_U = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_plus.txt')
        self.W_S_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_min.txt')
        self.W_U_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_plus.txt')
        self.W_U_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_min.txt')

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
        thrustMagnitude = load_spacecraft_properties(spacecraft_name)[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1],first_state_on_manifold[2], first_state_on_manifold[3],first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
        for row in self.W_S_plus.xs(0).iloc[::-1].iterrows():
             self.T_along_0_W_S_plus.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
             self.C_along_0_W_S_plus.append(abs(iom_on_manifold-first_iom_on_manifold))

        first_state_on_manifold = self.W_S_min.xs(0).tail(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1],first_state_on_manifold[2], first_state_on_manifold[3],first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
        for row in self.W_S_min.xs(0).iloc[::-1].iterrows():
              self.T_along_0_W_S_min.append(abs(row[0]))
              state_on_manifold = row[1].values
              iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
              self.C_along_0_W_S_min.append(abs(iom_on_manifold-first_iom_on_manifold))

        first_state_on_manifold = self.W_U_plus.xs(0).head(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
        for row in self.W_U_plus.xs(0).iterrows():
             self.T_along_0_W_U_plus.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
             self.C_along_0_W_U_plus.append(abs(iom_on_manifold - first_iom_on_manifold))

        first_state_on_manifold = self.W_U_min.xs(0).head(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
        for row in self.W_U_min.xs(0).iterrows():
             self.T_along_0_W_U_min.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
             self.C_along_0_W_U_min.append(abs(iom_on_manifold-first_iom_on_manifold))

             for i in range(self.numberOfOrbitsPerManifold):
                 self.phase.append(i / self.numberOfOrbitsPerManifold)

                 # On orbit
                 state_on_orbit = self.eigenvectorLocationDf_S.xs(i).values
                 iom_on_orbit = computeIntegralOfMotion(state_on_orbit[0], state_on_orbit[1], state_on_orbit[2],state_on_orbit[3], state_on_orbit[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)

                 # W_S_plus
                 state_on_manifold = self.W_S_plus.xs(i).tail(1).values[0]
                 iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
                 self.C_diff_start_W_S_plus.append(abs(iom_on_manifold - iom_on_orbit))
                 state_on_manifold = self.W_S_plus.xs(i).head(1).values[0]
                 # either very close to 1-mu or dy
                 if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                    self.W_S_plus_dx.append(0)
                    self.W_S_plus_dy.append(abs(state_on_manifold[1]))
                 else:
                    self.W_S_plus_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                    self.W_S_plus_dy.append(0)

                 # W_S_min
                 state_on_manifold = self.W_S_min.xs(i).tail(1).values[0]
                 iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
                 self.C_diff_start_W_S_min.append(abs(iom_on_manifold - iom_on_orbit))
                 state_on_manifold = self.W_S_min.xs(i).head(1).values[0]
                 # either very close to 1-mu or dy
                 if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                    self.W_S_min_dx.append(0)
                    self.W_S_min_dy.append(abs(state_on_manifold[1]))
                 else:
                    self.W_S_min_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                    self.W_S_min_dy.append(0)

                 # W_U_plus
                 state_on_manifold = self.W_U_plus.xs(i).head(1).values[0]
                 iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
                 self.C_diff_start_W_U_plus.append(abs(iom_on_manifold - iom_on_orbit))
                 state_on_manifold = self.W_U_plus.xs(i).tail(1).values[0]
                 # either very close to 1-mu or dy
                 if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                    self.W_U_plus_dx.append(0)
                    self.W_U_plus_dy.append(abs(state_on_manifold[1]))
                 else:
                    self.W_U_plus_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                    self.W_U_plus_dy.append(0)

                 # W_U_min
                 state_on_manifold = self.W_U_min.xs(i).head(1).values[0]
                 iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrustMagnitude, thrust_restriction)
                 self.C_diff_start_W_U_min.append(abs(iom_on_manifold - iom_on_orbit))
                 state_on_manifold = self.W_U_min.xs(i).tail(1).values[0]
                 # either very close to 1-mu or dy
                 if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                    self.W_U_min_dx.append(0)
                    self.W_U_min_dy.append(abs(state_on_manifold[1]))
                 else:
                    self.W_U_min_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                    self.W_U_min_dy.append(0)
                 pass

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
                                   'W_S_plus': self.colorPaletteStable[int(0.9 * self.numberOfOrbitsPerManifold)],
                                   'W_S_min': self.colorPaletteStable[int(0.4 * self.numberOfOrbitsPerManifold)],
                                   'W_U_plus': self.colorPaletteUnstable[int(0.9 * self.numberOfOrbitsPerManifold)],
                                   'W_U_min': self.colorPaletteUnstable[int(0.4 * self.numberOfOrbitsPerManifold)],
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

        print(self.W_S_plus.xs(1)['x'])
        print(pd.DataFrame(np.zeros((len(self.W_S_plus.xs(1)['x']), 1)))['0'])
        #Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax0.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_min.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
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

        if (thrust_restriction == "left" or "right"):
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ' + self.spacecraftNameForTitle + ' ' + self.thrustRestrictionForTitle + ' ' + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ' + self.spacecraftNameForTitle + ' ' + self.thrustRestrictionForTitle + ' ' + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at H$_{\text{lt}}$ = ' + str(np.round(self.C, 3)),size=self.suptitleSize)


        fig.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + ' ' + str(self.spacecraftName) + ' ' + str(self.thrustRestriction) + '_manifold_subplots.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

if __name__ == '__main__':
    #help()
    low_dpi = False
    lagrange_points = [1]
    orbit_types = ['horizontal']
    c_levels = [3.05]
    thrust_restrictions = ['left']
    spacecraft_names = ['DeepSpace']

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                for thrust_restriction in thrust_restrictions:
                    for spacecraft_name in spacecraft_names:
                        display_augmented_validation = DisplayAugmentedValidation(orbit_type, lagrange_point,
                                                                              orbit_ids[orbit_type][lagrange_point][
                                                                                  c_level], thrust_restriction, spacecraft_name,
                                                                              low_dpi=low_dpi)

                        display_augmented_validation.plot_manifolds()
                        del display_augmented_validation