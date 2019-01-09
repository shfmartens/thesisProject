import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
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
    load_manifold_augmented, cr3bp_velocity, computeIntegralOfMotion, load_spacecraft_properties, computeThrustAngle

class DisplayAugmentedValidation:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id, thrust_restriction, spacecraft_name, thrust_magnitude, low_dpi=False):
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
        self.thrustMagnitudeFloat = float(thrust_magnitude)
        self.thrustMagnitudeForPlotTitle = str(np.format_float_scientific(self.thrustMagnitudeFloat, unique=False, precision=1, exp_digits=1))
        self.thrustMagnitudeForTitle = str(thrust_magnitude)


        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.lagrangePointNr = lagrange_point_nr

        self.thrustRestriction = thrust_restriction
        self.thrustMagnitude = thrust_magnitude
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
            '../../data/raw/orbits/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '.txt')
        self.C = computeJacobiEnergy(self.orbitDf.iloc[0]['x'], self.orbitDf.iloc[0]['y'],
                                     self.orbitDf.iloc[0]['z'], self.orbitDf.iloc[0]['xdot'],
                                     self.orbitDf.iloc[0]['ydot'], self.orbitDf.iloc[0]['zdot'])

        self.eigenvectorDf_S = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorDf_U = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_S_plus.txt')
        self.W_S_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_S_min.txt')
        self.W_U_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_U_plus.txt')
        self.W_U_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_U_min.txt')

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
        self.W_S_plus_dm = []
        self.W_S_min_dx = []
        self.W_S_min_dy = []
        self.W_S_min_dm = []
        self.W_U_plus_dx = []
        self.W_U_plus_dy = []
        self.W_U_min_dx = []
        self.W_U_min_dy = []

        self.W_S_plus_alpha = []
        self.W_S_min_alpha = []
        self.W_U_plus_alpha = []
        self.W_U_min_alpha = []

        first_state_on_manifold = self.W_S_plus.xs(0).tail(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1],first_state_on_manifold[2], first_state_on_manifold[3],first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrust_magnitude, thrust_restriction)
        for row in self.W_S_plus.xs(0).iloc[::-1].iterrows():
             self.T_along_0_W_S_plus.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
             self.C_along_0_W_S_plus.append(abs(iom_on_manifold-first_iom_on_manifold))

        first_state_on_manifold = self.W_S_min.xs(0).tail(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrust_magnitude, thrust_restriction)
        for row in self.W_S_min.xs(0).iloc[::-1].iterrows():
              self.T_along_0_W_S_min.append(abs(row[0]))
              state_on_manifold = row[1].values
              iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
              self.C_along_0_W_S_min.append(abs(iom_on_manifold-first_iom_on_manifold))

        first_state_on_manifold = self.W_U_plus.xs(0).head(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrust_magnitude, thrust_restriction)
        for row in self.W_U_plus.xs(0).iterrows():
             self.T_along_0_W_U_plus.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
             self.C_along_0_W_U_plus.append(abs(iom_on_manifold - first_iom_on_manifold))

        first_state_on_manifold = self.W_U_min.xs(0).head(1).values[0]
        first_iom_on_manifold = computeIntegralOfMotion(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5], first_state_on_manifold[6], thrust_magnitude, thrust_restriction)
        for row in self.W_U_min.xs(0).iterrows():
             self.T_along_0_W_U_min.append(abs(row[0]))
             state_on_manifold = row[1].values
             iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
             self.C_along_0_W_U_min.append(abs(iom_on_manifold-first_iom_on_manifold))

        for i in range(self.numberOfOrbitsPerManifold):
            self.phase.append(i / self.numberOfOrbitsPerManifold)

            # On orbit
            state_on_orbit = self.eigenvectorLocationDf_S.xs(i).values
            iom_on_orbit = computeJacobiEnergy(state_on_orbit[0], state_on_orbit[1], state_on_orbit[2],
                                                       state_on_orbit[3], state_on_orbit[4], state_on_orbit[5])

            # W_S_plus
            state_on_manifold = self.W_S_plus.xs(i).tail(1).values[0]
            iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
            self.C_diff_start_W_S_plus.append(abs(iom_on_manifold - iom_on_orbit))
            state_on_manifold = self.W_S_plus.xs(i).head(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                self.W_S_plus_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                #self.W_S_plus_dx.append(0)
                self.W_S_plus_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_S_plus_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                self.W_S_plus_dy.append(abs(state_on_manifold[1]))
                #self.W_S_plus_dy.append(0)
            final_dm = abs(state_on_manifold[6]-1.0)
            self.W_S_plus_dm.append(final_dm)

            # W_S_min
            state_on_manifold = self.W_S_min.xs(i).tail(1).values[0]
            iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
            self.C_diff_start_W_S_min.append(abs(iom_on_manifold - iom_on_orbit))
            state_on_manifold = self.W_S_min.xs(i).head(1).values[0]
            # either very close to 1-mu or dy
            if abs(state_on_manifold[1]) < abs(state_on_manifold[0] - (1 - self.massParameter)):
                self.W_S_min_dx.append(0)
                self.W_S_min_dy.append(abs(state_on_manifold[1]))
            else:
                self.W_S_min_dx.append(abs(state_on_manifold[0] - (1 - self.massParameter)))
                self.W_S_min_dy.append(0)
            final_dm = abs(state_on_manifold[6] - 1.0)
            self.W_S_min_dm.append(final_dm)

            # W_U_plus
            state_on_manifold = self.W_U_plus.xs(i).head(1).values[0]
            iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
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
            iom_on_manifold = computeIntegralOfMotion(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2], state_on_manifold[3], state_on_manifold[4], state_on_manifold[5], state_on_manifold[6], thrust_magnitude, thrust_restriction)
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

    def plot_manifolds(self): #ADAPT self.orbitType if statements argument to horizontal to get rid of xz and yz projections
        # Plot: subplots
        if self.orbitType == 'horizontal2':
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
            if self.orbitType != 'horizontal2':
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

        #Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax0.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal2':
                ax1.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal2':
                ax1.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal2':
                ax1.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax0.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal2':
                ax1.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
        plot_alpha = 1
        line_width = 2
        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax3.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        if self.orbitType != 'horizontal2':
           ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
           ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)

        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.set_zlabel('z [-]')
        ax0.set_zlim([-0.4, 0.4])
        ax0.grid(True, which='both', ls=':')
        ax0.view_init(30, -120)

        if self.orbitType != 'horizontal2':
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
        #ax3.set_xlim([-1.0, 1.2])
        #ax3.set_ylim([-0.8, 0.8])
        ax3.grid(True, which='both', ls=':')



        fig.tight_layout()
        if self.orbitType != 'horizontal2':
            fig.subplots_adjust(top=0.9)
        else:
            fig.subplots_adjust(top=0.8)

        # Plot zero velocity surface
        x_range = np.arange(ax3.get_xlim()[0], ax3.get_xlim()[1], 0.001)
        y_range = np.arange(ax3.get_ylim()[0], ax3.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax3.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',alpha=0.5)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ '+ '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Orthographic projection at C = ' + str(np.round(self.C, 3)),size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ '+ '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' +  self.thrustMagnitudeForPlotTitle + '}$' + ' - Orthographic projection at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ '+ '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Orthographic projection at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)

        fig.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + str(self.spacecraftName) + '_' + self.thrustMagnitudeForTitle + '_' + str(self.thrustRestriction) + '_manifold_orthographic.pdf',
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
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lagrangePointNr == 2:
            if self.orbitType == 'horizontal':
                ax.set_xlim([1 - self.massParameter, 1.45])
                ax.set_ylim(-0.15, 0.15)
            else:
                ax.set_xlim([1 - self.massParameter, 1.25])
                ax.set_ylim(-0.05, 0.05)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L2']

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        # Plot zero velocity surface
        x_range = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 0.001)
        y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',alpha=0.5)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        # ax.annotate('\\textbf{Unstable exterior} $\\mathbf{ \mathcal{W}^{U+}}$',
        #             xy=(1.44, -0.1), xycoords='data',
        #             xytext=(-100, 50), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_U_plus'], linewidth=2))
        #
        # ax.annotate('\\textbf{Stable exterior} $\\mathbf{ \mathcal{W}^{S+}}$',
        #             xy=(1.37, 0.025), xycoords='data',
        #             xytext=(20, 50), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_S_plus'], linewidth=2))
        #
        # ax.annotate('\\textbf{Unstable interior} $\\mathbf{ \mathcal{W}^{U-}}$',
        #             xy=(1.01, 0.11), xycoords='data',
        #             xytext=(50, 10), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_U_min'], linewidth=2))
        # ax.annotate('\\textbf{Stable interior} $\\mathbf{ \mathcal{W}^{S-}}$',
        #             xy=(1.1, -0.11), xycoords='data',
        #             xytext=(-150, -10), textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->", color=self.plottingColors['W_S_min'], linewidth=2))

        # plt.show()
        fig.savefig('../../data/figures/manifolds/augmented/L' + str(
            self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + ' ' + str(self.spacecraftName) + ' ' + self.thrustMagnitudeForTitle + ' ' + str(self.thrustRestriction) + '_manifold_zoom.pdf',
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

        if self.lagrangePointNr ==1:
            W_S_plus_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_plus.txt')
            W_S_min_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_min.txt')
            W_U_plus_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_plus.txt')
            W_U_min_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_min.txt')
        else:
            W_S_plus_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_plus.txt')
            W_S_min_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_S_min.txt')
            W_U_plus_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_plus.txt')
            W_U_min_Lother = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + spacecraft_name + '_' + thrust_restriction + '_W_U_min.txt')

        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax.plot(W_S_plus_Lother.xs(manifold_orbit_number)['x'], W_S_plus_Lother.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_S_min_Lother.xs(manifold_orbit_number)['x'], W_S_min_Lother.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_plus_Lother.xs(manifold_orbit_number)['x'], W_U_plus_Lother.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(W_U_min_Lother.xs(manifold_orbit_number)['x'], W_U_min_Lother.xs(manifold_orbit_number)['y'],
                    color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2

        if self.lagrangePointNr == 1:
            orbitDf_Lother = load_orbit('../../data/raw/orbits/augmented/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitId) + '.txt')
        else:
            orbitDf_Lother = load_orbit('../../data/raw/orbits/augmented/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitId) + '.txt')

        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha, linewidth=line_width)
        ax.plot(orbitDf_Lother['x'], orbitDf_Lother['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
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

        plt.suptitle('$L_1, L_2$ ' + self.orbitTypeForTitle + ' ' + self.spacecraftNameForTitle + ' ' + str(self.ThrustMagnitudeForTitle) + self.thrustRestrictionForTitle + ' ' + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(np.round(self.C, 3)),
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
        fig.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + self.spacecraftName + '_' + str(self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_subplots_total.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass

    def plot_manifold_individual(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.gca()
        ax1 = fig.gca()
        ax2 = fig.gca()
        ax3 = fig.gca()
        ax4 = fig.gca()
        ax5 = fig.gca()
        ax6 = fig.gca()
        ax7 = fig.gca()
        ax8 = fig.gca()
        ax9 = fig.gca()
        ax10 = fig.gca()
        ax11 = fig.gca()


        if self.orbitType == 'horizontal':
            fig = plt.figure(figsize=self.figSize)
            ax0 = fig.add_subplot(2, 2, 1)
            ax3 = fig.add_subplot(2, 2, 2)
            ax6 = fig.add_subplot(2, 2, 3)
            ax9 = fig.add_subplot(2, 2, 4)
        else:
            fig = plt.figure(figsize=self.figSize)
            ax0 = fig.add_subplot(4, 3, 1)
            ax1 = fig.add_subplot(4, 3, 2)
            ax2 = fig.add_subplot(4, 3, 3)
            ax3 = fig.add_subplot(4, 3, 4)
            ax4 = fig.add_subplot(4, 3, 5)
            ax5 = fig.add_subplot(4, 3, 6)
            ax6 = fig.add_subplot(4, 3, 7)
            ax7 = fig.add_subplot(4, 3, 8)
            ax8 = fig.add_subplot(4, 3, 9)
            ax9 = fig.add_subplot(4, 3, 10)
            ax10 = fig.add_subplot(4, 3, 11)
            ax11 = fig.add_subplot(4, 3, 12)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax6.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax9.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            if self.orbitType != 'horizontal':
                ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax2.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax5.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax7.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax8.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax10.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')
                ax11.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax0.contour(x, y, z, color='black')
            ax3.contourf(x, y, z, colors='black')
            ax6.contourf(x, y, z, colors='black')
            ax9.contourf(x, y, z, colors='black')
            if self.orbitType != 'horizontal':
                ax1.contourf(x, z, y, colors='black')
                ax2.contourf(y, z, x, colors='black')
                ax4.contourf(x, z, y, colors='black')
                ax5.contourf(y, z, x, colors='black')
                ax7.contourf(x, z, y, colors='black')
                ax8.contourf(y, z, x, colors='black')
                ax10.contourf(x, z, y, colors='black')
                ax11.contourf(y, z, x, colors='black')

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax0.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'],
                     color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax3.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'],
                     color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax6.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'],
                     color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax9.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'],
                     color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax1.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(self.W_S_plus.xs(manifold_orbit_number)['y'], self.W_S_plus.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax4.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax5.plot(self.W_S_min.xs(manifold_orbit_number)['y'], self.W_S_min.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax7.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax8.plot(self.W_U_plus.xs(manifold_orbit_number)['y'], self.W_U_plus.xs(manifold_orbit_number)['z'],
                         color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax10.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha,
                          linewidth=line_width)
                ax11.plot(self.W_U_min.xs(manifold_orbit_number)['y'], self.W_U_min.xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha,
                          linewidth=line_width)

        plot_alpha = 1
        line_width = 2
        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                 linewidth=line_width)
        ax3.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                 linewidth=line_width)
        ax6.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                 linewidth=line_width)
        ax9.plot(self.orbitDf['x'], self.orbitDf['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                 linewidth=line_width)
        if self.orbitType != 'horizontal':
            ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax4.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax5.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax7.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax8.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                     linewidth=line_width)
            ax10.plot(self.orbitDf['x'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)
            ax11.plot(self.orbitDf['y'], self.orbitDf['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)

        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('z [-]')
        # ax1.set_ylim([-0.4, 0.4])
        ax0.grid(True, which='both', ls=':')
        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('z [-]')
        # ax1.set_ylim([-0.4, 0.4])
        ax3.grid(True, which='both', ls=':')
        ax6.set_xlabel('x [-]')
        ax6.set_ylabel('z [-]')
        # ax1.set_ylim([-0.4, 0.4])
        ax6.grid(True, which='both', ls=':')
        ax9.set_xlabel('x [-]')
        ax9.set_ylabel('z [-]')
        # ax1.set_ylim([-0.4, 0.4])
        ax9.grid(True, which='both', ls=':')

        if self.orbitType != 'horizontal':
            ax1.set_xlabel('x [-]')
            ax1.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax1.grid(True, which='both', ls=':')
            ax2.set_xlabel('x [-]')
            ax2.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax2.grid(True, which='both', ls=':')
            ax4.set_xlabel('x [-]')
            ax4.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax4.grid(True, which='both', ls=':')
            ax5.set_xlabel('x [-]')
            ax5.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax5.grid(True, which='both', ls=':')
            ax7.set_xlabel('x [-]')
            ax7.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax7.grid(True, which='both', ls=':')
            ax8.set_xlabel('x [-]')
            ax8.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax8.grid(True, which='both', ls=':')
            ax10.set_xlabel('x [-]')
            ax10.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax10.grid(True, which='both', ls=':')
            ax11.set_xlabel('x [-]')
            ax11.set_ylabel('z [-]')
            # ax1.set_ylim([-0.4, 0.4])
            ax11.grid(True, which='both', ls=':')


        # Plot zero velocity surface
        x_range = np.arange(ax0.get_xlim()[0], ax0.get_xlim()[1], 0.001)
        y_range = np.arange(ax0.get_ylim()[0], ax0.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax0.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',
                            alpha=0.5)

        x_range = np.arange(ax3.get_xlim()[0], ax3.get_xlim()[1], 0.001)
        y_range = np.arange(ax3.get_ylim()[0], ax3.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax3.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',
                         alpha=0.5)

        x_range = np.arange(ax6.get_xlim()[0], ax6.get_xlim()[1], 0.001)
        y_range = np.arange(ax6.get_ylim()[0], ax6.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax6.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',
                         alpha=0.5)

        x_range = np.arange(ax9.get_xlim()[0], ax9.get_xlim()[1], 0.001)
        y_range = np.arange(ax9.get_ylim()[0], ax9.get_ylim()[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.C)
        if z_mesh.min() < 0:
            ax9.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',
                         alpha=0.5)

        # Plot titles for subplot
        ax0.set_title('$\mathcal{W^{S+}}$')
        ax3.set_title('$\mathcal{W^{S-}}$')
        ax6.set_title('$\mathcal{W^{U+}}$')
        ax9.set_title('$\mathcal{W^{U-}}$')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        fig.savefig(
            '../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_individual.pdf',
            transparent=True)
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

        fig.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvectors.pdf',
                    transparent=True)
#        fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector_subplots.png')
        plt.close()
        pass

    def plot_iom_validation(self):
        fig = plt.figure(figsize=self.figSize)

        gs2 = gs.GridSpec(3,2)
        ax0 = fig.add_subplot(gs2[0, 0:2])
        ax1 = fig.add_subplot(gs2[1, 0])
        ax2 = fig.add_subplot(gs2[1, 1])
        ax3 = fig.add_subplot(gs2[2, 0])
        ax4 = fig.add_subplot(gs2[2, 1])




        highlight_alpha = 0.2
        ylim = [1e-16, 1e-9]
        t_min = 0
        step_size = 0.05

        ax0.semilogy(self.phase, self.C_diff_start_W_S_plus, c=self.plottingColors['W_S_plus'],
                           label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S+}$')
        ax0.semilogy(self.phase, self.C_diff_start_W_S_min, c=self.plottingColors['W_S_min'],
                        label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
        ax0.semilogy(self.phase, self.C_diff_start_W_U_plus, c=self.plottingColors['W_U_plus'],
                            label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U+}$')
        ax0.semilogy(self.phase, self.C_diff_start_W_U_min, c=self.plottingColors['W_U_min'],
                           label='$\mathbf{X}^i_0 \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

        ax0.set_xlim([0, 1])
        ax0.set_xlabel('$\\tau$ [-]')
        ax0.set_ylim(ylim)
        ax0.set_ylabel('$|IOM(\mathbf{X^i_0}) - IOM(\mathbf{X^p})|$ [-]')
        ax0.set_title('Jacobi energy deviation between orbit and manifold')
        ax0.legend(frameon=True, loc='center left',  bbox_to_anchor=(1, 0.5))

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
            w_s_plus_first_iom = computeIntegralOfMotion(w_s_plus_first_state[0], w_s_plus_first_state[1],
                                                        w_s_plus_first_state[2], w_s_plus_first_state[3],
                                                        w_s_plus_first_state[4], w_s_plus_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_s_min_first_iom = computeIntegralOfMotion(w_s_min_first_state[0], w_s_min_first_state[1],
                                                       w_s_min_first_state[2], w_s_min_first_state[3],
                                                       w_s_min_first_state[4], w_s_min_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_u_plus_first_iom = computeIntegralOfMotion(w_u_plus_first_state[0], w_u_plus_first_state[1],
                                                        w_u_plus_first_state[2], w_u_plus_first_state[3],
                                                        w_u_plus_first_state[4], w_u_plus_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_u_min_first_iom = computeIntegralOfMotion(w_u_min_first_state[0], w_u_min_first_state[1],
                                                       w_u_min_first_state[2], w_u_min_first_state[3],
                                                       w_u_min_first_state[4], w_u_min_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)

            for row in self.W_S_plus.xs(i).iterrows():
                w_s_plus_t.append(abs(row[0]))
                w_s_plus_state = row[1].values
                w_s_plus_iom = computeIntegralOfMotion(w_s_plus_state[0], w_s_plus_state[1], w_s_plus_state[2],
                                                      w_s_plus_state[3], w_s_plus_state[4], w_s_plus_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_s_plus_delta_j.append(w_s_plus_iom - w_s_plus_first_iom)
            for row in self.W_S_min.xs(i).iterrows():
                w_s_min_t.append(abs(row[0]))
                w_s_min_state = row[1].values
                w_s_min_iom = computeIntegralOfMotion(w_s_min_state[0], w_s_min_state[1], w_s_min_state[2],
                                                     w_s_min_state[3], w_s_min_state[4], w_s_min_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_s_min_delta_j.append(w_s_min_iom - w_s_min_first_iom)
            for row in self.W_U_plus.xs(i).iterrows():
                w_u_plus_t.append(abs(row[0]))
                w_u_plus_state = row[1].values
                w_u_plus_iom = computeIntegralOfMotion(w_u_plus_state[0], w_u_plus_state[1], w_u_plus_state[2],
                                                      w_u_plus_state[3], w_u_plus_state[4], w_u_plus_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_u_plus_delta_j.append(w_u_plus_iom - w_u_plus_first_iom)
            for row in self.W_U_min.xs(i).iterrows():
                w_u_min_t.append(abs(row[0]))
                w_u_min_state = row[1].values
                w_u_min_iom = computeIntegralOfMotion(w_u_min_state[0], w_u_min_state[1], w_u_min_state[2],
                                                     w_u_min_state[3], w_u_min_state[4], w_u_min_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_u_min_delta_j.append(w_u_min_iom - w_u_min_first_iom)

            w_s_plus_f = interp1d(w_s_plus_t, w_s_plus_delta_j)
            w_s_min_f = interp1d(w_s_min_t, w_s_min_delta_j)
            w_u_plus_f = interp1d(w_u_plus_t, w_u_plus_delta_j)
            w_u_min_f = interp1d(w_u_min_t, w_u_min_delta_j)

            w_s_plus_t_max = np.floor(max(w_s_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_s_min_t_max = np.floor(max(w_s_min_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_plus_t_max = np.floor(max(w_u_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_min_t_max = np.floor(max(w_u_min_t) * 1 / step_size) * step_size  # Round to nearest step-size

            w_s_plus_t_new = np.linspace(t_min, w_s_plus_t_max, np.round((w_s_plus_t_max - t_min) / step_size) + 1)
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
            #ax0.plot(w_s_plus_t, w_s_plus_delta_alpha, 'o')
        w_s_plus_df = w_s_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_s_min_df = w_s_min_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_plus_df = w_u_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_min_df = w_u_min_df.dropna(axis=0, how='all').fillna(method='ffill')

        # Plot W^S+
        y1 = w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1)
        y2 = w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1)

        ax1.fill_between(w_s_plus_df.mean(axis=1).index,y1=y1,y2=y2, where=y1 >= y2,facecolor=self.plottingColors['W_S_plus'], interpolate=True, alpha=highlight_alpha)
        l1, = ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\bar{IOM}_t^{S+} \pm 3\sigma_t^{S+} $', color=self.plottingColors['W_S_plus'], linestyle=':')
        l2, = ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{IOM}_t^{S+}$', color=self.plottingColors['W_S_plus'])
        ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_S_plus'], linestyle=':')
        ax1.set_ylabel('$IOM(\mathbf{X^i_t}) - IOM(\mathbf{X^p})$ [-]')
        ax1.set_title('IOM deviation on manifold ($\\forall i \in \mathcal{W}^{S+}$)', loc='right')

        # Plot W^S-
        ax2.fill_between(w_s_min_df.mean(axis=1).index,y1=w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1),y2=w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1),facecolor=self.plottingColors['W_S_min'], interpolate=True, alpha=highlight_alpha)
        l3, = ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\bar{IOM}_t^{S-} \pm 3\sigma_t^{S-}$', color=self.plottingColors['W_S_min'], linestyle=':')
        l4, = ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{IOM}_t^{S-}$',color=self.plottingColors['W_S_min'])
        ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2, l3, l4])
        ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_S_min'], linestyle=':')
        ax2.set_ylabel('$IOM(\mathbf{X^i_t}) - IOM(\mathbf{X^p})$ [-]')
        ax2.set_title('IOM deviation on manifold ($\\forall i \in \mathcal{W}^{S-}$)', loc='right')

        # Plot W^U+
        ax3.fill_between(w_u_plus_df.mean(axis=1).index,y1=w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1),y2=w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1),facecolor=self.plottingColors['W_U_plus'], interpolate=True, alpha=highlight_alpha)
        l5, = ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\bar{IOM}_t^{U+} \pm 3\sigma_t^{U+}$', color=self.plottingColors['W_U_plus'], linestyle=':')
        l6, = ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{IOM}_t^{U+}$',color=self.plottingColors['W_U_plus'])
        ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_U_plus'], linestyle=':')
        ax3.set_ylabel('$IOM(\mathbf{X^i_t}) - IOM(\mathbf{X^p})$  [-]')
        ax3.set_title('IOM deviation on manifold ($\\forall i \in \mathcal{W}^{U+}$)', loc='right')
        ax3.set_xlabel('$|t|$ [-]')

        # Plot W^U-
        ax4.fill_between(w_u_min_df.mean(axis=1).index,y1=w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),y2=w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna( method='ffill'),facecolor=self.plottingColors['W_U_min'], interpolate=True, alpha=highlight_alpha)
        l7, = ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\bar{IOM}_t^{U-} \pm 3\sigma_t^{U-}$', color=self.plottingColors['W_U_min'], linestyle=':')
        l8, = ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\bar{IOM}_t^{U-}$',color=self.plottingColors['W_U_min'])
        ax4.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l5, l6, l7, l8])
        ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_U_min'], linestyle=':')
        ax4.set_ylabel('$IOM(\mathbf{X^i_t}) - IOM(\mathbf{X^p})$  [-]')
        ax4.set_title('IOM deviation on manifold ($\\forall i \in \mathcal{W}^{U-}$)', loc='right')
        ax4.set_xlabel('$|t|$  [-]')

        ylim = [min(ax1.get_ylim()[0], ax3.get_ylim()[0], ax2.get_ylim()[0], ax4.get_ylim()[0]),
                max(ax1.get_ylim()[1], ax3.get_ylim()[1], ax2.get_ylim()[1], ax4.get_ylim()[1])]

        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)

        ax1.grid(True, which='both', ls=':')
        ax2.grid(True, which='both', ls=':')
        ax3.grid(True, which='both', ls=':')
        ax4.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - IOM verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - IOM verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - IOM verification at H$_{lt}$ = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_iom_validation.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_iom_validation.pdf',
                        transparent=True)
        plt.close()
        pass

    def plot_stopping_validation(self):
        fig = plt.figure(figsize=self.figSize)

        gs2 = gs.GridSpec(3, 1)
        ax0 = fig.add_subplot(gs2[0, 0])
        ax1 = fig.add_subplot(gs2[1, 0])
        ax2 = fig.add_subplot(gs2[2, 0])


        highlight_alpha = 0.2
        ylim = [1e-16, 1e-9]
        t_min = 0
        step_size = 0.05


        # TODO decide whether to filter out trajectories intersecting Moon
        w_s_plus_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_S_plus_dx}).set_index('phase')
        w_s_plus_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_S_plus_dy}).set_index('phase')
        w_s_plus_dm = pd.DataFrame({'phase': self.phase, 'dm': self.W_S_plus_dm}).set_index('phase')
        w_s_min_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_S_min_dx}).set_index('phase')
        w_s_min_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_S_min_dy}).set_index('phase')
        w_s_min_dm = pd.DataFrame({'phase': self.phase, 'dm': self.W_S_min_dm}).set_index('phase')
        w_u_plus_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_U_plus_dx}).set_index('phase')
        w_u_plus_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_U_plus_dy}).set_index('phase')
        w_u_min_dx = pd.DataFrame({'phase': self.phase, 'dx': self.W_U_min_dx}).set_index('phase')
        w_u_min_dy = pd.DataFrame({'phase': self.phase, 'dy': self.W_U_min_dy}).set_index('phase')

        if self.lagrangePointNr == 1:

            ax0.semilogy(w_s_plus_dx[w_s_plus_dx['dx'] < 1e-10], c=self.plottingColors['W_S_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax0.semilogy(w_s_min_dx[w_s_min_dx['dx'] < 1e-10], c=self.plottingColors['W_S_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            ax0.semilogy(w_u_plus_dx[w_u_plus_dx['dx']< 1e-10], c=self.plottingColors['W_U_plus'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            ax0.semilogy(w_u_min_dx[w_u_min_dx['dx'] < 1e-10], c=self.plottingColors['W_U_min'],
                               label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

            ax1.semilogy(w_s_plus_dy[w_s_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_S_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax1.semilogy(w_s_min_dy[w_s_min_dy['dy'] < 1e-10], c=self.plottingColors['W_S_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            ax1.semilogy(w_u_plus_dy[w_u_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_U_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            ax1.semilogy(w_u_min_dy[w_u_min_dy['dy'] < 1e-10], c=self.plottingColors['W_U_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

            ax2.semilogy(w_s_plus_dm[w_s_plus_dm['dm'] < 1e0], c=self.plottingColors['W_S_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax2.semilogy(w_s_min_dm[w_s_min_dm['dm'] < 1e-0], c=self.plottingColors['W_S_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')

        if self.lagrangePointNr == 2:

            ax0.semilogy(w_s_plus_dx[w_s_plus_dx['dx'] < 1e-10], c=self.plottingColors['W_S_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax0.semilogy(w_s_min_dy[w_s_min_dx['dx'] < 1e-10], c=self.plottingColors['W_S_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            ax0.semilogy(w_u_plus_dx[w_u_plus_dx['dx'] < 1e-10], c=self.plottingColors['W_U_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            ax0.semilogy(w_u_min_dy[w_u_min_dx['dx'] < 1e-10], c=self.plottingColors['W_U_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

            ax1.semilogy(w_s_plus_dx[w_s_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_S_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax1.semilogy(w_s_min_dy[w_s_min_dy['dy'] < 1e-10], c=self.plottingColors['W_S_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')
            ax1.semilogy(w_u_plus_dx[w_u_plus_dy['dy'] < 1e-10], c=self.plottingColors['W_U_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U+}$')
            ax1.semilogy(w_u_min_dy[w_u_min_dy['dy'] < 1e-10], c=self.plottingColors['W_U_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{U-}$', linestyle='--')

            ax2.semilogy(w_s_plus_dm[w_s_plus_dm['dm'] < 1e0], c=self.plottingColors['W_S_plus'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S+}$')
            ax2.semilogy(w_s_min_dm[w_s_min_dm['dm'] < 1e-0], c=self.plottingColors['W_S_min'],
                         label='$\mathbf{X}^i_n \; \\forall \; i \in \mathcal{W}^{S-}$', linestyle='--')

        ax0.set_ylabel('$|x^i_{t_f} - (1-\mu)|$ [-]')  # \; \\forall i =0, 1, \ldots m \in \mathcal{W}
        ax1.set_ylabel('$|y^i_{t_f}|$ [-]')  # \; \\forall i =0, 1, \ldots m \in \mathcal{W}
        ax2.set_ylabel('$|m^i_{t_f}-1|$ [-]')  # \; \\forall i =0, 1, \ldots m \in \mathcal{W}
        ax0.legend(frameon=True, loc='center left',  bbox_to_anchor=(1, 0.5))
        ax1.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
        # #ax0.invert_xaxis()
        ax0.set_xlim([0, 1])
        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax0.set_xlabel('$\\tau$ [-]')
        ax1.set_xlabel('$\\tau$ [-]')
        ax2.set_xlabel('$\\tau$ [-]')
        ax0.set_ylim(ylim)
        ax1.set_ylim(ylim)

        if self.lagrangePointNr == 1:
            ax0.set_title('Position deviation at $U_i \;  \\forall \; i = 2, 3$')
            ax1.set_title('Position deviation at $U_i \;  \\forall \; i = 1$')
            ax2.set_title('Mass deviation at $t_f \;  $')
        else:
            ax0.set_title('Position deviation at $U_i \;  \\forall \; i = 2, 3$')
            ax1.set_title('Position deviation at $U_i \;  \\forall \; i = 4$')
            ax2.set_title('Mass deviation at $t_f \;  $')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Stopping conditions verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Stopping conditions verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Stopping conditions verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_stopping_validation.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_stopping_validation.pdf',
                        transparent=True)
        plt.close()
        pass

    def plot_thrust_validation(self):
        fig = plt.figure(figsize=self.figSize)

        gs2 = gs.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs2[0, 0])
        ax2 = fig.add_subplot(gs2[0, 1])
        ax3 = fig.add_subplot(gs2[1, 0])
        ax4 = fig.add_subplot(gs2[1, 1])

        highlight_alpha = 0.2
        ylim = [1e-18, 1e-9]
        t_min = 0
        step_size = 0.05
        plt.close()

        w_s_plus_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_s_min_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_u_plus_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))
        w_u_min_df = pd.DataFrame(index=np.linspace(0, 100, 100 / 0.05 + 1))

        for i in range(self.numberOfOrbitsPerManifold):
            w_s_plus_t = []
            w_s_min_t = []
            w_u_plus_t = []
            w_u_min_t = []
            w_s_plus_delta_alpha = []
            w_s_min_delta_alpha = []
            w_u_plus_delta_alpha = []
            w_u_min_delta_alpha = []
            w_s_plus_first_state = self.W_S_plus.xs(0).head(1).values[0]
            w_s_min_first_state = self.W_S_min.xs(0).head(1).values[0]
            w_u_plus_first_state = self.W_U_plus.xs(0).head(1).values[0]
            w_u_min_first_state = self.W_U_min.xs(0).head(1).values[0]
            w_s_plus_first_alpha = computeThrustAngle(w_s_plus_first_state[3],
                                                         w_s_plus_first_state[4], w_s_plus_first_state[5],
                                                         w_s_plus_first_state[6], self.thrustMagnitude,
                                                         self.thrustRestriction)
            w_s_min_first_alpha = computeThrustAngle(w_s_min_first_state[3],
                                                         w_s_min_first_state[4], w_s_min_first_state[5],
                                                         w_s_min_first_state[6], self.thrustMagnitude,
                                                         self.thrustRestriction)
            w_u_plus_first_alpha = computeThrustAngle(w_u_plus_first_state[3],
                                                         w_u_plus_first_state[4], w_u_plus_first_state[5],
                                                         w_u_plus_first_state[6], self.thrustMagnitude,
                                                         self.thrustRestriction)
            w_u_min_first_alpha = computeThrustAngle(w_u_min_first_state[3],
                                                         w_u_min_first_state[4], w_u_min_first_state[5],
                                                         w_u_min_first_state[6], self.thrustMagnitude,
                                                         self.thrustRestriction)

            for row in self.W_S_plus.xs(i).iterrows():
                w_s_plus_t.append(abs(row[0]))
                w_s_plus_state = row[1].values
                w_s_plus_alpha = computeThrustAngle(w_s_plus_state[3],
                                                         w_s_plus_state[4], w_s_plus_state[5],
                                                         w_s_plus_state[6], self.thrustMagnitude,
                                                         self.thrustRestriction)
                w_s_plus_delta_alpha.append(w_s_plus_alpha - w_s_plus_first_alpha)
            for row in self.W_S_min.xs(i).iterrows():
                w_s_min_t.append(abs(row[0]))
                w_s_min_state = row[1].values
                w_s_min_alpha = computeThrustAngle(w_s_min_state[3],
                                                    w_s_min_state[4], w_s_min_state[5],
                                                    w_s_min_state[6], self.thrustMagnitude,
                                                    self.thrustRestriction)
                w_s_min_delta_alpha.append(w_s_min_alpha - w_s_min_first_alpha)
            for row in self.W_U_plus.xs(i).iterrows():
                w_u_plus_t.append(abs(row[0]))
                w_u_plus_state = row[1].values
                w_u_plus_alpha = computeThrustAngle(w_u_plus_state[3],
                                                    w_u_plus_state[4], w_u_plus_state[5],
                                                    w_u_plus_state[6], self.thrustMagnitude,
                                                    self.thrustRestriction)
                w_u_plus_delta_alpha.append(w_u_plus_alpha - w_u_plus_first_alpha)
            for row in self.W_U_min.xs(i).iterrows():
                w_u_min_t.append(abs(row[0]))
                w_u_min_state = row[1].values
                w_u_min_alpha = computeThrustAngle(w_u_min_state[3],
                                                   w_u_min_state[4], w_u_min_state[5],
                                                   w_u_min_state[6], self.thrustMagnitude,
                                                   self.thrustRestriction)
                w_u_min_delta_alpha.append(w_u_min_alpha - w_u_min_first_alpha)

            w_s_plus_f = interp1d(w_s_plus_t, w_s_plus_delta_alpha)
            w_s_min_f = interp1d(w_s_min_t, w_s_min_delta_alpha)
            w_u_plus_f = interp1d(w_u_plus_t, w_u_plus_delta_alpha)
            w_u_min_f = interp1d(w_u_min_t, w_u_min_delta_alpha)

            w_s_plus_t_max = np.floor(max(w_s_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_s_min_t_max = np.floor(max(w_s_min_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_plus_t_max = np.floor(max(w_u_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_min_t_max = np.floor(max(w_u_min_t) * 1 / step_size) * step_size  # Round to nearest step-size

            w_s_plus_t_new = np.linspace(t_min, w_s_plus_t_max, np.round((w_s_plus_t_max - t_min) / step_size) + 1)
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

        w_s_plus_df = w_s_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_s_min_df = w_s_min_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_plus_df = w_u_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_min_df = w_u_min_df.dropna(axis=0, how='all').fillna(method='ffill')

        # Plot W^S+
        y1 = w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1)
        y2 = w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1)

        ax1.fill_between(w_s_plus_df.mean(axis=1).index, y1=y1, y2=y2, where=y1 >= y2,
                         facecolor=self.plottingColors['W_S_plus'], interpolate=True, alpha=highlight_alpha)
        l1, = ax1.plot(
            w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),
            label='$\Delta \\alpha_t^{S+} \pm 3\sigma_t^{S+} $', color=self.plottingColors['W_S_plus'],
            linestyle=':')
        l2, = ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{S+}$',
                       color=self.plottingColors['W_S_plus'])
        ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),
                 color=self.plottingColors['W_S_plus'], linestyle=':')
        ax1.set_ylabel('$\\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$ [-]')
        ax1.set_title('Thrust pointing deviation on manifold ($\\forall i \in \mathcal{W}^{S+}$)', loc='right')

        # Plot W^S-
        ax2.fill_between(w_s_min_df.mean(axis=1).index,
                         y1=w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1),
                         y2=w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1),
                         facecolor=self.plottingColors['W_S_min'], interpolate=True, alpha=highlight_alpha)
        l3, = ax2.plot(
            w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),
            label='$\Delta \\alpha_t^{S-} \pm 3\sigma_t^{S-}$', color=self.plottingColors['W_S_min'], linestyle=':')
        l4, = ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{S-}$',
                       color=self.plottingColors['W_S_min'])
        ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2, l3, l4])
        ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),
                 color=self.plottingColors['W_S_min'], linestyle=':')
        ax2.set_ylabel('$\\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$ [-]')
        ax2.set_title('Thrust pointing deviation on manifold ($\\forall i \in \mathcal{W}^{S-}$)', loc='right')

        # Plot W^U+
        ax3.fill_between(w_u_plus_df.mean(axis=1).index,
                         y1=w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1),
                         y2=w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1),
                         facecolor=self.plottingColors['W_U_plus'], interpolate=True, alpha=highlight_alpha)
        l5, = ax3.plot(
            w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),
            label='$\Delta \\alpha_t^{U+} \pm 3\sigma_t^{U+}$', color=self.plottingColors['W_U_plus'], linestyle=':')
        l6, = ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{U+}$',
                       color=self.plottingColors['W_U_plus'])
        ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),
                 color=self.plottingColors['W_U_plus'], linestyle=':')
        ax3.set_ylabel('$\\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$  [-]')
        ax3.set_title('Thrust angle deviation on manifold ($\\forall i \in \mathcal{W}^{U+}$)', loc='right')
        ax3.set_xlabel('$|t|$ [-]')

        # Plot W^U-
        ax4.fill_between(w_u_min_df.mean(axis=1).index,
                         y1=w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(
                             method='ffill'),
                         y2=w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna(
                             method='ffill'), facecolor=self.plottingColors['W_U_min'], interpolate=True,
                         alpha=highlight_alpha)
        l7, = ax4.plot(
            w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),
            label='$\Delta \\alpha_t^{U-} \pm 3\sigma_t^{U-}$', color=self.plottingColors['W_U_min'], linestyle=':')
        l8, = ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{U-}$',
                       color=self.plottingColors['W_U_min'])
        ax4.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l5, l6, l7, l8])
        ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),
                 color=self.plottingColors['W_U_min'], linestyle=':')
        ax4.set_ylabel('$\\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$  [-]')
        ax4.set_title('Thrust angle deviation on manifold ($\\forall i \in \mathcal{W}^{U-}$)', loc='right')
        ax4.set_xlabel('$|t|$  [-]')

        ylim = [min(ax1.get_ylim()[0], ax3.get_ylim()[0], ax2.get_ylim()[0], ax4.get_ylim()[0]),
                max(ax1.get_ylim()[1], ax3.get_ylim()[1], ax2.get_ylim()[1], ax4.get_ylim()[1])]

        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)

        ax1.grid(True, which='both', ls=':')
        ax2.grid(True, which='both', ls=':')
        ax3.grid(True, which='both', ls=':')
        ax4.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing at H$_{lt}$ = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_pointing_validation.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_iom_validation.pdf',
                        transparent=True)
        plt.close()
        pass

    def plot_thrust_validation2(self):
        fig = plt.figure(figsize=self.figSize)

        gs2 = gs.GridSpec(2,2)
        ax1 = fig.add_subplot(gs2[0, 0])
        ax2 = fig.add_subplot(gs2[0, 1])
        ax3 = fig.add_subplot(gs2[1, 0])
        ax4 = fig.add_subplot(gs2[1, 1])




        highlight_alpha = 0.2
        ylim = [1e-18, 1e-9]
        t_min = 0
        step_size = 0.05

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
            w_s_plus_first_iom = computeThrustAngle(w_s_plus_first_state[3],
                                                        w_s_plus_first_state[4], w_s_plus_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_s_min_first_iom = computeThrustAngle(w_s_min_first_state[3],
                                                       w_s_min_first_state[4], w_s_min_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_u_plus_first_iom = computeThrustAngle(w_u_plus_first_state[3],
                                                        w_u_plus_first_state[4], w_u_plus_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)
            w_u_min_first_iom = computeThrustAngle(w_u_min_first_state[3],
                                                       w_u_min_first_state[4], w_u_min_first_state[5], w_s_plus_first_state[6], self.thrustMagnitude, self.thrustRestriction)

            for row in self.W_S_plus.xs(i).iterrows():
                w_s_plus_t.append(abs(row[0]))
                w_s_plus_state = row[1].values
                w_s_plus_iom = computeThrustAngle(w_s_plus_state[3], w_s_plus_state[4], w_s_plus_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_s_plus_delta_j.append(w_s_plus_iom - w_s_plus_first_iom)
            for row in self.W_S_min.xs(i).iterrows():
                w_s_min_t.append(abs(row[0]))
                w_s_min_state = row[1].values
                w_s_min_iom = computeThrustAngle(w_s_min_state[3], w_s_min_state[4], w_s_min_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_s_min_delta_j.append(w_s_min_iom - w_s_min_first_iom)
            for row in self.W_U_plus.xs(i).iterrows():
                w_u_plus_t.append(abs(row[0]))
                w_u_plus_state = row[1].values
                w_u_plus_iom = computeThrustAngle(w_u_plus_state[3], w_u_plus_state[4], w_u_plus_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_u_plus_delta_j.append(w_u_plus_iom - w_u_plus_first_iom)
            for row in self.W_U_min.xs(i).iterrows():
                w_u_min_t.append(abs(row[0]))
                w_u_min_state = row[1].values
                w_u_min_iom = computeThrustAngle(w_u_min_state[3], w_u_min_state[4], w_u_min_state[5], w_s_plus_state[6], self.thrustMagnitude, self.thrustRestriction)
                w_u_min_delta_j.append(w_u_min_iom - w_u_min_first_iom)

            w_s_plus_f = interp1d(w_s_plus_t, w_s_plus_delta_j)
            w_s_min_f = interp1d(w_s_min_t, w_s_min_delta_j)
            w_u_plus_f = interp1d(w_u_plus_t, w_u_plus_delta_j)
            w_u_min_f = interp1d(w_u_min_t, w_u_min_delta_j)

            w_s_plus_t_max = np.floor(max(w_s_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_s_min_t_max = np.floor(max(w_s_min_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_plus_t_max = np.floor(max(w_u_plus_t) * 1 / step_size) * step_size  # Round to nearest step-size
            w_u_min_t_max = np.floor(max(w_u_min_t) * 1 / step_size) * step_size  # Round to nearest step-size

            w_s_plus_t_new = np.linspace(t_min, w_s_plus_t_max, np.round((w_s_plus_t_max - t_min) / step_size) + 1)
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
            #ax0.plot(w_s_plus_t, w_s_plus_delta_alpha, 'o')
        w_s_plus_df = w_s_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_s_min_df = w_s_min_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_plus_df = w_u_plus_df.dropna(axis=0, how='all').fillna(method='ffill')
        w_u_min_df = w_u_min_df.dropna(axis=0, how='all').fillna(method='ffill')

        # Plot W^S+
        y1 = w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1)
        y2 = w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1)

        ax1.fill_between(w_s_plus_df.mean(axis=1).index,y1=y1,y2=y2, where=y1 >= y2,facecolor=self.plottingColors['W_S_plus'], interpolate=True, alpha=highlight_alpha)
        l1, = ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\alpha_t^{S+} \pm 3\sigma_t^{S+} $', color=self.plottingColors['W_S_plus'], linestyle=':')
        l2, = ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{S+}$', color=self.plottingColors['W_S_plus'])
        ax1.plot(w_s_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_plus_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_S_plus'], linestyle=':')
        ax1.set_ylabel('$ \\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$ [-]')
        ax1.set_title('$\\alpha $ deviation on manifold ($\\forall i \in \mathcal{W}^{S+}$)', loc='right')

        # Plot W^S-
        ax2.fill_between(w_s_min_df.mean(axis=1).index,y1=w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1),y2=w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1),facecolor=self.plottingColors['W_S_min'], interpolate=True, alpha=highlight_alpha)
        l3, = ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\alpha_t^{S-} \pm 3\sigma_t^{S-}$', color=self.plottingColors['W_S_min'], linestyle=':')
        l4, = ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{S-}$',color=self.plottingColors['W_S_min'])
        ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2, l3, l4])
        ax2.plot(w_s_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_s_min_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_S_min'], linestyle=':')
        ax2.set_ylabel('$ \\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$ [-]')
        ax2.set_title('$ \\alpha $ deviation on manifold ($\\forall i \in \mathcal{W}^{S-}$)', loc='right')

        # Plot W^U+
        ax3.fill_between(w_u_plus_df.mean(axis=1).index,y1=w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1),y2=w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1),facecolor=self.plottingColors['W_U_plus'], interpolate=True, alpha=highlight_alpha)
        l5, = ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\alpha_t^{U+} \pm 3\sigma_t^{U+}$', color=self.plottingColors['W_U_plus'], linestyle=':')
        l6, = ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{U+}$',color=self.plottingColors['W_U_plus'])
        ax3.plot(w_u_plus_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_plus_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_U_plus'], linestyle=':')
        ax3.set_ylabel('$\\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$  [-]')
        ax3.set_title('$\\alpha $ deviation on manifold ($\\forall i \in \mathcal{W}^{U+}$)', loc='right')
        ax3.set_xlabel('$|t|$ [-]')

        # Plot W^U-
        ax4.fill_between(w_u_min_df.mean(axis=1).index,y1=w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),y2=w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna( method='ffill'),facecolor=self.plottingColors['W_U_min'], interpolate=True, alpha=highlight_alpha)
        l7, = ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill') + 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),label='$\Delta \\alpha_t^{U-} \pm 3\sigma_t^{U-}$', color=self.plottingColors['W_U_min'], linestyle=':')
        l8, = ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill'), label='$\Delta \\alpha_t^{U-}$',color=self.plottingColors['W_U_min'])
        ax4.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l5, l6, l7, l8])
        ax4.plot(w_u_min_df.mean(axis=1).fillna(method='ffill') - 3 * w_u_min_df.std(axis=1).fillna(method='ffill'),color=self.plottingColors['W_U_min'], linestyle=':')
        ax4.set_ylabel('$ \\alpha (\mathbf{X^i_t}) - \\alpha (\mathbf{X^p})$  [-]')
        ax4.set_title('$\\alpha $ deviation on manifold ($\\forall i \in \mathcal{W}^{U-}$)', loc='right')
        ax4.set_xlabel('$|t|$  [-]')

        ylim = [min(ax1.get_ylim()[0], ax3.get_ylim()[0], ax2.get_ylim()[0], ax4.get_ylim()[0]),
                max(ax1.get_ylim()[1], ax3.get_ylim()[1], ax2.get_ylim()[1], ax4.get_ylim()[1])]

        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)

        ax1.grid(True, which='both', ls=':')
        ax2.grid(True, which='both', ls=':')
        ax3.grid(True, which='both', ls=':')
        ax4.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.85)

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing verification at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ const }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Thrust pointing verification at H$_{lt}$ = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)

        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_pointing_validation.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                self.orbitId) + '_' + self.spacecraftName + '_' + str(
                self.thrustMagnitudeForTitle) + '_' + self.thrustRestriction + '_manifold_pointing_validation.pdf',
                        transparent=True)
        plt.close()
        pass


if __name__ == '__main__':
    #help()
    low_dpi = True
    lagrange_points = [1]
    orbit_types = ['horizontal']
    c_levels = [3.05]
    thrust_restrictions = ['left']
    spacecraft_names = ['DeepSpace']
    thrust_magnitudes = ['0.000100']

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                for thrust_restriction in thrust_restrictions:
                    for spacecraft_name in spacecraft_names:
                        for thrust_magnitude in thrust_magnitudes:
                            display_augmented_validation = DisplayAugmentedValidation(orbit_type, lagrange_point,
                                                                              orbit_ids[orbit_type][lagrange_point][
                                                                                  c_level], thrust_restriction, spacecraft_name,
                                                                                  thrust_magnitude, low_dpi=low_dpi)
                            display_augmented_validation.plot_manifolds()
                            display_augmented_validation.plot_manifold_zoom()
                            display_augmented_validation.plot_manifold_individual()
                            display_augmented_validation.plot_eigenvectors()
                            display_augmented_validation.plot_iom_validation()
                            display_augmented_validation.plot_stopping_validation()
                            display_augmented_validation.plot_thrust_validation2()

                            del display_augmented_validation