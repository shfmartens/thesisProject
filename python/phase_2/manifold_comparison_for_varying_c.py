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
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored


class ManifoldComparisonForVaryingC:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id_per_c):
        print('=======================')
        print(str(orbit_type) + ' in L' + str(lagrange_point_nr))
        print('=======================')
        self.orbitType = orbit_type
        self.orbitIdPerC = orbit_id_per_c

        self.orbitTypeForTitle = orbit_type.capitalize()
        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.lagrangePointNr = lagrange_point_nr

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        initial_conditions_file_path = '../../data/raw/orbits/L' + str(lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

        self.C = []
        self.orbitDf = []
        self.W_S_plus = []
        self.W_S_min = []
        self.W_U_plus = []
        self.W_U_min = []

        for c_level in reversed(sorted(orbit_id_per_c)):
            orbit_id = orbit_id_per_c[c_level]
            # self.C.append(initial_conditions_incl_m_df.iloc[orbit_id][0])
            self.orbitDf.append(load_orbit(
                '../../data/raw/orbits/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '.txt'))

            self.C.append(computeJacobiEnergy(self.orbitDf[-1].iloc[0]['x'], self.orbitDf[-1].iloc[0]['y'],
                                              self.orbitDf[-1].iloc[0]['z'], self.orbitDf[-1].iloc[0]['xdot'],
                                              self.orbitDf[-1].iloc[0]['ydot'], self.orbitDf[-1].iloc[0]['zdot']))

            self.W_S_plus.append(load_manifold_refactored(
                '../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
                    orbit_id) + '_W_S_plus.txt'))
            self.W_S_min.append(load_manifold_refactored(
                '../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
                    orbit_id) + '_W_S_min.txt'))
            self.W_U_plus.append(load_manifold_refactored(
                '../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
                    orbit_id) + '_W_U_plus.txt'))
            self.W_U_min.append(load_manifold_refactored(
                '../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(
                    orbit_id) + '_W_U_min.txt'))

        self.numberOfOrbitsPerManifold = len(set(self.W_S_plus[0].index.get_level_values(0)))

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
                               'W_S_plus': self.colorPaletteStable[90],
                               'W_S_min': self.colorPaletteStable[40],
                               'W_U_plus': self.colorPaletteUnstable[90],
                               'W_U_min': self.colorPaletteUnstable[40],
                               'limit': 'black',
                               'orbit': 'navy'}
        self.suptitleSize = 20

        pass

    def plot_manifolds(self):
        # Plot: subplots
        if self.orbitType == 'horizontal':
           figsize = (self.figSize[0], self.figSize[1]/2)
           fig = plt.figure(figsize=figsize)
           ax00 = fig.add_subplot(2, 3, 1, projection='3d')
           ax01 = fig.add_subplot(2, 3, 2, projection='3d')
           ax02 = fig.add_subplot(2, 3, 3, projection='3d')
           ax10 = fig.add_subplot(2, 3, 4)
           ax11 = fig.add_subplot(2, 3, 5)
           ax12 = fig.add_subplot(2, 3, 6)
        else:
            figsize = self.figSize
            fig = plt.figure(figsize=figsize)
            ax00 = fig.add_subplot(4, 3, 1, projection='3d')
            ax01 = fig.add_subplot(4, 3, 2, projection='3d')
            ax02 = fig.add_subplot(4, 3, 3, projection='3d')
            ax10 = fig.add_subplot(4, 3, 4)
            ax11 = fig.add_subplot(4, 3, 5)
            ax12 = fig.add_subplot(4, 3, 6)
            ax20 = fig.add_subplot(4, 3, 7)
            ax21 = fig.add_subplot(4, 3, 8)
            ax22 = fig.add_subplot(4, 3, 9)
            ax30 = fig.add_subplot(4, 3, 10)
            ax31 = fig.add_subplot(4, 3, 11)
            ax32 = fig.add_subplot(4, 3, 12)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax00.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax01.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax02.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

            ax10.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         color='black', marker='x')
            ax11.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         color='black', marker='x')
            ax12.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         color='black', marker='x')
            if self.orbitType != 'horizontal':
                ax20.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')
                ax21.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')
                ax22.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')

                ax30.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')
                ax31.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')
                ax32.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                             color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax00.plot_surface(x, y, z, color='black')
            ax01.plot_surface(x, y, z, color='black')
            ax02.plot_surface(x, y, z, color='black')
            ax10.contourf(x, y, z, colors='black')
            ax11.contourf(x, y, z, colors='black')
            ax12.contourf(x, y, z, colors='black')
            if self.orbitType != 'horizontal':
                ax20.contourf(x, z, y, colors='black')
                ax21.contourf(x, z, y, colors='black')
                ax22.contourf(x, z, y, colors='black')
                ax30.contourf(y, z, x, colors='black')
                ax31.contourf(y, z, x, colors='black')
                ax32.contourf(y, z, x, colors='black')

        # Determine color for plot
        plot_alpha = 1
        line_width = 0.5
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax00.plot(self.W_S_plus[0].xs(manifold_orbit_number)['x'], self.W_S_plus[0].xs(manifold_orbit_number)['y'],
                      self.W_S_plus[0].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax01.plot(self.W_S_plus[1].xs(manifold_orbit_number)['x'], self.W_S_plus[1].xs(manifold_orbit_number)['y'],
                      self.W_S_plus[1].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax02.plot(self.W_S_plus[2].xs(manifold_orbit_number)['x'], self.W_S_plus[2].xs(manifold_orbit_number)['y'],
                      self.W_S_plus[2].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax10.plot(self.W_S_plus[0].xs(manifold_orbit_number)['x'], self.W_S_plus[0].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax11.plot(self.W_S_plus[1].xs(manifold_orbit_number)['x'], self.W_S_plus[1].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax12.plot(self.W_S_plus[2].xs(manifold_orbit_number)['x'], self.W_S_plus[2].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            if self.orbitType != 'horizontal':
                ax20.plot(self.W_S_plus[0].xs(manifold_orbit_number)['x'], self.W_S_plus[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax21.plot(self.W_S_plus[1].xs(manifold_orbit_number)['x'], self.W_S_plus[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax22.plot(self.W_S_plus[2].xs(manifold_orbit_number)['x'], self.W_S_plus[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

                ax30.plot(self.W_S_plus[0].xs(manifold_orbit_number)['y'], self.W_S_plus[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax31.plot(self.W_S_plus[1].xs(manifold_orbit_number)['y'], self.W_S_plus[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax32.plot(self.W_S_plus[2].xs(manifold_orbit_number)['y'], self.W_S_plus[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax00.plot(self.W_S_min[0].xs(manifold_orbit_number)['x'], self.W_S_min[0].xs(manifold_orbit_number)['y'],
                      self.W_S_min[0].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax01.plot(self.W_S_min[1].xs(manifold_orbit_number)['x'], self.W_S_min[1].xs(manifold_orbit_number)['y'],
                      self.W_S_min[1].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax02.plot(self.W_S_min[2].xs(manifold_orbit_number)['x'], self.W_S_min[2].xs(manifold_orbit_number)['y'],
                      self.W_S_min[2].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax10.plot(self.W_S_min[0].xs(manifold_orbit_number)['x'], self.W_S_min[0].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax11.plot(self.W_S_min[1].xs(manifold_orbit_number)['x'], self.W_S_min[1].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax12.plot(self.W_S_min[2].xs(manifold_orbit_number)['x'], self.W_S_min[2].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax20.plot(self.W_S_min[0].xs(manifold_orbit_number)['x'], self.W_S_min[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax21.plot(self.W_S_min[1].xs(manifold_orbit_number)['x'], self.W_S_min[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax22.plot(self.W_S_min[2].xs(manifold_orbit_number)['x'], self.W_S_min[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

                ax30.plot(self.W_S_min[0].xs(manifold_orbit_number)['y'], self.W_S_min[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax31.plot(self.W_S_min[1].xs(manifold_orbit_number)['y'], self.W_S_min[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax32.plot(self.W_S_min[2].xs(manifold_orbit_number)['y'], self.W_S_min[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax00.plot(self.W_U_plus[0].xs(manifold_orbit_number)['x'], self.W_U_plus[0].xs(manifold_orbit_number)['y'],
                      self.W_U_plus[0].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax01.plot(self.W_U_plus[1].xs(manifold_orbit_number)['x'], self.W_U_plus[1].xs(manifold_orbit_number)['y'],
                      self.W_U_plus[1].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax02.plot(self.W_U_plus[2].xs(manifold_orbit_number)['x'], self.W_U_plus[2].xs(manifold_orbit_number)['y'],
                      self.W_U_plus[2].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax10.plot(self.W_U_plus[0].xs(manifold_orbit_number)['x'], self.W_U_plus[0].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax11.plot(self.W_U_plus[1].xs(manifold_orbit_number)['x'], self.W_U_plus[1].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax12.plot(self.W_U_plus[2].xs(manifold_orbit_number)['x'], self.W_U_plus[2].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            if self.orbitType != 'horizontal':
                ax20.plot(self.W_U_plus[0].xs(manifold_orbit_number)['x'], self.W_U_plus[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax21.plot(self.W_U_plus[1].xs(manifold_orbit_number)['x'], self.W_U_plus[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax22.plot(self.W_U_plus[2].xs(manifold_orbit_number)['x'], self.W_U_plus[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

                ax30.plot(self.W_U_plus[0].xs(manifold_orbit_number)['y'], self.W_U_plus[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax31.plot(self.W_U_plus[1].xs(manifold_orbit_number)['y'], self.W_U_plus[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax32.plot(self.W_U_plus[2].xs(manifold_orbit_number)['y'], self.W_U_plus[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax00.plot(self.W_U_min[0].xs(manifold_orbit_number)['x'], self.W_U_min[0].xs(manifold_orbit_number)['y'],
                      self.W_U_min[0].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax01.plot(self.W_U_min[1].xs(manifold_orbit_number)['x'], self.W_U_min[1].xs(manifold_orbit_number)['y'],
                      self.W_U_min[1].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax02.plot(self.W_U_min[2].xs(manifold_orbit_number)['x'], self.W_U_min[2].xs(manifold_orbit_number)['y'],
                      self.W_U_min[2].xs(manifold_orbit_number)['z'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            ax10.plot(self.W_U_min[0].xs(manifold_orbit_number)['x'], self.W_U_min[0].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax11.plot(self.W_U_min[1].xs(manifold_orbit_number)['x'], self.W_U_min[1].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax12.plot(self.W_U_min[2].xs(manifold_orbit_number)['x'], self.W_U_min[2].xs(manifold_orbit_number)['y'],
                      color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

            if self.orbitType != 'horizontal':
                ax20.plot(self.W_U_min[0].xs(manifold_orbit_number)['x'], self.W_U_min[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax21.plot(self.W_U_min[1].xs(manifold_orbit_number)['x'], self.W_U_min[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax22.plot(self.W_U_min[2].xs(manifold_orbit_number)['x'], self.W_U_min[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

                ax30.plot(self.W_U_min[0].xs(manifold_orbit_number)['y'], self.W_U_min[0].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax31.plot(self.W_U_min[1].xs(manifold_orbit_number)['y'], self.W_U_min[1].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
                ax32.plot(self.W_U_min[2].xs(manifold_orbit_number)['y'], self.W_U_min[2].xs(manifold_orbit_number)['z'],
                          color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        plot_alpha = 1
        line_width = 2

        ax00.plot(self.orbitDf[0]['x'], self.orbitDf[0]['y'], self.orbitDf[0]['z'], color=self.plottingColors['orbit'],
                  alpha=plot_alpha, linewidth=line_width)
        ax01.plot(self.orbitDf[1]['x'], self.orbitDf[1]['y'], self.orbitDf[1]['z'], color=self.plottingColors['orbit'],
                  alpha=plot_alpha, linewidth=line_width)
        ax02.plot(self.orbitDf[2]['x'], self.orbitDf[2]['y'], self.orbitDf[2]['z'], color=self.plottingColors['orbit'],
                  alpha=plot_alpha, linewidth=line_width)

        ax10.plot(self.orbitDf[0]['x'], self.orbitDf[0]['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                  linewidth=line_width)
        ax11.plot(self.orbitDf[1]['x'], self.orbitDf[1]['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                  linewidth=line_width)
        ax12.plot(self.orbitDf[2]['x'], self.orbitDf[2]['y'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                  linewidth=line_width)
        if self.orbitType != 'horizontal':
            ax20.plot(self.orbitDf[0]['x'], self.orbitDf[0]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)
            ax21.plot(self.orbitDf[1]['x'], self.orbitDf[1]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)
            ax22.plot(self.orbitDf[2]['x'], self.orbitDf[2]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)

            ax30.plot(self.orbitDf[0]['y'], self.orbitDf[0]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)
            ax31.plot(self.orbitDf[1]['y'], self.orbitDf[1]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)
            ax32.plot(self.orbitDf[2]['y'], self.orbitDf[2]['z'], color=self.plottingColors['orbit'], alpha=plot_alpha,
                      linewidth=line_width)

        ax00.set_xlabel('x [-]')
        ax00.set_ylabel('y [-]')
        ax00.set_zlabel('z [-]')
        ax00.grid(True, which='both', ls=':')
        ax00.view_init(30, -120)
        ax00.set_title('C = ' + str(self.C[0]))
        ax01.set_xlabel('x [-]')
        ax01.set_ylabel('y [-]')
        ax01.set_zlabel('z [-]')
        ax01.grid(True, which='both', ls=':')
        ax01.view_init(30, -120)
        ax01.set_title('C = ' + str(self.C[1]))
        ax02.set_xlabel('x [-]')
        ax02.set_ylabel('y [-]')
        ax02.set_zlabel('z [-]')
        if self.orbitType == 'horizontal':
            ax00.set_zlim([-0.4, 0.4])
            ax01.set_zlim([-0.4, 0.4])
            ax02.set_zlim([-0.4, 0.4])
        ax02.grid(True, which='both', ls=':')
        ax02.view_init(30, -120)
        ax02.set_title('C = ' + str(self.C[2]))

        xlim = [min(ax00.get_xlim()[0], ax01.get_xlim()[0], ax02.get_xlim()[0]),
                max(ax00.get_xlim()[1], ax01.get_xlim()[1], ax02.get_xlim()[1])]
        ylim = [min(ax00.get_ylim()[0], ax01.get_ylim()[0], ax02.get_ylim()[0]),
                max(ax00.get_ylim()[1], ax01.get_ylim()[1], ax02.get_ylim()[1])]
        zlim = [min(ax00.get_zlim()[0], ax01.get_zlim()[0], ax02.get_zlim()[0]),
                max(ax00.get_zlim()[1], ax01.get_zlim()[1], ax02.get_zlim()[1])]
        ax00.set_xlim(xlim)
        ax01.set_xlim(xlim)
        ax02.set_xlim(xlim)
        ax00.set_ylim(ylim)
        ax01.set_ylim(ylim)
        ax02.set_ylim(ylim)
        ax00.set_zlim(zlim)
        ax01.set_zlim(zlim)
        ax02.set_zlim(zlim)

        ax11.set_xlabel('x [-]')
        ax10.set_ylabel('y [-]')
        xlim = [min(ax10.get_xlim()[0], ax11.get_xlim()[0], ax12.get_xlim()[0]),
                max(ax10.get_xlim()[1], ax11.get_xlim()[1], ax12.get_xlim()[1])]
        ylim = [min(ax10.get_ylim()[0], ax11.get_ylim()[0], ax12.get_ylim()[0]),
                max(ax10.get_ylim()[1], ax11.get_ylim()[1], ax12.get_ylim()[1])]
        ax10.set_xlim(xlim)
        ax11.set_xlim(xlim)
        ax12.set_xlim(xlim)
        ax10.set_ylim(ylim)
        ax11.set_ylim(ylim)
        ax12.set_ylim(ylim)
        ax10.grid(True, which='both', ls=':')
        ax11.grid(True, which='both', ls=':')
        ax12.grid(True, which='both', ls=':')

        if self.orbitType != 'horizontal':
            ax21.set_xlabel('x [-]')
            ax20.set_ylabel('z [-]')
            xlim = [min(ax20.get_xlim()[0], ax21.get_xlim()[0], ax22.get_xlim()[0]),
                    max(ax20.get_xlim()[1], ax21.get_xlim()[1], ax22.get_xlim()[1])]
            ylim = [min(ax20.get_ylim()[0], ax21.get_ylim()[0], ax22.get_ylim()[0]),
                    max(ax20.get_ylim()[1], ax21.get_ylim()[1], ax22.get_ylim()[1])]
            ax20.set_xlim(xlim)
            ax21.set_xlim(xlim)
            ax22.set_xlim(xlim)
            ax20.set_ylim(ylim)
            ax21.set_ylim(ylim)
            ax22.set_ylim(ylim)
            ax20.grid(True, which='both', ls=':')
            ax21.grid(True, which='both', ls=':')
            ax22.grid(True, which='both', ls=':')

            ax31.set_xlabel('y [-]')
            ax30.set_ylabel('z [-]')
            xlim = [min(ax30.get_xlim()[0], ax31.get_xlim()[0], ax32.get_xlim()[0]),
                    max(ax30.get_xlim()[1], ax31.get_xlim()[1], ax32.get_xlim()[1])]
            ylim = [min(ax30.get_ylim()[0], ax31.get_ylim()[0], ax32.get_ylim()[0]),
                    max(ax30.get_ylim()[1], ax31.get_ylim()[1], ax32.get_ylim()[1])]
            ax30.set_xlim(xlim)
            ax31.set_xlim(xlim)
            ax32.set_xlim(xlim)
            ax30.set_ylim(ylim)
            ax31.set_ylim(ylim)
            ax32.set_ylim(ylim)
            ax30.grid(True, which='both', ls=':')
            ax31.grid(True, which='both', ls=':')
            ax32.grid(True, which='both', ls=':')

        fig.tight_layout()
        if self.orbitType == 'horizontal':
            fig.subplots_adjust(top=0.8)
        else:
            fig.subplots_adjust(top=0.9)

        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle +
                     ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial comparison', size=self.suptitleSize)

        fig.savefig('../../data/figures/manifolds/refined_for_c/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_manifold_comparison.pdf',
                    transparent=True)
        # fig.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/0901/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_manifold_subplots.png')
        plt.close()
        pass


if __name__ == '__main__':
    lagrange_points = [1, 2]
    orbit_types = ['horizontal', 'vertical', 'halo']

    c_levels = [3.05, 3.1, 3.15]

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            manifold_comparison_for_varying_c = ManifoldComparisonForVaryingC(orbit_type, lagrange_point, orbit_ids[orbit_type][lagrange_point])
            manifold_comparison_for_varying_c.plot_manifolds()
            # plt.show()
