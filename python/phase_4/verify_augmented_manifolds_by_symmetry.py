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
import time
import sys
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, \
    load_manifold_augmented, cr3bp_velocity, computeIntegralOfMotion, load_spacecraft_properties, computeThrustAngle, \
    computeMassRate


class VerifyManifoldsBySymmetry:
    def __init__(self, orbit_type, lagrange_point_nr, orbit_id, c_level,  thrust_restriction, spacecraft_name, thrust_magnitude, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        self.orbitType = orbit_type
        self.orbitId = orbit_id
        self.cLevel = c_level
        self.C = c_level
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

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.eigenvectorDf_S = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorDf_U = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_S_plus.txt')
        self.W_S_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_S_min.txt')
        self.W_U_plus = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_U_plus.txt')
        self.W_U_min = load_manifold_augmented('../../data/raw/manifolds/augmented/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_' + spacecraft_name + '_' + str(thrust_magnitude) + '_' + thrust_restriction + '_W_U_min.txt')

        self.orbitDf = load_orbit('../../data/raw/orbits/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_100.txt')

        self.numberOfOrbitsPerManifold = len(set(self.W_S_plus.index.get_level_values(0)))
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
        pass

    def plot_manifolds(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

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

            ax.contourf(x, y, z, colors='black')

        # Determine color for plot
        plot_alpha = 1
        line_width = 1
        for manifold_orbit_number in range(self.numberOfOrbitsPerManifold):
            ax.plot(self.W_S_plus.xs(manifold_orbit_number)['x'], self.W_S_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_S_min.xs(manifold_orbit_number)['x'], self.W_S_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteStable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_plus.xs(manifold_orbit_number)['x'], self.W_U_plus.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)
            ax.plot(self.W_U_min.xs(manifold_orbit_number)['x'], self.W_U_min.xs(manifold_orbit_number)['y'], color=self.colorPaletteUnstable[manifold_orbit_number], alpha=plot_alpha, linewidth=line_width)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')

        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview', size=self.suptitleSize)
        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \\alpha = ' + self.thrustRestriction + '}_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Spatial overview at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)

        if self.lowDPI:
            fig.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + ' ' + str(self.spacecraftName) + ' ' + self.thrustMagnitudeForTitle + ' ' + str(self.thrustRestriction) + '_manifold_zoom.png',transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + ' ' + str(self.spacecraftName) + ' ' + self.thrustMagnitudeForTitle + ' ' + str(self.thrustRestriction) + '_manifold_zoom.pdf',transparent=True)

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
        line_width = 1
        color = self.plottingColors['orbit']

        ax0.plot(self.orbitDf['x'], self.orbitDf['y'], color=color, linewidth=line_width)
        ax1.plot(self.orbitDf['x'], self.orbitDf['z'], color=color, linewidth=line_width)
        ax2.plot(self.orbitDf['y'], self.orbitDf['z'], color=color, linewidth=line_width)

        # Determine color for plot
        if self.orbitType == 'vertical':
            eigenvector_offset = 0.004
        else:
            eigenvector_offset = 0.02

        for idx in range(self.numberOfOrbitsPerManifold):
            # if self.numberOfOrbitsPerManifold > 10 and idx % 4 != 0:
            #     continue

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

            ax0.annotate("", xy=(x_S[0], y_S[0]), xytext=(x_S[1], y_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax1.annotate("", xy=(x_S[0], z_S[0]), xytext=(x_S[1], z_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax2.annotate("", xy=(y_S[0], z_S[0]), xytext=(y_S[1], z_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax0.annotate("", xy=(x_U[0], y_U[0]), xytext=(x_U[1], y_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            ax1.annotate("", xy=(x_U[0], z_U[0]), xytext=(x_U[1], z_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            ax2.annotate("", xy=(y_U[0], z_U[0]), xytext=(y_U[1], z_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))

            pass

        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        ax0.set_xlim(xlim[0] * 0.975, xlim[1] * 1.025)
        ax0.set_ylim(ylim[0] * 1.2, ylim[1] * 1.2)
        ax0.set_xlabel('x [-]')
        ax0.set_ylabel('y [-]')
        ax0.grid(True, which='both', ls=':')

        zlim = ax1.get_ylim()
        ax1.set_xlim(xlim[0] * 0.95, xlim[1] * 1.05)
        # ax1.set_ylim(zlim[0] * 1.1, zlim[1] * 1.1)
        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('z [-]')
        ax1.grid(True, which='both', ls=':')

        xlim = ax2.get_xlim()
        ax2.set_ylim(zlim[0] * 1.1, zlim[1] * 1.1)
        ax2.set_xlim(xlim[0] * 1.2, xlim[1] * 1.2)
        ax2.set_xlabel('y [-]')
        ax2.set_ylabel('z [-]')
        ax2.grid(True, which='both', ls=':')

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nr = 'L' + str(self.lagrangePointNr)
        ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')
        ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
        ax2.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.suptitle('$L_' + str(
            self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Orientation of (un)stable modes at C = ' + str(self.cLevel),
                     size=self.suptitleSize)
        # plt.show()
        # fig.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^S_i}{|\mathbf{v}^S_i|}, \mathbf{X_i} \pm \epsilon \\frac{\mathbf{v}^U_i}{|\mathbf{v}^U_i|} \}$ - Spatial overview', size=self.suptitleSize)
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector.pdf',
                        transparent=True)
        plt.close()
        pass

    def plot_eigenvectors_zoom(self):
        # Plot: subplots
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        # Determine color for plot
        plot_alpha = 1
        line_width = 2
        color = self.plottingColors['orbit']
        ax.plot(self.orbitDf['x'], self.orbitDf['y'], color=color, linewidth=line_width)

        # Determine color for plot
        if self.orbitType == 'vertical':
            eigenvector_offset = 0.004
        else:
            eigenvector_offset = 0.02

        for idx in range(self.numberOfOrbitsPerManifold):
            # if self.numberOfOrbitsPerManifold > 10 and idx % 2 != 0:
            #     continue

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

            ax.annotate("", xy=(x_S[0], y_S[0]), xytext=(x_S[1], y_S[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_S_plus'], shrinkA=0, shrinkB=0))
            ax.annotate("", xy=(x_U[0], y_U[0]), xytext=(x_U[1], y_U[1]), arrowprops=dict(arrowstyle='<->, head_width=1e-1, head_length=2e-1', color=self.plottingColors['W_U_plus'], shrinkA=0, shrinkB=0))
            pass

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] * 0.975, xlim[1] * 1.025)
        ax.set_ylim(ylim[0] * 1.2, ylim[1] * 1.2)
        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        fig.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + '  $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Orientation of (un)stable modes', size=self.suptitleSize)
        # plt.show()
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector_zoom.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_eigenvector_zoom.pdf',
                        transparent=True)
        plt.close()
        pass

    def show_phase_difference(self):

        if self.numberOfOrbitsPerManifold == 10:
            fig = plt.figure()
            ax = fig.gca()
            c = ['blue', 'red', 'green', 'yellow', 'pink', 'purple', 'grey', 'black', 'orange', 'magenta']
            ls = []
            for i in range(self.numberOfOrbitsPerManifold):
                ax.plot(self.W_S_plus.xs(i)['x'], self.W_S_plus.xs(i)['y'], color=c[i])
                ls.append(ax.plot(self.W_S_min.xs(i)['x'], self.W_S_min.xs(i)['y'], label='s'+str(i), color=c[i]))
                ax.plot(self.W_U_plus.xs(i)['x'], self.W_U_plus.xs(i)['y'], color=c[i])
                ls.append(ax.plot(self.W_U_min.xs(i)['x'], self.W_U_min.xs(i)['y'], label='u'+str(i), color=c[i]))
            ls = [a[0] for a in ls]
            ax.legend(handles=ls)
            plt.savefig(
                '../../data/figures/manifolds/refined_for_c/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    self.orbitId) + '_phase_difference_overview.pdf', transparent=True)
            plt.close()

        n = self.numberOfOrbitsPerManifold

        dt_min = []
        dx_min = []
        dy_min = []
        dz_min = []
        dxdot_min = []
        dydot_min = []
        dzdot_min = []

        dt_plus = []
        dx_plus = []
        dy_plus = []
        dz_plus = []
        dxdot_plus = []
        dydot_plus = []
        dzdot_plus = []

        jacobi_s_plus = []
        jacobi_s_min = []
        jacobi_u_plus = []
        jacobi_u_min = []

        for i in range(n):
            # Check whether the full trajectory has been integrated (not stopped due to exceeding Jacobi deviation)
            if self.lagrangePointNr == 2 and ((abs(self.W_S_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11) or (abs(self.W_U_min.xs((n - i) % n).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11)):
                dt_min.append(np.nan)
                dx_min.append(np.nan)
                dy_min.append(np.nan)
                dz_min.append(np.nan)
                dxdot_min.append(np.nan)
                dydot_min.append(np.nan)
                dzdot_min.append(np.nan)
                print('min')
                continue
            dt_min.append(self.W_S_min.xs(i).head(1).index.get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1).index.get_values()[0])
            dx_min.append(self.W_S_min.xs(i).head(1)['x'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['x'].get_values()[0])
            dy_min.append(self.W_S_min.xs(i).head(1)['y'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['y'].get_values()[0])
            dz_min.append(self.W_S_min.xs(i).head(1)['z'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['z'].get_values()[0])
            dxdot_min.append(self.W_S_min.xs(i).head(1)['xdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['xdot'].get_values()[0])
            dydot_min.append(self.W_S_min.xs(i).head(1)['ydot'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['ydot'].get_values()[0])
            dzdot_min.append(self.W_S_min.xs(i).head(1)['zdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['zdot'].get_values()[0])
            pass

        for i in range(n):
            # Check whether the full trajectory has been integrated (not stopped due to exceeding Jacobi deviation)
            if self.lagrangePointNr == 1 and ((
                    abs(self.W_S_plus.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11) or (
                    abs(self.W_U_plus.xs((n - i) % n).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11)):
                dt_plus.append(np.nan)
                dx_plus.append(np.nan)
                dy_plus.append(np.nan)
                dz_plus.append(np.nan)
                dxdot_plus.append(np.nan)
                dydot_plus.append(np.nan)
                dzdot_plus.append(np.nan)
                print('plus')
                continue
            dt_plus.append(self.W_S_plus.xs(i).head(1).index.get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1).index.get_values()[0])
            dx_plus.append(self.W_S_plus.xs(i).head(1)['x'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['x'].get_values()[0])
            dy_plus.append(self.W_S_plus.xs(i).head(1)['y'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['y'].get_values()[0])
            dz_plus.append(self.W_S_plus.xs(i).head(1)['z'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['z'].get_values()[0])
            dxdot_plus.append(self.W_S_plus.xs(i).head(1)['xdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['xdot'].get_values()[0])
            dydot_plus.append(self.W_S_plus.xs(i).head(1)['ydot'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['ydot'].get_values()[0])
            dzdot_plus.append(self.W_S_plus.xs(i).head(1)['zdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['zdot'].get_values()[0])
            pass
        for i in range(n):
            jacobi_s_plus.append(computeJacobiEnergy(self.W_S_plus.xs(i).head(1)['x'].get_values()[0],
                                                     self.W_S_plus.xs(i).head(1)['y'].get_values()[0],
                                                     self.W_S_plus.xs(i).head(1)['z'].get_values()[0],
                                                     self.W_S_plus.xs(i).head(1)['xdot'].get_values()[0],
                                                     self.W_S_plus.xs(i).head(1)['ydot'].get_values()[0],
                                                     self.W_S_plus.xs(i).head(1)['zdot'].get_values()[0]) - self.cLevel)
            jacobi_s_min.append(computeJacobiEnergy(self.W_S_min.xs(i).head(1)['x'].get_values()[0],
                                                    self.W_S_min.xs(i).head(1)['y'].get_values()[0],
                                                    self.W_S_min.xs(i).head(1)['z'].get_values()[0],
                                                    self.W_S_min.xs(i).head(1)['xdot'].get_values()[0],
                                                    self.W_S_min.xs(i).head(1)['ydot'].get_values()[0],
                                                    self.W_S_min.xs(i).head(1)['zdot'].get_values()[0]) - self.cLevel)
            jacobi_u_plus.append(computeJacobiEnergy(self.W_U_plus.xs((n - i) % n).tail(1)['x'].get_values()[0],
                                                     self.W_U_plus.xs((n - i) % n).tail(1)['y'].get_values()[0],
                                                     self.W_U_plus.xs((n - i) % n).tail(1)['z'].get_values()[0],
                                                     self.W_U_plus.xs((n - i) % n).tail(1)['xdot'].get_values()[0],
                                                     self.W_U_plus.xs((n - i) % n).tail(1)['ydot'].get_values()[0],
                                                     self.W_U_plus.xs((n - i) % n).tail(1)['zdot'].get_values()[0]) - self.cLevel)
            jacobi_u_min.append(computeJacobiEnergy(self.W_U_min.xs((n - i) % n).tail(1)['x'].get_values()[0],
                                                    self.W_U_min.xs((n - i) % n).tail(1)['y'].get_values()[0],
                                                    self.W_U_min.xs((n - i) % n).tail(1)['z'].get_values()[0],
                                                    self.W_U_min.xs((n - i) % n).tail(1)['xdot'].get_values()[0],
                                                    self.W_U_min.xs((n - i) % n).tail(1)['ydot'].get_values()[0],
                                                    self.W_U_min.xs((n - i) % n).tail(1)['zdot'].get_values()[0]) - self.cLevel)

        deviation_w_min = pd.DataFrame({'dt': dt_min, 'dx': dx_min, 'dy': dy_min, 'dz': dz_min, 'dxdot': dxdot_min, 'dydot': dydot_min, 'dzdot': dzdot_min})
        deviation_w_plus = pd.DataFrame({'dt': dt_plus, 'dx': dx_plus, 'dy': dy_plus, 'dz': dz_plus, 'dxdot': dxdot_plus, 'dydot': dydot_plus,'dzdot': dzdot_plus})

        fig2, axarr = plt.subplots(2, figsize=self.figSize, sharex=True)
        tau = [i/self.numberOfOrbitsPerManifold for i in range(self.numberOfOrbitsPerManifold)]
        l0, = axarr[0].plot(tau, deviation_w_min['dt'], c=self.plottingColors['limit'])
        l1, = axarr[0].plot(tau, deviation_w_min['dx'], c=self.plottingColors['tripleLine'][0])
        l2, = axarr[0].plot(tau, deviation_w_min['dy'], c=self.plottingColors['tripleLine'][1])
        l3, = axarr[0].plot(tau, deviation_w_min['dz'], c=self.plottingColors['tripleLine'][2])
        l4, = axarr[0].plot(tau, deviation_w_min['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        l5, = axarr[0].plot(tau, deviation_w_min['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        l6, = axarr[0].plot(tau, deviation_w_min['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        axarr[1].plot(tau, deviation_w_plus['dt'], c=self.plottingColors['limit'])
        axarr[1].plot(tau, deviation_w_plus['dx'], c=self.plottingColors['tripleLine'][0])
        axarr[1].plot(tau, deviation_w_plus['dy'], c=self.plottingColors['tripleLine'][1])
        axarr[1].plot(tau, deviation_w_plus['dz'], c=self.plottingColors['tripleLine'][2])
        axarr[1].plot(tau, deviation_w_plus['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        axarr[1].plot(tau, deviation_w_plus['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        axarr[1].plot(tau, deviation_w_plus['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        if self.lagrangePointNr == 1:
            axarr[0].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (at $U_1$)')
            axarr[1].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (at $U_2, U_3$)')
        else:
            axarr[0].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (at $U_2, U_3$)')
            axarr[1].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (at $U_4$)')

        # axarr[2].plot(tau, jacobi_s_plus, label='$\mathcal{W}^{S +}$')
        # axarr[2].plot(tau, jacobi_s_min, label='$\mathcal{W}^{S -}$')
        # axarr[2].plot(tau, jacobi_u_plus, label='$\mathcal{W}^{U +}$')
        # axarr[2].plot(tau, jacobi_u_min, label='$\mathcal{W}^{U -}$')
        # axarr[2].legend(frameon=True, bbox_to_anchor=(1, 1))

        text_min = '$\\bar{\mathbf{X}}^{S-} (t_{s0})= [$' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['x'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['y'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['z'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['xdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['ydot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1)['zdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '],  $ \\bar{t}_{s0} = $' + \
                   str(np.round(np.mean([self.W_S_min.xs(i).head(1).index.get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '\n$ \\bar{\mathbf{X}}^{U-} (t_{uf}) = [$' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['x'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['y'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['z'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['xdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['ydot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1)['zdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '],  $ \\bar{t}_{uf} = $' + \
                   str(np.round(np.mean([self.W_U_min.xs(i).tail(1).index.get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3))
        text_plus = '$\\bar{\mathbf{X}}^{S+} (t_{s0})= [$' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['x'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['y'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['z'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['xdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['ydot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1)['zdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '],  $ \\bar{t}_{s0} = $' + \
                    str(np.round(np.mean([self.W_S_plus.xs(i).head(1).index.get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '\n$ \\bar{\mathbf{X}}^{U+} (t_{uf})= [$' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['x'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['y'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['z'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['xdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['ydot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + ', ' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1)['zdot'].get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3)) + '],  $ \\bar{t}_{uf} = $' + \
                    str(np.round(np.mean([self.W_U_plus.xs(i).tail(1).index.get_values()[0] for i in range(self.numberOfOrbitsPerManifold)]), 3))

        axarr[0].text(0.5, 0.025, s=text_min, horizontalalignment='left', verticalalignment='bottom',
                      transform=axarr[0].transAxes, bbox=dict(boxstyle="round", fc="w", alpha=0.5))
        axarr[1].text(0.5, 0.025, s=text_plus, horizontalalignment='left', verticalalignment='bottom',
                      transform=axarr[1].transAxes, bbox=dict(boxstyle="round", fc="w", alpha=0.5))

        for i in range(2):
            axarr[i].grid(True, which='both', ls=':')
            axarr[i].set_xlabel('$\\tau_s$ [-]')
            axarr[i].set_xlim([0, 1])

        axarr[0].set_ylabel('$\\bar{\mathbf{X}}^{S-} (t_{s0}) - \\bar{\mathbf{X}}^{U-} (t_{uf}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')
        axarr[1].set_ylabel('$\\bar{\mathbf{X}}^{S+} (t_{s0}) - \\bar{\mathbf{X}}^{U+} (t_{uf}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')

        fig2.tight_layout()
        fig2.subplots_adjust(top=0.9, right=0.9)
        axarr[0].legend(handles=(l0, l1, l2, l3, l4, l5, l6),
                        labels=(
                        '$\delta t$', '$\delta x$', '$\delta y$', '$\delta z$', '$\delta \dot{x}$', '$\delta \dot{y}$',
                        '$\delta \dot{z}$'),
                        frameon=True, bbox_to_anchor=(1.1, 0.2))

        plt.suptitle('$L_' + str(
            self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Symmetry validation at C = ' + str(self.cLevel),
                     size=self.suptitleSize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_phase_difference.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/refined_for_c/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_phase_difference.pdf',
                        transparent=True)

        plt.close()
        pass

    def show_phase_difference_refactored(self):
        n = self.numberOfOrbitsPerManifold

        dt_min = []
        dx_min = []
        dy_min = []
        dz_min = []
        dxdot_min = []
        dydot_min = []
        dzdot_min = []

        dt_plus = []
        dx_plus = []
        dy_plus = []
        dz_plus = []
        dxdot_plus = []
        dydot_plus = []
        dzdot_plus = []

        dt_min_0 = []
        dx_min_0 = []
        dy_min_0 = []
        dz_min_0 = []
        dxdot_min_0 = []
        dydot_min_0 = []
        dzdot_min_0 = []

        dt_plus_0 = []
        dx_plus_0 = []
        dy_plus_0 = []
        dz_plus_0 = []
        dxdot_plus_0 = []
        dydot_plus_0 = []
        dzdot_plus_0 = []

        min_intersect = False
        plus_intersect = False

        for i in range(n):
            # Compute deviation at start of manifold
            dt_min_0.append(self.W_S_min.xs(i).tail(1).index.get_values()[0] + self.W_U_min.xs((n - i) % n).head(1).index.get_values()[0])
            dx_min_0.append(self.W_S_min.xs(i).tail(1)['x'].get_values()[0] - self.W_U_min.xs((n - i) % n).head(1)['x'].get_values()[0])
            dy_min_0.append(self.W_S_min.xs(i).tail(1)['y'].get_values()[0] + self.W_U_min.xs((n - i) % n).head(1)['y'].get_values()[0])
            dz_min_0.append(self.W_S_min.xs(i).tail(1)['z'].get_values()[0] - self.W_U_min.xs((n - i) % n).head(1)['z'].get_values()[0])
            dxdot_min_0.append(self.W_S_min.xs(i).tail(1)['xdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).head(1)['xdot'].get_values()[0])
            dydot_min_0.append(self.W_S_min.xs(i).tail(1)['ydot'].get_values()[0] - self.W_U_min.xs((n - i) % n).head(1)['ydot'].get_values()[0])
            dzdot_min_0.append(self.W_S_min.xs(i).tail(1)['zdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).head(1)['zdot'].get_values()[0])

            # Check whether the full trajectory has been integrated (not stopped due to exceeding Jacobi deviation)
            if self.lagrangePointNr == 2 and ((abs(self.W_S_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11) or (abs(self.W_U_min.xs((n - i) % n).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11)):
                min_intersect = True
                dt_min.append(np.nan)
                dx_min.append(np.nan)
                dy_min.append(np.nan)
                dz_min.append(np.nan)
                dxdot_min.append(np.nan)
                dydot_min.append(np.nan)
                dzdot_min.append(np.nan)
                print('min')
                continue
            dt_min.append(self.W_S_min.xs(i).head(1).index.get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1).index.get_values()[0])
            dx_min.append(self.W_S_min.xs(i).head(1)['x'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['x'].get_values()[0])
            dy_min.append(self.W_S_min.xs(i).head(1)['y'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['y'].get_values()[0])
            dz_min.append(self.W_S_min.xs(i).head(1)['z'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['z'].get_values()[0])
            dxdot_min.append(self.W_S_min.xs(i).head(1)['xdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['xdot'].get_values()[0])
            dydot_min.append(self.W_S_min.xs(i).head(1)['ydot'].get_values()[0] - self.W_U_min.xs((n - i) % n).tail(1)['ydot'].get_values()[0])
            dzdot_min.append(self.W_S_min.xs(i).head(1)['zdot'].get_values()[0] + self.W_U_min.xs((n - i) % n).tail(1)['zdot'].get_values()[0])
            pass

        for i in range(n):
            # Compute deviation at start of manifold
            dt_plus_0.append(self.W_S_plus.xs(i).tail(1).index.get_values()[0] + self.W_U_plus.xs((n - i) % n).head(1).index.get_values()[0])
            dx_plus_0.append(self.W_S_plus.xs(i).tail(1)['x'].get_values()[0] - self.W_U_plus.xs((n - i) % n).head(1)['x'].get_values()[0])
            dy_plus_0.append(self.W_S_plus.xs(i).tail(1)['y'].get_values()[0] + self.W_U_plus.xs((n - i) % n).head(1)['y'].get_values()[0])
            dz_plus_0.append(self.W_S_plus.xs(i).tail(1)['z'].get_values()[0] - self.W_U_plus.xs((n - i) % n).head(1)['z'].get_values()[0])
            dxdot_plus_0.append(self.W_S_plus.xs(i).tail(1)['xdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).head(1)['xdot'].get_values()[0])
            dydot_plus_0.append(self.W_S_plus.xs(i).tail(1)['ydot'].get_values()[0] - self.W_U_plus.xs((n - i) % n).head(1)['ydot'].get_values()[0])
            dzdot_plus_0.append(self.W_S_plus.xs(i).tail(1)['zdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).head(1)['zdot'].get_values()[0])

            # Check whether the full trajectory has been integrated (not stopped due to exceeding Jacobi deviation)
            if self.lagrangePointNr == 1 and ((abs(self.W_S_plus.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11) or (abs(self.W_U_plus.xs((n - i) % n).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1E-11)):
                plus_intersect = True
                dt_plus.append(np.nan)
                dx_plus.append(np.nan)
                dy_plus.append(np.nan)
                dz_plus.append(np.nan)
                dxdot_plus.append(np.nan)
                dydot_plus.append(np.nan)
                dzdot_plus.append(np.nan)
                print('plus')
                continue

            dt_plus.append(self.W_S_plus.xs(i).head(1).index.get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1).index.get_values()[0])
            dx_plus.append(self.W_S_plus.xs(i).head(1)['x'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['x'].get_values()[0])
            dy_plus.append(self.W_S_plus.xs(i).head(1)['y'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['y'].get_values()[0])
            dz_plus.append(self.W_S_plus.xs(i).head(1)['z'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['z'].get_values()[0])
            dxdot_plus.append(self.W_S_plus.xs(i).head(1)['xdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['xdot'].get_values()[0])
            dydot_plus.append(self.W_S_plus.xs(i).head(1)['ydot'].get_values()[0] - self.W_U_plus.xs((n - i) % n).tail(1)['ydot'].get_values()[0])
            dzdot_plus.append(self.W_S_plus.xs(i).head(1)['zdot'].get_values()[0] + self.W_U_plus.xs((n - i) % n).tail(1)['zdot'].get_values()[0])
            pass

        deviation_w_min = pd.DataFrame({'dt': dt_min, 'dx': dx_min, 'dy': dy_min, 'dz': dz_min, 'dxdot': dxdot_min, 'dydot': dydot_min, 'dzdot': dzdot_min})
        deviation_w_min_0 = pd.DataFrame({'dt': dt_min_0, 'dx': dx_min_0, 'dy': dy_min_0, 'dz': dz_min_0, 'dxdot': dxdot_min_0, 'dydot': dydot_min_0, 'dzdot': dzdot_min_0})
        deviation_w_plus = pd.DataFrame({'dt': dt_plus, 'dx': dx_plus, 'dy': dy_plus, 'dz': dz_plus, 'dxdot': dxdot_plus, 'dydot': dydot_plus,'dzdot': dzdot_plus})
        deviation_w_plus_0 = pd.DataFrame({'dt': dt_plus_0, 'dx': dx_plus_0, 'dy': dy_plus_0, 'dz': dz_plus_0, 'dxdot': dxdot_plus_0, 'dydot': dydot_plus_0, 'dzdot': dzdot_plus_0})

        fig2, axarr = plt.subplots(2, 2, figsize=self.figSize, sharex=True)
        tau = [i/self.numberOfOrbitsPerManifold for i in range(self.numberOfOrbitsPerManifold)]
        l0, = axarr[0, 1].plot(tau, deviation_w_min['dt'], c=self.plottingColors['limit'])
        l1, = axarr[0, 1].plot(tau, deviation_w_min['dx'], c=self.plottingColors['tripleLine'][0])
        l2, = axarr[0, 1].plot(tau, deviation_w_min['dy'], c=self.plottingColors['tripleLine'][1])
        l3, = axarr[0, 1].plot(tau, deviation_w_min['dz'], c=self.plottingColors['tripleLine'][2])
        l4, = axarr[0, 1].plot(tau, deviation_w_min['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        l5, = axarr[0, 1].plot(tau, deviation_w_min['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        l6, = axarr[0, 1].plot(tau, deviation_w_min['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        axarr[1, 1].plot(tau, deviation_w_min_0['dt'], c=self.plottingColors['limit'])
        axarr[1, 1].plot(tau, deviation_w_min_0['dx'], c=self.plottingColors['tripleLine'][0])
        axarr[1, 1].plot(tau, deviation_w_min_0['dy'], c=self.plottingColors['tripleLine'][1])
        axarr[1, 1].plot(tau, deviation_w_min_0['dz'], c=self.plottingColors['tripleLine'][2])
        axarr[1, 1].plot(tau, deviation_w_min_0['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        axarr[1, 1].plot(tau, deviation_w_min_0['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        axarr[1, 1].plot(tau, deviation_w_min_0['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        axarr[0, 0].plot(tau, deviation_w_plus['dt'], c=self.plottingColors['limit'])
        axarr[0, 0].plot(tau, deviation_w_plus['dx'], c=self.plottingColors['tripleLine'][0])
        axarr[0, 0].plot(tau, deviation_w_plus['dy'], c=self.plottingColors['tripleLine'][1])
        axarr[0, 0].plot(tau, deviation_w_plus['dz'], c=self.plottingColors['tripleLine'][2])
        axarr[0, 0].plot(tau, deviation_w_plus['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        axarr[0, 0].plot(tau, deviation_w_plus['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        axarr[0, 0].plot(tau, deviation_w_plus['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        axarr[1, 0].plot(tau, deviation_w_plus_0['dt'], c=self.plottingColors['limit'])
        axarr[1, 0].plot(tau, deviation_w_plus_0['dx'], c=self.plottingColors['tripleLine'][0])
        axarr[1, 0].plot(tau, deviation_w_plus_0['dy'], c=self.plottingColors['tripleLine'][1])
        axarr[1, 0].plot(tau, deviation_w_plus_0['dz'], c=self.plottingColors['tripleLine'][2])
        axarr[1, 0].plot(tau, deviation_w_plus_0['dxdot'], c=self.plottingColors['tripleLine'][0], linestyle=':')
        axarr[1, 0].plot(tau, deviation_w_plus_0['dydot'], c=self.plottingColors['tripleLine'][1], linestyle=':')
        axarr[1, 0].plot(tau, deviation_w_plus_0['dzdot'], c=self.plottingColors['tripleLine'][2], linestyle=':')

        if self.lagrangePointNr == 1:
            axarr[0, 1].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (at $U_1$)')
            axarr[0, 0].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (at $U_2, U_3$)')
            axarr[1, 1].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (near $L_1$)')
            axarr[1, 0].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (near $L_1$)')
        else:
            axarr[0, 1].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (at $U_2, U_3$)')
            axarr[0, 0].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (at $U_4$)')
            axarr[1, 1].set_title('$\mathcal{W}^{S -} \cap \mathcal{W}^{U -}$ (near $L_2$)')
            axarr[1, 0].set_title('$\mathcal{W}^{S +} \cap \mathcal{W}^{U +}$ (near $L_2$)')

        # ylim = [min(axarr[0, 0].get_ylim()[0], axarr[0, 1].get_ylim()[0], axarr[1, 0].get_ylim()[0], axarr[1, 1].get_ylim()[0]),
        #         max(axarr[0, 0].get_ylim()[1], axarr[0, 1].get_ylim()[1], axarr[1, 0].get_ylim()[1], axarr[1, 1].get_ylim()[1])]

        ylim = [min( axarr[1, 0].get_ylim()[0],axarr[1, 1].get_ylim()[0]),
                max(axarr[1, 0].get_ylim()[1],axarr[1, 1].get_ylim()[1])]

        axarr[0, 0].set_xlim([0, 1])
        y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=True)
        for i in range(2):
            axarr[1, i].set_xlabel('$\\tau_s$ [-]')
            for j in range(2):
                axarr[i, j].grid(True, which='both', ls=':')
                axarr[i, j].yaxis.set_major_formatter(y_formatter)
                axarr[i, j].set_ylim(ylim)
                # if min_intersect:
                #      axarr[i, j].set_ylim(axarr[0, 0].get_ylim()[0], axarr[0, 0].get_ylim()[1])
                # elif plus_intersect:
                #     axarr[i, j].set_ylim(axarr[0, 1].get_ylim()[0], axarr[0, 1].get_ylim()[1])
                # else:
                #     axarr[i, j].set_ylim(ylim)

        axarr[0, 1].set_ylabel('$\\bar{\mathbf{X}}^{S-} (t_{s0}) - \\bar{\mathbf{X}}^{U-} (t_{uf}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')
        axarr[0, 0].set_ylabel('$\\bar{\mathbf{X}}^{S+} (t_{s0}) - \\bar{\mathbf{X}}^{U+} (t_{uf}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')
        axarr[1, 1].set_ylabel('$\\bar{\mathbf{X}}^{S-} (t_{sf}) - \\bar{\mathbf{X}}^{U-} (t_{u0}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')
        axarr[1, 0].set_ylabel('$\\bar{\mathbf{X}}^{S+} (t_{sf}) - \\bar{\mathbf{X}}^{U+} (t_{u0}) \quad \\forall \quad \\tau_u = 1 - \\tau_s$ [-]')

        fig2.tight_layout()
        fig2.subplots_adjust(top=0.9, right=0.9)
        axarr[0, 1].legend(handles=(l0, l1, l2, l3, l4, l5, l6),
                           labels=(
                               '$\delta t$', '$\delta x$', '$\delta y$', '$\delta z$', '$\delta \dot{x}$', '$\delta \dot{y}$',
                               '$\delta \dot{z}$'),
                           frameon=True, loc='center left', bbox_to_anchor=(1, 0))

        # plot main title
        if self.thrustRestriction == 'left':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longleftarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Symmetry validation at C = ' + str(np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustRestriction == 'right':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \longrightarrow }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Symmetry validation at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        elif self.thrustMagnitude == '0.000000':
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{  }_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Symmetry validation at C = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        else:
            plt.suptitle('$L_' + str(
                self.lagrangePointNr) + '$ ' + '$\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}^{ \\alpha = ' + self.thrustRestriction + '}_{' + self.thrustMagnitudeForPlotTitle + '}$' + ' - Symmetry validation at H$_{lt}$ = ' + str(
                np.round(self.C, 3)), size=self.suptitleSize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.show()
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + str(self.spacecraftName) + '_' + self.thrustMagnitudeForTitle + '_' + str(self.thrustRestriction) + '_phase_difference.png',transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str(self.orbitId) + '_' + str(self.spacecraftName) + '_' + self.thrustMagnitudeForTitle + '_' + str(self.thrustRestriction) + '_phase_difference.pdf',transparent=True)

        plt.close()
        pass


if __name__ == '__main__':
    low_dpi = True
    lagrange_points = [1, 2]
    orbit_types = ['horizontal']
    c_levels = [3.05]
    thrust_restrictions = ['000.0','090.0','180.0','270.0']
    spacecraft_names = ['DeepSpace']
    thrust_magnitudes = ['0.010000']
    orbit_ids = {'horizontal': {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                for thrust_restriction in thrust_restrictions:
                    for spacecraft_name in spacecraft_names:
                        for thrust_magnitude in thrust_magnitudes:
                            verify_manifolds_by_symmetry = VerifyManifoldsBySymmetry(orbit_type, lagrange_point,
                                                                         orbit_ids[orbit_type][lagrange_point][c_level],
                                                                         c_level,thrust_restriction, spacecraft_name,
                                                                                  thrust_magnitude, low_dpi=low_dpi)
                            #verify_manifolds_by_symmetry.plot_manifolds()
                            #verify_manifolds_by_symmetry.plot_eigenvectors()
                            #verify_manifolds_by_symmetry.plot_eigenvectors_zoom()
                            #verify_manifolds_by_symmetry.show_phase_difference()
                            verify_manifolds_by_symmetry.show_phase_difference_refactored()
                            # plt.show()
                            del verify_manifolds_by_symmetry