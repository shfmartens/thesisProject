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


class OverviewUnwindTime:
    def __init__(self, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.maxEigenvalueDeviation = 1.0e-3
        self.numberOfOrbitsPerManifold = 100

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)

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

    def plot_overview(self):

        lagrange_points = [1, 2]
        orbit_types = ['horizontal', 'halo', 'vertical']
        c_levels = [3.05, 3.1, 3.15]

        orbit_ids = {'horizontal': {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                     'halo': {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                     'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

        resultsplus = {'horizontal': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}},
                       'halo': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}},
                       'vertical': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}}}

        resultsmin = {'horizontal': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}},
                      'halo': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}},
                      'vertical': {1: {3.05: [], 3.1: [], 3.15: []}, 2: {3.05: [], 3.1: [], 3.15: []}}}

        linestyles = {1: '-', 2: '--'}

        fig, axarr = plt.subplots(1, 2, figsize=self.figSize, sharex=True, sharey=True)
        # fig, axarr = plt.subplots(1, 2, figsize=(self.figSize[0], self.figSize[1]/2), sharex=True, sharey=True)

        for lagrange_point in lagrange_points:
            for idx, orbit_type in enumerate(orbit_types):

                orbit_type_for_label = orbit_type.capitalize()
                if (orbit_type_for_label == 'Horizontal') or (orbit_type_for_label == 'Vertical'):
                    orbit_type_for_label += ' Lyapunov'

                label = '$L_' + str(lagrange_point) + '$ ' + orbit_type_for_label

                for c_level in c_levels:
                    df = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point)
                                                  + '_' + orbit_type + '_'
                                                  + str(orbit_ids[orbit_type][lagrange_point][c_level]) + '_W_U_plus.txt')
                    resultsplus[orbit_type][lagrange_point][c_level] = [df.xs(i).index.get_level_values(0)[-1] for i in range(100)]

                    df = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point)
                                                  + '_' + orbit_type + '_'
                                                  + str(orbit_ids[orbit_type][lagrange_point][c_level]) + '_W_U_min.txt')
                    resultsmin[orbit_type][lagrange_point][c_level] = [df.xs(i).index.get_level_values(0)[-1] for i in range(100)]

                plot_data_plus = [np.mean(resultsplus[orbit_type][lagrange_point][c_levels[0]]),
                                  np.mean(resultsplus[orbit_type][lagrange_point][c_levels[1]]),
                                  np.mean(resultsplus[orbit_type][lagrange_point][c_levels[2]])]

                plot_data_min = [np.mean(resultsmin[orbit_type][lagrange_point][c_levels[0]]),
                                 np.mean(resultsmin[orbit_type][lagrange_point][c_levels[1]]),
                                 np.mean(resultsmin[orbit_type][lagrange_point][c_levels[2]])]

                axarr[0].plot(c_levels, plot_data_plus, linestyle=linestyles[lagrange_point],
                             c=self.plottingColors['tripleLine'][idx])
                axarr[1].plot(c_levels, plot_data_min, label=label, linestyle=linestyles[lagrange_point],
                             c=self.plottingColors['tripleLine'][idx])

        axarr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5))

        for i in range(2):
            axarr[i].set_xlabel('$C$ [-]')
            axarr[i].set_ylabel('$|T|$ [-]')
            axarr[i].grid(True, which='both', ls=':')
        axarr[0].set_xlim([3.05, 3.15])
        axarr[0].set_title('Mean total integration period $\{\mathcal{W}^+\}$')
        axarr[1].set_title('Mean total integration period $\{\mathcal{W}^-\}$')
        fig.tight_layout()

        fig.subplots_adjust(top=0.8, right=0.8)

        fig.suptitle('Families overview - Orbital energy and time to unwind', size=self.suptitleSize)

        # plt.show()
        if self.lowDPI:
            plt.savefig('../../data/figures/manifolds/refined_for_c/overview_families_orbital_energy_unwind.png',
                        transparent=True, dpi=self.dpi)
        else:
            plt.savefig('../../data/figures/manifolds/refined_for_c/overview_families_orbital_energy_unwind.pdf',
                        transparent=True)

        pass


if __name__ == '__main__':
    low_dpi = True
    lagrange_points = [1, 2]
    orbit_types = ['horizontal', 'vertical', 'halo']
    c_levels = [3.05, 3.1, 3.15]

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    overview_unwind_time = OverviewUnwindTime(low_dpi=low_dpi)
    overview_unwind_time.plot_overview()
