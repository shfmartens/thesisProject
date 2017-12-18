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
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
import time
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored


class HorizontalPhasePortrait:
    def __init__(self, orbit_ids):
        self.orbitIds = orbit_ids
        self.numberOfOrbitsPerManifold = 100

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.suptitleSize = 20
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
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[int((n_colors-1)/2)], sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}

        pass

    def plot_portrait_L1_to_L2(self):
        fig, axarr = plt.subplots(1, 2, figsize=(self.figSize[0], self.figSize[1]/2), sharex=True, sharey=True)

        for idx, c_level in enumerate([3.1, 3.15]):
            w_u_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L1_horizontal_' +
                                                str(orbit_ids['horizontal'][1][c_level]) + '_W_U_plus.txt')
            w_s_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L2_horizontal_' +
                                               str(orbit_ids['horizontal'][2][c_level]) + '_W_S_min.txt')

            ls_u = []
            ls_s = []
            for i in range(self.numberOfOrbitsPerManifold):
                if abs(w_u_plus.xs(i).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_u.append(w_u_plus.xs(i).tail(1))
                if abs(w_s_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_s.append(w_s_min.xs(i).head(1))

            # Add start as endpoint for loop plot
            if abs(w_u_plus.xs(0).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_u.append(w_u_plus.xs(0).tail(1))
            if abs(w_s_min.xs(0).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_s.append(w_s_min.xs(0).head(1))

            w_u_plus_poincare = pd.concat(ls_u)
            w_s_min_poincare = pd.concat(ls_s)

            l1, = axarr[idx].plot(w_u_plus_poincare['y'], w_u_plus_poincare['ydot'], c='r', label='$\mathcal{W}^{U+}$')
            l2, = axarr[idx].plot(w_s_min_poincare['y'], w_s_min_poincare['ydot'], c='g', label='$\mathcal{W}^{S-}$')
            axarr[idx].set_title('$C$ = ' + str(c_level))

        axarr[0].set_ylim([-1, 1])
        axarr[1].set_xlim([-0.175, -0.0025])

        axarr[0].set_xlabel('$y$ [-]')
        axarr[1].set_xlabel('$y$ [-]')
        axarr[0].set_ylabel('$\dot{y}$ [-]')

        axarr[0].grid(True, which='both', ls=':')
        axarr[1].grid(True, which='both', ls=':')

        axarr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2])
        fig.tight_layout()
        fig.subplots_adjust(top=0.8, right=0.9)
        plt.suptitle('Planar Poincaré phase portrait at $\mathbf{U}_2$', size=self.suptitleSize)
        # plt.show()
        plt.savefig('../../data/figures/poincare_sections/planar_poincare_phase_portrait_l1_to_l2.pdf',
                    transparent=True)
        plt.close()
        pass

    def plot_portrait_L2_to_L1(self):
        fig, axarr = plt.subplots(1, 2, figsize=(self.figSize[0], self.figSize[1]/2), sharex=True, sharey=True)

        for idx, c_level in enumerate([3.1, 3.15]):
            w_u_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L2_horizontal_' +
                                                str(orbit_ids['horizontal'][2][c_level]) + '_W_U_min.txt')
            w_s_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L1_horizontal_' +
                                               str(orbit_ids['horizontal'][1][c_level]) + '_W_S_plus.txt')

            ls_u = []
            ls_s = []
            for i in range(self.numberOfOrbitsPerManifold):
                if abs(w_u_plus.xs(i).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_u.append(w_u_plus.xs(i).tail(1))
                if abs(w_s_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_s.append(w_s_min.xs(i).head(1))

            # Add start as endpoint for loop plot
            if abs(w_u_plus.xs(0).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_u.append(w_u_plus.xs(0).tail(1))
            if abs(w_s_min.xs(0).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_s.append(w_s_min.xs(0).head(1))

            w_u_plus_poincare = pd.concat(ls_u)
            w_s_min_poincare = pd.concat(ls_s)

            l1, = axarr[idx].plot(w_u_plus_poincare['y'], w_u_plus_poincare['ydot'], c='r', label='$\mathcal{W}^{U+}$')
            l2, = axarr[idx].plot(w_s_min_poincare['y'], w_s_min_poincare['ydot'], c='g', label='$\mathcal{W}^{S-}$')
            axarr[idx].set_title('$C$ = ' + str(c_level))

        axarr[0].set_ylim([-1, 1])
        axarr[1].set_xlim([0.0025, 0.175])

        axarr[0].set_xlabel('$y$ [-]')
        axarr[1].set_xlabel('$y$ [-]')
        axarr[0].set_ylabel('$\dot{y}$ [-]')

        axarr[0].grid(True, which='both', ls=':')
        axarr[1].grid(True, which='both', ls=':')

        axarr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2])
        fig.tight_layout()
        fig.subplots_adjust(top=0.8, right=0.9)
        plt.suptitle('Planar Poincaré phase portrait at $\mathbf{U}_3$', size=self.suptitleSize)
        plt.show()
        plt.savefig('../../data/figures/poincare_sections/planar_poincare_phase_portrait_l2_to_l1.pdf',
                    transparent=True)
        plt.close()
        pass

    def plot_portrait_cycle(self):
        fig, axarr = plt.subplots(2, 2, figsize=self.figSize, sharey=True)

        for idx, c_level in enumerate([3.1, 3.15]):
            w_u_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L1_horizontal_' +
                                                str(orbit_ids['horizontal'][1][c_level]) + '_W_U_plus.txt')
            w_s_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L2_horizontal_' +
                                               str(orbit_ids['horizontal'][2][c_level]) + '_W_S_min.txt')

            ls_u = []
            ls_s = []
            for i in range(self.numberOfOrbitsPerManifold):
                if abs(w_u_plus.xs(i).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_u.append(w_u_plus.xs(i).tail(1))
                if abs(w_s_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_s.append(w_s_min.xs(i).head(1))

            # Add start as endpoint for loop plot
            if abs(w_u_plus.xs(0).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_u.append(w_u_plus.xs(0).tail(1))
            if abs(w_s_min.xs(0).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_s.append(w_s_min.xs(0).head(1))

            w_u_plus_poincare = pd.concat(ls_u)
            w_s_min_poincare = pd.concat(ls_s)

            l1, = axarr[1, idx].plot(w_u_plus_poincare['y'], w_u_plus_poincare['ydot'], c='r', label='$\mathcal{W}^{U+}$ at $\mathbf{U}_2$')
            l2, = axarr[1, idx].plot(w_s_min_poincare['y'], w_s_min_poincare['ydot'], c='g', label='$\mathcal{W}^{S-}$ at $\mathbf{U}_2$')
            axarr[0, idx].set_title('$C$ = ' + str(c_level))

        axarr[1, 0].set_ylim([-1, 1])
        axarr[1, 0].set_xlim([-0.175, -0.0025])
        axarr[1, 1].set_xlim([-0.175, -0.0025])

        axarr[1, 0].set_ylabel('$\dot{y}$ [-]')
        axarr[1, 1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2])

        for idx, c_level in enumerate([3.1, 3.15]):
            w_u_plus = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L2_horizontal_' +
                                                str(orbit_ids['horizontal'][2][c_level]) + '_W_U_min.txt')
            w_s_min = load_manifold_refactored('../../data/raw/manifolds/refined_for_c/L1_horizontal_' +
                                               str(orbit_ids['horizontal'][1][c_level]) + '_W_S_plus.txt')

            ls_u = []
            ls_s = []
            for i in range(self.numberOfOrbitsPerManifold):
                if abs(w_u_plus.xs(i).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_u.append(w_u_plus.xs(i).tail(1))
                if abs(w_s_min.xs(i).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                    pass
                else:
                    ls_s.append(w_s_min.xs(i).head(1))

            # Add start as endpoint for loop plot
            if abs(w_u_plus.xs(0).tail(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_u.append(w_u_plus.xs(0).tail(1))
            if abs(w_s_min.xs(0).head(1)['x'].get_values()[0] - (1 - self.massParameter)) > 1e-11:
                pass
            else:
                ls_s.append(w_s_min.xs(0).head(1))

            w_u_plus_poincare = pd.concat(ls_u)
            w_s_min_poincare = pd.concat(ls_s)

            l1, = axarr[0, idx].plot(w_u_plus_poincare['y'], w_u_plus_poincare['ydot'], c='r', label='$\mathcal{W}^{U-}$ at $\mathbf{U}_3$')
            l2, = axarr[0, idx].plot(w_s_min_poincare['y'], w_s_min_poincare['ydot'], c='g', label='$\mathcal{W}^{S+}$ at $\mathbf{U}_3$')
            # axarr[1, idx].set_title('$C$ = ' + str(c_level))

        axarr[0, 0].set_ylim([-1, 1])
        axarr[0, 0].set_xlim([0.0025, 0.175])
        axarr[0, 1].set_xlim([0.0025, 0.175])

        axarr[1, 0].set_xlabel('$y$ [-]')
        axarr[1, 1].set_xlabel('$y$ [-]')
        axarr[0, 0].set_ylabel('$\dot{y}$ [-]')

        for i in range(2):
            for j in range(2):
                axarr[i, j].grid(True, which='both', ls=':')

        axarr[0, 1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), handles=[l1, l2])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.875)
        plt.suptitle('Planar Poincaré phase portrait at $\mathbf{U}_2$ and $\mathbf{U}_3$', size=self.suptitleSize)
        # plt.show()
        plt.savefig('../../data/figures/poincare_sections/planar_poincare_phase_portrait_cycle.pdf',
                    transparent=True)
        plt.close()
        pass

if __name__ == '__main__':
    c_levels = [3.05, 3.1, 3.15]
    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}}}

    horizontal_phase_portrait = HorizontalPhasePortrait(orbit_ids)
    # horizontal_phase_portrait.plot_portrait_L1_to_L2()
    # horizontal_phase_portrait.plot_portrait_L2_to_L1()
    horizontal_phase_portrait.plot_portrait_cycle()

