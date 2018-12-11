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
        self.eigenvectorDf_U = pd.read_table( '../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_S = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
        self.eigenvectorLocationDf_U = pd.read_table('../../data/raw/manifolds/refined_for_c/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))

        self.W_S_plus = load_manifold_refactored('../../data/raw/manifolds/L' + str(lagrange_point_nr) + '_' + orbit_type + '_' + str(orbit_id) + '_W_S_plus.txt')
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
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1],first_state_on_manifold[2], first_state_on_manifold[3],first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_plus.xs(0).iloc[::-1].iterrows():
            self.T_along_0_W_S_plus.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
            self.C_along_0_W_S_plus.append(abs(jacobi_on_manifold-first_jacobi_on_manifold))

        first_state_on_manifold = self.W_S_min.xs(0).tail(1).values[0]
        first_jacobi_on_manifold = computeJacobiEnergy(first_state_on_manifold[0], first_state_on_manifold[1], first_state_on_manifold[2], first_state_on_manifold[3], first_state_on_manifold[4], first_state_on_manifold[5])
        for row in self.W_S_min.xs(0).iloc[::-1].iterrows():
            self.T_along_0_W_S_min.append(abs(row[0]))
            state_on_manifold = row[1].values
            jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1], state_on_manifold[2],state_on_manifold[3], state_on_manifold[4], state_on_manifold[5])
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
                self.phase.append(i / self.numberOfOrbitsPerManifold)

                # On orbit
                state_on_orbit = self.eigenvectorLocationDf_S.xs(i).values
                jacobi_on_orbit = computeJacobiEnergy(state_on_orbit[0], state_on_orbit[1], state_on_orbit[2],state_on_orbit[3], state_on_orbit[4], state_on_orbit[5])

                # W_S_plus
                state_on_manifold = self.W_S_plus.xs(i).tail(1).values[0]
                jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4],state_on_manifold[5])
                self.C_diff_start_W_S_plus.append(abs(jacobi_on_manifold - jacobi_on_orbit))
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
                jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4],state_on_manifold[5])
                self.C_diff_start_W_S_min.append(abs(jacobi_on_manifold - jacobi_on_orbit))
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
                jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4],state_on_manifold[5])
                self.C_diff_start_W_U_plus.append(abs(jacobi_on_manifold - jacobi_on_orbit))
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
                jacobi_on_manifold = computeJacobiEnergy(state_on_manifold[0], state_on_manifold[1],state_on_manifold[2],state_on_manifold[3], state_on_manifold[4],state_on_manifold[5])
                self.C_diff_start_W_U_min.append(abs(jacobi_on_manifold - jacobi_on_orbit))
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


if __name__ == '__main__':
    help()
    low_dpi = False
    lagrange_points = [1, 2]
    orbit_types = ['horizontal']
    c_levels = [3.05, 3.1, 3.15]

    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}}}

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for c_level in c_levels:
                display_augmented_validation = DisplayAugmentedValidation(orbit_type, lagrange_point,
                                                                              orbit_ids[orbit_type][lagrange_point][
                                                                                  c_level],
                                                                              low_dpi=low_dpi)

                del display_augmented_validation