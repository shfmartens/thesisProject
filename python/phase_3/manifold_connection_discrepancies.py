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
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm


class ManifoldConnectionDiscrepancies:
    def __init__(self, w_s, w_u):
        self.stableManifold = load_manifold('../../data/raw/manifolds/' + w_s + '.txt')
        self.unstableManifold = load_manifold('../../data/raw/manifolds/' + w_u + '.txt')
        self.numberOfOrbitsPerManifold = len(set(self.stableManifold.index.get_level_values(0)))

        # Assemble states at Poincaré section (end of file)
        ls_s = []
        ls_u = []

        # plt.figure()
        ls_dx = []
        ls_dy = []
        ls_dz = []
        ls_dxdot = []
        ls_dydot = []
        ls_dzdot = []

        for i in range(self.numberOfOrbitsPerManifold):
                for j in range(self.numberOfOrbitsPerManifold):
                    # ls_s.append(self.stableManifold.xs(i).tail(1))
                    # ls_u.append(self.unstableManifold.xs(i).tail(1))

                    ls_dx.append(self.stableManifold.xs(i).tail(1)['x'].values-self.unstableManifold.xs(j).tail(1)['x'].values)
                    ls_dy.append(self.stableManifold.xs(i).tail(1)['y'].values - self.unstableManifold.xs(j).tail(1)['y'].values)
                    ls_dz.append(self.stableManifold.xs(i).tail(1)['z'].values - self.unstableManifold.xs(j).tail(1)['z'].values)

                    ls_dxdot.append(self.stableManifold.xs(i).tail(1)['xdot'].values - self.unstableManifold.xs(j).tail(1)['xdot'].values)
                    ls_dydot.append(self.stableManifold.xs(i).tail(1)['ydot'].values - self.unstableManifold.xs(j).tail(1)['ydot'].values)
                    ls_dzdot.append(self.stableManifold.xs(i).tail(1)['zdot'].values - self.unstableManifold.xs(j).tail(1)['zdot'].values)

        f, axarr = plt.subplots(2)
        axarr[0].scatter(list(range(len(ls_dx))), ls_dx)
        axarr[0].scatter(list(range(len(ls_dx))), ls_dy)
        axarr[0].scatter(list(range(len(ls_dx))), ls_dz)
        axarr[1].scatter(list(range(len(ls_dx))), ls_dxdot)
        axarr[1].scatter(list(range(len(ls_dx))), ls_dydot)
        axarr[1].scatter(list(range(len(ls_dx))), ls_dzdot)

        axarr[0].set_ylim([-0.01, 0.01])
        axarr[1].set_ylim([-0.01, 0.01])
        plt.show()

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
        pass


if __name__ == '__main__':
    w_s = 'L1_vertical_1159_W_U_plus'
    w_u = 'L2_vertical_1275_W_S_min'

    manifold_connection_discrepancies = ManifoldConnectionDiscrepancies(w_s=w_s, w_u=w_u)
