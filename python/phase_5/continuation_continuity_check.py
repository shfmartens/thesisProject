import numpy as np
import pandas as pd
import json
import matplotlib
from decimal import *
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
import time
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{gensymb}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
load_initial_conditions_incl_M, load_manifold

from load_data_augmented import load_orbit_augmented, load_lagrange_points_location_augmented, load_states_continuation

class numericalContinuation:
    def __init__(self, orbit_type, lagrange_point_nr, acceleration_magnitude, alpha, beta, low_dpi):
        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = beta
        self.figSize =  (7 * (1 + np.sqrt(5)) / 2, 7)
        self.dpi = 150



    def plot_energy_spacing(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('orbit$_{number}$ [-]')
        ax.set_ylabel('$H_lt$ [-]')
        ax.grid(True, which='both', ls=':')

        df = load_states_continuation('../../data/raw/orbits/augmented/varying_hamiltonian/L' + str(self.lagrangePointNr) + \
                                      '_' + str(self.orbitType) + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                      str("{:14.13f}".format(self.alpha)) + '_' + str("{:14.13f}".format(self.beta)) + '_states_continuation.txt')

        ax.plot(df['orbitID'],df['hlt'],color='blue', linewidth=1, label='energy')
        ax.set_xlim([0, len(df['orbitID'])])
        ax.set_ylim([min(df['hlt']), max(df['hlt'])])

        ax.set_title('Numerical Continuation spacing verification')

        fig.savefig('../../data/figures/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                    '_spacing_verification.png', transparent=True, dpi=self.dpi)

        pass


if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.00000]
    alphas = [0.0]
    betas = [0.0]

    low_dpi = True


    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for alpha in alphas:
                    for beta in betas:
                        numerical_Continuation = numericalContinuation(orbit_type, lagrange_point, acceleration_magnitude, \
                                        alpha, beta, low_dpi)
                        numerical_Continuation.plot_energy_spacing()
            del numerical_Continuation
