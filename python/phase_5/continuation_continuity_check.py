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

from load_data_augmented import load_orbit_augmented, load_lagrange_points_location_augmented, load_states_continuation, \
     load_differential_correction

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
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)


        ax1.set_xlabel('orbit$_{number}$ [-]')
        ax1.set_ylabel('$H_{lt}$ [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('orbit$_{number}$ [-]')
        ax2.set_ylabel('$a_{lt}$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('orbit$_{number}$ [-]')
        ax3.set_ylabel('$\\alpha$ [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('orbit$_{number}$ [-]')
        ax4.set_ylabel('Number of iterations [-]')
        ax4.grid(True, which='both', ls=':')


        df = load_states_continuation('../../data/raw/orbits/augmented/varying_hamiltonian/L' + str(self.lagrangePointNr) + \
                                      '_' + str(self.orbitType) + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                      str("{:14.13f}".format(self.alpha)) + '_' + str("{:14.13f}".format(self.beta)) + '_states_continuation.txt')

        df_corrections = load_differential_correction('../../data/raw/orbits/augmented/varying_hamiltonian/L' + str(self.lagrangePointNr) + \
                                      '_' + str(self.orbitType) + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                      str("{:14.13f}".format(self.alpha)) + '_' + str("{:14.13f}".format(self.beta)) + '_differential_correction.txt')

        ax1.plot(df['orbitID'],df['hlt'],color='blue', linewidth=1, label='energy')
        ax2.semilogy(df['orbitID'],df_corrections['alt'],color='blue', linewidth=1, label='energy')
        ax3.plot(df['orbitID'],df_corrections['alpha'],color='blue', linewidth=1, label='energy')
        ax4.plot(df['orbitID'],df_corrections['iterations'],color='blue', linewidth=1, label='energy')



        ax1.set_xlim([0, len(df['orbitID'])])
        ax1.set_ylim([min(df['hlt']), max(df['hlt'])])

        ax2.set_xlim([0, len(df['orbitID'])])
        ax2.set_ylim([1.0e-5,5.0e-1])

        ax3.set_xlim([0, len(df['orbitID'])])
        ax3.set_ylim([0, 360])

        ax4.set_xlim([0, len(df['orbitID'])])        
        ax4.set_ylim([0, max(df_corrections['iterations']*2)])


        ax1.set_title('$H_{lt}$ evolution')
        ax2.set_title('$a_{lt}$ evolution')
        ax3.set_title('$\\alpha$ evolution')
        ax4.set_title('Number of TLT cycles until convergence')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)




        fig.suptitle('L'+str(self.lagrangePointNr)+'($a_{lt} = ' + str("{:2.1e}".format(self.accelerationMagnitude))+ \
                     '$,$\\alpha = ' + str("{:3.1f}".format(self.alpha)) + '$) Family evolution')


        fig.savefig('../../data/figures/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                    '_spacing_verification.png', transparent=True, dpi=self.dpi)

        pass


if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1,2]
    acceleration_magnitudes = [0.001000]
    alphas = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
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
