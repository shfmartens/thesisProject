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

from load_data_augmented import load_orbit_augmented, load_lagrange_points_location_augmented, load_differential_correction, \
    load_states_continuation, load_initial_conditions_augmented_incl_M

class DisplayFamilyProperties:
    def __init__(self,orbit_type, lagrange_point_nr,  acceleration_magnitude, alpha, beta, varying_quantity, low_dpi):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = beta
        self.varyingQuantity = varying_quantity
        self.lowDPI = low_dpi

        self.orbitTypeForTitle = orbit_type.capitalize()
        if self.orbitTypeForTitle == 'Horizontal' or self.orbitTypeForTitle == 'Vertical':
            self.orbitTypeForTitle += ' Lyapunov'

        self.dpi = 150
        self.suptitleSize = 20

        self.overviewArray = []
        # check if acceleration Magnitude and or alpha is length
        if self.varyingQuantity == 'Hamiltonian':
            if len(self.accelerationMagnitude) > 1:
                self.overviewArray = self.accelerationMagnitude
                self.titleVariable = self.alpha[0]
            else:
                self.overviewArray = self.alpha
                self.titleVariable = self.accelerationMagnitude[0]
            self.file_directory = '../../data/raw/orbits/augmented/varying_hamiltonian/'

            # Colour schemes
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
                                   'fifthLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                                 sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 1.5)],
                                                 sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                                 sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 3)],
                                                 sns.color_palette("viridis", n_colors)[0]],
                                   'limit': 'black'}



    def plot_family_overview(self):
        fig = plt.figure(figsize=(7 * (1 + np.sqrt(5)) / 2, 7))
        ax = fig.gca()


        ax.set_xlabel('T [-]')
        ax.set_ylabel('$H_{lt}$ [-]')
        ax.grid(True, which='both', ls=':')


        for i in range(len(self.overviewArray)):
            if len(self.accelerationMagnitude) > 1:
                init_conditions_df = load_initial_conditions_augmented_incl_M(self.file_directory + 'L'+str(self.lagrangePointNr)+'_'+self.orbitType \
                      +'_' + str("{:12.11f}".format(self.overviewArray[i])) + '_' + str("{:12.11f}".format(self.titleVariable)) \
                      + '_' + str("{:12.11f}".format(self.beta)) + '_initial_conditions.txt')

                ax.plot(init_conditions_df[1].values, init_conditions_df[0].values,
                        label='$a_{lt} = + ' + str(self.overviewArray[i]),
                        linestyle='-', color=self.plottingColors['tripleLine'][i])

        if self.varyingQuantity == 'Hamiltonian':
            if len(self.accelerationMagnitude) > 1:
                plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + '($\\alpha = ' + str(
                self.alpha[0]) + ' ^{\\circ}$) Overview ' + '- Hamiltonian and period ', size=self.suptitleSize)
            else:
                plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + '($a_{lt} = ' + str(
                    self.accelerationMagnitude[0]) + '$) Overview ' + '- Hamiltonian and period ', size=self.suptitleSize)

        if self.varyingQuantity == 'Hamiltonian':
            if len(self.accelerationMagnitude) > 1:
                if self.lowDPI:
                    plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_alpha_' + str(
                    "{:7.6f}".format(self.alpha[0])) + '_overview_families_orbital_energy_period.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
                else:
                    plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_alpha_' + str(
                    "{:7.6f}".format(self.alpha[0])) + '_overview_families_orbital_energy_period.pdf', transparent=True, bbox_inches='tight')
            else:
                if self.lowDPI:
                    plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_acc_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude[0])) + '_overview_families_orbital_energy_period.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
                else:
                    plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_acc_' + str("{:7.6f}".format(self.accelerationMagnitude[0])) + '_overview_families_orbital_energy_period.png',transparent=True)


        pass

if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.1]
    alphas = [0.0,60.0,120.0,180.0,240.0,300.0]
    betas = [0.0]
    low_dpi = True
    varying_quantities = 'Hamiltonian'

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
                    for beta in betas:
                        display_family_properties = DisplayFamilyProperties(orbit_type, lagrange_point, acceleration_magnitudes, \
                                    alphas, beta, varying_quantities, low_dpi)

                        display_family_properties.plot_family_overview()

                        del display_family_properties



