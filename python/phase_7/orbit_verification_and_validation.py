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
    load_states_continuation

class DisplayPeriodicSolutions:
    def __init__(self,orbit_type, lagrange_point_nr,  acceleration_magnitude, alpha, beta, varying_quantity, low_dpi):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = beta
        self.varyingQuantity = varying_quantity
        self.lowDPI = low_dpi

        print('=======================')
        print('L' + str(self.lagrangePointNr) + '_' + self.orbitType + ' (acc = ' + str(self.accelerationMagnitude) \
                + ', alpha = ' + str(self.alpha), 'beta = ' + str(self.beta) + ')' )
        print('=======================')

        self.orbitTypeForTitle = orbit_type.capitalize()
        if self.orbitTypeForTitle == 'Horizontal' or self.orbitTypeForTitle == 'Vertical':
            self.orbitTypeForTitle += ' Lyapunov'

        #  =========== Extract relevant data from text files  ===============
        self.hamiltonian_filepath = '../../data/raw/orbits/augmented/varying_hamiltonian/'
        self.acceleration_filepath = '../../data/raw/orbits/augmented/varying_acceleration/'
        self.alpha_filepath = '../../data/raw/orbits/augmented/varying_alpha/'

        # define filenames of states_continuation and differential correction
        self.continuation_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
            + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:14.13f}".format(self.alpha)) + '_' + \
            str("{:14.13f}".format(self.beta)) + '_states_continuation.txt')

        self.correction_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                         + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                         str("{:14.13f}".format(self.alpha)) + '_' + \
                                         str("{:14.13f}".format(self.beta)) + '_differential_correction.txt')

        if self.varyingQuantity == 'Hamiltonian':
            statesContinuation_df = load_states_continuation(self.hamiltonian_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.hamiltonian_filepath + self.continuation_fileName)

        # Generate the lists with hamiltonians, periods and number of iterations
        self.Hlt = []
        self.alphaContinuation = []
        self.accelerationContinuation = []

        self.T = []
        self.orbitsId = []
        self.numberOfIterations = []

        for row in statesContinuation_df.iterrows():
            self.orbitsId.append(row[0])
            if row[1][1] < 0:
                self.Hlt.append(row[1][1])
                self.accelerationContinuation.append(row[1][8])
                self.alphaContinuation.append(row[1][9])

        for row in differentialCorrections_df.iterrows():
            self.T.append(row[1][2])
            self.numberOfIterations.append(row[1][0])

        # Determine which parameter is the varying parameter
        if self.varyingQuantity == 'Hamiltonian':
            self.continuationParameter = self.Hlt
        if self.varyingQuantity == 'Acceleration':
            self.continuationParameter = self.accelerationContinuation
        if self.varyingQuantity == 'Alpha':
            self.continuationParameter = self.alphaContinuation

        # Determine heatmap for level of the continuation parameter
        self.numberOfPlotColorIndices = len(self.continuationParameter)
        self.plotColorIndexBasedOnContinuation = []
        for hamiltonian in self.Hlt:
            self.plotColorIndexBasedOnContinuation.append(int(np.round(
                (hamiltonian - min(self.continuationParameter)) / (max(self.continuationParameter) - min(self.continuationParameter)) * (
                                self.numberOfPlotColorIndices - 1))))



        #  =========== Plot layout settings ===============

        # plot specific spacing properties
        self.orbitSpacingFactor = 1

        # scale properties
        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        # Colour schemes
        n_colors = 6
        self.plottingColors = {'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'fifthLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 1.5)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 3)],
                                             sns.color_palette("viridis", n_colors)[0]]}
        self.plotAlpha = 1
        self.lineWidth = 0.5

        pass
    def plot_families(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Overview',size=self.suptitleSize)



        # Plot libration point
        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],color='black', marker='x')

        # Plot bodies
        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.contourf(xM, yM, zM, colors='black')
        ax.contourf(xE, yE, zE, colors='black')


        continuation_normalized = [(value - min(self.continuationParameter)) / (max(self.continuationParameter) - min(self.continuationParameter)) for value in self.continuationParameter]
        colors = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r"))(continuation_normalized)
        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r", len(self.continuationParameter))),
                                   norm=plt.Normalize(vmin=min(self.continuationParameter), vmax=max(self.continuationParameter)), )

        orbitIdsPlot = list(range(0, len(self.continuationParameter), self.orbitSpacingFactor))
        if orbitIdsPlot != len(self.continuationParameter):
            orbitIdsPlot.append(len(self.continuationParameter) - 1)

        for i in orbitIdsPlot:
            plot_color = colors[self.plotColorIndexBasedOnContinuation[i]]

            df = load_orbit('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' \
                            + str("{:14.13f}".format(self.alpha)) + '_' \
                            + str("{:14.13f}".format(self.beta)) + '_' \
                            + str("{:14.13f}".format(self.Hlt[i])) + '_.txt')

            ax.plot(df['x'], df['y'], color=plot_color, alpha=self.plotAlpha, linewidth=self.lineWidth)


        minimumX = min(df['x'])
        minimumY = min(df['y'])
        maximumX = max(df['x'])
        maximumY = max(df['y'])


        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        sm.set_array([])
        cax, kw = matplotlib.colorbar.make_axes([ax])

        cbar = plt.colorbar(sm, cax=cax, label='$H_{lt}$ [-]', format='%1.4f', **kw)

        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_hamiltonian/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.pdf', transparent=True)
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_acceleration/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.pdf', transparent=True)
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_alpha/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_orthographic_projection.pdf', transparent=True)

        plt.close()
        pass

    def plot_periodicity_validation(self):

        pass



if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.0]
    alphas = [0.0]
    betas = [0.0]
    low_dpi = True
    varying_quantities = ['Hamiltonian']

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for alpha in alphas:
                    for beta in betas:
                        for varying_quantity in varying_quantities:
                            display_periodic_solutions = DisplayPeriodicSolutions(orbit_type, lagrange_point, acceleration_magnitude, \
                                         alpha, beta, varying_quantity, low_dpi)

                            display_periodic_solutions.plot_families()
                            del display_periodic_solutions

