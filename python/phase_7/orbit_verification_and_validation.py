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

        self.monodromy_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                       + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                       str("{:14.13f}".format(self.alpha)) + '_' + \
                                       str("{:14.13f}".format(self.beta)) + '_initial_conditions.txt')

        if self.varyingQuantity == 'Hamiltonian':
            statesContinuation_df = load_states_continuation(self.hamiltonian_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.hamiltonian_filepath + self.correction_fileName)
            initial_conditions_incl_m_df = load_initial_conditions_augmented_incl_M(self.hamiltonian_filepath + self.monodromy_fileName)

        # Generate the lists with hamiltonians, periods and number of iterations and deviations after convergence
        self.Hlt = []
        self.alphaContinuation = []
        self.accelerationContinuation = []

        self.T = []
        self.orbitsId = []
        self.numberOfIterations = []
        self.positionDeviationAfterConvergence = []
        self.velocityDeviationAfterConvergence = []
        self.velocityInteriorDeviationAfterConvergence = []
        self.velocityExteriorDeviationAfterConvergence = []
        self.periodDeviationAfterConvergence = []



        for row in statesContinuation_df.iterrows():
            self.orbitsId.append(row[0])
            if row[1][1] < 0:
                self.Hlt.append(row[1][1])
                self.accelerationContinuation.append(row[1][8])
                self.alphaContinuation.append(row[1][9])

        for row in differentialCorrections_df.iterrows():
            self.numberOfIterations.append(row[1][0])
            self.T.append(row[1][2])
            self.positionDeviationAfterConvergence.append(row[1][3])
            self.velocityDeviationAfterConvergence.append(row[1][4])
            self.velocityInteriorDeviationAfterConvergence.append(row[1][5])
            self.velocityExteriorDeviationAfterConvergence.append(row[1][6])
            self.periodDeviationAfterConvergence.append(row[1][7])


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

        # Determine deviations at full period

        self.deviation_x = []
        self.deviation_y = []
        self.deviation_z = []
        self.deviation_xdot = []
        self.deviation_ydot = []
        self.deviation_zdot = []
        self.deviation_position_norm = []
        self.deviation_velocity_norm = []


        for i in range(len(self.continuationParameter)):
            orbit_df = load_orbit_augmented('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
            + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:14.13f}".format(self.alpha)) + '_' + \
            str("{:14.13f}".format(self.beta))+ '_' + str("{:14.13f}".format(self.Hlt[i])) + '_.txt')

            initial_state = orbit_df.head(1).values[0]
            terminal_state = orbit_df.tail(1).values[0]

            xDeviation = terminal_state[1]-initial_state[1]
            yDeviation = terminal_state[2]-initial_state[2]
            zDeviation = terminal_state[3]-initial_state[3]
            xdotDeviation = terminal_state[4]-initial_state[4]
            ydotDeviation = terminal_state[5]-initial_state[5]
            zdotDeviation = terminal_state[6]-initial_state[6]

            self.deviation_x.append(np.abs(xDeviation))
            self.deviation_y.append(np.abs(yDeviation))
            self.deviation_z.append(np.abs(zDeviation))
            self.deviation_xdot.append(np.abs(xdotDeviation))
            self.deviation_ydot.append(np.abs(ydotDeviation))
            self.deviation_zdot.append(np.abs(zdotDeviation))
            self.deviation_position_norm.append(np.sqrt(xDeviation ** 2 + yDeviation  ** 2 + zDeviation ** 2 ))
            self.deviation_velocity_norm.append(np.sqrt(xdotDeviation ** 2 + ydotDeviation  ** 2 + zdotDeviation ** 2 ))

        # Analyse monodromy matrix
        self.maxEigenvalueDeviation = 1.0e-3  # Changed from 1e-3
        for row in initial_conditions_incl_m_df.iterrows():
            M = np.matrix(
                [list(row[1][12:18]), list(row[1][22:28]), list(row[1][32:38]), list(row[1][42:48]), list(row[1][52:58]),
                 list(row[1][62:68])])

            eigenvalue = np.linalg.eigvals(M)

        sorting_indices = [-1, -1, -1, -1, -1, -1]
        idx_real_one = []
        # Find indices of the first pair of real eigenvalue equal to one
        for idx, l in enumerate(eigenvalue):
            print(idx)
            print(l)
            if abs(l.imag) < self.maxEigenvalueDeviation:
                if abs(l.real - 1.0) < self.maxEigenvalueDeviation:
                    if sorting_indices[2] == -1:
                        sorting_indices[2] = idx
                        idx_real_one.append(idx)
                    elif sorting_indices[3] == -1:
                        sorting_indices[3] = idx
                        idx_real_one.append(idx)

        # Find indices of the pair of largest/smallest real eigenvalue (corresponding to the unstable/stable subspace)
        for idx, l in enumerate(eigenvalue):
                if idx == (sorting_indices[2] or sorting_indices[3]):
                    continue
                if abs(l.imag) < self.maxEigenvalueDeviation:
                    if abs(l.real) == max(abs(eigenvalue.real)):
                            sorting_indices[0] = idx
                    elif abs(abs(l.real) - 1.0 / max(abs(eigenvalue.real))) < self.maxEigenvalueDeviation:
                            sorting_indices[5] = idx


        missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))
        print(sorting_indices)
        print(set(sorting_indices))
        print(missing_indices)

        if eigenvalue.imag[missing_indices[0]] > eigenvalue.imag[missing_indices[1]]:
            sorting_indices[1] = missing_indices[0]
            sorting_indices[4] = missing_indices[1]
        else:
            sorting_indices[1] = missing_indices[1]
            sorting_indices[4] = missing_indices[0]




        #  =========== Plot layout settings ===============

        # plot specific spacing properties
        self.orbitSpacingFactor = 1

        # scale properties
        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1

        #label properties
        self.numberOfXTicks = 5

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        # Colour schemes
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'fifthLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 1.5)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 3)],
                                             sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}
        self.plotAlpha = 1
        self.lineWidth = 0.5

        pass
    def plot_families(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')



        if self.varyingQuantity == 'Hamiltonian':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Overview',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Overview', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Overview', size=self.suptitleSize)




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
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_hamiltonian/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.pdf', transparent=True)
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_acceleration/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.pdf', transparent=True)
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/varying_alpha/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.pdf', transparent=True)

        plt.close()
        pass

    def plot_periodicity_validation(self):
        f, arr = plt.subplots(3, 2, figsize=self.figSize)
        linewidth = 1
        ylim = [1e-16, 1e-8]

        xlim = [min(self.continuationParameter), max(self.continuationParameter)]
        xticks = (np.linspace(min(self.Hlt), max(self.Hlt), num=self.numberOfXTicks))

        arr[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 0].xaxis.set_ticks(xticks)
        arr[0, 0].set_title('Position deviation after convergence')
        arr[0, 0].semilogy(self.continuationParameter, self.positionDeviationAfterConvergence, linewidth=linewidth, c=self.plottingColors['singleLine'],label='$\\sqrt{(\\sum_{i=1}^n \\Delta x_{i}^2 + \\Delta y_{i}^2 + \\Delta z_{i}^2) }$')
        arr[0, 0].legend(frameon=True, loc='upper right')
        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_ylim(ylim)
        arr[0, 0].semilogy(self.continuationParameter, 1e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        arr[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 1].xaxis.set_ticks(xticks)
        arr[0, 1].set_title('Position deviation at full period')
        arr[0, 1].semilogy(self.continuationParameter, self.deviation_x, linewidth=linewidth, c=self.plottingColors['tripleLine'][0],label='$|{x}(T) - {x}(0)|$')
        arr[0, 1].semilogy(self.continuationParameter, self.deviation_y, linewidth=linewidth, c=self.plottingColors['tripleLine'][1],label='$|{y}(T) - {y}(0)|$')
        arr[0, 1].legend(frameon=True, loc='lower right')
        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_ylim(ylim)

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].set_title('Velocity deviation after convergence')
        arr[1, 0].semilogy(self.continuationParameter, self.velocityDeviationAfterConvergence, linewidth=linewidth, c=self.plottingColors['singleLine'], label='$\\sqrt{(\\sum_{i=1}^n \\Delta x_{i}^2 + \\Delta y_{i}^2 + \\Delta z_{i}^2) }$')
        arr[1, 0].legend(frameon=True, loc='upper right')
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim(ylim)
        arr[1, 0].semilogy(self.continuationParameter, 5e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].set_title('Velocity deviation at full period')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_xdot, linewidth=linewidth,c=self.plottingColors['tripleLine'][0], label='$|\dot{x}(T) - \dot{x}(0)|$')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_ydot, linewidth=linewidth,c=self.plottingColors['tripleLine'][1], label='$|\dot{y}(T) - \dot{y}(0)|$')
        arr[1, 1].legend(frameon=True, loc='lower right')
        arr[1, 1].set_xlim(xlim)
        arr[1, 1].set_ylim(ylim)

        arr[2, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 0].xaxis.set_ticks(xticks)
        arr[2, 0].set_title('Time deviation at full period')
        arr[2, 0].set_xlim(xlim)
        arr[2, 0].set_ylim(ylim)
        arr[2, 0].semilogy(self.continuationParameter, self.periodDeviationAfterConvergence, linewidth=linewidth, c=self.plottingColors['singleLine'],label='$\\sqrt{(\\sum_{i=1}^n \\Delta t_{i}^2 }$')
        arr[2, 0].semilogy(self.continuationParameter, 1.0e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 0].legend(frameon=True, loc='upper right')

        arr[2, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 1].xaxis.set_ticks(xticks)
        arr[2, 1].set_title('Number of correction cycles')
        arr[2, 1].set_xlim(xlim)
        arr[2, 1].set_ylim(0, 10)
        arr[2, 1].plot(self.continuationParameter, self.numberOfIterations, linewidth=linewidth, c=self.plottingColors['singleLine'])

        arr[0, 0].set_ylabel('$\Delta \mathbf{r}$ [-]')
        arr[1, 0].set_ylabel('$\Delta \mathbf{V}$ [-]')
        arr[2, 0].set_ylabel('$\Delta t$ [-]')
        arr[2, 1].set_ylabel('Number of iterations [-]')
        if self.varyingQuantity == 'Hamiltonian':
            arr[2, 0].set_xlabel('$H_{lt}$ [-]')
            arr[2, 1].set_xlabel('$H_{lt}$ [-]')
        if self.varyingQuantity == 'Alpha':
            arr[2, 0].set_xlabel('$\\alpha$ [-]')
            arr[2, 1].set_xlabel('$\\alpha$ [-]')
        if self.varyingQuantity == 'Acceleration':
            arr[2, 0].set_xlabel('$a_{lt}$ [-]')
            arr[2, 1].set_xlabel('$a_{lt}$ [-]')



        for i in range(3):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)


        if self.varyingQuantity == 'Hamiltonian':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Periodicity constraints verification',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Periodicity constraints verification', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Periodicity constraints verification', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, bbox_inches='tight')

        plt.close()
        pass

    def plot_monodromy_analysis(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        xlim = [min(self.continuationParameter), max(self.continuationParameter)]

        xticks = (np.linspace(min(self.Hlt), max(self.Hlt), num=self.numberOfXTicks))

        arr[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 0].xaxis.set_ticks(xticks)
        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_ylim([1e-4, 1e4])
        arr[0, 0].set_title('$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        arr[0, 0].set_ylabel('Eigenvalues module [-]')

        arr[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 1].xaxis.set_ticks(xticks)
        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_ylim([1e-14, 1e-6])
        arr[0, 1].set_ylabel('$| 1 - Det(\mathbf{M}) |$ [-]')
        arr[0, 1].set_title('Error in determinant ')

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].set_ylabel('Order of linear instability [-]')
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim([0, 3])

        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].set_xlim(xlim)
        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1, 1].set_ylabel(' $||\lambda_3|-1|$ [-]')
        arr[1, 1].set_title('Error in eigenvalue pair denoting periodicity')
        arr[1, 0].set_xlabel('x [-]')
        arr[1, 1].set_xlabel('x [-]')
        arr[1, 0].set_title('Order of linear instability')

        if self.varyingQuantity == 'Hamiltonian':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Monodromy matrix eigensystem validation',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, bbox_inches='tight')

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
                            display_periodic_solutions.plot_periodicity_validation()
                            display_periodic_solutions.plot_monodromy_analysis()


                            del display_periodic_solutions

