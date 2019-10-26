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
    load_states_continuation, load_initial_conditions_augmented_incl_M, load_states_continuation_length, compute_phase

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
            + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:12.11f}".format(self.alpha)) + '_' + \
            str("{:12.11f}".format(self.beta)) + '_states_continuation.txt')

        self.correction_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                         + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
                                         str("{:12.11f}".format(self.alpha)) + '_' + \
                                         str("{:12.11f}".format(self.beta)) + '_differential_correction.txt')

        self.monodromy_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                       + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
                                       str("{:12.11f}".format(self.alpha)) + '_' + \
                                       str("{:12.11f}".format(self.beta)) + '_initial_conditions.txt')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            statesContinuation_df = load_states_continuation(self.hamiltonian_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.hamiltonian_filepath + self.correction_fileName)
            initial_conditions_incl_m_df = load_initial_conditions_augmented_incl_M(self.hamiltonian_filepath + self.monodromy_fileName)
        # Generate the lists with hamiltonians, periods and number of iterations and deviations after convergence
        self.Hlt = []
        self.alphaContinuation = []
        self.accelerationContinuation = []
        self.x = []
        self.phase = []
        self.y = []
        self.T = []
        self.maxSegmentError = []
        self.maxDeltaError = []
        self.orbitsId = []
        self.numberOfIterations = []
        self.positionDeviationAfterConvergence = []
        self.velocityDeviationAfterConvergence = []
        self.velocityInteriorDeviationAfterConvergence = []
        self.velocityExteriorDeviationAfterConvergence = []
        self.stateDeviationAfterConvergence = []
        self.phaseDeviationAfterConvergence = []
        self.totalDeviationAfterConvergence = []

        self.xPhaseHalf = []
        self.yPhaseHalf = []
        self.zPhaseHalf = []
        self.xdotPhaseHalf = []
        self.ydotPhaseHalf = []
        self.zdotPhaseHalf = []

        self.numberOfCollocationPoints = []
        for row in statesContinuation_df.iterrows():
            self.orbitsId.append(row[1][0]+1)
            self.Hlt.append(row[1][1])
            self.x.append(row[1][3])
            self.phase.append(compute_phase(row[1][3],row[1][4],self.lagrangePointNr))
            self.y.append(row[1][4])
            self.accelerationContinuation.append(row[1][8])
            self.alphaContinuation.append(row[1][9]/180.0*np.pi)
            self.numberOfCollocationPoints.append(row[1][13])

        for row in differentialCorrections_df.iterrows():
            self.numberOfIterations.append(row[1][0])
            self.maxSegmentError.append(row[1][1])
            self.maxDeltaError.append(row[1][2])
            self.T.append(row[1][4])
            self.positionDeviationAfterConvergence.append(row[1][5])
            self.velocityDeviationAfterConvergence.append(row[1][6])
            self.velocityInteriorDeviationAfterConvergence.append(row[1][7])
            self.velocityExteriorDeviationAfterConvergence.append(row[1][8])
            self.stateDeviationAfterConvergence.append(np.sqrt(row[1][5] ** 2 + row[1][6] ** 2))
            self.phaseDeviationAfterConvergence.append(np.sqrt(row[1][9]**2))
            self.totalDeviationAfterConvergence.append(np.sqrt(row[1][5] ** 2 + row[1][6] ** 2 + row[1][9] ** 2))

            self.xPhaseHalf.append(row[1][20])
            self.yPhaseHalf.append(row[1][21])
            self.zPhaseHalf.append(row[1][22])
            self.xdotPhaseHalf.append(row[1][23])
            self.ydotPhaseHalf.append(row[1][24])
            self.zdotPhaseHalf.append(row[1][25])




        # Determine which parameter is the varying parameter
        if self.varyingQuantity == 'xcor':
            self.continuationParameter = self.x
        if self.varyingQuantity == 'Hamiltonian':
            self.continuationParameter = self.Hlt
        if self.varyingQuantity == 'Acceleration':
            self.continuationParameter = self.accelerationContinuation
        if self.varyingQuantity == 'Alpha':
            self.continuationParameter = self.alphaContinuation

        # Determine heatmap for level of the continuation parameter
        self.numberOfPlotColorIndices = len(self.continuationParameter)
        self.plotColorIndexBasedOnContinuation = []
        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'x':
             for hamiltonian in self.Hlt:
                 self.plotColorIndexBasedOnContinuation.append(int(np.round(
                     (hamiltonian - min(self.Hlt)) / (max(self.Hlt) - min(self.Hlt)) * (
                                     self.numberOfPlotColorIndices - 1))))
        else:
            for hamiltonian in self.continuationParameter:
                 self.plotColorIndexBasedOnContinuation.append(int(np.round(
                     (hamiltonian - min(self.continuationParameter)) / (max(self.continuationParameter) - min(self.continuationParameter)) * (
                                     self.numberOfPlotColorIndices - 1))))
                # for xcoor in self.x:
                #     self.plotColorIndexBasedOnContinuation.append(int(np.round(
                #         (xcoor - min(self.continuationParameter)) / (
                #                     max(self.continuationParameter) - min(self.continuationParameter)) * (
                #                 self.numberOfPlotColorIndices - 1))))

        # Determine deviations at full period

        self.deviation_x = []
        self.deviation_y = []
        self.deviation_z = []
        self.deviation_xdot = []
        self.deviation_ydot = []
        self.deviation_zdot = []
        self.deviation_position_norm = []
        self.deviation_velocity_norm = []
        self.deviation_state_norm = []



        for i in range(len(self.continuationParameter)):
            orbit_df = load_orbit_augmented('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
            + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:12.11f}".format(self.alpha)) + '_' + \
            str("{:12.11f}".format(self.beta))+ '_' + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')

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
            self.deviation_state_norm.append(np.sqrt(xDeviation ** 2 + yDeviation  ** 2 + zDeviation ** 2  + \
                                                    xdotDeviation ** 2 + ydotDeviation  ** 2 + zdotDeviation ** 2 ))

        # Analyse monodromy matrix
        self.eigenvalues = []
        self.D = []
        self.orderOfLinearInstability = []
        self.orbitIdBifurcations = []
        self.lambda1 = []
        self.lambda2 = []
        self.lambda3 = []
        self.lambda4 = []
        self.lambda5 = []
        self.lambda6 = []
        self.eigenvalues= []
        self.v1 = []
        self.v2 = []
        self.v3 = []

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

            if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                sorting_indices[1] = missing_indices[0]
                sorting_indices[4] = missing_indices[1]
            else:
                sorting_indices[1] = missing_indices[1]
                sorting_indices[4] = missing_indices[0]

            if len(sorting_indices) > len(set(sorting_indices)):
                print('\nWARNING: SORTING INDEX IS NOT UNIQUE FOR ' + self.orbitType + ' AT L' + str(
                     self.lagrangePointNr))
                if len(idx_real_one) != 2:
                    idx_real_one = []
                    # Find indices of the first pair of real eigenvalue equal to one
                    for idx, l in enumerate(eigenvalue):
                        if abs(l.imag) < 2 * self.maxEigenvalueDeviation:
                            if abs(l.real - 1.0) < 2 * self.maxEigenvalueDeviation:
                                if sorting_indices[2] == -1:
                                    sorting_indices[2] = idx
                                    idx_real_one.append(idx)
                                elif sorting_indices[3] == -1:
                                    sorting_indices[3] = idx
                                    idx_real_one.append(idx)

                if len(idx_real_one) == 2:
                    sorting_indices = [-1, -1, -1, -1, -1, -1]
                    sorting_indices[2] = idx_real_one[0]
                    sorting_indices[3] = idx_real_one[1]
                    # Assume two times real one and two conjugate pairs
                    for idx, l in enumerate(eigenvalue):
                        # min(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_real_one))], deg=True)))
                        # if abs(np.angle(l, deg=True))%180 == min(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_real_one))], deg=True)) %180):
                        if l.real == eigenvalue[list(set(range(6)) - set(idx_real_one))].real.max():
                            if l.imag > 0:
                                sorting_indices[0] = idx
                            elif l.imag < 0:
                                sorting_indices[5] = idx
                        # if abs(np.angle(l, deg=True))%180 == max(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_real_one))], deg=True)) %180):
                        if l.real == eigenvalue[list(set(range(6)) - set(idx_real_one))].real.min():
                            if l.imag > 0:
                                sorting_indices[1] = idx
                            elif l.imag < 0:
                                sorting_indices[4] = idx
                        print(sorting_indices)

                if len(sorting_indices) > len(set(sorting_indices)):
                    print('\nWARNING: SORTING INDEX IS STILL NOT UNIQUE')
                    # Sorting eigenvalues from largest to smallest norm, excluding real one

                    # Sorting based on previous phase
                    if len(idx_real_one) == 2:
                        sorting_indices = [-1, -1, -1, -1, -1, -1]
                        sorting_indices[2] = idx_real_one[0]
                        sorting_indices[3] = idx_real_one[1]

                        # Assume two times real one and two conjugate pairs
                        for idx, l in enumerate(eigenvalue[list(set(range(6)) - set(idx_real_one))]):
                            print(idx)
                            if abs(l.real - self.lambda1[-1].real) == min(
                                    abs(eigenvalue.real - self.lambda1[-1].real)) and abs(
                                    l.imag - self.lambda1[-1].imag) == min(
                                    abs(eigenvalue.imag - self.lambda1[-1].imag)):
                                sorting_indices[0] = idx
                            if abs(l.real - self.lambda2[-1].real) == min(
                                    abs(eigenvalue.real - self.lambda2[-1].real)) and abs(
                                    l.imag - self.lambda2[-1].imag) == min(
                                    abs(eigenvalue.imag - self.lambda2[-1].imag)):
                                sorting_indices[1] = idx
                            if abs(l.real - self.lambda5[-1].real) == min(
                                    abs(eigenvalue.real - self.lambda5[-1].real)) and abs(
                                    l.imag - self.lambda5[-1].imag) == min(
                                    abs(eigenvalue.imag - self.lambda5[-1].imag)):
                                sorting_indices[4] = idx
                            if abs(l.real - self.lambda6[-1].real) == min(
                                    abs(eigenvalue.real - self.lambda6[-1].real)) and abs(
                                    l.imag - self.lambda6[-1].imag) == min(
                                    abs(eigenvalue.imag - self.lambda6[-1].imag)):
                                sorting_indices[5] = idx
                            print(sorting_indices)

                    pass
                if (sorting_indices[1] and sorting_indices[4]) == -1:
                    # Fill two missing values
                    two_missing_indices = list(set(list(range(-1, 6))) - set(sorting_indices))
                    if abs(eigenvalue[two_missing_indices[0]].real) > abs(eigenvalue[two_missing_indices[1]].real):
                        sorting_indices[1] = two_missing_indices[0]
                        sorting_indices[4] = two_missing_indices[1]
                    else:
                        sorting_indices[1] = two_missing_indices[1]
                        sorting_indices[4] = two_missing_indices[0]
                    print(sorting_indices)
                if (sorting_indices[0] and sorting_indices[5]) == -1:
                    # Fill two missing values
                    two_missing_indices = list(set(list(range(-1, 6))) - set(sorting_indices))
                    print(eigenvalue)
                    print(sorting_indices)
                    print(two_missing_indices)
                    # TODO quick fix for extended v-l
                    # if len(two_missing_indices)==1:
                    #     print('odd that only one index remains')
                    #     if sorting_indices[0] == -1:
                    #         sorting_indices[0] = two_missing_indices[0]
                    #     else:
                    #         sorting_indices[5] = two_missing_indices[0]
                    # sorting_indices = abs(eigenvalue).argsort()[::-1]
                    if abs(eigenvalue[two_missing_indices[0]].real) > abs(eigenvalue[two_missing_indices[1]].real):
                        sorting_indices[0] = two_missing_indices[0]
                        sorting_indices[5] = two_missing_indices[1]
                    else:
                        sorting_indices[0] = two_missing_indices[1]
                        sorting_indices[5] = two_missing_indices[0]
                    print(sorting_indices)

                if len(sorting_indices) > len(set(sorting_indices)):
                    print('\nWARNING: SORTING INDEX IS STILL STILL NOT UNIQUE')
                    # Sorting eigenvalues from largest to smallest norm, excluding real one
                    sorting_indices = abs(eigenvalue).argsort()[::-1]
                print(eigenvalue[sorting_indices])

            self.eigenvalues.append(eigenvalue[sorting_indices])
            self.lambda1.append(eigenvalue[sorting_indices[0]])
            self.lambda2.append(eigenvalue[sorting_indices[1]])
            self.lambda3.append(eigenvalue[sorting_indices[2]])
            self.lambda4.append(eigenvalue[sorting_indices[3]])
            self.lambda5.append(eigenvalue[sorting_indices[4]])
            self.lambda6.append(eigenvalue[sorting_indices[5]])

            # Determine order of linear instability
            reduction = 0
            for i in range(6):
                if (abs(eigenvalue[i]) - 1.0) < 1e-2:
                    reduction += 1

            if len(self.orderOfLinearInstability) > 0:
                # Check for a bifurcation, when the order of linear instability changes
                if (6 - reduction) != self.orderOfLinearInstability[-1]:
                    self.orbitIdBifurcations.append(row[0])

            self.orderOfLinearInstability.append(6 - reduction)
            self.v1.append(abs(eigenvalue[sorting_indices[0]] + eigenvalue[sorting_indices[5]]) / 2)
            self.v2.append(abs(eigenvalue[sorting_indices[1]] + eigenvalue[sorting_indices[4]]) / 2)
            self.v3.append(abs(eigenvalue[sorting_indices[2]] + eigenvalue[sorting_indices[3]]) / 2)
            self.D.append(np.linalg.det(M))

        print('Index for bifurcations: ')
        print(self.orbitIdBifurcations)






        #  =========== Plot layout settings ===============

        # plot specific spacing properties
        self.orbitSpacingFactor = 50

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
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                                'singleLine': sns.color_palette("viridis", n_colors)[0],
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
                            + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(self.alpha)) + '_' \
                            + str("{:12.11f}".format(self.beta)) + '_' \
                            + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')

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

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
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
        ylim2 = [1e-19, 1e-1]
        xlim = [min(self.continuationParameter), max(self.continuationParameter)]
        xticks = (np.linspace(min(self.continuationParameter), max(self.continuationParameter), num=self.numberOfXTicks))

        arr[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 0].xaxis.set_ticks(xticks)
        arr[0, 0].set_title('Defect vector magnitude after convergence')
        arr[0, 0].semilogy(self.continuationParameter, self.totalDeviationAfterConvergence, linewidth=linewidth, c=self.plottingColors['singleLine'],label='$||F||$')
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

        arr[2, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 0].xaxis.set_ticks(xticks)
        arr[2, 0].set_title('Distribution of errors over collocated trajectory')
        arr[2, 0].semilogy(self.continuationParameter, self.maxDeltaError, linewidth=linewidth, c=self.plottingColors['singleLine'], label='max($e_{i}$)-min($e_{i}$)')
        arr[2, 0].legend(frameon=True, loc='upper right')
        arr[2, 0].set_xlim(xlim)
        arr[2, 0].set_ylim(ylim)
        arr[2, 0].semilogy(self.continuationParameter, 1e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].set_title('Velocity deviation at full period')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_xdot, linewidth=linewidth,c=self.plottingColors['tripleLine'][0], label='$|\dot{x}(T) - \dot{x}(0)|$')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_ydot, linewidth=linewidth,c=self.plottingColors['tripleLine'][1], label='$|\dot{y}(T) - \dot{y}(0)|$')
        arr[1, 1].legend(frameon=True, loc='lower right')
        arr[1, 1].set_xlim(xlim)
        arr[1, 1].set_ylim(ylim)

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].set_title('Maximum number of corrections')
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim([0, 150])
        arr[1, 0].plot(self.continuationParameter, self.numberOfIterations, linewidth=linewidth, c=self.plottingColors['singleLine'],label='Number of corrections')
        arr[1, 0].legend(frameon=True, loc='upper right')

        arr[2, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 1].xaxis.set_ticks(xticks)
        arr[2, 1].set_title('Maximum collocation segment error')
        arr[2, 1].set_xlim(xlim)
        arr[2, 1].set_ylim([10e-15,1.0e-5])
        arr[2, 1].tick_params(axis='y', labelcolor=self.plottingColors['tripleLine'][0])
        arr[2, 1].semilogy(self.continuationParameter, self.maxSegmentError, linewidth=linewidth, c=self.plottingColors['tripleLine'][0])
        ax2 = arr[2, 1].twinx()
        ax2.tick_params(axis='y', labelcolor=self.plottingColors['tripleLine'][1])
        ax2.plot(self.continuationParameter, self.numberOfCollocationPoints, linewidth=linewidth,color=self.plottingColors['tripleLine'][1])
        ax2.set_ylim([0,70])
        ax2.set_xlim(xlim)
        ax2.grid(b=None)


        arr[0, 0].set_ylabel('$||F||$ [-]')
        arr[0, 1].set_ylabel('$\Delta \mathbf{R}$ [-]')
        arr[2, 0].set_ylabel('max($e_{i}$) - min($e_{i}$)[-]')
        arr[1, 1].set_ylabel('$\Delta \mathbf{V}$ [-]')
        arr[1, 0].set_ylabel('Number of iterations [-]')
        arr[2, 1].set_ylabel('max($e_{i}$)')
        ax2.set_ylabel('Number of collocation points [-]')
        if self.varyingQuantity == 'xcor':
            arr[2, 0].set_xlabel('x [-]')
            arr[2, 1].set_xlabel('x [-]')
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
        ax2.grid(False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Periodicity constraints verification',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Periodicity constraints verification', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Periodicity constraints verification', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
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

        xticks = (np.linspace(min(self.continuationParameter), max(self.continuationParameter), num=self.numberOfXTicks))

        l1 = [abs(entry) for entry in self.lambda1]
        l2 = [abs(entry) for entry in self.lambda2]
        l3 = [abs(entry) for entry in self.lambda3]
        l4 = [abs(entry) for entry in self.lambda4]
        l5 = [abs(entry) for entry in self.lambda5]
        l6 = [abs(entry) for entry in self.lambda6]

        arr[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 0].xaxis.set_ticks(xticks)
        print(len(self.continuationParameter))

        arr[0, 0].semilogy(self.continuationParameter, l1, c=self.plottingColors['lambda1'])
        arr[0, 0].semilogy(self.continuationParameter, l2, c=self.plottingColors['lambda2'])
        arr[0, 0].semilogy(self.continuationParameter, l3, c=self.plottingColors['lambda3'])
        arr[0, 0].semilogy(self.continuationParameter, l4, c=self.plottingColors['lambda4'])
        arr[0, 0].semilogy(self.continuationParameter, l5, c=self.plottingColors['lambda5'])
        arr[0, 0].semilogy(self.continuationParameter, l6, c=self.plottingColors['lambda6'])
        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_ylim([1e-4, 1e4])
        arr[0, 0].set_title('$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        arr[0, 0].set_ylabel('Eigenvalues module [-]')

        d = [abs(entry - 1) for entry in self.D]
        arr[0, 1].semilogy(self.continuationParameter, d, c=self.plottingColors['singleLine'], linewidth=1)
        arr[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 1].xaxis.set_ticks(xticks)
        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_ylim([1e-14, 1e-6])
        arr[0, 1].set_ylabel('$| 1 - Det(\mathbf{M}) |$ [-]')
        arr[0, 1].set_title('Error in determinant ')
        arr[0, 1].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')


        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].scatter(self.continuationParameter, self.orderOfLinearInstability, s=size, c=self.plottingColors['singleLine'])
        arr[1, 0].set_ylabel('Order of linear instability [-]')
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim([0, 3])

        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].set_xlim(xlim)
        l3zoom = [abs(entry - 1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[1, 1].semilogy(self.continuationParameter, l3zoom, c=self.plottingColors['doubleLine'][0], linewidth=1)
        arr[1, 1].semilogy(self.continuationParameter, l4zoom, c=self.plottingColors['doubleLine'][1], linewidth=1, linestyle=':')
        arr[1, 1].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        if self.varyingQuantity == 'Hamiltonian':
            arr[1, 0].set_xlabel('$H_{lt}$ [-]')
            arr[1, 1].set_xlabel('$H_{lt}$ [-]')
        if self.varyingQuantity == 'xcor':
            arr[1, 0].set_xlabel('x [-]')
            arr[1, 1].set_xlabel('x [-]')



        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1, 1].set_ylabel(' $||\lambda_3|-1|$ [-]')
        arr[1, 1].set_title('Error in eigenvalue pair denoting periodicity')

        arr[1, 0].set_title('Order of linear instability')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Monodromy matrix eigensystem validation',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
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

    def plot_stability(self):
        unit_circle_1 = plt.Circle((0, 0), 1, color='grey', fill=False)
        unit_circle_2 = plt.Circle((0, 0), 1, color='grey', fill=False)

        size = 7

        xlim = [min(self.continuationParameter), max(self.continuationParameter)]
        xticks = (np.linspace(min(self.Hlt), max(self.Hlt), num=self.numberOfXTicks))



        f, arr = plt.subplots(3, 3, figsize=self.figSize)


        arr[0, 0].scatter(np.real(self.lambda1), np.imag(self.lambda1), c=self.plottingColors['lambda1'], s=size)
        arr[0, 0].scatter(np.real(self.lambda6), np.imag(self.lambda6), c=self.plottingColors['lambda6'], s=size)
        arr[0, 0].set_xlim([0, 3000])
        arr[0, 0].set_ylim([-1000, 1000])
        arr[0, 0].set_title('$\lambda_1, 1/\lambda_1$')
        arr[0, 0].set_xlabel('Re [-]')
        arr[0, 0].set_ylabel('Im [-]')

        arr[0, 1].scatter(np.real(self.lambda2), np.imag(self.lambda2), c=self.plottingColors['lambda2'], s=size)
        arr[0, 1].scatter(np.real(self.lambda5), np.imag(self.lambda5), c=self.plottingColors['lambda5'], s=size)
        arr[0, 1].set_xlim([-8, 2])
        arr[0, 1].set_ylim([-4, 4])
        arr[0, 1].set_title('$\lambda_2, 1/\lambda_2$')
        arr[0, 1].set_xlabel('Re [-]')
        arr[0, 1].add_artist(unit_circle_1)


        arr[0, 2].scatter(np.real(self.lambda3), np.imag(self.lambda3), c=self.plottingColors['lambda3'], s=size)
        arr[0, 2].scatter(np.real(self.lambda4), np.imag(self.lambda4), c=self.plottingColors['lambda4'], s=size)
        arr[0, 2].set_xlim([-1.5, 1.5])
        arr[0, 2].set_ylim([-1, 1])
        arr[0, 2].set_title('$\lambda_3, 1/\lambda_3$')
        arr[0, 2].set_xlabel('Re [-]')
        arr[0, 2].add_artist(unit_circle_2)

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].scatter(self.continuationParameter, np.angle(self.lambda1, deg=True), c=self.plottingColors['lambda1'], s=size)
        arr[1, 0].scatter(self.continuationParameter, np.angle(self.lambda6, deg=True), c=self.plottingColors['lambda6'], s=size)
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim([-180, 180])
        arr[1, 0].set_ylabel('Phase [$^\circ$]')

        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].scatter(self.continuationParameter, np.angle(self.lambda2, deg=True), c=self.plottingColors['lambda2'], s=size)
        arr[1, 1].scatter(self.continuationParameter, np.angle(self.lambda5, deg=True), c=self.plottingColors['lambda5'], s=size)
        arr[1, 1].set_xlim(xlim)
        arr[1, 1].set_ylim([-180, 180])

        arr[1, 2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 2].xaxis.set_ticks(xticks)
        arr[1, 2].scatter(self.continuationParameter, np.angle(self.lambda3, deg=True), c=self.plottingColors['lambda3'], s=size)
        arr[1, 2].scatter(self.continuationParameter, np.angle(self.lambda4, deg=True), c=self.plottingColors['lambda4'], s=size)
        arr[1, 2].set_xlim(xlim)
        arr[1, 2].set_ylim([-180, 180])

        arr[2, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 0].xaxis.set_ticks(xticks)
        arr[2, 0].semilogy(self.continuationParameter, self.v1, c=self.plottingColors['lambda6'])
        arr[2, 0].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 0].set_xlim(xlim)
        arr[2, 0].set_ylim([1e-1, 1e4])
        arr[2, 0].set_ylabel('Stability index [-]')
        arr[2, 0].set_title('$v_1$')

        arr[2, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 1].xaxis.set_ticks(xticks)
        arr[2, 1].semilogy(self.continuationParameter, self.v2, c=self.plottingColors['lambda5'])
        arr[2, 1].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 1].set_xlim(xlim)
        arr[2, 1].set_ylim([1e-1, 1e1])
        arr[2, 1].set_title('$v_2$')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            xlabel = '$H_{lt}$ [-]'
        if self.varyingQuantity == 'Acceleration':
            xlabel = '$a_{lt}$ [-]'
        if self.varyingQuantity == 'Alpha':
            xlabel = '$\\alpha$ [-]'


        arr[2, 0].set_xlabel(xlabel)
        arr[2, 1].set_xlabel(xlabel)
        arr[2, 2].set_xlabel(xlabel)

        arr[2, 2].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 2].xaxis.set_ticks(xticks)
        arr[2, 2].semilogy(self.continuationParameter, self.v3, c=self.plottingColors['lambda4'])
        arr[2, 2].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 2].set_xlim(xlim)
        arr[2, 2].set_ylim([1e-1, 1e1])
        arr[2, 2].set_title('$v_3$')

        for i in range(3):
            for j in range(3):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Eigenvalues $\lambda_i$ \& stability indices $v_i$',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Eigenvalues $\lambda_i$ \& stability indices $v_i$', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Eigenvalues $\lambda_i$ \& stability indices $v_i$', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, bbox_inches='tight')


        pass

    def plot_continuation_procedure(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        xlim = [1,len(self.orbitsId)]
        ylimSpacing = (max(self.Hlt)-min(self.Hlt))*0.05


        arr[0,0].plot(self.orbitsId,self.Hlt,c=self.plottingColors['singleLine'], linewidth=1,label='$H_{lt}$ [-]')
        arr[0,0].set_xlim(xlim)
        arr[0,0].set_ylim(min(self.Hlt)-ylimSpacing,max(self.Hlt)+ylimSpacing)
        arr[0,0].set_title('$H_{lt}$ evolution')
        arr[0,0].set_xlabel('orbit Number [-]')
        arr[0,1].set_ylabel('$H_{lt}$ [-]')
        arr[0,0].legend(frameon=True, loc='upper right')




        arr[0,1].plot(self.orbitsId,self.alphaContinuation,c=self.plottingColors['singleLine'], linewidth=1,label='$\\alpha$ [-]')
        arr[0,1].set_xlim(xlim)
        arr[0,1].set_ylim([0, 2*np.pi])
        arr[0,1].set_title('$\\alpha$ evolution')
        arr[0,1].set_xlabel('orbit Number [-]')
        arr[0,1].set_ylabel('$\\alpha$ [-]')
        arr[0,1].legend(frameon=True, loc='upper right')



        arr[1,0].plot(self.orbitsId,self.accelerationContinuation,c=self.plottingColors['singleLine'], linewidth=1,label='$a_{lt}$ [-]')
        arr[1,0].set_xlim(xlim)
        arr[1,0].set_ylim([0, 0.1])
        arr[1,0].set_title('$a_{lt}$ evolution')
        arr[1,0].set_xlabel('orbit Number [-]')
        arr[1,0].set_ylabel('$a_{lt}$ [-]')
        arr[1,0].legend(frameon=True, loc='upper right')



        arr[1,1].plot(self.orbitsId,self.x,c=self.plottingColors['tripleLine'][0], linewidth=1,label='$x$ [-]')
        arr[1,1].plot(self.orbitsId,self.y,c=self.plottingColors['tripleLine'][1], linewidth=1,label='$y$ [-]')
        #arr[1,1].plot(self.orbitsId,self.phase,c=self.plottingColors['tripleLine'][2], linewidth=1)
        arr[1,1].set_xlim(xlim)
        arr[1,1].set_ylim([-1,1])
        arr[1,1].set_title('Spatial and phase evolution')
        arr[1,1].set_xlabel('orbit Number [-]')
        arr[1,1].set_ylabel('$x$ [-], $y$ [-]')


        ax2 = arr[1, 1].twinx()
        ax2.tick_params(axis='phase [-]', labelcolor=self.plottingColors['tripleLine'][2])
        ax2.plot(self.orbitsId, self.phase, linewidth=1,color=self.plottingColors['tripleLine'][2],label='$\\phi$ [-]')
        ax2.set_ylim([0, 2*np.pi])
        ax2.set_xlim(xlim)
        ax2.grid(b=None)
        arr[1,1].legend(frameon=True, loc='upper right')
        ax2.legend(frameon=True, loc='lower right')




        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Numerical continuation validation ',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Numerical continuation validation', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Numerical continuation validation', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, bbox_inches='tight')



        pass

    def plot_increment_of_orbits(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        xlim = [1, len(self.orbitsId)]

        arr[0, 0].plot(self.orbitsId, self.x, c=self.plottingColors['doubleLine'][0], linewidth=1, label='$x$ [-]')
        arr[0, 0].plot(self.orbitsId, self.y, c=self.plottingColors['doubleLine'][1], linewidth=1, label='$y$ [-]')

        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_title('Coordinate evolution of initial condition')
        arr[0, 0].set_xlabel('orbit Number [-]')
        arr[0, 0].set_ylabel('$x$ [-], $y$ [-]')
        arr[0, 0].legend(frameon=True, loc='upper right')

        arr[0, 1].plot(self.orbitsId, self.xPhaseHalf, c=self.plottingColors['doubleLine'][0], linewidth=1, label='$x$ [-]')
        arr[0, 1].plot(self.orbitsId, self.yPhaseHalf, c=self.plottingColors['doubleLine'][1], linewidth=1, label='$y$ [-]')

        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_title('Coordinate evolution of $\\frac{\\phi}{2}$')
        arr[0, 1].set_xlabel('orbit Number [-]')
        arr[0, 1].set_ylabel('$x$ [-], $y$ [-]')
        arr[0, 1].legend(frameon=True, loc='upper right')

        xIncrement = []
        yIncrement = []
        normIncrement = []
        xIncrementPhaseHalf = []
        yIncrementPhaseHalf = []
        normIncrementPhaseHalf = []

        orbitIdNew = []
        for i in range(len(self.orbitsId)):
            if i > 0:
                xIncrement.append(self.x[i]-self.x[i-1])
                yIncrement.append(self.y[i]-self.y[i-1])
                normIncrement.append(  np.sqrt( (self.x[i]-self.x[i-1])**2 + ( self.y[i]-self.y[i-1])**2 ))

                xIncrementPhaseHalf.append(self.xPhaseHalf[i] - self.xPhaseHalf[i - 1])
                yIncrementPhaseHalf.append(self.yPhaseHalf[i] - self.yPhaseHalf[i - 1])
                normIncrementPhaseHalf.append( np.sqrt(( self.xPhaseHalf[i]-self.xPhaseHalf[i-1])**2 + ( self.yPhaseHalf[i]-self.yPhaseHalf[i-1])**2) )
                orbitIdNew.append(i)

        #arr[1, 0].plot(orbitIdNew, xIncrement, c=self.plottingColors['tripleLine'][0], linewidth=1,label='$\\Delta x$ [-]')
        #arr[1, 0].plot(orbitIdNew, yIncrement, c=self.plottingColors['tripleLine'][1], linewidth=1,label='$\\Delta y$ [-]')
        arr[1, 0].semilogy(orbitIdNew, normIncrement, c=self.plottingColors['tripleLine'][2], linewidth=1,label='$\\Delta R$ [-]')
        arr[1,1].set_ylim([1.0e-5,1.0e-3])

        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_title('Increment evolution of initial condition')
        arr[1, 0].set_xlabel('orbit Number [-]')
        arr[1, 0].set_ylabel('$\\Delta x$ [-], $\\Delta y$ [-], $\\Delta R$ [-]')
        arr[1, 0].set_ylabel('$\\Delta x$ [-]')

        arr[1, 0].legend(frameon=True, loc='upper right')

        #arr[1, 1].plot(orbitIdNew, xIncrementPhaseHalf, c=self.plottingColors['tripleLine'][0], linewidth=1,label='$\\Delta x$ [-]')
        #arr[1, 1].plot(orbitIdNew, yIncrementPhaseHalf, c=self.plottingColors['tripleLine'][1], linewidth=1,label='$\\Delta y$ [-]')
        arr[1, 1].semilogy(orbitIdNew, normIncrementPhaseHalf, c=self.plottingColors['tripleLine'][2], linewidth=1,label='$\\Delta R$ [-]')

        arr[1, 1].set_xlim(xlim)
        arr[1,1].set_ylim([1.0e-5,1.0e-3])

        arr[1, 1].set_title('Increment evolution of $\\frac{\\phi}{2}$')
        arr[1, 1].set_xlabel('orbit Number [-]')
        arr[1, 1].set_ylabel('$\\Delta x$ [-], $\\Delta y$ [-], $\\Delta R$ [-]')
        arr[1, 1].set_ylabel('$\\Delta R$ [-]')

        arr[1, 1].legend(frameon=True, loc='upper right')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        print('1300')
        print(normIncrement[1299])
        print('1350')
        print(normIncrement[1349])
        print('1400')
        print(normIncrement[1399])
        print('1450')
        print(normIncrement[1449])
        print('1500')
        print(normIncrement[1499])

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Spatial evolution analysis ',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Spatial evolution analysis ', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.1f}".format(self.accelerationMagnitude)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + ' - Spatial evolution analysis ', size=self.suptitleSize)



        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, dpi=self.dpi,
                            bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, dpi=self.dpi,
                            bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, dpi=self.dpi,
                            bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                    "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, bbox_inches='tight')

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

                            #display_periodic_solutions.plot_families()
                            #display_periodic_solutions.plot_periodicity_validation()
                            #display_periodic_solutions.plot_monodromy_analysis()
                            #display_periodic_solutions.plot_stability()
                            #display_periodic_solutions.plot_continuation_procedure()
                            display_periodic_solutions.plot_increment_of_orbits()


                            del display_periodic_solutions

