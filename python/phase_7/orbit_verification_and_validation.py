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
    def __init__(self,orbit_type, lagrange_point_nr,  acceleration_magnitude, alpha, Hamiltonian, varying_quantity, low_dpi, \
                 plot_as_x_coordinate, plot_as_family_number):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.Hamiltonian = Hamiltonian
        self.beta = 0.0
        self.varyingQuantity = varying_quantity
        self.lowDPI = low_dpi
        self.plotXCoordinate = plot_as_x_coordinate
        self.plotFamilyNumbers = plot_as_family_number


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
        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            quantity1 = self.accelerationMagnitude
            quantity2 = self.alpha
            quantity3 = self.beta

        if self.varyingQuantity == 'Acceleration':
            quantity1 = self.alpha
            quantity2 = self.beta
            quantity3 = self.Hamiltonian

        if self.varyingQuantity == 'Alpha':
            quantity1 = self.accelerationMagnitude
            quantity2 = self.beta
            quantity3 = self.Hamiltonian


        self.continuation_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                + '_' + str("{:12.11f}".format(quantity1)) + '_' + \
                str("{:12.11f}".format(quantity2)) + '_' + \
                str("{:12.11f}".format(quantity3)) + '_states_continuation.txt')

        self.correction_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                         + '_' + str("{:12.11f}".format(quantity1)) + '_' + \
                                         str("{:12.11f}".format(quantity2)) + '_' + \
                                         str("{:12.11f}".format(quantity3)) + '_differential_correction.txt')

        self.monodromy_fileName = str('L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                       + '_' + str("{:12.11f}".format(quantity1)) + '_' + \
                                       str("{:12.11f}".format(quantity2)) + '_' + \
                                       str("{:12.11f}".format(quantity3)) + '_initial_conditions.txt')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            statesContinuation_df = load_states_continuation(self.hamiltonian_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.hamiltonian_filepath + self.correction_fileName)
            initial_conditions_incl_m_df = load_initial_conditions_augmented_incl_M(self.hamiltonian_filepath + self.monodromy_fileName)

        if self.varyingQuantity == 'Acceleration':
            statesContinuation_df = load_states_continuation(self.acceleration_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.acceleration_filepath + self.correction_fileName)
            initial_conditions_incl_m_df = load_initial_conditions_augmented_incl_M(self.acceleration_filepath + self.monodromy_fileName)

        if self.varyingQuantity == 'Alpha':
            statesContinuation_df = load_states_continuation(self.alpha_filepath + self.continuation_fileName)
            differentialCorrections_df = load_differential_correction(self.alpha_filepath + self.correction_fileName)
            initial_conditions_incl_m_df = load_initial_conditions_augmented_incl_M(self.alpha_filepath + self.monodromy_fileName)

            #print(statesContinuation_df['alpha'][320:361])

            counter = 0.0
            for i in range(len(statesContinuation_df['orbitID'])):
                statesContinuation_df['orbitID'][i] = counter
                counter = counter + 1.0



        # Generate the lists with hamiltonians, periods and number of iterations and deviations after convergence
        self.Hlt = []
        self.alphaContinuation = []
        self.alphaContinuationRad = []
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

        self.xdot = []
        self.ydot = []

        self.numberOfCollocationPoints = []
        for row in statesContinuation_df.iterrows():
            self.orbitsId.append(row[1][0]+1)
            self.Hlt.append(row[1][1])
            self.x.append(row[1][3])
            self.phase.append(compute_phase(row[1][3],row[1][4],self.lagrangePointNr))
            self.y.append(row[1][4])
            self.accelerationContinuation.append(row[1][9])
            self.alphaContinuation.append(row[1][10])
            self.alphaContinuationRad.append(row[1][10]/180.0*np.pi)
            self.numberOfCollocationPoints.append(row[1][13])

            self.xdot.append(row[1][6])
            self.ydot.append(row[1][7])

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
        if self.plotXCoordinate == False and self.plotFamilyNumbers == False:
            if self.varyingQuantity == 'Hamiltonian':
                self.continuationParameter = self.Hlt
                self.colorbarLabel = '$H_{lt}$ [-]'
            if self.varyingQuantity == 'Acceleration':
                self.continuationParameter = self.accelerationContinuation
                self.colorbarLabel = '$a_{lt}$ [-]'
            if self.varyingQuantity == 'Alpha':
                self.continuationParameter = self.alphaContinuation
                self.colorbarLabel = '$\\alpha$ [$rad$]'
        elif self.plotXCoordinate == True and self.plotFamilyNumbers == True:
            print('BOTH X COORDINATE AND FAMILY NUMBERS HAVE BEEN SELECTED AS CONTINUATION PARAMETERS, TAKE ORBITID AS CONTINUATION PARAMETER')
            self.continuationParameter = self.x
        elif self.plotXCoordinate == True and self.plotFamilyNumbers == False:
             self.continuationParameter = self.x
             self.colorbarLabel = '$x$ [-]'
        else:
            self.continuationParameter = self.orbitsId
            self.colorbarLabel = 'Orbit Number [-]'

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
            if self.varyingQuantity == 'Hamiltonian':
                orbitDFString = '../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
            + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:12.11f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.beta))+ '_' + str("{:12.11f}".format(self.Hlt[i])) + '_.txt'
            if self.varyingQuantity == 'Acceleration':
                orbitDFString = '../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + str(
                    self.orbitType) \
                                + '_' + str("{:12.11f}".format(self.accelerationContinuation[i])) + '_' + \
                                str("{:12.11f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.beta)) + '_' + str("{:12.11f}".format(self.Hlt[i])) + '_.txt'
            if self.varyingQuantity == 'Alpha':
                orbitDFString = '../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + str(
                    self.orbitType) + '_' + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' + \
                                str("{:12.11f}".format(self.alphaContinuation[i])) + '_' + str("{:12.11f}".format(self.beta)) + '_' + str("{:12.11f}".format(self.Hlt[i])) + '_.txt'

            orbit_df = load_orbit_augmented(orbitDFString)

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
        counter_temp = 0
        for row in initial_conditions_incl_m_df.iterrows():
            M = np.matrix(
                [list(row[1][12:18]), list(row[1][22:28]), list(row[1][32:38]), list(row[1][42:48]), list(row[1][52:58]),
                 list(row[1][62:68])])

            eigenvalue = np.linalg.eigvals(M)

            sorting_indices = [-1, -1, -1, -1, -1, -1]
            idx_in_plane = []
            idx_manifolds = []
            idx_out_plane = []


            if counter_temp > 1938 and counter_temp < 1959:
                print ('family member: ' + str(counter_temp))
                print ('M: ' + str(M))
                print('eigenvalues: ' + str(eigenvalue))
                print('sorting_indices: ' + str(sorting_indices))

            if counter_temp > 2035 and counter_temp < 2039:
                print('member ' + str(counter_temp) + ' eigenvalues: ' + str(eigenvalue))

            # Find indices of the first pair of real eigenvalue equal to one
            for idx, l in enumerate(eigenvalue):
                if abs(l.imag) < self.maxEigenvalueDeviation:
                    if abs(l.real - 1.0) < self.maxEigenvalueDeviation:
                        if sorting_indices[2] == -1:
                            sorting_indices[2] = idx
                            idx_in_plane.append(idx)
                        elif sorting_indices[3] == -1:
                            sorting_indices[3] = idx
                            idx_in_plane.append(idx)

            # if counter_temp == 1011:
            #     print('sorting_indices: ' + str(sorting_indices))
            #     print('idx_in_plane: ' + str(idx_in_plane))
            #     print('idx_manifolds: ' + str(idx_manifolds))
            #     print('idx_out_plane: ' + str(idx_out_plane))

            no_manifolds_on_positive_axes = True
            unstable_manifold_on_negative_axes = False

            for idx, l in enumerate(eigenvalue):
                #Check if it is a real eigenvector with magnitude larger than 1.0
                if abs(l.imag) < self.maxEigenvalueDeviation and abs(abs(l)-1.0) > self.maxEigenvalueDeviation:
                    if l.real > 0.0:
                        no_manifolds_on_positive_axes = False

            # if no_manifolds_on_positive_axes == True:
            #     print('counter_temp: ' + str(counter_temp) + ' There are no Manifolds on positive_axes!!')



            # Find indices of the pair of largest/smallest real eigenvalue (corresponding to the unstable/stable subspace)
            for idx, l in enumerate(eigenvalue):
                if idx == (sorting_indices[2] or sorting_indices[3]):
                    continue
                if abs(l.imag) < self.maxEigenvalueDeviation:
                    if abs(l.real) == max(abs(eigenvalue.real)):
                            sorting_indices[0] = idx
                            idx_manifolds.append(idx)
                            if l.real < 0.0:
                                #print('counter_temp:' + str(counter_temp) + 'UNSTABLE MANIFOLD ON NEGATIVE AXES')
                                unstable_manifold_on_negative_axes = True
                    elif abs(abs(l.real) - 1.0 / max(abs(eigenvalue.real))) < self.maxEigenvalueDeviation:
                            sorting_indices[5] = idx
                            idx_manifolds.append(idx)

            # if counter_temp == 1011:
            #     print('sorting_indices: ' + str(sorting_indices))
            #     print('idx_in_plane: ' + str(idx_in_plane))
            #     print('idx_manifolds: ' + str(idx_manifolds))
            #     print('idx_out_plane: ' + str(idx_out_plane))

            missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))

            # if counter_temp == 1011:
            #     print('sorting_indices: ' + str(sorting_indices))
            #     print('missing_indices: ' + str(missing_indices))

            if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                sorting_indices[1] = missing_indices[0]
                sorting_indices[4] = missing_indices[1]
                idx_out_plane.append(missing_indices[0])
                idx_out_plane.append(missing_indices[1])

            else:
                sorting_indices[1] = missing_indices[1]
                sorting_indices[4] = missing_indices[0]
                idx_out_plane.append(missing_indices[1])
                idx_out_plane.append(missing_indices[0])

            # if counter_temp == 1011:
            #     print('sorting_indices: ' + str(sorting_indices))
            #     print('idx_in_plane: ' + str(idx_in_plane))
            #     print('idx_manifolds: ' + str(idx_manifolds))
            #     print('idx_out_plane: ' + str(idx_out_plane))
            #
            #     print('len(sorting_indices): ' + str(len(sorting_indices)))
            #     print('len(set(sorting_indices)): ' + str(len(set(sorting_indices))))

            if len(sorting_indices) > len(set(sorting_indices)) or len(idx_in_plane) == 1:
                print('\nWARNING: SORTING INDEX IS NOT UNIQUE FOR ' + self.orbitType + ' AT L' + str(
                     self.lagrangePointNr) + 'counter_temp is: ' + str(counter_temp) )

                # Determine if all eigenvalues are on real axis or unit circle
                eigenvalues_module_1 = 0

                for idx, l in enumerate(eigenvalue):
                    # print('l: '  + str(l))
                    # print('abs(l): '  + str(abs(l)))
                    # print('self.maxEigenvalueDeviation: '  + str(self.maxEigenvalueDeviation))

                    if abs(abs(l) - 1.0) < self.maxEigenvalueDeviation:
                        eigenvalues_module_1 = eigenvalues_module_1 + 1


                #print('eigenvalues_module_1: ' + str(eigenvalues_module_1))

                # If not the case and manifolds are present: assume that in_center_subspace exceeds threshold deviation
                if eigenvalues_module_1 != 6 and (len(idx_manifolds) == 2 or len(idx_in_plane) == 1):

                    # restart setting but start with invariant manifolds

                    sorting_indices = [-1, -1, -1, -1, -1, -1]
                    idx_in_plane = []
                    idx_manifolds = []
                    idx_out_plane = []

                    # if counter_temp == 1011:
                    #     print('resetted the sorting, start with manifolds')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                    # Find indices of the pair of largest/smallest real eigenvalue (corresponding to the unstable/stable subspace)
                    for idx, l in enumerate(eigenvalue):
                        if abs(l.imag) < self.maxEigenvalueDeviation:
                            if abs(l.real) == max(abs(eigenvalue.real)):
                                sorting_indices[0] = idx
                                idx_manifolds.append(idx)
                            elif abs(abs(l.real) - 1.0 / max(abs(eigenvalue.real))) < self.maxEigenvalueDeviation:
                                sorting_indices[5] = idx
                                idx_manifolds.append(idx)

                    # if counter_temp == 1011:
                    #     print('manifolds selected')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                    # Select the in-plane moduli with relaxed constraints
                    for idx, l in enumerate(eigenvalue):
                        if idx == (sorting_indices[0] or sorting_indices[5]):
                            continue
                        if abs(l.real - 1.0) < 3.0 * self.maxEigenvalueDeviation:
                            if sorting_indices[2] == -1:
                                sorting_indices[2] = idx
                                idx_in_plane.append(idx)
                            elif sorting_indices[3] == -1:
                                sorting_indices[3] = idx
                                idx_in_plane.append(idx)

                    # if counter_temp == 1011:
                    #     print('in-plane moduli')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                    # Select the in-plane moduli with relaxed constraints
                    missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))

                    # if counter_temp == 1011:
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('missing_indices: ' + str(missing_indices))

                    if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                        sorting_indices[1] = missing_indices[0]
                        sorting_indices[4] = missing_indices[1]
                        idx_out_plane.append(missing_indices[0])
                        idx_out_plane.append(missing_indices[1])

                    else:
                        sorting_indices[1] = missing_indices[1]
                        sorting_indices[4] = missing_indices[0]
                        idx_out_plane.append(missing_indices[1])
                        idx_out_plane.append(missing_indices[0])


                    # if counter_temp == 1011:
                    #     print('out-plane moduli')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                # If so: start with in-plane by selecting pair with real component closest to one!
                if eigenvalues_module_1 == 6:

                    # restart sorting setting
                    sorting_indices = [-1, -1, -1, -1, -1, -1]
                    idx_in_plane = []
                    idx_manifolds = []
                    idx_out_plane = []

                    # if counter_temp == 1266:
                    #     print('START ALTERNATIVE SELECTION!')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))
                    #     print('eigenvalues: ' + str(eigenvalue))

                    # Find indices of the in-plane component (with real value close to +1)
                    for idx, l in enumerate(eigenvalue):
                        if abs(l.real - 1.0) < self.maxEigenvalueDeviation:
                            if sorting_indices[2] == -1:
                                sorting_indices[2] = idx
                                idx_in_plane.append(idx)
                            elif sorting_indices[3] == -1:
                                sorting_indices[3] = idx
                                idx_in_plane.append(idx)

                    # if counter_temp == 1266:
                    #     print('in-plane moduli SELECTED')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                    previousLambda2  = self.lambda2[-1]
                    previousLambda5 = self.lambda5[-1]

                    PhaseDiscrepancy = 4.0*np.pi
                    idx_minimal_2 = 0
                    for idx, l in enumerate(eigenvalue):
                        if idx == (sorting_indices[2] or sorting_indices[3]):
                            continue
                        if abs(np.angle(previousLambda2) - np.angle(l)) < PhaseDiscrepancy:
                            PhaseDiscrepancy = abs(np.angle(previousLambda2) - np.angle(l))
                            idx_minimal_2 = idx

                    sorting_indices[1] = idx_minimal_2
                    idx_out_plane.append(idx_minimal_2)

                    PhaseDiscrepancy = 4.0 * np.pi
                    idx_minimal_5 = 0
                    for idx, l in enumerate(eigenvalue):
                        if idx == (sorting_indices[2] or sorting_indices[3]):
                            continue
                        if abs(np.angle(previousLambda5) - np.angle(l)) < PhaseDiscrepancy:
                            PhaseDiscrepancy = abs(np.angle(previousLambda2) - np.angle(l))
                            idx_minimal_5 = idx

                    sorting_indices[4] = idx_minimal_5
                    idx_out_plane.append(idx_minimal_5)

                    # if counter_temp == 1266:
                    #     print('out-plane moduli SELECTED')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

                    # Determine the manifolds!
                    missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))

                    # if counter_temp == 1266:
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('missing_indices: ' + str(missing_indices))

                    if eigenvalue.imag[missing_indices[0]] > eigenvalue.imag[missing_indices[1]]:
                        sorting_indices[0] = missing_indices[0]
                        sorting_indices[5] = missing_indices[1]
                        idx_manifolds.append(missing_indices[0])
                        idx_manifolds.append(missing_indices[1])

                    else:
                        sorting_indices[0] = missing_indices[1]
                        sorting_indices[5] = missing_indices[0]
                        idx_manifolds.append(missing_indices[1])
                        idx_manifolds.append(missing_indices[0])

                    # if counter_temp == 1266:
                    #     print('Manifolds SELECTED')
                    #     print('sorting_indices: ' + str(sorting_indices))
                    #     print('idx_in_plane: ' + str(idx_in_plane))
                    #     print('idx_manifolds: ' + str(idx_manifolds))
                    #     print('idx_out_plane: ' + str(idx_out_plane))

            # if counter_temp > 1011:
            #     print('unstable_manifold_on_negative_axes :' + str(unstable_manifold_on_negative_axes))
            #     print('no_manifolds_on_positive_axes :' + str(no_manifolds_on_positive_axes))
            #     print(eigenvalue)

            l2_180 = False
            if self.varyingQuantity == 'Hamiltonian' and self.lagrangePointNr == 2 and self.accelerationMagnitude > 0.09 and self.alpha > 170.0 and self.alpha < 190.0:
                l2_180 = True

            # In case there are positive real eigenvalues not on unit axes but negative out-of-plane real lamda's are selected
            if unstable_manifold_on_negative_axes == True and no_manifolds_on_positive_axes == False and l2_180 == False :
                sorting_indices = [-1, -1, -1, -1, -1, -1]
                idx_in_plane = []
                idx_manifolds = []
                idx_out_plane = []

                # print('SORT AGAIN WITH DIFFERENT RULE FOR MANIFOLDS!')
                # print('family member: ' + str(counter_temp))
                # print('M: ' + str(M))
                # print('eigenvalues: ' + str(eigenvalue))
                # print('sorting_indices: ' + str(sorting_indices))

                # Find indices of the first pair of real eigenvalue equal to one
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation:
                        if abs(l.real - 1.0) < self.maxEigenvalueDeviation:
                            if sorting_indices[2] == -1:
                                sorting_indices[2] = idx
                                idx_in_plane.append(idx)
                            elif sorting_indices[3] == -1:
                                sorting_indices[3] = idx
                                idx_in_plane.append(idx)

                # print('IN PLANE SELECTED!')
                # print('sorting_indices: ' + str(sorting_indices))
                # print('idx_in_plane: ' + str(idx_in_plane))
                # print('idx_manifolds: ' + str(idx_manifolds))
                # print('idx_out_plane: ' + str(idx_out_plane))

                minimum_lambda = 1.0e6
                minimum_idx = 0
                maximum_lambda = -10
                maximum_idx = 0

                # Find indices of the pair of largest/smallest real eigenvalue (corresponding to the unstable/stable subspace)
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation and l.real > 0.0:
                        #print(l.real)
                        if l.real < minimum_lambda:
                            minimum_idx = idx
                            minimum_lambda = l.real
                        if l.real > maximum_lambda:
                            maximum_idx = idx
                            maximum_lambda = l.real

                sorting_indices[0] = maximum_idx
                idx_manifolds.append(maximum_idx)
                sorting_indices[5] = minimum_idx
                idx_manifolds.append(minimum_idx)

                # print('MANIFOLDS SELECTED')
                # print('sorting_indices: ' + str(sorting_indices))
                # print('idx_in_plane: ' + str(idx_in_plane))
                # print('idx_manifolds: ' + str(idx_manifolds))
                # print('idx_out_plane: ' + str(idx_out_plane))

                missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))


                # print('sorting_indices: ' + str(sorting_indices))
                # print('missing_indices: ' + str(missing_indices))

                if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                    sorting_indices[1] = missing_indices[0]
                    sorting_indices[4] = missing_indices[1]
                    idx_out_plane.append(missing_indices[0])
                    idx_out_plane.append(missing_indices[1])

                else:
                    sorting_indices[1] = missing_indices[1]
                    sorting_indices[4] = missing_indices[0]
                    idx_out_plane.append(missing_indices[1])
                    idx_out_plane.append(missing_indices[0])

                # print('OUT OF PLANE SELECTED')
                # print('sorting_indices: ' + str(sorting_indices))
                # print('idx_in_plane: ' + str(idx_in_plane))
                # print('idx_manifolds: ' + str(idx_manifolds))
                # print('idx_out_plane: ' + str(idx_out_plane))

            if l2_180 == True and counter_temp == 2036:
                sorting_indices = [-1, -1, -1, -1, -1, -1]
                idx_in_plane = []
                idx_manifolds = []
                idx_out_plane = []

                # Find indices of the first pair of real eigenvalue equal to one
                ref_deviation = 80
                lambda1_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real-1.24) < ref_deviation:
                            ref_deviation = l.real-1.24
                            lambda1_sorting = lambda1_sorting

                sorting_indices[0] = lambda1_sorting

                ref_deviation = 80
                lambda2_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real + 0.22) < ref_deviation:
                            ref_deviation = l.real + 0.22
                            lambda2_sorting = idx

                sorting_indices[1] = lambda2_sorting


                ref_deviation = 80
                lambda3_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real -0.9986) < ref_deviation:
                            ref_deviation = l.real -0.9986
                            lambda3_sorting = idx

                sorting_indices[3] = lambda3_sorting

                ref_deviation = 80
                lambda4_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real - 1.0014) < ref_deviation:
                            ref_deviation =l.real - 1.0014
                            lambda4_sorting = idx

                sorting_indices[2] = lambda4_sorting

                ref_deviation = 80
                lambda5_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real +4.45) < ref_deviation:
                            ref_deviation =l.real +4.45
                            lambda5_sorting = idx

                sorting_indices[4] = lambda5_sorting

                ref_deviation = 80
                lambda6_sorting = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < 3.0 * self.maxEigenvalueDeviation:
                        if abs(l.real - 0.8042537) < ref_deviation:
                            ref_deviation = l.real - 0.8042537
                            lambda6_sorting = idx

                sorting_indices[5] = lambda6_sorting

                print('FINAL ADAP: ' + str(sorting_indices))


            if l2_180 == True and counter_temp > 1610 and counter_temp < 1629:

                sorting_indices = [-1, -1, -1, -1, -1, -1]
                idx_in_plane = []
                idx_manifolds = []
                idx_out_plane = []

                # Find indices of the first pair of real eigenvalue equal to one
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation or (
                            abs(l.imag) < 3.0 * self.maxEigenvalueDeviation and counter_temp > 2034):
                        if abs(l.real - 1.0) < 3.0 * self.maxEigenvalueDeviation:
                            if sorting_indices[2] == -1:
                                sorting_indices[2] = idx
                                idx_in_plane.append(idx)
                            elif sorting_indices[3] == -1:
                                sorting_indices[3] = idx
                                idx_in_plane.append(idx)

                # Find indices of the pair of largest/smallest positive real eigenvalue (corresponding to the unstable/stable subspace)
                max_lambda = -10
                max_index = 0
                minimum_lambda = 1000
                min_index = 0
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation and l.real > 0.0:
                            if l.real > max_lambda:
                                    max_lambda = l.real
                                    max_index = idx
                            if l.real < minimum_lambda:
                                    minimum_lambda = l.real
                                    min_index = idx

                sorting_indices[0] = max_index
                idx_manifolds.append(max_index)
                sorting_indices[5] = min_index
                idx_manifolds.append(min_index)

                missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))

                if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                    sorting_indices[1] = missing_indices[0]
                    sorting_indices[4] = missing_indices[1]
                    idx_out_plane.append(missing_indices[0])
                    idx_out_plane.append(missing_indices[1])

                else:
                    sorting_indices[1] = missing_indices[1]
                    sorting_indices[4] = missing_indices[0]
                    idx_out_plane.append(missing_indices[1])
                    idx_out_plane.append(missing_indices[0])






            # In case there are positive real eigenvalues not on unit axes but negative out-of-plane real lamda's are selected
            if unstable_manifold_on_negative_axes == True and no_manifolds_on_positive_axes == True and l2_180 == True:

                sorting_indices = [-1, -1, -1, -1, -1, -1]
                idx_in_plane = []
                idx_manifolds = []
                idx_out_plane = []

                if counter_temp > 2034:
                    print('L2 0.1 180 CASE!')
                    print('family member: ' + str(counter_temp))
                    print('M: ' + str(M))
                    print('eigenvalues: ' + str(eigenvalue))
                    print('sorting_indices: ' + str(sorting_indices))

                # Find indices of the first pair of real eigenvalue equal to one
                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation or (abs(l.imag) < 3.0*self.maxEigenvalueDeviation and counter_temp > 2034):
                        if abs(l.real - 1.0) < 3.0*self.maxEigenvalueDeviation:
                            if sorting_indices[2] == -1:
                                sorting_indices[2] = idx
                                idx_in_plane.append(idx)
                            elif sorting_indices[3] == -1:
                                sorting_indices[3] = idx
                                idx_in_plane.append(idx)

                if counter_temp > 2034:
                    print('IN PLANE SELECTED!')
                    print('sorting_indices: ' + str(sorting_indices))
                    print('idx_in_plane: ' + str(idx_in_plane))
                    print('idx_manifolds: ' + str(idx_manifolds))
                    print('idx_out_plane: ' + str(idx_out_plane))

                counter_negative_axes = 0

                for idx, l in enumerate(eigenvalue):
                    if abs(l.imag) < self.maxEigenvalueDeviation:
                        if abs(abs(l.real)-1.0) > self.maxEigenvalueDeviation:
                            counter_negative_axes = counter_negative_axes + 1

                minimum_lambda = 10
                maximum_lambda = -1.0e6

                if counter_temp > 2034:
                    print('Number of neg axes: ' + str(counter_negative_axes))
                    print('sorting_indices: ' + str(sorting_indices))
                    print('idx_in_plane: ' + str(idx_in_plane))
                    print('idx_manifolds: ' + str(idx_manifolds))
                    print('idx_out_plane: ' + str(idx_out_plane))

                if counter_negative_axes == 2:
                    for idx, l in enumerate(eigenvalue):
                        if abs(l.imag) < self.maxEigenvalueDeviation and l.real < 0.0:
                            #print(l.real)
                            if l.real < minimum_lambda:
                                minimum_idx = idx
                                minimum_lambda = l.real
                            if l.real > maximum_lambda:
                                maximum_idx = idx
                                maximum_lambda = l.real

                    sorting_indices[1] = maximum_idx
                    idx_out_plane.append(maximum_idx)
                    sorting_indices[4] = minimum_idx
                    idx_out_plane.append(minimum_idx)



                    missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))



                    if counter_temp > 2034:
                        print('SELECTED OUT PLANE: axes num ' + str(counter_negative_axes))
                        print('sorting_indices: ' + str(sorting_indices))
                        print('idx_in_plane: ' + str(idx_in_plane))
                        print('idx_manifolds: ' + str(idx_manifolds))
                        print('idx_out_plane: ' + str(idx_out_plane))
                        print('sorting_indices: ' + str(sorting_indices))
                        print('missing_indices: ' + str(missing_indices))

                    if eigenvalue.imag[missing_indices[0]] > eigenvalue.imag[missing_indices[1]]:
                        sorting_indices[0] = missing_indices[0]
                        sorting_indices[5] = missing_indices[1]
                        idx_manifolds.append(missing_indices[0])
                        idx_manifolds.append(missing_indices[1])

                    else:
                        sorting_indices[0] = missing_indices[1]
                        sorting_indices[5] = missing_indices[0]
                        idx_out_plane.append(missing_indices[1])
                        idx_out_plane.append(missing_indices[0])

                    if counter_temp > 2034:
                        print('SELECTED Manifolds: axes num ' + str(counter_negative_axes))
                        print('sorting_indices: ' + str(sorting_indices))
                        print('idx_in_plane: ' + str(idx_in_plane))
                        print('idx_manifolds: ' + str(idx_manifolds))
                        print('idx_out_plane: ' + str(idx_out_plane))
                        print(eigenvalue)



                if counter_negative_axes == 4:

                    for idx, l in enumerate(eigenvalue):
                        if abs(l.imag) < self.maxEigenvalueDeviation and l.real < 0.0:
                            #print(l.real)
                            if l.real < minimum_lambda:
                                minimum_idx = idx
                                minimum_lambda = l.real
                            if l.real > maximum_lambda:
                                maximum_idx = idx
                                maximum_lambda = l.real

                    sorting_indices[1] = maximum_idx
                    idx_manifolds.append(maximum_idx)
                    sorting_indices[4] = minimum_idx
                    idx_manifolds.append(minimum_idx)

                    missing_indices = sorted(list(set(list(range(-1, 6))) - set(sorting_indices)))

                    # print('sorting_indices: ' + str(sorting_indices))
                    # print('missing_indices: ' + str(missing_indices))

                    if eigenvalue.real[missing_indices[0]] > eigenvalue.real[missing_indices[1]]:
                        sorting_indices[0] = missing_indices[0]
                        sorting_indices[5] = missing_indices[1]
                        idx_out_plane.append(missing_indices[0])
                        idx_out_plane.append(missing_indices[1])

                    else:
                        sorting_indices[0] = missing_indices[1]
                        sorting_indices[5] = missing_indices[0]
                        idx_out_plane.append(missing_indices[1])
                        idx_out_plane.append(missing_indices[0])

            l2_300 = False
            if self.varyingQuantity == 'Hamiltonian' and self.lagrangePointNr == 2 and self.accelerationMagnitude > 0.09 and self.alpha > 290:
                l2_300 = True
            if counter_temp >1951 and counter_temp < 1955 and l2_300 == True:

                if counter_temp == 1952:
                    sorting_indices = [0, 5, 2,3, 4, 1]

                if counter_temp == 1953:
                    sorting_indices = [0, 5, 3,2, 4, 1]

                if counter_temp == 1954:
                    sorting_indices = [0, 5, 2,3, 4, 1]

                print('FINAL ADAP: ' + str(sorting_indices))



                ## Sort complex values

                # if len(idx_in_plane) != 2:
                #     print('len(idx_in_plane) != 2')
                #     idx_in_plane = []
                #     # Find indices of the first pair of real eigenvalue equal to one
                #     for idx, l in enumerate(eigenvalue):
                #         if abs(l.imag) < 2 * self.maxEigenvalueDeviation:
                #             if abs(l.real - 1.0) < 2 * self.maxEigenvalueDeviation:
                #                 if sorting_indices[2] == -1:
                #                     sorting_indices[2] = idx
                #                     idx_in_plane.append(idx)
                #                 elif sorting_indices[3] == -1:
                #                     sorting_indices[3] = idx
                #                     idx_in_plane.append(idx)
                #         print(sorting_indices)
                #
                # if len(idx_in_plane) == 2:
                #     print('len(idx_in_plane) == 2')
                #
                #     sorting_indices = [-1, -1, -1, -1, -1, -1]
                #     sorting_indices[2] = idx_in_plane[0]
                #     sorting_indices[3] = idx_in_plane[1]
                #     # Assume two times real one and two conjugate pairs
                #     for idx, l in enumerate(eigenvalue):
                #         # min(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_in_plane))], deg=True)))
                #         # if abs(np.angle(l, deg=True))%180 == min(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_in_plane))], deg=True)) %180):
                #         if l.real == eigenvalue[list(set(range(6)) - set(idx_in_plane))].real.max():
                #             if l.imag > 0:
                #                 sorting_indices[0] = idx
                #             elif l.imag < 0:
                #                 sorting_indices[5] = idx
                #         # if abs(np.angle(l, deg=True))%180 == max(abs(np.angle(eigenvalue[list(set(range(6)) - set(idx_in_plane))], deg=True)) %180):
                #         if l.real == eigenvalue[list(set(range(6)) - set(idx_in_plane))].real.min():
                #             if l.imag > 0:
                #                 sorting_indices[1] = idx
                #             elif l.imag < 0:
                #                 sorting_indices[4] = idx
                #         print(sorting_indices)
                #
                # if len(sorting_indices) > len(set(sorting_indices)):
                #     print('len(sorting_indices) > len(set(sorting_indices))')
                #
                #     print('\nWARNING: SORTING INDEX IS STILL NOT UNIQUE')
                #     # Sorting eigenvalues from largest to smallest norm, excluding real one
                #
                #     # Sorting based on previous phase
                #     if len(idx_in_plane) == 2:
                #         sorting_indices = [-1, -1, -1, -1, -1, -1]
                #         sorting_indices[2] = idx_in_plane[0]
                #         sorting_indices[3] = idx_in_plane[1]
                #
                #         # Assume two times real one and two conjugate pairs
                #         for idx, l in enumerate(eigenvalue[list(set(range(6)) - set(idx_in_plane))]):
                #             print(idx)
                #             if abs(l.real - self.lambda1[-1].real) == min(
                #                     abs(eigenvalue.real - self.lambda1[-1].real)) and abs(
                #                     l.imag - self.lambda1[-1].imag) == min(
                #                     abs(eigenvalue.imag - self.lambda1[-1].imag)):
                #                 sorting_indices[0] = idx
                #             if abs(l.real - self.lambda2[-1].real) == min(
                #                     abs(eigenvalue.real - self.lambda2[-1].real)) and abs(
                #                     l.imag - self.lambda2[-1].imag) == min(
                #                     abs(eigenvalue.imag - self.lambda2[-1].imag)):
                #                 sorting_indices[1] = idx
                #             if abs(l.real - self.lambda5[-1].real) == min(
                #                     abs(eigenvalue.real - self.lambda5[-1].real)) and abs(
                #                     l.imag - self.lambda5[-1].imag) == min(
                #                     abs(eigenvalue.imag - self.lambda5[-1].imag)):
                #                 sorting_indices[4] = idx
                #             if abs(l.real - self.lambda6[-1].real) == min(
                #                     abs(eigenvalue.real - self.lambda6[-1].real)) and abs(
                #                     l.imag - self.lambda6[-1].imag) == min(
                #                     abs(eigenvalue.imag - self.lambda6[-1].imag)):
                #                 sorting_indices[5] = idx
                #             print(sorting_indices)
                #
                #     pass
                # if (sorting_indices[1] and sorting_indices[4]) == -1:
                #     # Fill two missing values
                #     two_missing_indices = list(set(list(range(-1, 6))) - set(sorting_indices))
                #     if abs(eigenvalue[two_missing_indices[0]].real) > abs(eigenvalue[two_missing_indices[1]].real):
                #         sorting_indices[1] = two_missing_indices[0]
                #         sorting_indices[4] = two_missing_indices[1]
                #     else:
                #         sorting_indices[1] = two_missing_indices[1]
                #         sorting_indices[4] = two_missing_indices[0]
                #     print(sorting_indices)
                # if (sorting_indices[0] and sorting_indices[5]) == -1:
                #     # Fill two missing values
                #     two_missing_indices = list(set(list(range(-1, 6))) - set(sorting_indices))
                #     print(eigenvalue)
                #     print(sorting_indices)
                #     print(two_missing_indices)
                #     # TODO quick fix for extended v-l
                #     # if len(two_missing_indices)==1:
                #     #     print('odd that only one index remains')
                #     #     if sorting_indices[0] == -1:
                #     #         sorting_indices[0] = two_missing_indices[0]
                #     #     else:
                #     #         sorting_indices[5] = two_missing_indices[0]
                #     # sorting_indices = abs(eigenvalue).argsort()[::-1]
                #     if abs(eigenvalue[two_missing_indices[0]].real) > abs(eigenvalue[two_missing_indices[1]].real):
                #         sorting_indices[0] = two_missing_indices[0]
                #         sorting_indices[5] = two_missing_indices[1]
                #     else:
                #         sorting_indices[0] = two_missing_indices[1]
                #         sorting_indices[5] = two_missing_indices[0]
                #     print(sorting_indices)

                if len(sorting_indices) > len(set(sorting_indices)):
                    print('\nWARNING: SORTING INDEX IS STILL STILL NOT UNIQUE for counter temp: ' + str(counter_temp))
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
            self.v1.append((abs(eigenvalue[sorting_indices[0]]) + abs(eigenvalue[sorting_indices[5]])) / 2)
            self.v2.append((abs(eigenvalue[sorting_indices[1]]) + abs(eigenvalue[sorting_indices[4]])) / 2)
            self.v3.append((abs(eigenvalue[sorting_indices[2]]) + abs(eigenvalue[sorting_indices[3]])) / 2)
            self.D.append(np.linalg.det(M))

            counter_temp = counter_temp + 1

        # if self.alpha == 300:
        #     print('lambda2: final ' + str(self.lambda2[1011]))
        #     print('lambda5: final ' + str(self.lambda5[1011]))



        print('Index for bifurcations: ')
        print(self.orbitIdBifurcations)

        self.incrementPhaseNul = np.sqrt( (self.x[len(self.x)-2]-self.x[len(self.x)-1]) ** 2\
                                 + (self.y[len(self.y)-2]-self.y[len(self.y)-1]) ** 2)

        self.incrementPhaseHalf = np.sqrt((self.xPhaseHalf[len(self.xPhaseHalf) - 2] - self.xPhaseHalf[len(self.xPhaseHalf) - 1]) ** 2 \
                                         + (self.yPhaseHalf[len(self.yPhaseHalf) - 2] - self.yPhaseHalf[len(self.yPhaseHalf) - 1]) ** 2)

        print('== Check termination reason: ==')
        print('Number of members: ' + str(len(self.continuationParameter)))
        print('FM deviation x: ' + str(self.deviation_x[-1]))
        print('FM deviation y: ' + str(self.deviation_y[-1]))
        print('FM deviation xdot: ' + str(self.deviation_xdot[-1]))
        print('FM deviation ydot: ' + str(self.deviation_ydot[-1]))
        print('FM lambda3: ' + str(self.lambda3[-1]))
        print('FM D: ' + str(self.D[-1]))
        print('FM incrementPhaseNul: ' + str(self.incrementPhaseNul))
        print('FM incrementPhaseHalf: ' + str(self.incrementPhaseHalf))
        print('FM Change in increment: ' + str(self.incrementPhaseHalf/self.incrementPhaseNul))

        # Write statements! to show the reason
        if self.deviation_x[-1] > 1.0e-9:
            print('Continuation procedure terminated due to violation of X-coordinate periodicity condition')
        elif self.deviation_y[-1] > 1.0e-9:
            print('Continuation procedure terminated due to violation of Y-coordinate periodicity condition')
        elif self.deviation_xdot[-1] > 1.0e-9:
            print('Continuation procedure terminated due to violation of XDOT-coordinate periodicity condition')
        elif self.deviation_ydot[-1] > 1.0e-9:
            print('Continuation procedure terminated due to violation of YDOT-coordinate periodicity condition')
        elif (self.lagrangePointNr == 1 and np.abs(np.real(self.lambda3[-1]) - 1) > 1.0e-3 ) or (self.lagrangePointNr == 2 and np.abs(np.abs(self.lambda3[-1]) - 1) > 1.0e-3):
            print('Continuation procedure terminated due to error in lambda3 periodicity ')
        elif np.abs(self.D[-1] - 1) > 1.0e-3:
            print('Continuation procedure terminated due to error in determinant')
        elif np.abs(self.incrementPhaseHalf/self.incrementPhaseNul) < 0.1:
            print('Continuation procedure terminated due to Decrease in spacing of family')
        elif len(self.continuationParameter) == 3000 or len(self.continuationParameter) == 1000 or len(self.continuationParameter) == 4500:
            print('Continuation procedure terminated likely due to maximumNumberOfMembers')
        else:
            print('All termination conditions are adhered to')
            print('Continuation procedure terminated likely due to reversing of continuation condition!')


        print('====HAMILTONIAN RANGE===')
        print('Hlt min:' + str(min(self.Hlt)))
        print('Hlt min:' + str(max(self.Hlt)))




        # Print number of members
        # Print properties of final member

            # Phase increment
            # Error in eigenvalue of periodicity
            # Monodromy
            # if number is 2500
        # if none of them: unclear: due to reversing of continuation condition !!!!




        #  =========== Plot layout settings ===============

        # plot specific spacing properties
        if self.varyingQuantity == 'Hamiltonian':
            self.orbitSpacingFactor = 100
        elif self.varyingQuantity == 'Acceleration':
            self.orbitSpacingFactor = 1
        else:
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

        self.figSizeThird = (7 * (1 + np.sqrt(5)) / 2, 3.5*0.75)

        self.figSizeCont = (7 * (1 + np.sqrt(5)) / 2, 7.0*0.70)



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

        print(len(self.continuationParameter))

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        # Add bodies

        # Add libration point orbits


        if self.varyingQuantity == 'Hamiltonian':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.3f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Overview',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Overview', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.1f}".format(self.accelerationMagnitude))  + '$ - Overview', size=self.suptitleSize)




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

            if self.varyingQuantity == 'Hamiltonian':
                df = load_orbit('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(self.alpha)) + '_' \
                            + str("{:12.11f}".format(self.beta)) + '_' \
                            + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')
            if self.varyingQuantity == 'Acceleration':
                # print('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                #     + str("{:12.11f}".format(self.accelerationContinuation[i])) + '_' \
                #     + str("{:12.11f}".format(self.alpha)) + '_' \
                #     + str("{:12.11f}".format(self.beta)) + '_' \
                #     + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')
                df = load_orbit(
                    '../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:12.11f}".format(self.accelerationContinuation[i])) + '_' \
                    + str("{:12.11f}".format(self.alpha)) + '_' \
                    + str("{:12.11f}".format(self.beta)) + '_' \
                    + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')
            if self.varyingQuantity == 'Alpha':
                # print('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                #     + str("{:12.11f}".format(self.accelerationContinuation[i])) + '_' \
                #     + str("{:12.11f}".format(self.alpha)) + '_' \
                #     + str("{:12.11f}".format(self.beta)) + '_' \
                #     + str("{:12.11f}".format(self.Hlt[i])) + '_.txt')
                df = load_orbit(
                    '../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' \
                    + str("{:12.11f}".format(self.alphaContinuation[i])) + '_' \
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

        if self.varyingQuantity == 'Alpha':
            ax.set_xlim([-0.1,1.3])
            ax.set_ylim([-1.2,1.2])


        sm.set_array([])
        cax, kw = matplotlib.colorbar.make_axes([ax])

        cbar = plt.colorbar(sm, cax=cax, label=self.colorbarLabel, format='%1.4f', **kw)
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                fig.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

            else:
                fig.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                    "{:7.6f}".format(self.alpha)) + \
                            '_planar_projection.png', transparent=True, dpi=300, bbox_inches='tight')

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
        arr[0, 0].semilogy(self.continuationParameter, self.totalDeviationAfterConvergence, linewidth=linewidth, c=self.plottingColors['singleLine'],label='$||\\mathbf{F}||$')
        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_ylim(ylim)
        arr[0, 0].semilogy(self.continuationParameter, 1e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[0, 0].legend(frameon=True, loc='upper left')


        arr[0, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0, 1].xaxis.set_ticks(xticks)
        arr[0, 1].set_title('Position deviation at full period')
        arr[0, 1].semilogy(self.continuationParameter, self.deviation_x, linewidth=linewidth, c=self.plottingColors['tripleLine'][0],label='$|{x}(T) - {x}(0)|$')
        arr[0, 1].semilogy(self.continuationParameter, self.deviation_y, linewidth=linewidth, c=self.plottingColors['tripleLine'][1],label='$|{y}(T) - {y}(0)|$')
        arr[0, 1].semilogy(self.continuationParameter, 1e-9 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=0.5, linestyle='--')


        arr[0, 1].legend(frameon=True, loc='lower left',markerscale=11)
        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_ylim(ylim)

        arr[2, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 0].xaxis.set_ticks(xticks)
        arr[2, 0].set_title('Distribution of errors over collocated trajectory')
        arr[2, 0].semilogy(self.continuationParameter, self.maxDeltaError, linewidth=linewidth, c=self.plottingColors['singleLine'], label='max($e_{i}$)-min($e_{i}$)')
        arr[2, 0].set_xlim(xlim)
        arr[2, 0].set_ylim(ylim)
        arr[2, 0].semilogy(self.continuationParameter, 1e-12 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
        arr[2, 0].legend(frameon=True, loc='upper left')


        arr[1, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 1].xaxis.set_ticks(xticks)
        arr[1, 1].set_title('Velocity deviation at full period')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_xdot, linewidth=linewidth,c=self.plottingColors['tripleLine'][0], label='$|\dot{x}(T) - \dot{x}(0)|$')
        arr[1, 1].semilogy(self.continuationParameter, self.deviation_ydot, linewidth=linewidth,c=self.plottingColors['tripleLine'][1], label='$|\dot{y}(T) - \dot{y}(0)|$')
        arr[1, 1].set_xlim(xlim)
        arr[1, 1].set_ylim(ylim)
        arr[1, 1].semilogy(self.continuationParameter, 1e-9 * np.ones(len(self.continuationParameter)),color=self.plottingColors['limit'], linewidth=0.5, linestyle='--')
        arr[1, 1].legend(frameon=True, loc='lower left',markerscale=11)

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].set_title('Maximum number of corrections')
        arr[1, 0].set_xlim(xlim)
        if max(self.numberOfIterations) < 150:
            arr[1, 0].set_ylim([0, 150])
        else:
            arr[1, 0].set_ylim([0,max(self.numberOfIterations)+5])
        arr[1, 0].plot(self.continuationParameter, self.numberOfIterations, linewidth=linewidth, c=self.plottingColors['singleLine'],label='Number of corrections')
        arr[1, 0].legend(frameon=True, loc='upper left')

        arr[2, 1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[2, 1].xaxis.set_ticks(xticks)
        arr[2, 1].set_title('Maximum collocation segment error')
        arr[2, 1].set_xlim(xlim)
        arr[2, 1].set_ylim([10e-15,1.0e-5])
        arr[2, 1].tick_params(axis='y', labelcolor=self.plottingColors['tripleLine'][0])
        lns1 = arr[2, 1].semilogy(self.continuationParameter, self.maxSegmentError, linewidth=linewidth, c=self.plottingColors['tripleLine'][0],label='$e_{i}$')
        ax2 = arr[2, 1].twinx()
        ax2.tick_params(axis='y', labelcolor=self.plottingColors['tripleLine'][1])
        lns2 = ax2.plot(self.continuationParameter, self.numberOfCollocationPoints, linewidth=linewidth,color=self.plottingColors['tripleLine'][1],label='Number of nodes')
        ax2.set_ylim([0,70])
        ax2.set_xlim(xlim)
        ax2.grid(b=None)

        # added these three lines
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        arr[2,1].legend(lns, labs, frameon=True, loc='upper left')

        arr[0, 0].set_ylabel('$||\\mathbf{F}||$ [-]')
        arr[0, 1].set_ylabel('$\Delta \\bar{R}$ [-]')
        arr[2, 0].set_ylabel('max($e_{i}$) - min($e_{i}$)[-]')
        arr[1, 1].set_ylabel('$\Delta \\bar{V}$ [-]')
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
                if self.varyingQuantity == 'Alpha' and plot_as_x_coordinate == False and plot_as_family_number == False:
                    arr[i,j].set_xticks([0,90,180,270,360])
                    arr[i,j].set_xticklabels(['$0$','$\\frac{1}{2}\\pi$','$\\pi$','$\\frac{3}{2}\\pi$','$2\\pi$'])

        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 1:
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.05:
                xcoords = [135.0, 260.0]
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [93.0,123.0,237.0,267.0]
            if self.Hamiltonian == -1.525 and self.accelerationMagnitude == 0.1:
                xcoords = [110, 146.0, 214.0, 253.0]
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.1:
                xcoords = [126.0, 250.0]
        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 2:
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [61,96,109,251,265,301]

            for i in range(3):
                for j in range(2):
                    for xc in xcoords:
                        arr[i,j].axvline(x=xc,color='red',linestyle='--',linewidth=0.5)

        ax2.grid(False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.alpha < 59.0:
                alphaTitle = '0'
            elif self.alpha > 1.0 and self.alpha < 61.0:
                alphaTitle = '\\frac{1}{3}\\pi'
            elif self.alpha > 61.0 and self.alpha < 121.0:
                alphaTitle = '\\frac{2}{3}\\pi'
            elif self.alpha > 121.0 and self.alpha < 181.0:
                alphaTitle = '\\pi'
            elif self.alpha > 181.0 and self.alpha < 241.0:
                alphaTitle = '\\frac{4}{3}\\pi'
            elif self.alpha > 241.0 and self.alpha < 301.0:
                alphaTitle = '\\frac{5}{3}\\pi'


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + ' - Periodicity constraints verification',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + ' - Periodicity constraints verification', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.2f}".format(self.accelerationMagnitude))  + '$) - Periodicity constraints verification', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_periodicity_constraints.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_periodicity_constraints.png', transparent=True, dpi=300, bbox_inches='tight')

        plt.close()
        pass

    def plot_monodromy_analysis(self):
        f, arr = plt.subplots(1, 2, figsize=self.figSizeThird)
        size = 7

        xlim = [min(self.continuationParameter), max(self.continuationParameter)]

        xticks = (np.linspace(min(self.continuationParameter), max(self.continuationParameter), num=self.numberOfXTicks))

        l1 = [abs(entry) for entry in self.lambda1]
        l2 = [abs(entry) for entry in self.lambda2]
        l3 = [abs(entry) for entry in self.lambda3]
        l4 = [abs(entry) for entry in self.lambda4]
        l5 = [abs(entry) for entry in self.lambda5]
        l6 = [abs(entry) for entry in self.lambda6]

        print('lambda1: ' + str(self.lambda1[1265:1280]))
        print('lambda2: ' + str(self.lambda2[1265:1280]))
        print('lambda3: ' + str(self.lambda3[1265:1280]))
        print('lambda4: ' + str(self.lambda4[1265:1280]))
        print('lambda5: ' + str(self.lambda5[1265:1280]))
        print('lambda6: ' + str(self.lambda6[1265:1280]))

        print('v1: ' + str(self.v1[1265:1280]))
        print('v2: ' + str(self.v2[1265:1280]))
        print('v3: ' + str(self.v3[1265:1280]))

        # arr[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        # arr[0, 0].xaxis.set_ticks(xticks)
        # print(len(self.continuationParameter))


        # arr[0, 0].semilogy(self.continuationParameter, l1, c=self.plottingColors['lambda1'])
        # arr[0, 0].semilogy(self.continuationParameter, l2, c=self.plottingColors['lambda2'])
        # arr[0, 0].semilogy(self.continuationParameter, l3, c=self.plottingColors['lambda3'])
        # arr[0, 0].semilogy(self.continuationParameter, l4, c=self.plottingColors['lambda4'])
        # arr[0, 0].semilogy(self.continuationParameter, l5, c=self.plottingColors['lambda5'])
        # arr[0, 0].semilogy(self.continuationParameter, l6, c=self.plottingColors['lambda6'])
        # arr[0, 0].set_xlim(xlim)
        # arr[0, 0].set_ylim([1e-4, 1e4])
        # arr[0, 0].set_title('$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = |1/\lambda_3| \geq |1/\lambda_2| \geq |1/\lambda_1|$')
        # arr[0, 0].set_ylabel('Eigenvalues module [-]')

        d = [abs(entry - 1) for entry in self.D]
        arr[0].semilogy(self.continuationParameter, d, c=self.plottingColors['singleLine'], linewidth=1)
        arr[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0].xaxis.set_ticks(xticks)
        arr[0].set_xlim(xlim)
        arr[0].set_ylim([1e-14, 1e-5])
        arr[0].set_ylabel('$| 1 - $det($M$)$|$ [-]')
        arr[0].set_title('Error in determinant ')
        arr[0].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
        if self.varyingQuantity == 'Alpha' and plot_as_x_coordinate == False and plot_as_family_number == False:
            arr[0].set_xticks([0, 90, 180, 270, 360])
            arr[0].set_xticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])
            arr[1].set_xticks([0, 90, 180, 270, 360])
            arr[1].set_xticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])

        # arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        # arr[1, 0].xaxis.set_ticks(xticks)
        # arr[1, 0].scatter(self.continuationParameter, self.orderOfLinearInstability, s=size, c=self.plottingColors['singleLine'])
        # arr[1, 0].set_ylabel('Order of linear instability [-]')
        # arr[1, 0].set_xlim(xlim)
        # arr[1, 0].set_ylim([0, 3])

        arr[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1].xaxis.set_ticks(xticks)
        arr[1].set_xlim(xlim)
        l3zoom = [abs(entry - 1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[1].semilogy(self.continuationParameter, l3zoom, c=self.plottingColors['doubleLine'][0], linewidth=1,label='$||\\lambda_3 || $')
        arr[1].semilogy(self.continuationParameter, l4zoom, c=self.plottingColors['doubleLine'][1], linewidth=1, linestyle=':',label='$||\\lambda_4 || $')
        arr[1].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        if self.varyingQuantity == 'Hamiltonian':
            arr[0].set_xlabel('$H_{lt}$ [-]')
            arr[1].set_xlabel('$H_{lt}$ [-]')
        if self.varyingQuantity == 'xcor':
            arr[0].set_xlabel('x [-]')
            arr[1].set_xlabel('x [-]')


        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[1].set_ylabel(' $|||\lambda_3||-1|$ [-]')
        arr[1].set_title('Error in eigenvalue pair denoting periodicity')
        arr[1].legend(frameon=True, loc='center left',bbox_to_anchor=(1, 0.5),markerscale=15)

        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 1:
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.05:
                xcoords = [135.0, 260.0]
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [93.0, 123.0, 237.0, 267.0]
            if self.Hamiltonian == -1.525 and self.accelerationMagnitude == 0.1:
                xcoords = [110, 146.0, 214.0, 253.0]
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.1:
                xcoords = [126.0, 250.0]
        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 2:
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [61.0, 96.0, 109.0, 251.0, 265.0, 301.0]

            for i in range(2):
                for xc in xcoords:
                        arr[i].axvline(x=xc, color='red', linestyle='--', linewidth=0.5)

        #arr[1, 0].set_title('Order of linear instability')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.alpha < 59.0:
                alphaTitle = '0'
            elif self.alpha > 1.0 and self.alpha < 61.0:
                alphaTitle = '\\frac{1}{3}\\pi'
            elif self.alpha > 61.0 and self.alpha < 121.0:
                alphaTitle = '\\frac{2}{3}\\pi'
            elif self.alpha > 121.0 and self.alpha < 181.0:
                alphaTitle = '\\pi'
            elif self.alpha > 181.0 and self.alpha < 241.0:
                alphaTitle = '\\frac{4}{3}\\pi'
            elif self.alpha > 241.0 and self.alpha < 301.0:
                alphaTitle = '\\frac{5}{3}\\pi'

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + '- Monodromy matrix eigensystem validation',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.2f}".format(self.accelerationMagnitude))  + '$) - Monodromy matrix eigensystem validation', size=self.suptitleSize)

        plt.tight_layout()
        plt.subplots_adjust(top=0.82)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:3.2f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:3.3f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=300,bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:3.2f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_monodromy_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_monodromy_analysis.png', transparent=True, dpi=300, bbox_inches='tight')

        pass

    def plot_monodromy_analysis_old(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)
        size = 7

        print('=========== EXTRACTING QUANTIFICATION INFO ================')
        print('Varying quantity: ' + str(self.varyingQuantity) )
        print('Family: L'+str(self.lagrangePointNr) + ', Hlt: ' + str(self.Hamiltonian) + ', alt: ' +str(self.accelerationMagnitude) + ', alpha: ' + str(self.alpha))
        print('# Members: ' + str(len(self.continuationParameter)))
        print('Bifurcation indices: ' + str(self.orbitIdBifurcations))
        bifurcationHamiltonian = []
        for i in self.orbitIdBifurcations:
            bifurcationHamiltonian.append(self.continuationParameter[i])
            print('member: ' + str(i) + ' contintuation  value: ' + str("{:2.12f}".format(self.continuationParameter[i])) )

        print('Continuation Parameter value at bifurcations: ' + str(bifurcationHamiltonian))
        print('Minimum Continuation parameter ' + str("{:2.12f}".format(min(self.continuationParameter))) + ' at member: ' + str(self.continuationParameter.index(min(self.continuationParameter))))
        print('Maximum Continuation parameter ' + str("{:2.12f}".format(max(self.continuationParameter))) + ' at member: ' + str(self.continuationParameter.index(max(self.continuationParameter))))

        print('Minimum T ' + str("{:2.12f}".format(min(self.T))) + ' at member: ' + str(self.T.index(min(self.T))))
        print('Maximum T ' + str("{:2.12f}".format(max(self.T))) + ' at member: ' + str(self.T.index(max(self.T))))

        print('Minimum v1 ' + str(min(self.v1)) + ' at member: ' + str(self.v1.index(min(self.v1))))
        print('Maximum v1 ' + str(max(self.v1)) + ' at member: ' + str(self.v1.index(max(self.v1))))

        #print('alpha Cont: ' +str(self.alphaContinuation))
        # print('local extrema max: ' + str(max(self.continuationParameter[800:1500])) + ' at member: ' + str(self.continuationParameter[800:1500].index(max(self.continuationParameter[800:1500]))+800))
        # print('local extrema minimum: ' + str(min(self.continuationParameter[1000:4449])) + ' at member: ' + str(self.continuationParameter[1000:4449].index(min(self.continuationParameter[1000:4449]))+1500))
        #
        # print('max T: ' + str(max(self.T)) + ' at member: ' + str(self.T.index(max(self.T))))
        # #print('Hamiltonian of max T: ' + str(self.continuationParameter))

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
        arr[1,1].semilogy(self.continuationParameter, d, c=self.plottingColors['singleLine'], linewidth=1)
        arr[1,1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1,1].xaxis.set_ticks(xticks)
        arr[1,1].set_xlim(xlim)
        arr[1,1].set_ylim([1e-14, 1e-5])
        arr[1,1].set_ylabel('$| 1 - Det(\mathbf{M}) |$ [-]')
        arr[1,1].set_title('Error in determinant ')
        arr[1,1].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
        if self.varyingQuantity == 'Alpha' and plot_as_x_coordinate == False and plot_as_family_number == False:
            arr[0,0].set_xticks([0, 90, 180, 270, 360])
            arr[0,0].set_xticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])
            arr[1,1].set_xticks([0, 90, 180, 270, 360])
            arr[1,1].set_xticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])

        arr[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[1, 0].xaxis.set_ticks(xticks)
        arr[1, 0].scatter(self.continuationParameter, self.orderOfLinearInstability, s=size, c=self.plottingColors['singleLine'])
        arr[1, 0].set_ylabel('Order of linear instability [-]')
        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_ylim([0, 3])

        arr[0,1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.4f'))
        arr[0,1].xaxis.set_ticks(xticks)
        arr[0,1].set_xlim(xlim)
        l3zoom = [abs(entry - 1) for entry in l3]
        l4zoom = [abs(entry - 1) for entry in l4]
        arr[0,1].semilogy(self.continuationParameter, l3zoom, c=self.plottingColors['doubleLine'][0], linewidth=1,label='$||\\lambda_3 || $')
        arr[0,1].semilogy(self.continuationParameter, l4zoom, c=self.plottingColors['doubleLine'][1], linewidth=1, linestyle=':',label='$||\\lambda_4 || $')
        arr[0,1].semilogy(self.continuationParameter, 1.0e-3 * np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')

        if self.varyingQuantity == 'Hamiltonian':
            arr[0,0].set_xlabel('$H_{lt}$ [-]')
            arr[0,1].set_xlabel('$H_{lt}$ [-]')
        if self.varyingQuantity == 'xcor':
            arr[0,0].set_xlabel('x [-]')
            arr[0,1].set_xlabel('x [-]')


        # arr[1, 1].set_ylim([0, 1.5e-3])
        arr[0,1].set_ylabel(' $|||\lambda_3||-1|$ [-]')
        arr[0,1].set_title('Error in eigenvalue pair denoting periodicity')
        arr[0,1].legend(frameon=True, loc='center left',bbox_to_anchor=(1, 0.5),markerscale=15)

        #arr[1, 0].set_title('Order of linear instability')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + '- Monodromy matrix eigensystem validation',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + str(self.alpha) + ' ^{\\circ}$) ' + ' - Monodromy matrix eigensystem validation', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.3f}".format(self.accelerationMagnitude))  + '$) - Monodromy matrix eigensystem validation', size=self.suptitleSize)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:3.2f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:3.3f}".format(self.alpha)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=300,bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:3.2f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_monodromy_analysis_OLD.png', transparent=True, dpi=300, bbox_inches='tight')

        pass

    def plot_stability(self):
        unit_circle_1 = plt.Circle((0, 0), 1, color='grey', fill=False)
        unit_circle_2 = plt.Circle((0, 0), 1, color='grey', fill=False)
        unit_circle_3 = plt.Circle((0, 0), 1, color='grey', fill=False)


        size = 7

        xlim = [min(self.continuationParameter), max(self.continuationParameter)]
        xticks = (np.linspace(min(self.Hlt), max(self.Hlt), num=self.numberOfXTicks))

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.alpha < 59.0:
                alphaTitle = '0'
            elif self.alpha > 1.0 and self.alpha < 61.0:
                alphaTitle = '\\frac{1}{3}\\pi'
            elif self.alpha > 61.0 and self.alpha < 121.0:
                alphaTitle = '\\frac{2}{3}\\pi'
            elif self.alpha > 121.0 and self.alpha < 181.0:
                alphaTitle = '\\pi'
            elif self.alpha > 181.0 and self.alpha < 241.0:
                alphaTitle = '\\frac{4}{3}\\pi'
            elif self.alpha > 241.0 and self.alpha < 301.0:
                alphaTitle = '\\frac{5}{3}\\pi'

        f, arr = plt.subplots(3, 3, figsize=self.figSize)


        arr[0, 0].scatter(np.real(self.lambda1), np.imag(self.lambda1), c=self.plottingColors['lambda1'], s=size)
        arr[0, 0].scatter(np.real(self.lambda6), np.imag(self.lambda6), c=self.plottingColors['lambda6'], s=size)
        arr[0, 0].set_xlim([-4, 4])
        arr[0, 0].set_ylim([-4, 4])
        arr[0, 0].set_title('$\lambda_1, 1/\lambda_1$')
        arr[0, 0].set_xlabel('Re [-]')
        arr[0, 0].set_ylabel('Im [-]')
        arr[0, 0].add_artist(unit_circle_3)


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
                if self.varyingQuantity == 'Alpha' and plot_as_x_coordinate == False and plot_as_family_number == False and i > 0:
                    arr[i,j].set_xticks([0,90,180,270,360])
                    arr[i,j].set_xticklabels(['$0$','$\\frac{1}{2}\\pi$','$\\pi$','$\\frac{3}{2}\\pi$','$2\\pi$'])


        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.3f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + '- Eigenvalues $\lambda_i$ \& stability indices $v_i$',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + str(self.alpha) + '$ rad) ' + ' - Eigenvalues $\lambda_i$ \& stability indices $v_i$', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.3f}".format(self.accelerationMagnitude))  + '$) - Eigenvalues $\lambda_i$ \& stability indices $v_i$', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True,dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_stability.png', transparent=True,dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_stability.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_stability.png', transparent=True,dpi=300, bbox_inches='tight')


        pass

    def plot_continuation_procedure(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSizeCont)
        size = 7
        #print(self.Hlt)

        xlim = [1,len(self.orbitsId)]
        ylimSpacing = (max(self.Hlt)-min(self.Hlt))*1.0

        arr[0,0].plot(self.orbitsId,self.Hlt,c=self.plottingColors['singleLine'], linewidth=1,label='$H_{lt}$ [-]')
        arr[0,0].set_xlim(xlim)
        arr[0,0].set_ylim(min(self.Hlt)-ylimSpacing,max(self.Hlt)+ylimSpacing)
        arr[0,0].set_title('$H_{lt}$ evolution')
        arr[0,0].set_xlabel('orbit Number [-]')
        arr[0,1].set_ylabel('$H_{lt}$ [-]')
        arr[0,0].legend(frameon=True, loc='upper left')
        if self.varyingQuantity != 'Hamiltonian':
            arr[0,0].plot(self.orbitsId, self.Hamiltonian* np.ones(len(self.continuationParameter)), color=self.plottingColors['limit'], linewidth=1, linestyle='--')
            # arr[0,0].set_yticks([self.Hamiltonian])
            # arr[0,0].set_yticklabels(str(self.Hamiltonian))





        arr[0,1].plot(self.orbitsId,self.alphaContinuationRad,c=self.plottingColors['singleLine'], linewidth=1,label='$\\alpha$ [-]')
        arr[0,1].set_xlim(xlim)
        arr[0,1].set_ylim([-0.1, 2*np.pi])
        arr[0,1].set_title('$\\alpha$ evolution')
        arr[0,1].set_xlabel('orbit Number [-]')
        arr[0,1].set_ylabel('$\\alpha$ [-]')
        arr[0,1].legend(frameon=True, loc='center left',bbox_to_anchor=(1, 0.5))
        ticksLocators1 = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi]
        labels1 = ('$0$', '$ \\frac{1}{2}\\pi $', '$0$', '$ \\frac{3}{2}\\pi $', '$2 \\pi $')
        arr[0, 1].set_yticks(ticksLocators1, minor=False)
        arr[0, 1].set_yticklabels(labels1, fontdict=None, minor=False)




        arr[1,0].plot(self.orbitsId,self.accelerationContinuation,c=self.plottingColors['singleLine'], linewidth=1,label='$a_{lt}$ [-]')
        arr[1,0].set_xlim(xlim)
        arr[1,0].set_ylim([-0.01, 0.11])
        arr[1,0].set_title('$a_{lt}$ evolution')
        arr[1,0].set_xlabel('orbit Number [-]')
        arr[1,0].set_ylabel('$a_{lt}$ [-]')
        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Alpha':
            if self.accelerationMagnitude < 0.06:
                arr[1,0].legend(frameon=True, loc='upper left')
            else:
                arr[1, 0].legend(frameon=True, loc='lower left')
        else:
            arr[1, 0].legend(frameon=True, loc='upper left')



        lns0 = arr[1,1].plot(self.orbitsId,self.x,c=self.plottingColors['tripleLine'][0], linewidth=1,label='$x$ [-]')
        lns1 = arr[1,1].plot(self.orbitsId,self.y,c=self.plottingColors['tripleLine'][1], linewidth=1,label='$y$ [-]')
        #arr[1,1].plot(self.orbitsId,self.phase,c=self.plottingColors['tripleLine'][2], linewidth=1)
        arr[1,1].set_xlim(xlim)
        arr[1,1].set_ylim([-1.0,2.0])
        if self.varyingQuantity == 'Hamiltonian':
            arr[1,1].set_title('Spatial and phase evolution')
        else:
            arr[1, 1].set_title('Spatial evolution')
            lns = lns0 + lns1
        arr[1,1].set_xlabel('orbit Number [-]')
        arr[1,1].set_ylabel('$x$ [-], $y$ [-]')


        if self.varyingQuantity == 'Hamiltonian':
            ax2 = arr[1, 1].twinx()
            ax2.tick_params(axis='phase [-]', labelcolor=self.plottingColors['tripleLine'][2])
            lns2 = ax2.plot(self.orbitsId, self.phase, linewidth=1,color=self.plottingColors['tripleLine'][2],label='$\\phi$ [-]')
            ax2.set_ylim([-0.05, 2*np.pi+0.05])
            ax2.set_xlim(xlim)
            ax2.grid(b=None)
            #arr[1,1].legend(frameon=True, loc='lower right')
            #ax2.legend(frameon=True, loc='lower right')

            ticksLocators = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi]
            labels = ('$0$', '$ \\frac{1}{2}\\pi $', '$0$', '$ \\frac{3}{2}\\pi $', '$2 \\pi $')
            ax2.set_yticks(ticksLocators, minor=False)
            ax2.set_yticklabels(labels, fontdict=None, minor=False)

        # added these three lines

            lns = lns0 + lns1 + lns2
        labs = [l.get_label() for l in lns]
        arr[1, 1].legend(lns, labs, frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5), markerscale=15)

        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')
                # if self.varyingQuantity == 'Alpha' and plot_as_x_coordinate == False and plot_as_family_number == False:
                #     arr[i,j].set_xticks([0,90,180,270,360])
                #     arr[i,j].set_xticklabels(['$0$','$\\frac{1}{2}\\pi$','$\\pi$','$\\frac{3}{2}\\pi$','$2\\pi$'])

        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 1:
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.05:
                xcoords = [135.0, 260.0]
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [93.0, 123.0, 124.0, 153.0]
            if self.Hamiltonian == -1.525 and self.accelerationMagnitude == 0.1:
                xcoords = [110, 146.0, 147.0, 185.0]
            if self.Hamiltonian == -1.50 and self.accelerationMagnitude == 0.1:
                xcoords = [126.0, 250.0]
        if self.varyingQuantity == 'Alpha' and self.lagrangePointNr == 2:
            if self.Hamiltonian == -1.55 and self.accelerationMagnitude == 0.1:
                xcoords = [61, 96, 111, 113, 125, 161]

            for i in range(2):
                for j in range(2):
                    for xc in xcoords:
                        arr[i, j].axvline(x=xc, color='red', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.alpha < 59.0:
                alphaTitle = '0'
            elif self.alpha > 1.0 and self.alpha < 61.0:
                alphaTitle = '\\frac{1}{3}\\pi'
            elif self.alpha > 61.0 and self.alpha < 121.0:
                alphaTitle = '\\frac{2}{3}\\pi'
            elif self.alpha > 121.0 and self.alpha < 181.0:
                alphaTitle = '\\pi'
            elif self.alpha > 181.0 and self.alpha < 241.0:
                alphaTitle = '\\frac{4}{3}\\pi'
            elif self.alpha > 241.0 and self.alpha < 301.0:
                alphaTitle = '\\frac{5}{3}\\pi'

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + '- Numerical continuation verification ',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + ' - Numerical continuation verification', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.2f}".format(self.accelerationMagnitude))  + '$) - Numerical continuation validation', size=self.suptitleSize)


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True,dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Acceleration':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + '_continuation_analysis.png', transparent=True,dpi=300, bbox_inches='tight')
        if self.varyingQuantity == 'Alpha':
            if self.lowDPI:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_continuation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight')
            else:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str(
                "{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_continuation_analysis.png', transparent=True,dpi=300, bbox_inches='tight')



        pass

    def plot_increment_of_orbits(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSizeCont)
        size = 7

        xlim = [1, len(self.orbitsId)]

        arr[0, 0].plot(self.orbitsId, self.x, c=self.plottingColors['doubleLine'][0], linewidth=1, label='$x$ [-]')
        arr[0, 0].plot(self.orbitsId, self.y, c=self.plottingColors['doubleLine'][1], linewidth=1, label='$y$ [-]')

        arr[0, 0].set_xlim(xlim)
        arr[0, 0].set_title('Coordinate evolution of initial condition')
        arr[0, 0].set_xlabel('orbit Number [-]')
        arr[0, 0].set_ylabel('$x$ [-], $y$ [-]')
        #arr[0, 0].legend(frameon=True, loc='upper right')

        arr[0, 1].plot(self.orbitsId, self.xPhaseHalf, c=self.plottingColors['doubleLine'][0], linewidth=1, label='$x$ [-]')
        arr[0, 1].plot(self.orbitsId, self.yPhaseHalf, c=self.plottingColors['doubleLine'][1], linewidth=1, label='$y$ [-]')

        arr[0, 1].set_xlim(xlim)
        arr[0, 1].set_title('Coordinate evolution of $\\frac{\\phi}{2}$')
        arr[0, 1].set_xlabel('orbit Number [-]')
        arr[0, 1].set_ylabel('$x$ [-], $y$ [-]')
        arr[0, 1].legend(frameon=True, loc='center left',bbox_to_anchor=(1, 0.5),markerscale=15)
        arr[1, 0].set_ylim([1.0e-5,1.0e-3])


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
        arr[1, 0].semilogy(orbitIdNew, normIncrement, c=self.plottingColors['tripleLine'][2], linewidth=1,label='$||\\Delta \\bar{R}||$ [-]')
        arr[1,1].set_ylim([1.0e-5,1.0e-3])

        arr[1, 0].set_xlim(xlim)
        arr[1, 0].set_title('Increment evolution of initial condition')
        arr[1, 0].set_xlabel('orbit Number [-]')
        arr[1, 0].set_ylabel('$||\\Delta \\bar{R}||$ [-]')

        #arr[1, 0].legend(frameon=True, loc='upper right')

        #arr[1, 1].plot(orbitIdNew, xIncrementPhaseHalf, c=self.plottingColors['tripleLine'][0], linewidth=1,label='$\\Delta x$ [-]')
        #arr[1, 1].plot(orbitIdNew, yIncrementPhaseHalf, c=self.plottingColors['tripleLine'][1], linewidth=1,label='$\\Delta y$ [-]')
        arr[1, 1].semilogy(orbitIdNew, normIncrementPhaseHalf, c=self.plottingColors['tripleLine'][2], linewidth=1,label='$||\\Delta \\bar{R}||$ [-]')
        arr[1, 1].semilogy(self.continuationParameter, 1e-5 * np.ones(len(self.continuationParameter)),
                           color=self.plottingColors['limit'], linewidth=0.5, linestyle='--')

        arr[1, 1].set_xlim(xlim)
        arr[1,1].set_ylim([1.0e-5,1.0e-3])

        arr[1, 1].set_title('Increment evolution of $\\frac{\\phi}{2}$')
        arr[1, 1].set_xlabel('orbit Number [-]')
        #arr[1, 1].set_ylabel('$\\Delta x$ [-], $\\Delta y$ [-], $\\Delta R$ [-]')
        arr[1, 1].set_ylabel('$||\\Delta \\bar{R}||$ [-]')

        arr[1, 1].legend(frameon=True, loc='center left',bbox_to_anchor=(1, 0.5),markerscale=15)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.alpha < 59.0:
                alphaTitle = '0'
            elif self.alpha > 1.0 and self.alpha < 61.0:
                alphaTitle = '\\frac{1}{3}\\pi'
            elif self.alpha > 61.0 and self.alpha < 121.0:
                alphaTitle = '\\frac{2}{3}\\pi'
            elif self.alpha > 121.0 and self.alpha < 181.0:
                alphaTitle = '\\pi'
            elif self.alpha > 181.0 and self.alpha < 241.0:
                alphaTitle = '\\frac{4}{3}\\pi'
            elif self.alpha > 241.0 and self.alpha < 301.0:
                alphaTitle = '\\frac{5}{3}\\pi'


        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'xcor':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + '- Spatial evolution analysis ',size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + alphaTitle + '$ rad) ' + ' - Spatial evolution analysis ', size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt} = ' + str("{:3.2f}".format(self.accelerationMagnitude))  + '$) - Spatial evolution analysis ', size=self.suptitleSize)



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
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, dpi=300, bbox_inches='tight')
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
                    "{:7.6f}".format(self.alpha)) + '_spatial_analysis.png', transparent=True, dpi=300, bbox_inches='tight')
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
                "{:7.6f}".format(self.Hamiltonian)) + '_' + str(
                "{:7.6f}".format(self.accelerationMagnitude)) +'_spatial_analysis.png', transparent=True, dpi=300, bbox_inches='tight')

        pass



if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [2]
    acceleration_magnitudes = [0.0]
    alphas = [180.0]
    Hamiltonians = [-1.55]
    low_dpi = False
    varying_quantities = ['Hamiltonian']
    plot_as_x_coordinate  = False
    plot_as_family_number = False




    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for alpha in alphas:
                    for Hamiltonian in Hamiltonians:
                        for varying_quantity in varying_quantities:
                            display_periodic_solutions = DisplayPeriodicSolutions(orbit_type, lagrange_point, acceleration_magnitude, \
                                         alpha, Hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)

                            #display_periodic_solutions.plot_families()
                            display_periodic_solutions.plot_periodicity_validation()
                            display_periodic_solutions.plot_monodromy_analysis()
                            #display_periodic_solutions.plot_monodromy_analysis_old()
                            #display_periodic_solutions.plot_stability()
                            display_periodic_solutions.plot_continuation_procedure()
                            #display_periodic_solutions.plot_increment_of_orbits()


                            del display_periodic_solutions

