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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib2tikz import save as tikz_save
from textwrap import wrap
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

sys.path.append('../phase_7')

from  orbit_verification_and_validation import *

class presentationAnimations:
    def __init__(self, lagrange_point_nr,  acceleration_magnitude, alpha, hamiltonian, varying_quantity, orbit_objects, low_dpi,plot_as_x_coordinate,plot_as_family_number):
        self.lagrangePointNr = lagrange_point_nr
        self.orbitType = 'horizontal'
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = 0.0
        self.varyingQuantity = varying_quantity
        self.orbitObjects = orbit_objects
        self.Hamiltonian = hamiltonian

        if plot_as_x_coordinate == False and plot_as_family_number == False:
            if varying_quantity == 'Hamiltonian':
                self.continuationLabel = '$H_{lt}$ [-]'
            if varying_quantity == 'Alpha':
                self.continuationLabel = '$\\alpha$ [-]'
            if varying_quantity == 'Acceleration':
                self.continuationLabel = '$a_{lt}$ [-]'
        elif plot_as_x_coordinate == True and plot_as_family_number == False:
            self.continuationLabel = '$x$ [-]'
        else:
            self.continuationLabel = 'Orbit number [-]'

        if varying_quantity == 'Hamiltonian' or varying_quantity == 'Acceleration':
            self.subplotTitle = '$\\alpha$ = '
            self.subPlotTitleValueList = []
            for i in range(len(self.orbitObjects)):
                self.subPlotTitleValueList.append(orbit_objects[i].alpha)
        if varying_quantity == 'Alpha':
            print('test')
            self.subplotTitle = '$a_{lt}$ =  '
            self.subplotTitleTwo = '$H_{lt}$ =  '
            self.subPlotTitleValueList = []
            self.subPlotTitleValueListTwo = []
            for i in range(len(self.orbitObjects)):
                self.subPlotTitleValueList.append(orbit_objects[i].accelerationMagnitude)
                self.subPlotTitleValueListTwo.append(orbit_objects[i].Hamiltonian)


        # plot specific spacing properties
        self.orbitSpacingFactor = 50

        # scale properties
        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1

        # label properties
        self.numberOfXTicks = 5

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        self.figSizeLarge = (7 * (1 + np.sqrt(5)) / 2, 7*1.5)
        self.figRatioLarge = (7 * (1 + np.sqrt(5)) / 2) / (7*1.5)

        self.figureRatioSix = (7 * (1 + np.sqrt(5)) / 2) * 2/ (7*3)


        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figSizeWidePaper = (7 * (1 + np.sqrt(5)) / 2, 3.5 / 2)

        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        # Colour schemes
        n_colors = 3
        n_colors_4 = 4
        n_colors_6 = 6
        n_colors_9 = 9
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
                               'fourFamilies': [sns.color_palette("viridis", n_colors_4)[0],
                                             sns.color_palette("viridis", n_colors_4)[1],
                                             sns.color_palette("viridis", n_colors_4)[2],
                                             sns.color_palette("viridis", n_colors_4)[3]],
                                'sixFamilies': [sns.color_palette("viridis", n_colors_6)[0],
                                                sns.color_palette("viridis", n_colors_6)[1],
                                                sns.color_palette("viridis", n_colors_6)[2],
                                                 sns.color_palette("viridis", n_colors_6)[3],
                                                 sns.color_palette("viridis", n_colors_6)[4],
                                                sns.color_palette("viridis", n_colors_6)[5]],
                               'nineFamilies': [sns.color_palette("viridis", n_colors_9)[0],
                                               sns.color_palette("viridis", n_colors_9)[1],
                                               sns.color_palette("viridis", n_colors_9)[2],
                                               sns.color_palette("viridis", n_colors_9)[3],
                                               sns.color_palette("viridis", n_colors_9)[4],
                                               sns.color_palette("viridis", n_colors_9)[5],
                                               sns.color_palette("viridis", n_colors_9)[6],
                                               sns.color_palette("viridis", n_colors_9)[7],
                                               sns.color_palette("viridis", n_colors_9)[8]],
                               'limit': 'black'}
        self.plotAlpha = 1
        self.lineWidth = 0.5

        pass

    def cover_page_picture(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.add_subplot(111)

        lagrange_points_df = load_lagrange_points_location()

        lagrange_point_nrs = ['L1','L2','L3','L4','L5']

        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        # Plot bodies and equilibria
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

        # Determine complete the bounds of the colormap
        continuationParameter_min = 50000
        continuationParameter_max = -50000

        for i in range(1):
                if min(self.orbitObjects[i].continuationParameter) < continuationParameter_min:
                    continuationParameter_min = min(self.orbitObjects[i].continuationParameter)
                if max(self.orbitObjects[i].continuationParameter) > continuationParameter_max:
                    continuationParameter_max = max(self.orbitObjects[i].continuationParameter)

        print('i: ' + str(i))
        print('continuationParameter_min: ' + str(continuationParameter_min))
        print('continuationParameter_max: ' + str(continuationParameter_max))

        # Create the colourbar for all instances!
        if varying_quantity != 'Alpha':
            sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",
                                   (2000))),norm=plt.Normalize(vmin=continuationParameter_min, vmax=continuationParameter_max))
        else:
            sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",
                                                                                               (2000))),norm=plt.Normalize(vmin=0,vmax=360.0))


        minimum_x = 1000
        minimum_x2 = 1000
        maximum_x = -1000

        minimum_y = 1000
        maximum_y = -1000

        objectCounter = 0
        continuation_normalized_orbit = []
        continuation_normalized_orbit = [(value - continuationParameter_min) / (continuationParameter_max - continuationParameter_min) \
        for value in self.orbitObjects[objectCounter].continuationParameter]

        number_of_colors_orbit = len(self.orbitObjects[objectCounter].continuationParameter)

        colors_orbit = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r", number_of_colors_orbit))(continuation_normalized_orbit)

        numberOfPlotColorIndices_Orbit = len(self.orbitObjects[objectCounter].continuationParameter)
        plotColorIndexBasedOnVariable_Orbit = []

        for variable in self.orbitObjects[objectCounter].continuationParameter:
            plotColorIndexBasedOnVariable_Orbit.append( int(np.round(((variable - continuationParameter_min) / (continuationParameter_max - continuationParameter_min)) * (number_of_colors_orbit - 1))))

        orbitIdsPlot_orbit = []
        orbitIdsPlot_orbit = list(range(0, len(self.orbitObjects[objectCounter].continuationParameter), self.orbitObjects[objectCounter].orbitSpacingFactor))
        if orbitIdsPlot_orbit != len(self.orbitObjects[objectCounter].continuationParameter):
            orbitIdsPlot_orbit.append(len(self.orbitObjects[objectCounter].continuationParameter) - 1)
        print('')
        print('number of plots: ' + str(len(orbitIdsPlot_orbit)))
        print('')
        for k in orbitIdsPlot_orbit:
            #plot_color = colors_orbit[plotColorIndexBasedOnVariable_Orbit[k]]
            plot_color = colors_orbit[k]

            if self.varyingQuantity == 'Hamiltonian':
                df1String = '../../data/raw/orbits/augmented/L' + str(self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                + str("{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
                + str("{:12.11f}".format(self.orbitObjects[objectCounter].alpha)) + '_' \
                + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
            if self.varyingQuantity == 'Acceleration':
                df1String = '../../data/raw/orbits/augmented/L' + str(
                    self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                            + str(
                    "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationContinuation[k])) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].alpha)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
            if self.varyingQuantity == 'Alpha':
                df1String = '../../data/raw/orbits/augmented/L' + str(
                    self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].alphaContinuation[k])) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
            df1 = load_orbit(df1String)
            ax.plot(df1['x'], df1['y'], color=plot_color, alpha=self.orbitObjects[objectCounter].plotAlpha, linewidth=self.orbitObjects[objectCounter].lineWidth)

            if min(df1['x']) < minimum_x:
                    minimum_x = min(df1['x'])
                    # print('objectCounter: ' + str(objectCounter))
                    # print('alpha continuation xminimum at angle: ' + str(self.orbitObjects[objectCounter].alphaContinuation[k]))
                    # print('minimum x value: ' + str(minimum_x))


            if min(df1['y']) < minimum_y:
                 minimum_y = min(df1['y'])
            if max(df1['x']) > maximum_x:
                maximum_x = max(df1['x'])
            if max(df1['y']) > maximum_y:
                maximum_y = max(df1['y'])
            if min(df1['x']) < minimum_x2 and self.orbitObjects[objectCounter].alphaContinuation[k] > 179.0:
                minimum_x2 = min(df1['x'])
                # print('objectCounter: ' + str(objectCounter))
                # print('alpha continuation xminimum at angle: ' + str(self.orbitObjects[objectCounter].alphaContinuation[k]))
                # print('minimum x2 value: ' + str(minimum_x2))
        if self.varyingQuantity == 'Hamiltonian':
            for k in self.orbitObjects[objectCounter].orbitIdBifurcations:
                plot_color = colors_orbit[k]
                print('ObjectCounter: ' +str(objectCounter) + ' index bifurcation : ' + str(k))

                if self.orbitObjects[objectCounter].varyingQuantity == 'Hamiltonian':
                    df1String = '../../data/raw/orbits/augmented/L' + str(
                        self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                            + str(
                        "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].alpha)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                            + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
                if self.orbitObjects[objectCounter].varyingQuantity == 'Acceleration':
                    df1String = '../../data/raw/orbits/augmented/L' + str(
                        self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                                    + str(
                            "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationContinuation[k])) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].alpha)) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
                if self.orbitObjects[objectCounter].varyingQuantity == 'Alpha':
                    df1String = '../../data/raw/orbits/augmented/L' + str(
                            self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                                    + str(
                            "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
                                    + str(
                            "{:12.11f}".format(self.orbitObjects[objectCounter].alphaContinuation[k])) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
                    df1 = load_orbit(df1String)
                    ax.plot(df1['x'], df1['y'], color=plot_color,
                                   alpha=self.orbitObjects[objectCounter].plotAlpha,
                                   linewidth=2)

        objectCounter = objectCounter + 1


        xMiddle = minimum_x + ( maximum_x - minimum_x ) / 2
        yMiddle = minimum_y + ( maximum_y - minimum_y ) / 2

        scaleDistance = max((maximum_y - minimum_y), (maximum_x - minimum_x))



        plt.savefig('../../data/figures/cover_page.pdf', transparent=True)
        plt.close()

if __name__ == '__main__':
    lagrange_point_nr = 1
    acceleration_magnitude = 0.1
    alpha = 0.0
    hamiltonian = -1.55
    varying_quantity = 'Hamiltonian'
    low_dpi = False
    plot_as_x_coordinate = False
    plot_as_family_number = False

    orbit1 = DisplayPeriodicSolutions('horizontal', 1, 0.01, 60.0, -1.525, 'Acceleration',low_dpi, plot_as_x_coordinate, plot_as_family_number)
    orbit2 = DisplayPeriodicSolutions('horizontal', 2, 0.05, 120.0, -1.55, 'Hamiltonian', low_dpi, plot_as_x_coordinate, plot_as_family_number)

    my_objects = []
    my_objects.append(orbit1)
    my_objects.append(orbit2)


    presentation_animations = presentationAnimations(lagrange_point_nr, acceleration_magnitude,
                                                                        alpha, hamiltonian, varying_quantity,
                                                                        my_objects, low_dpi, plot_as_x_coordinate,
                                                                        plot_as_family_number)

    presentation_animations.cover_page_picture()

    del presentation_animations