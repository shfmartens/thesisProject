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

from  orbit_verification_and_validation import *


class PeriodicSolutionsCharacterization:
    def __init__(self, lagrange_point_nr,  acceleration_magnitude, alpha, hamiltonian, varying_quantity, orbit_objects, low_dpi,plot_as_x_coordinate,plot_as_family_number):

        self.lagrangePointNr = lagrange_point_nr
        self.orbitType = 'horizontal'
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = 0.0
        self.varyingQuantity = varying_quantity
        self.orbitObjects = orbit_objects
        self.Hamiltonian = hamiltonian

        #  =========== Plot layout settings ===============

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

        self.figSizeThird = (7 * (1 + np.sqrt(5)) / 2, 3.5 * 0.75)

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

    def ballistic_graphical_projection(self):
        fig = plt.figure(figsize=self.figSizeWide)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        # Plot libration point
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1','L2','L3','L4','L5']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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

        ax1.contourf(xM, yM, zM, colors='black')
        ax1.contourf(xE, yE, zE, colors='black')
        ax2.contourf(xM, yM, zM, colors='black')
        ax2.contourf(xE, yE, zE, colors='black')

        # Create new color maps
        Orbit1 = self.orbitObjects[0]
        Orbit2 = self.orbitObjects[1]

        Hlt_min = min(min(Orbit1.continuationParameter), min(Orbit2.continuationParameter))
        Hlt_max = max(max(Orbit1.continuationParameter), max(Orbit2.continuationParameter))

        continuation_normalized_orbit1 = [(value - Hlt_min) / (Hlt_max - Hlt_min) for value in Orbit1.continuationParameter]
        continuation_normalized_orbit2 = [(value - Hlt_min) / (Hlt_max - Hlt_min) for value in Orbit2.continuationParameter]

        #print(continuation_normalized_orbit1)

        number_of_colors_orbit1 = len(Orbit1.continuationParameter)
        number_of_colors_orbit2 = len(Orbit2.continuationParameter)


        colors_orbit1 = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",number_of_colors_orbit1))(continuation_normalized_orbit1)
        colors_orbit2 = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",number_of_colors_orbit2))(continuation_normalized_orbit2)
        print('length of colors_orbit_1:' +str(len(colors_orbit1)))
        print(colors_orbit1.tolist())

        numberOfPlotColorIndices_Orbit1 = len(Orbit1.continuationParameter)
        numberOfPlotColorIndices_Orbit2 = len(Orbit2.continuationParameter)


        plotColorIndexBasedOnHlt_Orbit1 = []
        plotColorIndexBasedOnHlt_Orbit2 = []

        #numberOfPlotColorIndices_Orbit2
        #numberOfPlotColorIndices_Orbit1
        for hamiltonian in Orbit1.continuationParameter:
            plotColorIndexBasedOnHlt_Orbit1.append(  \
                int( np.round ( ( (hamiltonian - Hlt_min) / (Hlt_max - Hlt_min) ) * (number_of_colors_orbit1 - 1) ) )     )

        for hamiltonian in Orbit2.continuationParameter:
            plotColorIndexBasedOnHlt_Orbit2.append( \
                int(np.round(((hamiltonian - Hlt_min) / (Hlt_max - Hlt_min)) * (number_of_colors_orbit2 - 1))))

        print(plotColorIndexBasedOnHlt_Orbit1)
        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",
             (number_of_colors_orbit1 ))),norm=plt.Normalize(vmin=Hlt_min, vmax=Hlt_max))


        orbitIdsPlot_orbit1 = list(range(0, len(Orbit1.continuationParameter), Orbit1.orbitSpacingFactor))

        if orbitIdsPlot_orbit1 != len(Orbit1.continuationParameter):
            orbitIdsPlot_orbit1.append(len(Orbit1.continuationParameter) - 1)

        orbitIdsPlot_orbit2 = list(range(0, len(Orbit2.continuationParameter), Orbit2.orbitSpacingFactor))
        if orbitIdsPlot_orbit2 != len(Orbit2.continuationParameter):
            orbitIdsPlot_orbit2.append(len(Orbit2.continuationParameter) - 1)

        for i in orbitIdsPlot_orbit1:
            plot_color1 = colors_orbit1[i]
            print(i)
            print(plotColorIndexBasedOnHlt_Orbit1[i])
            print(plot_color1)

            df1 = load_orbit('../../data/raw/orbits/augmented/L' + str(Orbit1.lagrangePointNr) + '_' + Orbit1.orbitType + '_' \
                            + str("{:12.11f}".format(Orbit1.accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(Orbit1.alpha)) + '_' \
                            + str("{:12.11f}".format(Orbit1.beta)) + '_' \
                            + str("{:12.11f}".format(Orbit1.Hlt[i])) + '_.txt')

            ax1.plot(df1['x'], df1['y'], color=plot_color1, alpha=Orbit1.plotAlpha, linewidth=Orbit1.lineWidth)

        # Plot the bifurcations
        for i in Orbit1.orbitIdBifurcations:
            plot_color1 = colors_orbit1[i]

            linewidthBifurcation = 2

            df1BF = load_orbit(
                '../../data/raw/orbits/augmented/L' + str(Orbit1.lagrangePointNr) + '_' + Orbit1.orbitType + '_' \
                + str("{:12.11f}".format(Orbit1.accelerationMagnitude)) + '_' \
                + str("{:12.11f}".format(Orbit1.alpha)) + '_' \
                + str("{:12.11f}".format(Orbit1.beta)) + '_' \
                + str("{:12.11f}".format(Orbit1.Hlt[i])) + '_.txt')

            ax1.plot(df1BF['x'], df1BF['y'], color=plot_color1, alpha=Orbit1.plotAlpha, linewidth=linewidthBifurcation)


        for i in orbitIdsPlot_orbit2:
            plot_color2 = colors_orbit2[plotColorIndexBasedOnHlt_Orbit2[i]]

            df2 = load_orbit('../../data/raw/orbits/augmented/L' + str(Orbit2.lagrangePointNr) + '_' + Orbit2.orbitType + '_' \
                            + str("{:12.11f}".format(Orbit2.accelerationMagnitude)) + '_' \
                            + str("{:12.11f}".format(Orbit2.alpha)) + '_' \
                            + str("{:12.11f}".format(Orbit2.beta)) + '_' \
                            + str("{:12.11f}".format(Orbit2.Hlt[i])) + '_.txt')

            ax2.plot(df2['x'], df2['y'], color=plot_color2, alpha=Orbit2.plotAlpha, linewidth=Orbit2.lineWidth)

        for i in Orbit2.orbitIdBifurcations:
            plot_color2 = colors_orbit2[i]

            linewidthBifurcation = 2

            df2BF = load_orbit(
                '../../data/raw/orbits/augmented/L' + str(Orbit2.lagrangePointNr) + '_' + Orbit2.orbitType + '_' \
                + str("{:12.11f}".format(Orbit2.accelerationMagnitude)) + '_' \
                + str("{:12.11f}".format(Orbit2.alpha)) + '_' \
                + str("{:12.11f}".format(Orbit2.beta)) + '_' \
                + str("{:12.11f}".format(Orbit2.Hlt[i])) + '_.txt')

            ax2.plot(df2BF['x'], df2BF['y'], color=plot_color2, alpha=Orbit2.plotAlpha, linewidth=linewidthBifurcation)

        sm.set_array([])

        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("right", size="2%", pad=0.2)

        ### test test
        position_handle = ax1.get_position().bounds
        position_handle2 = ax1.get_position().bounds

        colourbar_base = position_handle2[1]
        colourbar_height = position_handle[1] + position_handle[3] - colourbar_base

        # axColorbar = fig.add_axes([0.951, 0.07, 0.02, 0.93])
        axColorbar = fig.add_axes([0.935, colourbar_base + 0.026, 0.018, colourbar_height + 0.030])


        axColorbar.get_xaxis().set_visible(False)
        axColorbar.get_yaxis().set_visible(False)
        axColorbar.set_visible(False)

        divider = make_axes_locatable(axColorbar)

        cax = divider.append_axes("left", size="100%", pad=0.0)

        # cbar = plt.colorbar(sm, cax=cax, label='$|| \\lambda ||$ [-]', ticks=self.cbarTicksAngle)
        cbar = plt.colorbar(sm, cax=cax, format='%.2f', label=self.continuationLabel)

        # cbar.set_ticklabels(self.cbarTicksAngleLabels)

        fig.subplots_adjust(left=0.055, right=0.933, bottom=0.07, top=1.0)


        ## Create loops for continuationParameter label!

        minimumX1 = min(df1['x'])
        minimumY1 = min(df1['y'])

        minimumX2 = min(df2['x'])
        minimumY2 = min(df2['y'])

        maximumX1 = max(df1['x'])
        maximumY1 = max(df1['y'])

        maximumX2 = max(df2['x'])
        maximumY2 = max(df2['y'])

        Xmiddle1 = minimumX1 + (maximumX1 - minimumX1) / 2.0
        Ymiddle1 = minimumY1 + (maximumY1 - minimumY1) / 2.0

        Xmiddle2 = minimumX2 + (maximumX2 - minimumX2) / 2.0
        Ymiddle2 = minimumY2 + (maximumY2 - minimumY2) / 2.0

        scaleDistance1 = max((maximumY1 - minimumY1), (maximumX1 - minimumX1))
        scaleDistance2 = max((maximumY2 - minimumY2), (maximumX2 - minimumX2))

        scaleDistance = max(scaleDistance1,scaleDistance2)


        ax1.set_xlim([(Xmiddle1 - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                     (Xmiddle1 + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim(
            [Ymiddle1 - 0.5 * scaleDistance * self.spacingFactor, Ymiddle1 + 0.5 * scaleDistance * self.spacingFactor])

        # ax2.set_xlim([(Xmiddle2 - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
        #               (Xmiddle2 + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        # ax2.set_ylim([Ymiddle2 - 0.5 * scaleDistance * self.spacingFactor, Ymiddle2 + 0.5 * scaleDistance * self.spacingFactor])
        ax2.set_xlim([(Xmiddle1 - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle1 + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle2 - 0.5 * scaleDistance * self.spacingFactor, Ymiddle2 + 0.5 * scaleDistance * self.spacingFactor])

        ax1.set_aspect(1.0)
        ax2.set_aspect(1.0)


        #fig.tight_layout()

        if self.lowDpi:
            fig.savefig('../../data/figures/orbits/ballistic_projection.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/orbits/ballistic_projection.png', transparent=True, dpi=300)
        pass

    def ballistic_bifurcation_analysis(self):
        fig = plt.figure(figsize=self.figSizeWide)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        unit_circle_1 = plt.Circle((0, 0), 1, color='grey', fill=False)
        ax2.add_artist(unit_circle_1)

        size = 7

        Orbit1 = self.orbitObjects[0]
        Orbit2 = self.orbitObjects[1]

        continuationParameter_min = min(min(Orbit1.continuationParameter),min(Orbit2.continuationParameter))
        continuationParameter_max = max(max(Orbit1.continuationParameter),max(Orbit2.continuationParameter))

        realpart1_max = max(max(np.real(Orbit1.lambda1) ),max(np.real(Orbit2.lambda1)),max(np.real(Orbit1.lambda6) ),max(np.real(Orbit2.lambda6) ))
        realpart1_min = min(min(np.real(Orbit1.lambda1) ),min(np.real(Orbit2.lambda1)),min(np.real(Orbit1.lambda6) ),min(np.real(Orbit2.lambda6) ))

        realpart2_max = max(max(np.real(Orbit1.lambda2)), max(np.real(Orbit2.lambda2)), max(np.real(Orbit1.lambda5)),max(np.real(Orbit2.lambda5)))
        realpart2_min = min(min(np.real(Orbit1.lambda2)), min(np.real(Orbit2.lambda2)), min(np.real(Orbit1.lambda5)),min(np.real(Orbit2.lambda5)))

        #xlim1 = [realpart1_min,realpart1_max]
        #xlim2 = [realpart2_min,realpart2_max]
        xlim3 = [continuationParameter_min, continuationParameter_max]

        Orbit1_l1 = [abs(entry) for entry in Orbit1.lambda1]
        Orbit1_l2 = [abs(entry) for entry in Orbit1.lambda2]
        Orbit1_l3 = [abs(entry) for entry in Orbit1.lambda3]
        Orbit1_l4 = [abs(entry) for entry in Orbit1.lambda4]
        Orbit1_l5 = [abs(entry) for entry in Orbit1.lambda5]
        Orbit1_l6 = [abs(entry) for entry in Orbit1.lambda6]

        Orbit2_l1 = [abs(entry) for entry in Orbit2.lambda1]
        Orbit2_l2 = [abs(entry) for entry in Orbit2.lambda2]
        Orbit2_l3 = [abs(entry) for entry in Orbit2.lambda3]
        Orbit2_l4 = [abs(entry) for entry in Orbit2.lambda4]
        Orbit2_l5 = [abs(entry) for entry in Orbit2.lambda5]
        Orbit2_l6 = [abs(entry) for entry in Orbit2.lambda6]

        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l1, c=self.plottingColors['lambda1'],label='$|\lambda_{1}|$')
        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l2, c=self.plottingColors['lambda2'],label='$|\lambda_{2}|$')
        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l3, c=self.plottingColors['lambda3'],label='$|\lambda_{3}|$')
        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l4, c=self.plottingColors['lambda4'],label='$\\frac{1}{|\lambda_{3}|}$')
        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l5, c=self.plottingColors['lambda5'],label='$\\frac{1}{|\lambda_{2}|}$')
        ax1.semilogy(Orbit1.continuationParameter, Orbit1_l6, c=self.plottingColors['lambda6'],label='$\\frac{1}{|\lambda_{1}|}$')
        ax1.set_title('$|\lambda_1| \geq |\lambda_2| \geq |\lambda_3| = 1 = 1/|\lambda_3| \geq 1/|\lambda_2| \geq 1/|\lambda_1|$')


        ax2.scatter(Orbit1.continuationParameter, np.angle(Orbit1.lambda2, deg=True), c=self.plottingColors['lambda2'], s=size)
        ax2.scatter(Orbit1.continuationParameter, np.angle(Orbit1.lambda5, deg=True), c=self.plottingColors['lambda5'], s=size)

        ax2.set_title('$\\lambda_2, 1/\\lambda_2$')


        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l1, c=self.plottingColors['lambda1'])
        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l2, c=self.plottingColors['lambda2'])
        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l3, c=self.plottingColors['lambda3'])
        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l4, c=self.plottingColors['lambda4'])
        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l5, c=self.plottingColors['lambda5'])
        ax3.semilogy(Orbit2.continuationParameter, Orbit2_l6, c=self.plottingColors['lambda6'])



        ax4.scatter(Orbit2.continuationParameter, np.angle(Orbit2.lambda2, deg=True), c=self.plottingColors['lambda2'], s=size)
        ax4.scatter(Orbit2.continuationParameter, np.angle(Orbit2.lambda5, deg=True), c=self.plottingColors['lambda5'], s=size)




        ax1.set_ylabel('Eigenvalues Module [-]')
        ax1.grid(True, which='both', ls=':')
        ax1.set_xlim(xlim3)
        #ax1.set_title('$ \\lambda_{1} $, $1/\\lambda_{1}$ ')

        ax2.set_ylabel('Phase [rad]')
        ax2.grid(True, which='both', ls=':')
        ax2.set_xlim(xlim3)
        ax2.set_ylim([-180.0,180.0])


        ax3.set_xlabel(self.continuationLabel)
        ax3.set_ylabel('Eigenvalues Module [-]')
        ax3.grid(True, which='both', ls=':')
        ax3.set_xlim(xlim3)


        ax4.set_xlabel(self.continuationLabel)
        ax4.set_ylabel('Phase [rad]')
        ax4.set_xlim(xlim3)
        ax4.set_ylim([-180.0,180.0])

        yticksLocators = [-180,-90,0,90,180]
        ylabels = ('-$\\pi$','-$\\frac{\\pi}{2}$',0,'-$\\frac{\\pi}{2}$','$\\pi$')
        ax2.set_yticks(yticksLocators, minor=False)
        ax4.set_yticks(yticksLocators, minor=False)
        ax2.set_yticklabels(ylabels, fontdict=None, minor=False)
        ax4.set_yticklabels(ylabels, fontdict=None, minor=False)

        ax1.set_ylim([1e-4,1e4])
        ax1.set_ylim([1e-4,1e4])

        yticksLocators2 = [1.0e-4, 1.0e-2, 1.0e0, 1.0e2, 1.0e4]
        ylabels2 = ('$10^{-4}$', '$10^{-2}$', '$10^{0}$', '$10^{2}$', '$10^{4}$')
        ax1.set_yticks(yticksLocators2, minor=False)
        ax3.set_yticks(yticksLocators2, minor=False)
        ax1.set_yticklabels(ylabels2, fontdict=None, minor=False)
        ax3.set_yticklabels(ylabels2, fontdict=None, minor=False)


        ax4.grid(True, which='both', ls=':')
        ax4.set_xlim(xlim3)


        lgd = ax1.legend(frameon=True, loc='center left',bbox_to_anchor=(2.165, -0.2),markerscale=10)

        fig.tight_layout()
        plt.subplots_adjust(right=0.91)


        if self.lowDpi:
            fig.savefig('../../data/figures/orbits/ballistic_bifurcation.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/orbits/ballistic_bifurcation.png', transparent=True, dpi=300)
        pass

    def ballistic_stability_analysis(self):
        fig = plt.figure(figsize=self.figSizeWidePaper)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        Orbit1 = self.orbitObjects[0]
        Orbit2 = self.orbitObjects[1]

        continuationParameter_min = min(min(Orbit1.continuationParameter), min(Orbit2.continuationParameter))
        continuationParameter_max = max(max(Orbit1.continuationParameter), max(Orbit2.continuationParameter))

        xlim = [continuationParameter_min, continuationParameter_max]

        ax1.semilogy(Orbit1.continuationParameter, Orbit1.v1, c=self.plottingColors['lambda6'],label='$L_{1}$')
        ax1.semilogy(Orbit2.continuationParameter, Orbit2.v1, c=self.plottingColors['lambda3'],label='$L_{2}$')
        ax1.set_title('$\\nu_{1}$')

        ax2.plot(Orbit1.T, Orbit1.Hlt, c=self.plottingColors['lambda6'],label='$L_{1}$')
        ax2.plot(Orbit2.T, Orbit2.Hlt, c=self.plottingColors['lambda3'],label='$L_{2}$')
        #ax2.set_title('$\\nu_{2}$')

        ax1.axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        #ax2.axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')



        ax1.set_ylabel('Stability Index [-]')
        ax2.set_ylabel('$H_{lt}$ [-]')

        ax1.set_xlabel(self.continuationLabel)
        ax1.grid(True, which='both', ls=':')
        ax1.set_xlim(xlim)
        #ax1.set_ylim([])


        ax2.set_xlabel('$T$ [-]')

        ax2.grid(True, which='both', ls=':')
        #ax1.set_ylim([])


        lgd = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1.0, 0.5), markerscale=10)

        fig.tight_layout()
        plt.subplots_adjust(right=0.92)


        if self.lowDpi:
            fig.savefig('../../data/figures/orbits/stability_analysis.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/orbits/stability_analysis.png', transparent=True, dpi=300)
        pass

    def graphical_projection(self):


        numberOfPlots = len(self.orbitObjects)
        print(numberOfPlots)
        if numberOfPlots == 4:
            f, arr = plt.subplots(2, 2, figsize=self.figSize)
            rownNumber =  2
            columnNumber =  2

        if numberOfPlots == 6:
            f, arr = plt.subplots(2, 3, figsize=self.figSize)
            rownNumber = 2
            columnNumber = 3

        if numberOfPlots == 9:
            f, arr = plt.subplots(3, 3, figsize=self.figSizeLarge)
            rownNumber = 3
            columnNumber = 3

        # Build the subtitles and labels
        objectCounter = 0
        for k in range(rownNumber):
            for j in range(columnNumber):

                if k == 1:
                    arr[k,j].set_xlabel('x [-]')
                if j == 0:
                    arr[k,j].set_ylabel('y [-]')
                arr[k,j].grid(True, which='both', ls=':')
                if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
                    if self.subPlotTitleValueList[objectCounter] > -0.1 and self.subPlotTitleValueList[
                        objectCounter] < 0.1:
                        alphaRadians = '$0$'
                    if self.subPlotTitleValueList[objectCounter] > 59.9 and self.subPlotTitleValueList[
                        objectCounter] < 60.1:
                        alphaRadians = '$\\frac{1}{3}\\pi$'
                    if self.subPlotTitleValueList[objectCounter] > 119.9 and self.subPlotTitleValueList[
                        objectCounter] < 120.1:
                        alphaRadians = '$\\frac{2}{3}\\pi$'
                    if self.subPlotTitleValueList[objectCounter] > 179.9 and self.subPlotTitleValueList[
                        objectCounter] < 180.1:
                        alphaRadians = '$\\pi$'
                    if self.subPlotTitleValueList[objectCounter] > 239.9 and self.subPlotTitleValueList[
                        objectCounter] < 240.1:
                        alphaRadians = '$\\frac{4}{3}\\pi$'
                    if self.subPlotTitleValueList[objectCounter] > 299.9 and self.subPlotTitleValueList[
                        objectCounter] < 300.1:
                        alphaRadians = '$\\frac{5}{3}\\pi$'
                    subtitleString = self.subplotTitle + alphaRadians
                else:
                    subtitleString = str(objectCounter + 1) + '. '  + self.subplotTitleTwo + str(
                    "{:2.3f}".format(self.subPlotTitleValueListTwo[objectCounter])) + ' ' \
                                    + self.subplotTitle + str(
                    "{:2.3f}".format(self.subPlotTitleValueList[objectCounter]))

                arr[k, j].set_title(subtitleString)
                objectCounter = objectCounter + 1


        ## Plot bodies and natural libration points
                # Plot libration point
                # lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
                # if self.lagrangePointNr == 1:
                #     lagrange_point_nrs = ['L1', 'L2']
                # if self.lagrangePointNr == 2:
                #     lagrange_point_nrs = ['L1', 'L2']

                # Plot libration point
                lagrange_points_df = load_lagrange_points_location()
                lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']

                for lagrange_point_nr in lagrange_point_nrs:
                    for i in range(rownNumber):
                        for j in range(columnNumber):
                            arr[i,j].scatter(lagrange_points_df[lagrange_point_nr]['x'],lagrange_points_df[lagrange_point_nr]['y'],color='black', marker='x')



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

                for i in range(rownNumber):
                    for j in range(columnNumber):
                        arr[i,j].contourf(xM, yM, zM, colors='black')
                        arr[i,j].contourf(xE, yE, zE, colors='black')


        # Determine complete the bounds of the colormap
        continuationParameter_min = 50000
        continuationParameter_max = -50000

        for i in range(numberOfPlots):


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
        for i in range(rownNumber):
            for j in range(columnNumber):
                #print('objectCounter: ' +str(objectCounter) )
                continuation_normalized_orbit = []
                continuation_normalized_orbit = [
                    (value - continuationParameter_min) / (continuationParameter_max - continuationParameter_min) \
                    for value in self.orbitObjects[objectCounter].continuationParameter]

                number_of_colors_orbit = len(self.orbitObjects[objectCounter].continuationParameter)

                colors_orbit = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r", number_of_colors_orbit))(
                    continuation_normalized_orbit)

                numberOfPlotColorIndices_Orbit = len(self.orbitObjects[objectCounter].continuationParameter)
                #print('numberOfPlotColorIndices_Orbit: ' + str(numberOfPlotColorIndices_Orbit))
                plotColorIndexBasedOnVariable_Orbit = []

                for variable in self.orbitObjects[objectCounter].continuationParameter:
                    plotColorIndexBasedOnVariable_Orbit.append( \
                        int(np.round(
                            ((variable - continuationParameter_min) / (continuationParameter_max - continuationParameter_min)) * (number_of_colors_orbit - 1))))

                orbitIdsPlot_orbit = []
                orbitIdsPlot_orbit = list(range(0, len(self.orbitObjects[objectCounter].continuationParameter), self.orbitObjects[objectCounter].orbitSpacingFactor))
                if orbitIdsPlot_orbit != len(self.orbitObjects[objectCounter].continuationParameter):
                    orbitIdsPlot_orbit.append(len(self.orbitObjects[objectCounter].continuationParameter) - 1)

                #print(plotColorIndexBasedOnVariable_Orbit)



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
                    arr[i,j].plot(df1['x'], df1['y'], color=plot_color, alpha=self.orbitObjects[objectCounter].plotAlpha, linewidth=self.orbitObjects[objectCounter].lineWidth)

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

                        if self.varyingQuantity == 'Hamiltonian':
                            df1String = '../../data/raw/orbits/augmented/L' + str(
                                self.orbitObjects[objectCounter].lagrangePointNr) + '_horizontal_' \
                                    + str(
                                "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
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
                                    + str(
                            "{:12.11f}".format(self.orbitObjects[objectCounter].accelerationMagnitude)) + '_' \
                                    + str(
                            "{:12.11f}".format(self.orbitObjects[objectCounter].alphaContinuation[k])) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].beta)) + '_' \
                                    + str("{:12.11f}".format(self.orbitObjects[objectCounter].Hlt[k])) + '_.txt'
                        df1 = load_orbit(df1String)
                        arr[i, j].plot(df1['x'], df1['y'], color=plot_color,
                                   alpha=self.orbitObjects[objectCounter].plotAlpha,
                                   linewidth=2)

                objectCounter = objectCounter + 1


        xMiddle = minimum_x + ( maximum_x - minimum_x ) / 2
        yMiddle = minimum_y + ( maximum_y - minimum_y ) / 2

        scaleDistance = max((maximum_y - minimum_y), (maximum_x - minimum_x))

        #plt.subplots_adjust(top=0.83)

        for i in range(rownNumber):
            for j in range(columnNumber):
                if numberOfPlots == 4:
                    arr[i,j].set_xlim([(xMiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(xMiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
                    arr[i,j].set_ylim([yMiddle - 0.5 * scaleDistance * self.spacingFactor, yMiddle + 0.5 * scaleDistance * self.spacingFactor])
                if numberOfPlots == 6:
                    arr[i, j].set_xlim([(xMiddle - 0.5 * scaleDistance * self.figureRatioSix * self.spacingFactor),
                                        (xMiddle + 0.5 * scaleDistance * self.figureRatioSix * self.spacingFactor)])
                    arr[i, j].set_ylim([yMiddle - 0.5 * scaleDistance * self.spacingFactor,
                                        yMiddle + 0.5 * scaleDistance * self.spacingFactor])
                if numberOfPlots == 9:
                    arr[i, j].set_xlim([(xMiddle - 0.5 * scaleDistance * self.figRatioLarge * self.spacingFactor),
                                        (xMiddle + 0.5 * scaleDistance * self.figRatioLarge * self.spacingFactor)])
                    arr[i, j].set_ylim([yMiddle - 0.5 * scaleDistance * self.spacingFactor,
                                        yMiddle + 0.5 * scaleDistance * self.spacingFactor])

        sm.set_array([])

        position_handle = arr[0,columnNumber-1].get_position().bounds
        position_handle2 = arr[1, columnNumber-1].get_position().bounds

        colourbar_base = position_handle2[1]
        colourbar_height = position_handle[1]+position_handle[3]-colourbar_base

        colour_base2 = 0.073
        colourbar_height2 = 0.9385 - colour_base2
        if numberOfPlots == 6 or numberOfPlots == 4:
            colour_base2 = 0.073
            colourbar_height2 = 0.9385 - colour_base2
            axColorbar = f.add_axes([0.92, colour_base2, 0.02, colourbar_height2])
        else:
            colour_base2 = 0.06
            colourbar_height2 = 0.89
            axColorbar = f.add_axes([0.92, colour_base2, 0.02, colourbar_height2])
        axColorbar.get_xaxis().set_visible(False)
        axColorbar.get_yaxis().set_visible(False)
        axColorbar.set_visible(False)

        divider = make_axes_locatable(axColorbar)

        cax = divider.append_axes("left", size="100%", pad=0.0)

        plt.subplots_adjust(left=0.065,bottom=0.05,top=0.96)


        cbar = plt.colorbar(sm, cax=cax, label=self.continuationLabel)

        if self.varyingQuantity == 'Alpha':
            cbar.set_ticks([0,90,180,270,360])
            cbar.set_ticklabels(['$0$','$\\frac{1}{2}\\pi$','$\\pi$','$\\frac{3}{2}\\pi$','$2\\pi$'])


        # print('===== TEST ASPECT RATIO THINGS ====')
        # print('position handle of plot: ' + str(position_handle2))
        # print('Width of plot: ' + str(position_handle2[2]))
        # print('Height of plot: ' + str(position_handle2[3]))
        # print('Width/Height: ' + str(position_handle2[2]/position_handle2[3]))
        # print('Height/Width: ' + str(position_handle2[3]/position_handle2[2]))
        # print('self.figureRation: ' + str(self.figureRatio))
        # print('self.figureRatioSix: ' + str(self.figureRatioSix))



        for k in range(rownNumber):
            for j in range(columnNumber):
                arr[k, j].set_aspect('equal')



        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_graphical_projection.png', transparent=True, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_graphical_projection.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Acceleration':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_graphical_projection.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_graphical_projection.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Alpha':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_graphical_projection.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_graphical_projection.png', transparent=True, dpi=300)

        plt.close()
        pass

    def bifurcation_analysis(self):
        numberOfPlots = len(self.orbitObjects)
        print(numberOfPlots)
        if numberOfPlots == 4:
            f, arr = plt.subplots(4, 2, figsize=self.figSize)
            rownNumber = 4
            columnNumber = 2

        if numberOfPlots == 6:
            f, arr = plt.subplots(4, 3, figsize=self.figSize)
            rownNumber = 4
            columnNumber = 3

        if numberOfPlots == 9:
            f, arr = plt.subplots(6, 3, figsize=self.figSizeLarge)
            rownNumber = 6
            columnNumber = 3

        # Build the subtitles and labels
        objectCounter = 0
        for k in range(rownNumber):
            for j in range(columnNumber):
                if (k == 3 and numberOfPlots == 6) or (k == 5 and numberOfPlots == 9):
                    arr[k, j].set_xlabel(self.continuationLabel)
                if j == 0 and (k == 0 or k == 2 or (k == 4 and numberOfPlots == 9) ):
                    arr[k, j].set_ylabel('Eigenvalues Module [-]')
                if j == 0 and (k == 1 or k == 3 or (k == 5 and numberOfPlots == 9)):
                    arr[k, j].set_ylabel('Phase [rad]')

                arr[k, j].grid(True, which='both', ls=':')
                if k == 0 or k == 2 or (k == 4 and numberOfPlots == 9) :
                    if self.varyingQuantity != 'Hamiltonian' and self.varyingQuantity != 'Alpha':
                        subtitleString = self.subplotTitle + str(
                            "{:4.1f}".format(self.subPlotTitleValueList[objectCounter]))
                    elif self.varyingQuantity == 'Hamiltonian':
                        if self.subPlotTitleValueList[objectCounter] > -0.1 and self.subPlotTitleValueList[
                            objectCounter] < 0.1:
                            alphaRadians = '$0$'
                        if self.subPlotTitleValueList[objectCounter] > 59.9 and self.subPlotTitleValueList[
                            objectCounter] < 60.1:
                            alphaRadians = '$\\frac{1}{3}\\pi$'
                        if self.subPlotTitleValueList[objectCounter] > 119.9 and self.subPlotTitleValueList[
                            objectCounter] < 120.1:
                            alphaRadians = '$\\frac{2}{3}\\pi$'
                        if self.subPlotTitleValueList[objectCounter] > 179.9 and self.subPlotTitleValueList[
                            objectCounter] < 180.1:
                            alphaRadians = '$\\pi$'
                        if self.subPlotTitleValueList[objectCounter] > 239.9 and self.subPlotTitleValueList[
                            objectCounter] < 240.1:
                            alphaRadians = '$\\frac{4}{3}\\pi$'
                        if self.subPlotTitleValueList[objectCounter] > 299.9 and self.subPlotTitleValueList[
                            objectCounter] < 300.1:
                            alphaRadians = '$\\frac{5}{3}\\pi$'
                        subtitleString = self.subplotTitle + alphaRadians
                    else:
                        subtitleString = self.subplotTitleTwo + str(
                            "{:2.3f}".format(self.subPlotTitleValueListTwo[objectCounter])) + ' ' \
                                         + self.subplotTitle + str(
                            "{:2.3f}".format(self.subPlotTitleValueList[objectCounter]))

                    arr[k, j].set_title(subtitleString)
                    objectCounter = objectCounter + 1


        # Determine complete the bounds of the colormap
        continuationParameter_min = 50000
        continuationParameter_max = -50000

        for i in range(numberOfPlots):

            if min(self.orbitObjects[i].continuationParameter) < continuationParameter_min:
                continuationParameter_min = min(self.orbitObjects[i].continuationParameter)
            if max(self.orbitObjects[i].continuationParameter) > continuationParameter_max:
                continuationParameter_max = max(self.orbitObjects[i].continuationParameter)

        objectCounter = 0
        for i in range(rownNumber):
            for j in range(columnNumber):

                if i == 0 or i == 2 or (i == 4 and numberOfPlots == 9):
                    Orbit_l1 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda1]
                    Orbit_l2 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda2]
                    Orbit_l3 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda3]
                    Orbit_l4 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda4]
                    Orbit_l5 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda5]
                    Orbit_l6 = [abs(entry) for entry in self.orbitObjects[objectCounter].lambda6]

                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l1, c=self.plottingColors['lambda1'],label='$|\lambda_{1}|$')
                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l2, c=self.plottingColors['lambda2'],label='$|\lambda_{2}|$')
                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l3, c=self.plottingColors['lambda3'],label='$|\lambda_{3}|$')
                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l4, c=self.plottingColors['lambda4'],label='$|\\frac{1}{\lambda_{1}}|$')
                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l5, c=self.plottingColors['lambda5'],label='$|\\frac{1}{\lambda_{2}}|$')
                    arr[i,j].semilogy(self.orbitObjects[objectCounter].continuationParameter, Orbit_l6, c=self.plottingColors['lambda6'],label='$|\\frac{1}{\lambda_{3}}|$')

                    size = 3
                    arr[i+1, j].scatter(self.orbitObjects[objectCounter].continuationParameter,np.angle(self.orbitObjects[objectCounter].lambda2, deg=True),c=self.plottingColors['lambda2'], s=size)
                    arr[i+1, j].scatter(self.orbitObjects[objectCounter].continuationParameter,np.angle(self.orbitObjects[objectCounter].lambda5, deg=True),c=self.plottingColors['lambda5'], s=size)

                    arr[i,j].set_xlim([continuationParameter_min,continuationParameter_max])
                    arr[i+1,j].set_xlim([continuationParameter_min,continuationParameter_max])
                    arr[i,j].set_ylim([1.0e-4,1.0e4])
                    arr[i,j].set_yticks([1.0e-4,1.0e-2,1.0e0,1.0e2,1.0e4])
                    arr[i, j].set_yticklabels(('$10^{-4}$','$10^{-2}$','$10^{0\\phantom{-}}$','$10^{2\\phantom{-}}$','$10^{4\\phantom{-}}$'))
                    arr[i+1,j].set_ylim([-180, 180])
                    arr[i+1, j].set_yticks([-180.0, -90.0, 0.0, 90.0, 180.0])
                    arr[i+1, j].set_yticklabels(('$-\\pi$', '$-\\frac{\\pi}{2}$', '$0$', '$\\frac{\\pi}{2}$', '$\\pi$'))

                    objectCounter = objectCounter + 1

        plt.subplots_adjust(left=0.06,bottom=0.065,top=0.95,hspace=0.5)

        if numberOfPlots == 4 or numberOfPlots == 6:
            lgd = arr[2,columnNumber-1].legend(frameon=True, loc='center left', bbox_to_anchor=(1.05, 1.3), markerscale=10)
        else:
            lgd = arr[2, columnNumber - 1].legend(frameon=True, loc='center left', bbox_to_anchor=(1.05, -0.2),markerscale=10)

        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_bifurcation_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_bifurcation_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Acceleration':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_bifurcation_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_bifurcation_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Alpha':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_bifurcation_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_bifurcation_analysis.png', transparent=True, dpi=300)

        plt.close()
        pass

    def stability_analysis(self):
        numberOfPlots = len(self.orbitObjects)
        f, arr = plt.subplots(1, 2, figsize=self.figSizeWidePaper)

        arr[0].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        #arr[1].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')

        # Determine complete the bounds of the colormap
        continuationParameter_min = 50000
        continuationParameter_max = -50000

        for i in range(numberOfPlots):
            if min(self.orbitObjects[i].continuationParameter) < continuationParameter_min:
                continuationParameter_min = min(self.orbitObjects[i].continuationParameter)
            if max(self.orbitObjects[i].continuationParameter) > continuationParameter_max:
                continuationParameter_max = max(self.orbitObjects[i].continuationParameter)

        if len(self.orbitObjects) == 4:
            colour_family = 'fourFamilies'
        elif len(self.orbitObjects) == 6:
            colour_family = 'sixFamilies'
        else:
            colour_family = 'nineFamilies'

        objectCounter = 0
        for i in range(len(self.orbitObjects)):


            if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
                if self.subPlotTitleValueList[objectCounter] > -0.1 and self.subPlotTitleValueList[
                    objectCounter] < 0.1:
                    alphaRadians = '$0$'
                    stabilityLineStyle = '-'
                if self.subPlotTitleValueList[objectCounter] > 59.9 and self.subPlotTitleValueList[
                    objectCounter] < 60.1:
                    alphaRadians = '$\\frac{1}{3}\\pi$'
                    stabilityLineStyle = '-'
                if self.subPlotTitleValueList[objectCounter] > 119.9 and self.subPlotTitleValueList[
                    objectCounter] < 120.1:
                    alphaRadians = '$\\frac{2}{3}\\pi$'
                    stabilityLineStyle = '-'
                if self.subPlotTitleValueList[objectCounter] > 179.9 and self.subPlotTitleValueList[
                    objectCounter] < 180.1:
                    alphaRadians = '$\\pi$'
                    stabilityLineStyle = '-'
                if self.subPlotTitleValueList[objectCounter] > 239.9 and self.subPlotTitleValueList[
                    objectCounter] < 240.1:
                    alphaRadians = '$\\frac{4}{3}\\pi$'
                    stabilityLineStyle = '--'
                if self.subPlotTitleValueList[objectCounter] > 299.9 and self.subPlotTitleValueList[
                    objectCounter] < 300.1:
                    alphaRadians = '$\\frac{5}{3}\\pi$'
                    stabilityLineStyle = '--'
                subtitleString = self.subplotTitle + alphaRadians
            else:
                # subtitleString = self.subplotTitleTwo + str(
                #     "{:2.3f}".format(self.subPlotTitleValueListTwo[objectCounter])) + ' \n' \
                #                  + self.subplotTitle + str(
                #     "{:2.3f}".format(self.subPlotTitleValueList[objectCounter]))
                subtitleString = str(objectCounter+1)
                stabilityLineStyle = '-'

            objectCounter = objectCounter + 1

            arr[0].semilogy(self.orbitObjects[i].continuationParameter, self.orbitObjects[i].v1, c=self.plottingColors[colour_family][i], linestyle = stabilityLineStyle, label=subtitleString)
            #arr[1].semilogy(self.orbitObjects[i].continuationParameter, self.orbitObjects[i].v2, c=self.plottingColors[colour_family][i], label=subtitleString)
            if self.varyingQuantity == 'Hamiltonian':
                arr[1].plot(self.orbitObjects[i].T, self.orbitObjects[i].Hlt, c=self.plottingColors[colour_family][i], linestyle = stabilityLineStyle, label=subtitleString)
                arr[1].set_xlabel('$T$ [-]')
                arr[1].set_ylabel('$H_{lt}$ [-]')
            elif self.varyingQuantity == 'Acceleration':
                arr[1].plot(self.orbitObjects[i].continuationParameter, self.orbitObjects[i].T, c=self.plottingColors[colour_family][i], linestyle = stabilityLineStyle, label=subtitleString)
                arr[1].set_xlabel('$a_{lt}$ [-]')
                arr[1].set_ylabel('$T$ [-]')
            else:
                arr[1].plot(self.orbitObjects[i].continuationParameter, self.orbitObjects[i].T,
                            c=self.plottingColors[colour_family][i], label=subtitleString)
                arr[1].set_xlabel('$\\alpha$ [rad]')
                arr[1].set_ylabel('$T$ [-]')

            if i == len(self.orbitObjects) - 1:
                arr[0].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
                #arr[1].axhline(1, c=self.plottingColors['limit'], linewidth=1, linestyle='--')

        arr[0].set_xlim([continuationParameter_min, continuationParameter_max])
        #arr[1].set_xlim([continuationParameter_min, continuationParameter_max])
        arr[0].set_ylim([1.0e-1, 1.0e4])
        #arr[1].set_ylim([1.0e-1, 1.0e1])
        arr[0].set_xlabel(self.continuationLabel)
        #arr[1].set_xlabel(self.continuationLabel)
        arr[1].set_xlim([0,0.1])

        arr[0].set_ylabel('Stability Index [-]')

        arr[0].set_title('$\\nu_{1}$')
        #arr[1].set_title('$\\nu_{2}$')


        f.subplots_adjust(left=0.06,bottom=0.25,top=0.89)


        lgd = arr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})


        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_stability_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_stability_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Acceleration':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_stability_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_stability_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Alpha':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_stability_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_stability_analysis.png', transparent=True, dpi=300)

        plt.close()
        pass

    def hamiltonian_domain_analysis(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax = fig.gca()

        numberOfPlots=len(self.orbitObjects)

        # Determine complete the bounds of the colormap
        hamiltonian_min = 50000
        hamiltonian_max = -50000

        for i in range(numberOfPlots):
            if min(self.orbitObjects[i].Hlt) < hamiltonian_min:
                hamiltonian_min = min(self.orbitObjects[i].Hlt)
            if max(self.orbitObjects[i].Hlt) > hamiltonian_max:
                hamiltonian_max = max(self.orbitObjects[i].Hlt)

        orbitalPeriod_min = 50000
        orbitalPeriod_max = -50000

        for i in range(numberOfPlots):
            if min(self.orbitObjects[i].T) < orbitalPeriod_min:
                orbitalPeriod_min = min(self.orbitObjects[i].T)
            if max(self.orbitObjects[i].T) > orbitalPeriod_max:
                orbitalPeriod_max = max(self.orbitObjects[i].T)

        ax.set_xlim([orbitalPeriod_min-0.01,orbitalPeriod_max+0.01])
        ax.set_ylim([hamiltonian_min-0.1,hamiltonian_max+0.1])


        if len(self.orbitObjects) == 4:
            colour_family = 'fourFamilies'
        else:
            colour_family = 'sixFamilies'

        objectCounter = 0

        for i in range(numberOfPlots):
            # parameter for label:
            if self.varyingQuantity != 'Hamiltonian':
                subtitleString = self.subplotTitle + str(
                    "{:4.1f}".format(self.subPlotTitleValueList[objectCounter]))
            else:
                if self.subPlotTitleValueList[objectCounter] > -0.1 and self.subPlotTitleValueList[
                    objectCounter] < 0.1:
                    alphaRadians = '$0$'
                if self.subPlotTitleValueList[objectCounter] > 59.9 and self.subPlotTitleValueList[
                    objectCounter] < 60.1:
                    alphaRadians = '$\\frac{1}{3}\\pi$'
                if self.subPlotTitleValueList[objectCounter] > 119.9 and self.subPlotTitleValueList[
                    objectCounter] < 120.1:
                    alphaRadians = '$\\frac{2}{3}\\pi$'
                if self.subPlotTitleValueList[objectCounter] > 179.9 and self.subPlotTitleValueList[
                    objectCounter] < 180.1:
                    alphaRadians = '$\\pi$'
                if self.subPlotTitleValueList[objectCounter] > 239.9 and self.subPlotTitleValueList[
                    objectCounter] < 240.1:
                    alphaRadians = '$\\frac{4}{3}\\pi$'
                if self.subPlotTitleValueList[objectCounter] > 299.9 and self.subPlotTitleValueList[
                    objectCounter] < 300.1:
                    alphaRadians = '$\\frac{5}{3}\\pi$'
                subtitleString = self.subplotTitle + alphaRadians
            objectCounter = objectCounter + 1

            ax.plot(self.orbitObjects[i].T, self.orbitObjects[i].Hlt,c=self.plottingColors[colour_family][i], label=subtitleString)

        lgd = ax.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=10)

        plt.subplots_adjust(bottom=0.14,top=0.96,left=0.08)
        ax.set_xlabel('T [-]')
        ax.set_ylabel('$H_{lt}$ [-]')


        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_hamiltonian_domain_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_hamiltonian_domain_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Acceleration':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_hamiltonian_domain_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_hamiltonian_domain_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Alpha':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_hamiltonian_domain_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_hamiltonian_domain_analysis.png', transparent=True, dpi=300)

        plt.close()
        pass

    def shooting_conditions_analysis(self):
        f, arr = plt.subplots(1, 2, figsize=self.figSizeWide)

        arr[0].set_xlabel('x [-]')
        arr[0].set_ylabel('y [-]')
        arr[0].set_title('Projection of shooting conditions')

        # Plot libration point
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']

        for lagrange_point_nr in lagrange_point_nrs:
            arr[0].scatter(lagrange_points_df[lagrange_point_nr]['x'],
                            lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

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

        arr[0].contourf(xM, yM, zM, colors='black')
        arr[0].contourf(xE, yE, zE, colors='black')

        if self.varyingQuantity == 'Hamiltonian' or self.varyingQuantity == 'Acceleration':
            if self.orbitObjects[0].alpha < 59.0:
                alphaTitle1 = '0'
            elif self.orbitObjects[0].alpha > 1.0 and self.orbitObjects[0].alpha < 61.0:
                alphaTitle1 = '\\frac{1}{3}\\pi'
            elif self.orbitObjects[0].alpha > 61.0 and self.orbitObjects[0].alpha < 121.0:
                alphaTitle1 = '\\frac{2}{3}\\pi'
            elif self.orbitObjects[0].alpha > 121.0 and self.orbitObjects[0].alpha < 181.0:
                alphaTitle1 = '\\pi'
            elif self.orbitObjects[0].alpha > 181.0 and self.orbitObjects[0].alpha < 241.0:
                alphaTitle1 = '\\frac{4}{3}\\pi'
            elif self.orbitObjects[0].alpha > 241.0 and self.orbitObjects[0].alpha < 301.0:
                alphaTitle1 = '\\frac{5}{3}\\pi'

            if self.orbitObjects[1].alpha < 59.0:
                alphaTitle2 = '0'
            elif self.orbitObjects[1].alpha > 1.0 and self.orbitObjects[1].alpha < 61.0:
                alphaTitle2 = '\\frac{1}{3}\\pi'
            elif self.orbitObjects[1].alpha > 61.0 and self.orbitObjects[1].alpha < 121.0:
                alphaTitle2 = '\\frac{2}{3}\\pi'
            elif self.orbitObjects[1].alpha > 121.0 and self.orbitObjects[1].alpha < 181.0:
                alphaTitle2 = '\\pi'
            elif self.orbitObjects[1].alpha > 181.0 and self.orbitObjects[1].alpha < 241.0:
                alphaTitle2 = '\\frac{4}{3}\\pi'
            elif self.orbitObjects[1].alpha > 241.0 and self.orbitObjects[1].alpha < 301.0:
                alphaTitle2 = '\\frac{5}{3}\\pi'

            # Create new color maps
            Orbit1 = self.orbitObjects[0]
            Orbit2 = self.orbitObjects[1]

            Hlt_min = min(min(Orbit1.continuationParameter), min(Orbit2.continuationParameter))
            Hlt_max = max(max(Orbit1.continuationParameter), max(Orbit2.continuationParameter))

            continuation_normalized_orbit1 = [(value - Hlt_min) / (Hlt_max - Hlt_min) for value in
                                                  Orbit1.continuationParameter]
            continuation_normalized_orbit2 = [(value - Hlt_min) / (Hlt_max - Hlt_min) for value in
                                                  Orbit2.continuationParameter]

            # print(continuation_normalized_orbit1)

            number_of_colors_orbit1 = len(Orbit1.continuationParameter)
            number_of_colors_orbit2 = len(Orbit2.continuationParameter)

            colors_orbit1 = matplotlib.colors.ListedColormap(
                    sns.color_palette("viridis_r", number_of_colors_orbit1))(continuation_normalized_orbit1)
            colors_orbit2 = matplotlib.colors.ListedColormap(
                    sns.color_palette("viridis_r", number_of_colors_orbit2))(continuation_normalized_orbit2)
            print('length of colors_orbit_1:' + str(len(colors_orbit1)))
            #print(colors_orbit1.tolist())


            #print(plotColorIndexBasedOnHlt_Orbit1)
            sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",
                                                                                                   (
                                                                                                       number_of_colors_orbit1))),
                                           norm=plt.Normalize(vmin=Hlt_min, vmax=Hlt_max))

            orbitIdsPlot_orbit1 = list(range(0, len(Orbit1.continuationParameter), 1))


            arr[0].scatter(self.orbitObjects[0].x, self.orbitObjects[0].y, c=colors_orbit1,s=2)
            arr[0].scatter(self.orbitObjects[1].x, self.orbitObjects[1].y, c=colors_orbit2,s=2)

            sm.set_array([])

            position_handle = arr[0].get_position().bounds
            position_handle2 = arr[0].get_position().bounds

            colourbar_base = position_handle2[1] +0.02
            colourbar_height = position_handle[1] + position_handle[3] - colourbar_base -0.047


            axColorbar = f.add_axes([0.4, colourbar_base, 0.02, colourbar_height])
            axColorbar.get_xaxis().set_visible(False)
            axColorbar.get_yaxis().set_visible(False)
            axColorbar.set_visible(False)

            divider = make_axes_locatable(axColorbar)

            cax = divider.append_axes("left", size="100%", pad=0.0)

            #plt.subplots_adjust(left=0.065, bottom=0.05, top=0.96)

            cbar = plt.colorbar(sm, cax=cax, label=self.continuationLabel)

            minimum_x = min(min(self.orbitObjects[0].x),min(self.orbitObjects[1].x))
            minimum_y = min(min(self.orbitObjects[0].y),min(self.orbitObjects[1].y))

            maximum_x = max(max(self.orbitObjects[0].x),max(self.orbitObjects[1].x))
            maximum_y = max(max(self.orbitObjects[0].y),max(self.orbitObjects[1].y))

            xMiddle = minimum_x + (maximum_x - minimum_x) / 2
            yMiddle = minimum_y + (maximum_y - minimum_y) / 2

            scaleDistance = max((maximum_y - minimum_y), (maximum_x - minimum_x))

            ### Compute deviations

            minimum_length = min(len(self.orbitObjects[0].continuationParameter),len(self.orbitObjects[1].continuationParameter))
            maximum_length = max(len(self.orbitObjects[0].continuationParameter),len(self.orbitObjects[1].continuationParameter))
            print(minimum_length)

            position_deviation = []
            x_deviation = []
            y_deviation = []
            velocity_deviation = []
            continuation_parameter_deviation = []

            for i in range(minimum_length):
                position_deviation.append(np.sqrt( (self.orbitObjects[0].x[i]-self.orbitObjects[1].x[i]) ** 2 + \
                                              (self.orbitObjects[0].y[i] - self.orbitObjects[1].y[i]) ** 2))
                x_deviation.append(np.sqrt((self.orbitObjects[0].x[i] - self.orbitObjects[1].x[i]) ** 2 ))
                y_deviation.append(np.sqrt((self.orbitObjects[0].y[i] - -1.0 *self.orbitObjects[1].y[i]) ** 2 ))
                velocity_deviation.append(np.sqrt((self.orbitObjects[0].xdot[i] - self.orbitObjects[1].xdot[i]) ** 2 + \
                                                      (self.orbitObjects[0].ydot[i] - self.orbitObjects[1].ydot[i]) ** 2))
                continuation_parameter_deviation.append(
                    np.sqrt((self.orbitObjects[0].continuationParameter[i] - self.orbitObjects[1].continuationParameter[i]) ** 2 ))

            short_object_index = 0
            long_object_index = 0
            if maximum_length > minimum_length:
                if len(self.orbitObjects[0].continuationParameter) > len(self.orbitObjects[1].continuationParameter):
                    short_object_index = 1
                    long_object_index = 0
                else:
                    short_object_index = 0
                    long_object_index = 1

                for i in range(minimum_length,maximum_length):
                    position_deviation.append(np.sqrt((self.orbitObjects[long_object_index].x[i]) ** 2 + \
                                                      (self.orbitObjects[long_object_index].y[i]) ** 2))
                    x_deviation.append(np.sqrt(( self.orbitObjects[long_object_index].x[i]) ** 2))
                    y_deviation.append(np.sqrt((self.orbitObjects[long_object_index].y[i] ) ** 2))
                    velocity_deviation.append(
                    np.sqrt((self.orbitObjects[long_object_index].xdot[i]) ** 2 + \
                                ( self.orbitObjects[long_object_index].ydot[i]) ** 2))
                    continuation_parameter_deviation.append(
                        np.sqrt((self.orbitObjects[long_object_index].continuationParameter[i]) ** 2))




            if self.varyingQuantity == 'Hamiltonian':
                thirdLabel = '$|\\Delta H_{lt}|$'
            else:
                thirdLabel = '$|\\Delta a_{lt}|$'

            #arr[1].plot(self.orbitObjects[long_object_index].orbitsId[0:maximum_lengthh],position_deviation,color=self.plottingColors['tripleLine'][0],label='||$\\Delta \\bar{R}||$')
            arr[1].semilogy(self.orbitObjects[long_object_index].orbitsId[0:maximum_length], x_deviation,color=self.plottingColors['tripleLine'][0], label='$|\\Delta x|$')
            arr[1].semilogy(self.orbitObjects[long_object_index].orbitsId[0:maximum_length], y_deviation,color=self.plottingColors['tripleLine'][1], label='$|y^{i}_{'+str(alphaTitle1)+'}+y^{i}_{'+str(alphaTitle2)+'}|$')

            #arr[1].plot(self.orbitObjects[long_object_index].orbitsId[0:maximum_length],velocity_deviation,color=self.plottingColors['tripleLine'][1],label='||$\\Delta \\bar{V}||$')
            arr[1].plot(self.orbitObjects[long_object_index].orbitsId[0:maximum_length], continuation_parameter_deviation,color=self.plottingColors['tripleLine'][2], label=thirdLabel)
            lgd = arr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.83), markerscale=8)
            arr[1].set_ylim([1.0e-12,1.0e1])
            arr[1].set_xlim([0,maximum_length])

        if self.varyingQuantity == 'Alpha':
            Orbit1 = self.orbitObjects[0]

            Hlt_min = 0.0
            Hlt_max = 360.0

            continuation_normalized_orbit1 = [(value - Hlt_min) / (Hlt_max - Hlt_min) for value in
                                              Orbit1.continuationParameter]

            # print(continuation_normalized_orbit1)

            number_of_colors_orbit1 = len(Orbit1.continuationParameter)

            colors_orbit1 = matplotlib.colors.ListedColormap(
                sns.color_palette("viridis_r", number_of_colors_orbit1))(continuation_normalized_orbit1)

            sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r",
                                                                                               (number_of_colors_orbit1))),
                                       norm=plt.Normalize(vmin=Hlt_min, vmax=Hlt_max))

            orbitIdsPlot_orbit1 = list(range(0, len(Orbit1.continuationParameter), 1))

            arr[0].scatter(self.orbitObjects[0].x, self.orbitObjects[0].y, c=colors_orbit1, s=2)

            sm.set_array([])

            position_handle = arr[0].get_position().bounds
            position_handle2 = arr[0].get_position().bounds

            colourbar_base = position_handle2[1] + 0.02
            colourbar_height = position_handle[1] + position_handle[3] - colourbar_base - 0.047

            axColorbar = f.add_axes([0.4, colourbar_base, 0.02, colourbar_height])
            axColorbar.get_xaxis().set_visible(False)
            axColorbar.get_yaxis().set_visible(False)
            axColorbar.set_visible(False)

            divider = make_axes_locatable(axColorbar)

            cax = divider.append_axes("left", size="100%", pad=0.0)

            # plt.subplots_adjust(left=0.065, bottom=0.05, top=0.96)

            cbar = plt.colorbar(sm, cax=cax, label=self.continuationLabel)

            if self.varyingQuantity == 'Alpha':
                cbar.set_ticks([0, 90, 180, 270, 360])
                cbar.set_ticklabels(['$0$', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$'])

            minimum_x = min(self.orbitObjects[0].x)
            minimum_y = min(self.orbitObjects[0].y)

            maximum_x = max(self.orbitObjects[0].x)
            maximum_y = max(self.orbitObjects[0].y)

            xMiddle = minimum_x + (maximum_x - minimum_x) / 2
            yMiddle = minimum_y + (maximum_y - minimum_y) / 2

            scaleDistance = max((maximum_y - minimum_y), (maximum_x - minimum_x))

            value_alpha = []
            index_alpha = []
            value_minus_alpha = []
            index_minus_alpha = []
            ## Obtain symmetry analysis [DEVIATIONS DUE TO PHASE SHIFT!]
            for i in range(len(Orbit1.alphaContinuation)):
                currentAlpha = Orbit1.alphaContinuation[i]
                # print(currentAlpha)
                if Orbit1.alphaContinuation[i] > 0.5 and Orbit1.alphaContinuation[i] < 179.5:
                    value_alpha.append(currentAlpha)
                    index_alpha.append(Orbit1.alphaContinuation.index(currentAlpha))

                    minus_alpha = 360.0 - currentAlpha
                    value_minus_alpha.append(minus_alpha)
                    index_minus_alpha.append(Orbit1.alphaContinuation.index(minus_alpha))

            value_list_length = min(len(value_alpha), len(value_minus_alpha))

            position_deviation = []
            x_deviation = []
            y_deviation = []
            velocity_deviation = []
            continuation_parameter_deviation = []

            for i in range(value_list_length):
                loop_index_alpha = index_alpha[i]
                loop_index_minus_alpha = index_minus_alpha[i]

                position_deviation.append(np.sqrt(
                    (self.orbitObjects[0].x[loop_index_alpha] - self.orbitObjects[0].x[loop_index_minus_alpha]) ** 2 + \
                    (self.orbitObjects[0].y[loop_index_alpha] - self.orbitObjects[0].y[loop_index_minus_alpha]) ** 2))
                x_deviation.append(np.sqrt(
                    (self.orbitObjects[0].x[loop_index_alpha] - self.orbitObjects[0].x[loop_index_minus_alpha]) ** 2))
                y_deviation.append(np.sqrt((self.orbitObjects[0].y[loop_index_alpha] - -1.0 * self.orbitObjects[0].y[
                    loop_index_minus_alpha]) ** 2))
                velocity_deviation.append(np.sqrt((self.orbitObjects[0].xdot[loop_index_alpha] -
                                                   self.orbitObjects[0].xdot[loop_index_minus_alpha]) ** 2 + \
                                                  (self.orbitObjects[0].ydot[loop_index_alpha] -
                                                   self.orbitObjects[0].ydot[loop_index_minus_alpha]) ** 2))
                continuation_parameter_deviation.append(
                    np.sqrt((self.orbitObjects[0].continuationParameter[loop_index_alpha] -
                             self.orbitObjects[0].continuationParameter[
                                 loop_index_minus_alpha]) ** 2))

            arr[1].semilogy(value_alpha,x_deviation,color=self.plottingColors['tripleLine'][0], label='$|\\Delta x|$')
            arr[1].semilogy(value_alpha,y_deviation,color=self.plottingColors['tripleLine'][1], label='$|y^{i}_{\\alpha}+y^{i}_{-\\alpha}|$')
            #arr[1].plot(value_alpha,continuation_parameter_deviation,color=self.plottingColors['tripleLine'][2],label='$|\\Delta \\alpha|$')
            arr[1].set_xlim([0,180.0])
            arr[1].set_ylim([1.0e-12, 1.0e1])
            lgd = arr[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.83), markerscale=8)
            print(y_deviation)

        # Plot the shooting conditions
        arr[0].set_xlim([(xMiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(xMiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        arr[0].set_ylim([yMiddle - 0.5 * scaleDistance * self.spacingFactor, yMiddle + 0.5 * scaleDistance * self.spacingFactor])


        #arr[0].set_aspect(1.0)

        if self.varyingQuantity == 'Hamiltonian':
            arr[1].set_xlabel('Orbit number [-]')
        elif self.varyingQuantity == 'Acceleration':
            arr[1].set_xlabel('Orbit number [-]')
        else:
            arr[1].set_xlabel('$\\alpha$ [rad]')

        arr[1].set_ylabel('$||\\Delta \\bar{R}||$, $||\\Delta \\bar{V}||$ [-]')
        arr[1].set_title('Deviation analysis')


        if self.varyingQuantity == 'Hamiltonian':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + ' ($a_{lt} = ' + str(
                "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + alphaTitle1 + '$ rad and $' + alphaTitle2 + '$ rad) ' + ' - Shooting symmetry verification',
                         size=self.suptitleSize)
        if self.varyingQuantity == 'Acceleration':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $\\alpha = ' + alphaTitle1 + '$ rad and $' + alphaTitle2 + '$ rad) ' + ' - Shooting symmetry verification',
                         size=self.suptitleSize)
        if self.varyingQuantity == 'Alpha':
            plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + ' ($H_{lt} = ' + str(
                "{:3.3f}".format(self.Hamiltonian)) + '$, $a_{lt}= ' + str("{:3.2f}".format(self.accelerationMagnitude)) + '$) ' + ' - Shooting symmetry verification',
                         size=self.suptitleSize)

        #plt.tight_layout()
        if self.varyingQuantity != 'Alpha':
            plt.subplots_adjust(left=0.065,bottom=0.13,top=0.83,wspace = 0.43,right=0.87)
        else:
            plt.subplots_adjust(left=0.069, bottom=0.13, top=0.83, wspace=0.43, right=0.87)



        if self.varyingQuantity == 'Hamiltonian':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_shooting_analysis.png', transparent=True, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig('../../data/figures/orbits/varying_hamiltonian/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:5.4f}".format(self.alpha)) + \
                            '_shooting_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Acceleration':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_shooting_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_acceleration/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.alpha)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_shooting_analysis.png', transparent=True, dpi=300)

        if self.varyingQuantity == 'Alpha':
            if self.lowDpi:
                plt.savefig('../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                            + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                            '_shooting_analysis.png', transparent=True, dpi=self.dpi)

            else:
                plt.savefig(
                    '../../data/figures/orbits/varying_alpha/L' + str(self.lagrangePointNr) + '_horizontal_' \
                    + str("{:5.4f}".format(self.accelerationMagnitude)) + '_' + str("{:12.11f}".format(self.Hamiltonian)) + \
                    '_shooting_analysis.png', transparent=True, dpi=300)

        plt.close()




if __name__ == '__main__':

    ballistic_planar_projection = True
    ballistic_bifurcation_analysis = False
    ballistic_stability_analysis = False
    graphical_projection = False
    bifurcation_analysis = False
    stability_analysis = False
    hamiltonian_domain_analysis = False
    shooting_analysis = False

    if ballistic_planar_projection == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.0
        alpha = 0.0
        beta = 0.0
        hamiltonian = 0.0
        varying_quantity = 'Hamiltonian'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False

        orbitL1 = DisplayPeriodicSolutions('horizontal',1,acceleration_magnitude,alpha,beta,varying_quantity,low_dpi, plot_as_x_coordinate, plot_as_family_number )
        orbitL2 = DisplayPeriodicSolutions('horizontal',2,acceleration_magnitude,alpha,beta,varying_quantity,low_dpi, plot_as_x_coordinate, plot_as_family_number )


        my_objects = []
        my_objects.append(orbitL1)
        my_objects.append(orbitL2)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr,acceleration_magnitude, alpha, hamiltonian,varying_quantity, my_objects, low_dpi,plot_as_x_coordinate,plot_as_family_number)

        #characterize_periodic_solutions.ballistic_graphical_projection()
        characterize_periodic_solutions.ballistic_bifurcation_analysis()
        #characterize_periodic_solutions.ballistic_stability_analysis()


        del characterize_periodic_solutions

    if ballistic_bifurcation_analysis == True:
        lagrange_point_nr = 1
        acceleration_magnitude = 0.05
        alpha = 0.0
        beta = 0.0
        hamiltonian = 0.0
        varying_quantity = 'Hamiltonian'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False

        orbitL1 = DisplayPeriodicSolutions('horizontal', 1, acceleration_magnitude, alpha, beta, varying_quantity,low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbitL2 = DisplayPeriodicSolutions('horizontal', 2, acceleration_magnitude, alpha, beta, varying_quantity,low_dpi,plot_as_x_coordinate,plot_as_family_number)

        my_objects = []
        my_objects.append(orbitL1)
        my_objects.append(orbitL2)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr, acceleration_magnitude,
                                                                            alpha, hamiltonian, varying_quantity, my_objects,
                                                                            low_dpi,plot_as_x_coordinate,plot_as_family_number)

        characterize_periodic_solutions.ballistic_bifurcation_analysis()

        del characterize_periodic_solutions

    if ballistic_stability_analysis == True:
        lagrange_point_nr = 1
        acceleration_magnitude = 0.1
        alpha = 0.0
        hamiltonian = 0.0
        varying_quantity = 'Hamiltonian'
        low_dpi = True
        plot_as_x_coordinate = False
        plot_as_family_number = False


        orbitL1 = DisplayPeriodicSolutions('horizontal', 1, acceleration_magnitude, alpha, hamiltonian, varying_quantity,low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbitL2 = DisplayPeriodicSolutions('horizontal', 2, acceleration_magnitude, alpha, hamiltonian,  varying_quantity,low_dpi,plot_as_x_coordinate,plot_as_family_number)

        my_objects = []
        my_objects.append(orbitL1)
        my_objects.append(orbitL2)
        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr, acceleration_magnitude,
                                                                            alpha, hamiltonian, varying_quantity, my_objects,
                                                                            low_dpi,plot_as_x_coordinate,plot_as_family_number)

        characterize_periodic_solutions.ballistic_stability_analysis()

        del characterize_periodic_solutions

    if  graphical_projection == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.01
        alpha = 0.0
        beta = 0.0
        hamiltonian = -1.50
        varying_quantity = 'Hamiltonian'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False



        orbit1 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbit2 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 60.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbit3 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 120.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbit4 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 180.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbit5 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 300.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        orbit6 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 240.0, hamiltonian, varying_quantity, low_dpi,plot_as_x_coordinate,plot_as_family_number)
        # orbit7 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, 0.1, alpha, hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        # orbit8 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, 0.1, alpha, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        # orbit9 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, 0.1, alpha, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)

        my_objects = []
        my_objects.append(orbit1)
        my_objects.append(orbit2)
        my_objects.append(orbit3)
        my_objects.append(orbit4)
        my_objects.append(orbit5)
        my_objects.append(orbit6)
        # my_objects.append(orbit7)
        # my_objects.append(orbit8)
        # my_objects.append(orbit9)


        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr, acceleration_magnitude,
                                                                            alpha, hamiltonian, varying_quantity, my_objects,low_dpi,plot_as_x_coordinate,plot_as_family_number)

        #characterize_periodic_solutions.graphical_projection()
        #characterize_periodic_solutions.bifurcation_analysis()
        characterize_periodic_solutions.stability_analysis()

        del characterize_periodic_solutions

    if bifurcation_analysis == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.01
        alpha = 0.0
        beta = 0.0
        hamiltonian = -1.50
        varying_quantity = 'Acceleration'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False

        orbit1 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit2 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 60.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit3 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 120.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit4 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 180.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit5 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 300.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit6 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 240.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit7 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit8 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit9 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)

        my_objects = []
        my_objects.append(orbit1)
        my_objects.append(orbit2)
        my_objects.append(orbit3)
        my_objects.append(orbit4)
        my_objects.append(orbit5)
        my_objects.append(orbit6)
        #my_objects.append(orbit7)
        #my_objects.append(orbit8)
        #my_objects.append(orbit9)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr,acceleration_magnitude,alpha, hamiltonian, varying_quantity,my_objects, low_dpi,plot_as_x_coordinate,plot_as_family_number)

        characterize_periodic_solutions.bifurcation_analysis()

        del characterize_periodic_solutions

    if stability_analysis == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.05
        alpha = 0.0
        beta = 0.0
        hamiltonian = -1.50
        varying_quantity = 'Hamiltonian'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False

        orbit1 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit2 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 60.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit3 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 120.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit4 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 180.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit5 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 300.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit6 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 240.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit7 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, alpha, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit8 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, alpha, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        #orbit9 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, alpha, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)

        my_objects = []
        my_objects.append(orbit1)
        my_objects.append(orbit2)
        my_objects.append(orbit3)
        my_objects.append(orbit4)
        my_objects.append(orbit5)
        my_objects.append(orbit6)
        #my_objects.append(orbit7)
        #my_objects.append(orbit8)
        #my_objects.append(orbit9)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr, acceleration_magnitude,alpha, hamiltonian, varying_quantity,my_objects, low_dpi, plot_as_x_coordinate,plot_as_family_number)

        characterize_periodic_solutions.stability_analysis()

        del characterize_periodic_solutions

    if hamiltonian_domain_analysis == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.0
        alpha = 0.0
        beta = 0.0
        hamiltonian = -1.55
        varying_quantity = 'Acceleration'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False

        orbit0 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, 0.0, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit1 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 0.0, hamiltonian,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit2 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 60.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit3 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 120.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit4 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 180.0,hamiltonian, varying_quantity, low_dpi, plot_as_x_coordinate,plot_as_family_number)
        orbit5 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 300.0, beta,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbit6 = DisplayPeriodicSolutions('horizontal', lagrange_point_nr, acceleration_magnitude, 240.0, beta,varying_quantity, low_dpi, plot_as_x_coordinate, plot_as_family_number)

        my_objects = []
        my_objects.append(orbit1)
        my_objects.append(orbit2)
        my_objects.append(orbit3)
        my_objects.append(orbit4)
        my_objects.append(orbit5)
        my_objects.append(orbit6)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr,acceleration_magnitude, alpha,hamiltonian, varying_quantity,my_objects, low_dpi,plot_as_x_coordinate,plot_as_family_number)

        characterize_periodic_solutions.hamiltonian_domain_analysis()

        del characterize_periodic_solutions

    if shooting_analysis == True:
        lagrange_point_nr = 2
        acceleration_magnitude = 0.1
        alpha = 0.0
        beta = 0.0
        hamiltonian = -1.55
        varying_quantity = 'Hamiltonian'
        low_dpi = False
        plot_as_x_coordinate = False
        plot_as_family_number = False


        orbitL1 = DisplayPeriodicSolutions('horizontal', 2, 0.1, 60.0, hamiltonian, varying_quantity,low_dpi, plot_as_x_coordinate, plot_as_family_number)
        orbitL2 = DisplayPeriodicSolutions('horizontal', 2, 0.1, 300.0, hamiltonian, varying_quantity,low_dpi, plot_as_x_coordinate, plot_as_family_number)

        my_objects = []
        my_objects.append(orbitL1)
        my_objects.append(orbitL2)

        characterize_periodic_solutions = PeriodicSolutionsCharacterization(lagrange_point_nr, acceleration_magnitude,
                                                                            alpha, hamiltonian, varying_quantity,
                                                                            my_objects,
                                                                            low_dpi, plot_as_x_coordinate,
                                                                            plot_as_family_number)

        characterize_periodic_solutions.shooting_conditions_analysis()




