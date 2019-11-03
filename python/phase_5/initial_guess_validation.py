
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

from load_data_augmented import load_orbit_augmented, load_lagrange_points_location_augmented, load_differential_correction

# from matplotlib.ticker import ScalarFormatter
# class ScalarFormatterForceFormat(ScalarFormatter):
#     def _set_format(self,vmin,vmax):  # Override function that finds format to use.
#         self.format = "%3.1f"  # Give format here


class initialGuessValidation:
    def __init__(self, lagrange_point_nr, orbit_type, acceleration_magnitude, alpha, amplitude, correction_time, low_dpi):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.amplitude = amplitude
        self.correctionTime = correction_time

        ## Check which effect is calculated
        if isinstance(self.amplitude, list):
            if self.orbitType == 'horizontal':
                self.numberOfSolutions = 6
                self.numberOfAmplitudes = len(self.amplitude)
                self.numberOfAxisTicks = 5
            else:
                self.numberOfSolutions = 4
                self.numberOfAmplitudes = len(self.amplitude)
                self.numberOfAxisTicks = 5
        elif isinstance(self.alpha, list):
            self.numberOfSolutions = 8
            self.numberOfAlphas = len(self.alpha)
            self.numberOfAxisTicks = 9
            self.numberOfAxisTicksOrtho = 5
        elif isinstance(self.accelerationMagnitude, list):
            self.numberOfSolutions = 6
            self.numberOfAccelerations = len(self.accelerationMagnitude)
            self.numberOfAxisTicks = 4
            self.numberOfAxisTicksOrtho = 5
        else:
            self.numberOfCorrectionPlots = 4
            self.numberOfSolutions = 4
            self.numberOfAxisTicks = 6
            self.numberOfAxisTicksOrtho = 6




        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150

        self.suptitleSize = 20

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        self.scaleDistanceY = 2.5
        self.scaleDistanceX = self.scaleDistanceY * self.figureRatio

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        self.figSizThree = (7 * (1 + np.sqrt(5)) / 2,7*1.5)


        self.scaleDistanceXWide = self.scaleDistanceY * self.figureRatioWide

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
        self.lineWidth=1

    def plot_amplitude_effect(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$|A|$ [-]')
        ax2.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax1.set_title('Initial guesses')
        ax2.set_title('State deviations at full period')

        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

        # Create information of orbits to be plotted
        orbitIdsPlot = list(range(0, len(self.amplitude), 1))
        deviation_list = []

        deviation_list = []
        deviation_corrected_list = []

        #indexPlotlist = np.linspace(0, len(self.amplitude) - 1, num=self.numberOfSolutions).tolist()
        indexPlotlist = [271, 286 ,311, 336, 361]
        #print(self.amplitude[336])

        Indexlist = 0

        # Compute the deviations and plot the desired intial guesses
        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
            + str("{:7.6f}".format(self.amplitude[i])) + '_' \
            + str("{:7.6f}".format(self.correctionTime)) + '_stateHistory.txt')



            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            deviation_list.append([self.amplitude[i], deltaR, deltaV])

            if i == indexPlotlist[Indexlist]:
                legendString = '$|A| = $' + str("{:2.1e}".format(self.amplitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', 6)[Indexlist], linewidth=1,label=legendString)
                Indexlist = Indexlist + 1

        # Create the ax2 plot
        deviation_df = pd.DataFrame(deviation_list, columns=['amplitude', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaR'], label='$|\\Delta R|$',color=sns.color_palette('viridis', 2)[0],linewidth=1)
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaV'], label='$|\\Delta V|$', color=sns.color_palette('viridis', 2)[1],linewidth=1)

        # Create axes formats and legends for ax1



        scaleDistance = max((max(df['x']) - min(df['x'])), (max(df['y']) - min(df['y'])))

        Xmiddle = min(df['x']) + (max(df['x']) - min(df['x'])) / 2
        Ymiddle = min(df['y']) + (max(df['y']) - min(df['y'])) / 2


        # figW, figH = ax1.get_figure().get_size_inches()
        # _, _, w, h = ax1.get_position().bounds
        # # Ratio of display units
        # disp_ratio = (figH * h) / (figW * w)

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks(xticks)

        ## Test for axes ratio
        #print(1 / ax1.get_data_ratio())

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

        lgd  = ax1.legend(frameon=True, loc='upper left',  bbox_to_anchor=(0, 1),prop={'size': 8})

        # Create the axes formats and legends for ax2
        ax2.set_xlim([min(deviation_df['amplitude']), max(deviation_df['amplitude'])])
        ax2.set_ylim([0, max(max(deviation_df['deltaV']),max(deviation_df['deltaR'])) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
        xticks2 = (np.linspace(min(deviation_df['amplitude']), max(deviation_df['amplitude']), num=self.numberOfAxisTicks))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']),max(deviation_df['deltaV'])*self.spacingFactor), num=self.numberOfAxisTicks))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2  = ax2.legend(frameon=True, loc='upper left',  bbox_to_anchor=(0, 1),prop={'size': 8})

        # Add a subtitle and do a tight layout
        suptitle = fig.suptitle('$L_{'+str(self.lagrangePointNr)+'}$ ($a_{lt} = '+ str("{:2.1e}".format(self.accelerationMagnitude))+' $, $\\alpha ='+ str("{:3.1f}".format(self.alpha)) +' $) Amplitude effect ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.83)


        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.correctionTime)) + \
                        '_amplitude_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd, lgd2, suptitle), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_amplitude_effect.pdf', transparent=True)

        plt.close()
        pass

    def plot_angle_effect(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$|A|$ [-]')
        ax2.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax1.set_title('Initial guesses')
        ax2.set_title('State deviations at full period')


        orbitIdsPlot = list(range(0, len(self.alpha), 1))

        # deviation_df = pd.DataFrame({'Amplitude': [], 'DeltaR': [], 'DeltaV': []})
        deviation_list = []

        #indexPlotlist = np.linspace(0, 315, num=self.numberOfSolutions).tolist()
        indexPlotlist = [0, 60, 120, 180, 240, 300]
        Indexlist = 0
        print(self.alpha[0])
        print(self.alpha[60])
        print(self.alpha[120])
        print(self.alpha[180])
        print(self.alpha[240])
        print(self.alpha[300])


        minimumX = 0.0
        minimumY = 0.0
        maximumX = 0.0
        maximumY = 0.0
        maximumDevR = 0.0
        maximumDevV = 0.0



        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha[i])) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str("{:7.6f}".format(self.correctionTime)) + '_stateHistory.txt')


            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])


            deviation_list.append([self.alpha[i], deltaR, deltaV])

            lagrange_points_df = load_lagrange_points_location()

            if self.lagrangePointNr == 1:
                lagrange_point_nrs = ['L1']
            if self.lagrangePointNr == 2:
                lagrange_point_nrs = ['L2']

            for lagrange_point_nr in lagrange_point_nrs:
                ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            color='black', marker='x')

            if i == indexPlotlist[Indexlist]:

                legendString = '$\\alpha = $' + str("{:4.1f}".format(self.alpha[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', 6)[Indexlist], linewidth=1, label= legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha[i])
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', 6)[Indexlist], marker='x')

                if Indexlist == 0.0:
                    minimumX = min(df['x'])
                    minimumY = min(df['y'])

                    maximumX = max(df['x'])
                    maximumY = max(df['y'])

                else:
                    minimumX_temp = min(df['x'])
                    minimumY_temp = min(df['y'])

                    maximumX_temp = max(df['x'])
                    maximumY_temp = max(df['y'])

                    if minimumX_temp < minimumX:
                        minimumX = minimumX_temp
                    if minimumY_temp < minimumY:
                        minimumY = minimumY_temp

                    if maximumX_temp > maximumX:
                        maximumX = maximumX_temp
                    if maximumY_temp > maximumY:
                        maximumY = maximumY_temp

                if Indexlist < len(indexPlotlist)-1:
                    Indexlist = Indexlist + 1

        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])


        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks1 = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),num=self.numberOfAxisTicksOrtho))
        ax1.xaxis.set_ticks(xticks1)

        lgd1 = ax1.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.7),prop={'size': 8})

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


        # Plot deviations ax2
        deviation_df = pd.DataFrame(deviation_list, columns=['alpha', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df['alpha'], deviation_df['deltaR'],
                 color=sns.color_palette('viridis', 2)[0], linewidth=1, label='$| \\Delta R |$ ')
        ax2.plot(deviation_df['alpha'], deviation_df['deltaV'],
                 color=sns.color_palette('viridis', 2)[1], linewidth=1, label='$| \\Delta V |$ ')

        ax2.set_xlim([0, 2*np.pi])
        ax2.set_ylim([0, max(deviation_df['deltaV']) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks2 = (np.linspace(0,360,num=7))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']), max(deviation_df['deltaV']) * self.spacingFactor),
                               num=self.numberOfAxisTicksOrtho))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2 = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.9),prop={'size': 8})

        # Add a subtitle and do a tight layout
        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = ' + str(
            "{:2.1e}".format(self.accelerationMagnitude)) + ' $, $|A| =' + str(
            "{:2.1e}".format(self.amplitude)) + ' $) Angle effect ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.83)

        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str("{:7.6f}".format(self.correctionTime)) + \
                        '_angle_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd1,lgd2,suptitle), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str("{:7.6f}".format(self.correctionTime)) + \
                        '_angle_effect.pdf', transparent=True)

        plt.close()
        pass

    def plot_acceleration_effect(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$|A|$ [-]')
        ax2.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax1.set_title('Initial guesses')
        ax2.set_title('State deviations at full period')

        orbitIdsPlot = list(range(0, len(self.accelerationMagnitude), 1))

        # deviation_df = pd.DataFrame({'Amplitude': [], 'DeltaR': [], 'DeltaV': []})
        deviation_list = []
        deviation_corrected_list = []

        #indexPlotlist = np.linspace(0, len(self.accelerationMagnitude) - 1, num=self.numberOfSolutions).tolist()
        indexPlotlist = [90,105,130,155,180]
        print(self.accelerationMagnitude[90])
        print(self.accelerationMagnitude[105])
        print(self.accelerationMagnitude[130])
        print(self.accelerationMagnitude[155])
        print(self.accelerationMagnitude[180])


        Indexlist = 0

        minimumX = 0.0
        minimumY = 0.0
        maximumX = 0.0
        maximumY = 0.0
        maximumDevR = 0.0
        maximumDevV = 0.0

        df = load_orbit_augmented(
            '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(0.000000)) + '_' + str("{:7.6f}".format(0.000000)) + '_' \
            + str("{:7.6f}".format(self.amplitude)) + '_' \
            + str("{:7.6f}".format(self.correctionTime)) + '_stateHistory.txt')

        deviations = df.head(1).values[0] - df.tail(1).values[0]
        deltaR = np.linalg.norm(deviations[1:4])
        deltaV = np.linalg.norm(deviations[4:7])

        deviation_list.append([0.000000, deltaR, deltaV])

        legendString = 'a$_{lt} = $' + str("{:2.1e}".format(0.000000))
        ax1.plot(df['x'], df['y'], color='black', linewidth=1, label=legendString)

        minimumX = min(df['x'])
        minimumY = min(df['y'])

        maximumX = max(df['x'])
        maximumY = max(df['y'])


        for i in orbitIdsPlot:
            df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude[i])) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str("{:7.6f}".format(self.correctionTime)) + '_stateHistory.txt')

            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            deviation_list.append([self.accelerationMagnitude[i], deltaR, deltaV])

            if i == indexPlotlist[Indexlist]:
                legendString = 'a$_{lt} = $' + str("{:2.1e}".format(self.accelerationMagnitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', 6)[Indexlist], linewidth=1, label= legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude[i], self.alpha)
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', 6)[Indexlist], marker='x')

                minimumX_temp = min(df['x'])
                minimumY_temp = min(df['y'])

                maximumX_temp = max(df['x'])
                maximumY_temp = max(df['y'])

                if minimumX_temp < minimumX:
                    minimumX = minimumX_temp
                if minimumY_temp < minimumY:
                    minimumY = minimumY_temp

                if maximumX_temp > maximumX:
                    maximumX = maximumX_temp
                if maximumY_temp > maximumY:
                    maximumY = maximumY_temp

                if Indexlist < len(indexPlotlist) - 1:
                    Indexlist = Indexlist + 1

        lagrange_points_df = load_lagrange_points_location_augmented(0.0, 0.0)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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


        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks(xticks)

        lgd1 = ax1.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.77),prop={'size': 7})


        deviation_df = pd.DataFrame(deviation_list, columns=['acceleration', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df['acceleration'], deviation_df['deltaR'], label='$\\Delta R$', color=sns.color_palette('viridis', 2)[0],linewidth=1)
        ax2.plot(deviation_df['acceleration'], deviation_df['deltaV'], label='$\\Delta V$', color=sns.color_palette('viridis', 2)[1],linewidth=1)

        ax2.set_xlim([min(deviation_df['acceleration']), max(deviation_df['acceleration'])])
        ax2.set_ylim([0, max(deviation_df['deltaV']) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks2 = (np.linspace(min(deviation_df['acceleration']), max(deviation_df['acceleration']), num=7))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']), max(deviation_df['deltaV']) * self.spacingFactor),
                               num=self.numberOfAxisTicksOrtho))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2 = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.9),prop={'size': 7})


        #fig.tight_layout()
        #fig.subplots_adjust(top=0.9, bottom=-0.1)

        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($\\alpha = ' + str(
            "{:3.1f}".format(self.alpha)) + ' $, $|A| =' + str(
            "{:2.1e}".format(self.amplitude)) + ' $) Acceleration effect ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.83)

        if self.lowDpi:
            #fig.savefig('../../data/figures/initial_guess/test.png',transparent=True, dpi=self.dpi)
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str("{:7.6f}".format(self.correctionTime)) + \
                        '_acceleration_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists =(lgd1, lgd2, suptitle), bbox_inches = 'tight')
        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str("{:7.6f}".format(self.correctionTime)) + '_acceleration_effect.pdf', transparent=True)

        plt.close()
        pass

    def plot_corrections_effect(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$|A|$ [-]')
        ax2.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax1.set_title('Initial guesses')
        ax2.set_title('State deviations at full period')

        orbitIdsPlot = list(range(0, len(self.correctionTime), 1))


        deviation_list = []

        indexPlotlist = np.linspace(0, len(self.correctionTime) - 1, num=self.numberOfSolutions).tolist()
        Indexlist = 0

        minimumX = 1000.0
        minimumY = 1000.0
        maximumX = -1000.0
        maximumY = -1000.0
        maximumDevR = 0.0
        maximumDevV = 0.0

        for i in orbitIdsPlot:
            df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str("{:7.6f}".format(self.correctionTime[i])) + '_stateHistory.txt')

            maneuver_array = np.loadtxt('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str("{:7.6f}".format(self.correctionTime[i])) + '_Maneuvers.txt')

            deltaVCORR = np.linalg.norm(maneuver_array)

            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deltaVTOT  = np.sqrt(deltaVCORR ** 2 + deltaV**2 )


            deviation_list.append([self.correctionTime[i], deltaR, deltaV, deltaVCORR, deltaVTOT])

            if i == indexPlotlist[Indexlist]:
                print(Indexlist)
                ax1.plot(df['x'], df['y'], color=self.plottingColors['fifthLine'][Indexlist], linewidth=1, label='$\\Delta t = '+ str(self.correctionTime[i])+' $')

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=self.plottingColors['fifthLine'][Indexlist], marker='x')




                if min(df['x']) < minimumX:
                    minimumX = min(df['x'])
                if min(df['y']) < minimumY:
                    minimumY = min(df['y'])
                if max(df['x']) > maximumX:
                    maximumX = max(df['x'])
                if max(df['y']) > maximumY:
                    maximumY = max(df['y'])

                if Indexlist < len(indexPlotlist) - 1:
                    Indexlist = Indexlist + 1



        lagrange_points_df = load_lagrange_points_location_augmented(0.0, 0.0)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])




        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%4.5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%3.5f'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        #ax1.ticklabel_format(style='sci', axis='x', scilimits=(6, 2))
        #ax1.ticklabel_format(style='sci', axis='y', scilimits=(6, 2))


        ax1.xaxis.set_ticks(xticks)

        lgd1 = ax1.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.83), prop={'size': 7})

        deviation_df = pd.DataFrame(deviation_list, columns=['correction', 'deltaR', 'deltaV','deltaVCORR','deltaVTOT'])
        ax2.plot(deviation_df['correction'], deviation_df['deltaR'], label='$\\Delta R$',
                 color=sns.color_palette('viridis', 4)[0], linewidth=1)
        ax2.plot(deviation_df['correction'], deviation_df['deltaV'], label='$\\Delta V FINAL$',
                 color=sns.color_palette('viridis', 4)[1], linewidth=1)
        ax2.plot(deviation_df['correction'], deviation_df['deltaVCORR'], label='$\\Delta V CORR$',
                     color=sns.color_palette('viridis', 4)[2], linewidth=1)
        ax2.plot(deviation_df['correction'], deviation_df['deltaVTOT'], label='$\\Delta V TOT$',
                     color=sns.color_palette('viridis', 4)[3], linewidth=1)

        ax2.set_xlim([min(deviation_df['correction']), max(deviation_df['correction'])])
        ax2.set_ylim([0, max(deviation_df['deltaVTOT']) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.3f'))
        xticks2 = (np.linspace(min(deviation_df['correction']), max(deviation_df['correction']), num=6))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']), max(deviation_df['deltaVTOT']) * self.spacingFactor),
                               num=self.numberOfAxisTicksOrtho))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2 = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.88), prop={'size': 5})

        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=-0.1)

        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = '+ str(self.accelerationMagnitude) +'$, $\\alpha = ' + str(
            "{:3.1f}".format(self.alpha)) + ' $, $|A| =' + str(
            "{:2.1e}".format(self.amplitude)) + ' $) Correction effect ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.83)

        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_'  \
                        + str("{:7.6f}".format(self.amplitude)) + \
                        '_correction_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd1, lgd2, suptitle),bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_correction_effect.png', transparent=True)

        plt.close()
        pass

    def plot_correction_time_effect(self):
        fig = plt.figure(figsize=self.figSize)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('x [-]')
        ax4.set_ylabel('y [-]')
        ax4.grid(True, which='both', ls=':')

        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']
        if self.lagrangePointNr == 3:
            lagrange_point_nrs = ['L3']
        if self.lagrangePointNr == 4:
            lagrange_point_nrs = ['L4']
        if self.lagrangePointNr == 5:
            lagrange_point_nrs = ['L5']


        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

            bodies_df = load_bodies_location()
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
            y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax1.contourf(x, y, z, colors='black')
            ax2.contourf(x, y, z, colors='black')
            ax3.contourf(x, y, z, colors='black')
            ax4.contourf(x, y, z, colors='black')


            # initialize the min and max x and y values
            min_x = 1000
            min_y = 1000
            max_x = -1000
            max_y = -1000

            for i in range(4):
                orbit_df = load_orbit_augmented('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_'\
                                            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                                            + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                                            + '_' + str("{:7.6f}".format(self.correctionTime[0])) \
                                            + '_stateHistory.txt')

                if i == 0:
                  ax1.plot(orbit_df['x'],orbit_df['y'],color=self.plottingColors['singleLine'],lineWidth=self.lineWidth,label='$\\Delta t = '+ str(self.correctionTime[0])+' $')
                  ax1.set_title('$\\Delta t_{correction} = ' + str(self.correctionTime[0]) + '$')
                  lgd1 = ax1.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

                  if min_x > min(orbit_df['x']):
                        min_x = min(orbit_df['x'])
                  if min_y > min(orbit_df['y']):
                      min_y = min(orbit_df['y'])
                  if max_x < max(orbit_df['x']):
                      max_x = max(orbit_df['x'])
                  if max_y < max(orbit_df['y']):
                      max_y = max(orbit_df['y'])

                if i == 1:
                  ax2.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],lineWidth=self.lineWidth ,label='$\\Delta t = '+ str(self.correctionTime[0])+' $')
                  ax2.set_title('$\\Delta t_{correction} = ' + str(self.correctionTime[0]) + '$')
                  lgd2 = ax2.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


                  if min_x > min(orbit_df['x']):
                        min_x = min(orbit_df['x'])
                  if min_y > min(orbit_df['y']):
                      min_y = min(orbit_df['y'])
                  if max_x < max(orbit_df['x']):
                      max_x = max(orbit_df['x'])
                  if max_y < max(orbit_df['y']):
                      max_y = max(orbit_df['y'])

                if i == 2:
                  ax3.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],lineWidth=self.lineWidth,label='$\\Delta t = '+ str(self.correctionTime[0])+' $')
                  ax3.set_title('$\\Delta t_{correction} = ' + str(self.correctionTime[0]) + '$')
                  lgd3 = ax3.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


                  if min_x > min(orbit_df['x']):
                        min_x = min(orbit_df['x'])
                  if min_y > min(orbit_df['y']):
                      min_y = min(orbit_df['y'])
                  if max_x < max(orbit_df['x']):
                      max_x = max(orbit_df['x'])
                  if max_y < max(orbit_df['y']):
                      max_y = max(orbit_df['y'])

                if i == 3:
                  ax4.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],lineWidth=self.lineWidth,label='$\\Delta t = '+ str(self.correctionTime[0])+' $')
                  ax4.set_title('$\\Delta t_{correction} = ' + str(self.correctionTime[0]) + '$')
                  lgd4 = ax4.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


                  if min_x > min(orbit_df['x']):
                        min_x = min(orbit_df['x'])
                  if min_y > min(orbit_df['y']):
                      min_y = min(orbit_df['y'])
                  if max_x < max(orbit_df['x']):
                      max_x = max(orbit_df['x'])
                  if max_y < max(orbit_df['y']):
                      max_y = max(orbit_df['y'])

            Xmiddle = min_x + (max_x - min_x) / 2.0
            Ymiddle = min_y + (max_y - min_y) / 2.0

            scaleDistance = max((max_y - min_y), (max_x - min_x))

            ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax3.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax3.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax4.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax4.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax3.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax3.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))
            ax4.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.5f'))

            suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = ' + str(
                "{:2.1e}".format(self.accelerationMagnitude)) + ' $, $\\alpha =' + str(
                "{:3.1f}".format(self.alpha)) + '$ $|A| = ' + str(
                "{:2.1e}".format(self.amplitude)) + ' $) Correction time influence ', size=self.suptitleSize)

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)


        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_'  \
                        + str("{:7.6f}".format(self.amplitude)) + \
                        '_correctionTime_effect.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_correctionTime_effect.png', transparent=True)


        pass

    def plot_vertical_capability(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$|A|$ [-]')
        ax2.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('y [-]')
        ax3.set_ylabel('z [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('x [-]')
        ax4.set_ylabel('z [-]')
        ax4.grid(True, which='both', ls=':')

        ax1.set_title('xy projection')
        ax2.set_title('State deviations at full period')
        ax3.set_title('yz projection')
        ax4.set_title('xz projection')

        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')
            ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')

        # Create information of orbits to be plotted
        orbitIdsPlot = list(range(0, len(self.amplitude), 1))
        deviation_list = []

        deviation_list = []
        deviation_corrected_list = []

        indexPlotlist = np.linspace(0, len(self.amplitude) - 1, num=self.numberOfSolutions).tolist()
        Indexlist = 0

        print(len(self.amplitude) - 1)

        print(indexPlotlist)



        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
            + str("{:7.6f}".format(self.amplitude[i])) + '_' \
            + str("{:7.6f}".format(self.correctionTime[i])) + '_initialGuess.txt')



            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            deviation_list.append([self.amplitude[i], deltaR, deltaV])

            if i == indexPlotlist[Indexlist]:
                print('test')
                legendString = '$|A| = $' + str("{:2.1e}".format(self.amplitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1, label=legendString)
                ax3.plot(df['y'], df['z'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1,label=legendString)
                ax4.plot(df['x'], df['z'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1, label=legendString)

                Indexlist = Indexlist + 1

        # Create the ax2 plot
        deviation_df = pd.DataFrame(deviation_list, columns=['amplitude', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaR'], label='$|\\Delta R|$',color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaV'], label='$|\\Delta V|$',color=sns.color_palette('viridis', 2)[1], linewidth=1)

        # Create the axes formats and legends for ax2
        ax2.set_xlim([min(deviation_df['amplitude']), max(deviation_df['amplitude'])])
        ax2.set_ylim([0, max(max(deviation_df['deltaV']), max(deviation_df['deltaR'])) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks2 = (np.linspace(min(deviation_df['amplitude']), max(deviation_df['amplitude']), num=self.numberOfAxisTicks))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']), max(deviation_df['deltaV']) * self.spacingFactor),num=self.numberOfAxisTicks))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2  = ax2.legend(frameon=True, loc='upper left',  bbox_to_anchor=(0, 1),prop={'size': 8})

        scaleDistance = max((max(df['x']) - min(df['x'])), (max(df['y']) - min(df['y'])), (max(df['z']) - min(df['z'])))


        Xmiddle = min(df['x']) + (max(df['x']) - min(df['x'])) / 2
        Ymiddle = min(df['y']) + (max(df['y']) - min(df['y'])) / 2
        Zmiddle = min(df['z']) + (max(df['z']) - min(df['z'])) / 2

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks(xticks)

        ax3.set_xlim([(Ymiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Ymiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim(
            [Zmiddle - 0.5 * scaleDistance * self.spacingFactor, Zmiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax3.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax3.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Ymiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Ymiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax3.xaxis.set_ticks(xticks)

        ax4.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax4.set_ylim(
            [Zmiddle - 0.5 * scaleDistance * self.spacingFactor, Zmiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax4.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax4.xaxis.set_ticks(xticks)

        lgd1  = ax1.legend(frameon=True, loc='upper left',  bbox_to_anchor=(0, 1),prop={'size': 8})

        suptitle = fig.suptitle('$L_{'+str(self.lagrangePointNr)+'}$ ($a_{lt} = '+ str("{:2.1e}".format(self.accelerationMagnitude))+' $, $\\alpha ='+ str("{:3.1f}".format(self.alpha)) +' $) Vertical Capability ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + \
                        '_vertical_capability.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd1, lgd2, suptitle), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                "{:7.6f}".format(self.alpha)) + \
                        '_vertical_capability.png', transparent=True)

        plt.close()
        pass

    def phase_space_plot(self):
        fig = plt.figure(figsize=self.figSizeWide)
        fig2 = plt.figure(figsize=self.figSizeWide)
        fig3 = plt.figure(figsize=self.figSizeWide)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax3 = fig2.add_subplot(1, 2, 1)
        ax4 = fig2.add_subplot(1, 2, 2)
        ax5 = fig3.add_subplot(1, 2, 1)
        ax6 = fig3.add_subplot(1, 2, 2)

        ax1title = '$L_{1}$($a_{lt}$ = 0.1, $\\alpha$=90.0$^{\\circ}$) approximate periodic solutions'
        ax1.set_title(ax1title)
        ax3title = '$L_{1}$($a_{lt}$ = 0.1, $||A||$=$1.0 \\cdot 10^{-4}$) approximate periodic solutions'
        ax3.set_title(ax3title)
        ax5title = '$L_{1}$($\\alpha$ = 120$^{\\circ}$, $||A||$=$1.0 \\cdot 10^{-4}$) approximate periodic solutions'
        ax5.set_title(ax5title)

        ax2title = 'Position and velocity deviation after one orbital revolution'
        ax2.set_title(ax2title)
        ax4title = 'Position and velocity deviation after one orbital revolution'
        ax4.set_title(ax4title)
        ax6title = 'Position and velocity deviation after one orbital revolution'
        ax6.set_title(ax6title)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$||A||$ [-]')
        ax2.set_ylabel('$||\\Delta \\bar{R}||$ [-], $||\\Delta \\bar{V}||$, [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('$\\alpha$ [rad]')
        ax4.set_ylabel('$||\\Delta \\bar{R}||$ [-], $||\\Delta \\bar{V}||$, [-]')
        ax4.grid(True, which='both', ls=':')

        ax5.set_xlabel('x [-]')
        ax5.set_ylabel('y [-]')
        ax5.grid(True, which='both', ls=':')

        ax6.set_xlabel('$a_{lt}$ [-]')
        ax6.set_ylabel('$||\\Delta \\bar{R}||$ [-], $||\\Delta \\bar{V}||$, [-]')
        ax6.grid(True, which='both', ls=':')

        suptitle = fig.suptitle(' In-plane amplitude effect on deviation at full period', size=self.suptitleSize)
        suptitle2 = fig2.suptitle('$\\alpha$ effect on deviation at full period', size=self.suptitleSize)
        suptitle3 = fig3.suptitle('$a_{lt}$ effect on deviation at full period', size=self.suptitleSize)


        ##### PLOT 1 ####

        # data plot 1
        acc1 = 0.1
        alpha1 = 90.0
        ampArray = np.linspace(1.0E-5,1.0E-4,num=91)
        ampArray2 = [1.0e-5,4.0e-5,7.0e-5,1.0e-4]

        min_x1 = 1000
        max_x1 = -1000
        min_y1 = 1000
        max_y1 = -1000

        for k in ampArray2:
            if k < 3.0e-5:
                legendString = '$|A| = 1.0 \\cdot 10^{-5}$'
                colourstamp = 0
            if k > 3.0e-5 and k < 5.0e-5:
                legendString = '$|A| = 4.0 \\cdot 10^{-5}$'
                colourstamp = 1
            if k > 5.0e-5 and k < 8.0e-5:
                legendString = '$|A| = 7.0 \\cdot 10^{-5}$'
                colourstamp = 2
            if k > 8.0e-5:
                legendString = '$|A| = 1.0 \\cdot 10^{-4}$'
                colourstamp = 3
            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(acc1)) + '_' \
                + str("{:7.6f}".format(alpha1)) + '_' + str("{:7.6f}".format(k)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            if min(orbit_df['x']) < min_x1:
                min_x1 = min(orbit_df['x'])
            if max(orbit_df['x']) > max_x1:
                max_x1 = max(orbit_df['x'])
            if min(orbit_df['y']) < min_y1:
                min_y1 = min(orbit_df['y'])
            if max(orbit_df['y']) > max_y1:
                max_y1 = max(orbit_df['y'])

            ax1.plot(orbit_df['x'],orbit_df['y'], color=sns.color_palette('viridis', 4)[colourstamp], linewidth=1,label=legendString )

        scaleDistance1 = max((max_x1 - min_x1),(max_y1 - min_y1))

        Xmiddle1 = min_x1 + (max_x1-min_x1) / 2
        Ymiddle1 = min_y1 + (max_y1-min_y1) / 2

        ax1.set_xlim([(Xmiddle1 - 0.5 * scaleDistance1 * self.figureRatio * self.spacingFactor),
                      (Xmiddle1 + 0.5 * scaleDistance1 * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle1 - 0.5 * scaleDistance1 * self.spacingFactor, Ymiddle1 + 0.5 * scaleDistance1 * self.spacingFactor])

        lgd1 = ax1.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

        lagrange_points_df = load_lagrange_points_location_augmented(acc1, alpha1)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

        deviation_list = []
        for i in ampArray:
            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(acc1)) + '_' \
                + str("{:7.6f}".format(alpha1)) + '_' + str("{:7.6f}".format(i)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            deviations = orbit_df.head(1).values[0] - orbit_df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deviation_list.append([i, deltaR, deltaV])



        deviation_df = pd.DataFrame(deviation_list, columns=['amplitude', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaR'], label='$|\\Delta R|$',color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax2.plot(deviation_df['amplitude'], deviation_df['deltaV'], label='$|\\Delta V|$',color=sns.color_palette('viridis', 2)[1], linewidth=1)

        ax2.set_xlim([min(deviation_df['amplitude']), max(deviation_df['amplitude'])])
        ax2.set_ylim([0, max(max(deviation_df['deltaV']),max(deviation_df['deltaR'])) * self.spacingFactor])

        ticksLocators = [0.00001, 0.00004, 0.00007, 0.0001]
        labels = ('$1.0 \\cdot 10^{-5}$', '$4.0 \\cdot 10^{-5}$', '$7.0 \\cdot 10^{-5}$', '$1.0 \\cdot 10^{-4}$')
        ax2.set_xticks(ticksLocators, minor=False)
        ax2.set_xticklabels(labels, fontdict=None, minor=False)

        ticksLocators = [0.0, 0.00008, 0.00016, 0.00025]
        labels = ('$0.0$', '$8.0 \\cdot 10^{-5}$', '$1.6 \\cdot 10^{-4}$', '$2.5 \\cdot 10^{-4}$')
        ax2.set_yticks(ticksLocators, minor=False)
        ax2.set_yticklabels(labels, fontdict=None, minor=False)

        lgd2 = ax2.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


        #### PLOT 2 ####
        acc2 = 0.1
        amp2 = 1.0e-4
        alphaArray = np.linspace(0, 359, num=360)
        alphaArray2 = [0,60.0,120,180,240,300]

        min_x2 = 1000
        max_x2 = -1000
        min_y2 = 1000
        max_y2 = -1000

        lagrange_points_df = load_lagrange_points_location_augmented(0.0, 0.0)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

        for k in alphaArray2:
            if k < 15:
                legendString = '$\\alpha = 0^{\\circ}$'
                colourstamp = 0
            if k > 30 and k < 80:
                legendString = '$\\alpha = 60^{\\circ}$'
                colourstamp = 1
            if k > 80 and k < 130:
                legendString = '$\\alpha = 120^{\\circ}$'
                colourstamp = 2
            if k > 130 and k < 190:
                legendString = '$\\alpha = 180^{\\circ}$'
                colourstamp = 3
            if k > 190 and k < 250:
                legendString = '$\\alpha = 240^{\\circ}$'
                colourstamp = 4
            if k > 250 and k < 310:
                legendString = '$\\alpha = 300^{\\circ}$'
                colourstamp = 5
            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(acc2)) + '_' \
                + str("{:7.6f}".format(k)) + '_' + str("{:7.6f}".format(amp2)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            if min(orbit_df['x']) < min_x2:
                min_x2 = min(orbit_df['x'])
            if max(orbit_df['x']) > max_x2:
                max_x2 = max(orbit_df['x'])
            if min(orbit_df['y']) < min_y2:
                min_y2 = min(orbit_df['y'])
            if max(orbit_df['y']) > max_y2:
                max_y2 = max(orbit_df['y'])

            ax3.plot(orbit_df['x'], orbit_df['y'], color=sns.color_palette('viridis', 6)[colourstamp], linewidth=1,
                     label=legendString)

            lagrange_points_df = load_lagrange_points_location_augmented(acc2, k)
            if self.lagrangePointNr == 1:
                lagrange_point_nrs = ['L1']
            if self.lagrangePointNr == 2:
                lagrange_point_nrs = ['L2']

            for lagrange_point_nr in lagrange_point_nrs:
                ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            color=sns.color_palette('viridis', 6)[colourstamp], marker='x',s=0.1)

        scaleDistance2 = max((max_x2 - min_x2), (max_y2 - min_y2))

        Xmiddle2 = min_x2 + (max_x2 - min_x2) / 2
        Ymiddle2 = min_y2 + (max_y2 - min_y2) / 2

        ax3.set_xlim([(Xmiddle2 - 0.5 * scaleDistance2 * self.figureRatio * self.spacingFactor),
                      (Xmiddle2 + 0.5 * scaleDistance2 * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle2 - 0.5 * scaleDistance2 * self.spacingFactor,
                      Ymiddle2 + 0.5 * scaleDistance2 * self.spacingFactor])

        lgd3 = ax3.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

        deviation_list2 = []
        for i in alphaArray:
            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(acc2)) + '_' \
                + str("{:7.6f}".format(i)) + '_' + str("{:7.6f}".format(amp2)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            deviations = orbit_df.head(1).values[0] - orbit_df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deviation_list2.append([(i*np.pi/180.0), deltaR, deltaV])

        deviation_df2 = pd.DataFrame(deviation_list2, columns=['amplitude', 'deltaR', 'deltaV'])
        ax4.plot(deviation_df2['amplitude'], deviation_df2['deltaR'], label='$|\\Delta R|$',
                 color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax4.plot(deviation_df2['amplitude'], deviation_df2['deltaV'], label='$|\\Delta V|$',
                 color=sns.color_palette('viridis', 2)[1], linewidth=1)

        ax4.set_xlim([0, 2*np.pi])
        ticksLocators4 = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
        labels4 = ('0', '$\\frac{1}{2}\\pi$', '$\\pi$', '$\\frac{3}{2}\\pi$', '$2\\pi$')
        ax4.set_xticks(ticksLocators4, minor=False)
        ax4.set_xticklabels(labels4, fontdict=None, minor=False)

        ax4.set_ylim([0, 0.0006])
        ticksLocators = [0.0, 0.00015, 0.00030, 0.00045, 0.0006]
        labels = ('$0.0$', '$1.5 \\cdot 10^{-4}$', '$3.0 \\cdot 10^{-4}$', '$4.5 \\cdot 10^{-4}$','$6.0 \\cdot 10^{-4}$')
        ax4.set_yticks(ticksLocators, minor=False)
        ax4.set_yticklabels(labels, fontdict=None, minor=False)

        lgd4 = ax4.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


        ## plot 3
        alpha3 = 120.0
        amp3 = 1.0e-4
        accArray = np.linspace(0, 0.1, num=101)
        accArray2 = [0.0, 0.025, 0.05, 0.075, 0.1]

        print(accArray2)
        min_x3 = 1000
        max_x3 = -1000
        min_y3 = 1000
        max_y3 = -1000

        colourstamp = -1
        for s in accArray2:

            colourstamp = colourstamp+1

            if colourstamp == 0:
                legendString = '$a_{lt} = 0.0$'
            if colourstamp == 1:
                legendString = '$a_{lt} = 0.025$'
            if colourstamp == 2:
                legendString = '$a_{lt} = 0.05$'
            if colourstamp == 3:
                legendString = '$a_{lt} = 0.075$'
            if colourstamp == 4:
                legendString = '$a_{lt} = 0.1$'


            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(s)) + '_' \
                + str("{:7.6f}".format(alpha3)) + '_' + str("{:7.6f}".format(amp3)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            if min(orbit_df['x']) < min_x3:
                min_x3 = min(orbit_df['x'])
            if max(orbit_df['x']) > max_x3:
                max_x3 = max(orbit_df['x'])
            if min(orbit_df['y']) < min_y3:
                min_y3 = min(orbit_df['y'])
            if max(orbit_df['y']) > max_y3:
                max_y3 = max(orbit_df['y'])

            ax5.plot(orbit_df['x'], orbit_df['y'], color=sns.color_palette('viridis', 5)[colourstamp], linewidth=1,
                     label=legendString)

            # lagrange_points_df = load_lagrange_points_location_augmented(s, alpha3)
            # if self.lagrangePointNr == 1:
            #     lagrange_point_nrs = ['L1']
            # if self.lagrangePointNr == 2:
            #     lagrange_point_nrs = ['L2']
            #
            # for lagrange_point_nr in lagrange_point_nrs:
            #     ax5.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #                 color=sns.color_palette('viridis', 5)[colourstamp], marker='x')

        scaleDistance3 = max((max_x3 - min_x3), (max_y3 - min_y3))

        Xmiddle3 = min_x3 + (max_x3 - min_x3) / 2
        Ymiddle3 = min_y3 + (max_y3 - min_y3) / 2

        ax5.set_xlim([(Xmiddle3 - 0.5 * scaleDistance3 * self.figureRatio * self.spacingFactor),
                      (Xmiddle3 + 0.5 * scaleDistance3 * self.figureRatio * self.spacingFactor)])
        ax5.set_ylim([Ymiddle3 - 0.5 * scaleDistance3 * self.spacingFactor,
                      Ymiddle3 + 0.5 * scaleDistance3 * self.spacingFactor])

        lgd5 = ax5.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

        deviation_list3 = []
        for i in accArray:
            orbit_df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(i)) + '_' \
                + str("{:7.6f}".format(alpha3)) + '_' + str("{:7.6f}".format(amp3)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            deviations = orbit_df.head(1).values[0] - orbit_df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deviation_list3.append([i, deltaR, deltaV])

        deviation_df3 = pd.DataFrame(deviation_list3, columns=['amplitude', 'deltaR', 'deltaV'])
        ax6.plot(deviation_df3['amplitude'], deviation_df3['deltaR'], label='$|\\Delta R|$',
                 color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax6.plot(deviation_df3['amplitude'], deviation_df3['deltaV'], label='$|\\Delta V|$',
                 color=sns.color_palette('viridis', 2)[1], linewidth=1)

        ax6.set_ylim([0, 0.0004])
        ticksLocators6 = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        labels6 = ('$0.0$','$1.0 \\cdot 10^{-4}$', '$2.0 \\cdot 10^{-4}$', '$3.0 \\cdot 10^{-4}$', '$4.0 \\cdot 10^{-4}$', '$5.0 \\cdot 10^{-4}$')
        ax6.set_yticks(ticksLocators6, minor=False)
        ax6.set_yticklabels(labels6, fontdict=None, minor=False)

        ax6.set_xlim([0, 0.1])
        ticksLocators6 = [0, 0.025, 0.05, 0.075, 0.1]
        labels6 = ('0', '0.025', '0.05', '0.075', '0.1')
        ax6.set_xticks(ticksLocators6, minor=False)
        ax6.set_xticklabels(labels6, fontdict=None, minor=False)

        lgd6 = ax6.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})


        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.8)
        fig3.tight_layout()
        fig3.subplots_adjust(top=0.8)

        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/phase_space_plot_amp.png', transparent=True, dpi=self.dpi)
            fig2.savefig('../../data/figures/initial_guess/phase_space_plot_alpha.png', transparent=True, dpi=self.dpi)
            fig3.savefig('../../data/figures/initial_guess/phase_space_plot_acc.png', transparent=True, dpi=self.dpi)

            # bbox_extra_artists=(lgd1, lgd2, suptitle), bbox_inches='tight'
        else:
            fig.savefig('../../data/figures/initial_guess/phase_space_plot', transparent=True)

        plt.close()
        pass
    def plot_correction_time_effect_report(self):
        fig = plt.figure(figsize=self.figSize)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1title = 'Approximate periodic solutions - $\\Delta t = 50.0$'
        ax1.set_title(ax1title)
        ax3title = 'Approximate periodic solutions - $\\Delta t = 5.0 \\cdot 10^{-2}$'
        ax3.set_title(ax3title)


        ax2title = 'Position and velocity deviation after one orbital revolution'
        ax2.set_title(ax2title)
        ax4title = 'Position and velocity deviation after one orbital revolution'
        ax4.set_title(ax4title)


        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$||A||$ [-]')
        ax2.set_ylabel('$||\\Delta \\bar{R}||$ [-], $||\\Delta \\bar{V}||$, [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('$||A||$ [-]')
        ax4.set_ylabel('$||\\Delta \\bar{R}||$ [-], $||\\Delta \\bar{V}||$, [-]')
        ax4.grid(True, which='both', ls=':')

        lagrange_points_df = load_lagrange_points_location_augmented(0.1, 90.0)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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
        ax3.contourf(xM, yM, zM, colors='black')
        ax3.contourf(xE, yE, zE, colors='black')

        amplitudeArray = np.linspace(1.0E-5,1.0E-4,num=91)
        amplitudeArray2 = np.linspace(1.0E-4, 1.0E-3, num=91)
        amplitudeArray3 = np.linspace(1.0E-3, 1.0E-2, num=91)
        amplitudeArray4 = np.linspace(1.0E-2, 1.0E-1, num=91)

        amplitudeArray1 = amplitudeArray[:-1]
        amplitudeArray2 = amplitudeArray2[:-1]
        amplitudeArray3 = amplitudeArray3[:-1]

        newArray = np.append(amplitudeArray, amplitudeArray2)
        newArray2 = np.append(newArray, amplitudeArray3)
        newArray3 = np.append(newArray2, amplitudeArray4)

        amplitudes = newArray3.tolist()
        amplitudes2 = [0.01, 0.04, 0.07, 0.1]

        deviation_list_50 = []
        deviation_list_005 = []

        min_x50 = 1000
        max_x50 = -1000
        min_y50 = 1000
        max_y50 = -1000

        min_x005 = 1000
        max_x005 = -1000
        min_y005 = 1000
        max_y005 = -1000

        colourstamp = -1
        for i in amplitudes2:

            colourstamp = colourstamp + 1

            if colourstamp == 0:
                legendString = '$a_{lt} = 0.0$'
            if colourstamp == 1:
                legendString = '$a_{lt} = 0.025$'
            if colourstamp == 2:
                legendString = '$a_{lt} = 0.05$'
            if colourstamp == 3:
                legendString = '$a_{lt} = 0.075$'
            if colourstamp == 4:
                legendString = '$a_{lt} = 0.1$'

            orbit_df_50 = load_orbit_augmented(
                '../../data/raw/initial_guess/final_plot/L' + str(
                    self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(0.1)) + '_' + str(
                    "{:7.6f}".format(90.0)) + '_' + str("{:7.6f}".format(i)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')

            ax1.plot(orbit_df_50['x'], orbit_df_50['y'], color=sns.color_palette('viridis', 6)[colourstamp], linewidth=1,label=legendString)

            if min(orbit_df_50['x']) < min_x50:
                min_x50 = min(orbit_df_50['x'])
            if max(orbit_df_50['x']) > max_x50:
                max_x50 = max(orbit_df_50['x'])
            if min(orbit_df_50['y']) < min_y50:
                min_y50 = min(orbit_df_50['y'])
            if max(orbit_df_50['y']) > max_y50:
                max_y50 = max(orbit_df_50['y'])


            orbit_df_005 = load_orbit_augmented('../../data/raw/initial_guess/final_plot/L' + str(
                self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(0.1)) + '_' + str(
                "{:7.6f}".format(90.0)) + '_' + str("{:7.6f}".format(i)) \
                                                + '_' + str("{:7.6f}".format(0.05)) \
                                                + '_stateHistory.txt')

            ax3.plot(orbit_df_005['x'], orbit_df_005['y'], color=sns.color_palette('viridis', 6)[colourstamp], linewidth=1,label=legendString)

            if min(orbit_df_005['x']) < min_x005:
                min_x005 = min(orbit_df_005['x'])
            if max(orbit_df_005['x']) > max_x005:
                max_x005 = max(orbit_df_005['x'])
            if min(orbit_df_005['y']) < min_y005:
                min_y005 = min(orbit_df_005['y'])
            if max(orbit_df_005['y']) > max_y005:
                max_y005 = max(orbit_df_005['y'])

        scaleDistance50 = max((max_x50 - min_x50), (max_y50 - min_y50))

        Xmiddle50 = min_x50 + (max_x50 - min_x50) / 2
        Ymiddle50 = min_y50 + (max_y50 - min_y50) / 2

        ax1.set_xlim([(Xmiddle50 - 0.5 * scaleDistance50 * self.figureRatio * self.spacingFactor),
                      (Xmiddle50 + 0.5 * scaleDistance50 * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle50 - 0.5 * scaleDistance50 * self.spacingFactor,
                      Ymiddle50 + 0.5 * scaleDistance50 * self.spacingFactor])

        scaleDistance005 = max((max_x005 - min_x005), (max_y005 - min_y005))

        Xmiddle005 = min_x005 + (max_x005 - min_x005) / 2
        Ymiddle005 = min_y005 + (max_y005 - min_y005) / 2

        ax3.set_xlim([(Xmiddle005 - 0.5 * scaleDistance005 * self.figureRatio * self.spacingFactor),
                      (Xmiddle005 + 0.5 * scaleDistance005 * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle005 - 0.5 * scaleDistance005 * self.spacingFactor,
                      Ymiddle005 + 0.5 * scaleDistance005 * self.spacingFactor])


        for i in amplitudes:
            orbit_df_50 = load_orbit_augmented(
                '../../data/raw/initial_guess/final_plot/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(0.1)) + '_' + str("{:7.6f}".format(90.0)) + '_' + str("{:7.6f}".format(i)) \
                + '_' + str("{:7.6f}".format(50.0)) \
                + '_stateHistory.txt')
            deviations = orbit_df_50.head(1).values[0] - orbit_df_50.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deviation_list_50.append([i, deltaR, deltaV])

            orbit_df_005 = load_orbit_augmented('../../data/raw/initial_guess/final_plot/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(0.1)) + '_' + str("{:7.6f}".format(90.0)) + '_' + str("{:7.6f}".format(i)) \
                + '_' + str("{:7.6f}".format(0.05)) \
                + '_stateHistory.txt')

            deviations = orbit_df_005.head(1).values[0] - orbit_df_005.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            maneuver_array = np.loadtxt('../../data/raw/initial_guess/final_plot/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(0.1)) + '_' + str("{:7.6f}".format(90.0)) + '_' + str("{:7.6f}".format(i)) \
                + '_' + str("{:7.6f}".format(0.05)) + '_Maneuvers.txt')

            deltaVCORR = np.linalg.norm(maneuver_array)

            deltaVTOT = np.sqrt(deltaVCORR ** 2 + deltaV ** 2)




            deviation_list_005.append([i, deltaR, deltaVTOT])

        deviation_df_50 = pd.DataFrame(deviation_list_50, columns=['amplitude', 'deltaR', 'deltaV'])
        ax2.plot(deviation_df_50['amplitude'], deviation_df_50['deltaR'], label='$|\\Delta R|$',
                 color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax2.plot(deviation_df_50['amplitude'], deviation_df_50['deltaV'], label='$|\\Delta V|$',
                 color=sns.color_palette('viridis', 2)[1], linewidth=1)

        deviation_df_005 = pd.DataFrame(deviation_list_005, columns=['amplitude', 'deltaR', 'deltaV'])
        ax4.plot(deviation_df_005['amplitude'], deviation_df_005['deltaR'], label='$|\\Delta R|$',
                 color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax4.plot(deviation_df_005['amplitude'], deviation_df_005['deltaV'], label='$|\\Delta V|$',
                 color=sns.color_palette('viridis', 2)[1], linewidth=1)

        print(max(deviation_df_005['deltaR']))

        ax2.set_xlim([min(deviation_df_50['amplitude']), max(deviation_df_50['amplitude'])])
        ax2.set_ylim([0, max(max(deviation_df_50['deltaV']), max(deviation_df_50['deltaR'])) * self.spacingFactor])

        ax4.set_xlim([min(deviation_df_005['amplitude']), max(deviation_df_005['amplitude'])])
        #ax4.set_ylim([10e-11,1.0])
        ax4.set_ylim([0, max(max(deviation_df_005['deltaV']), max(deviation_df_005['deltaR'])) * self.spacingFactor])


        ticksLocators = [0.00001, 0.025, 0.05 ,0.075, 0.1]
        labels = ('$1.0 \\cdot 10^{-5}$', '$2.5 \\cdot 10^{-2}$', '$5.0 \\cdot 10^{-2}$', '$7.5 \\cdot 10^{-2}$', '$1.0 \\cdot 10^{-1}$')
        ax2.set_xticks(ticksLocators, minor=False)
        ax2.set_xticklabels(labels, fontdict=None, minor=False)
        ax4.set_xticks(ticksLocators, minor=False)
        ax4.set_xticklabels(labels, fontdict=None, minor=False)

        lgd1 = ax1.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
        lgd2 = ax2.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
        lgd3 = ax3.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
        lgd4 = ax4.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})



        suptitle = fig.suptitle(' $L_{1}$($a_{lt}$ = 0.1, $\\alpha$=90.0$^{\\circ}$) - Correction interval effect', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lowDpi:
            fig.savefig('../../data/figures/initial_guess/final_correction_time_plot.png', transparent=True, dpi=self.dpi)

            # bbox_extra_artists=(lgd1, lgd2, suptitle), bbox_inches='tight'
        else:
            fig.savefig('../../data/figures/initial_guess/final_correction_time_plot', transparent=True)

        pass


if __name__ == '__main__':
    lagrange_point_nrs = [1]
    orbit_type = 'horizontal'
    alt_values = [0.1]
    angles = [0.0]
    amplitudes = [1.0E-5]
    correction_times = [50.0]
    low_dpi = False

    for lagrange_point_nr in lagrange_point_nrs:
     for alt_value in alt_values:
        for angle in angles:
            for correction_time in correction_times:
                initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
                angle, amplitudes, correction_time, low_dpi)

                initial_guess_validation.plot_correction_time_effect_report()
                initial_guess_validation.phase_space_plot()

                del initial_guess_validation

    # lagrange_point_nrs = [1]
    # orbit_type = 'horizontal'
    # alt_values = [0.1]
    # angles = [00.0]
    # amplitudes = [1.0E-5]
    # correction_times = [50.0]
    # low_dpi = True
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for alt_value in alt_values:
    #         for angle in angles:
    #             for amplitude in amplitudes:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                      angle, amplitude, correction_times, low_dpi)
    #
    #                 initial_guess_validation.plot_correction_time_effect()
    #                 initial_guess_validation.plot_corrections_effect()
    #
    #
    #
    #                 del initial_guess_validation
    #
    # lagrange_point_nrs = [1]
    # orbit_type = 'horizontal'
    # alt_values = [0.1]
    # angles = [90.0]
    #
    # amplitudeArray = np.linspace(1.0E-5,1.0E-4,num=91)
    # amplitudeArray2 = np.linspace(1.0E-4, 1.0E-3, num=91)
    # amplitudeArray3 = np.linspace(1.0E-3, 1.0E-2, num=91)
    # amplitudeArray4 = np.linspace(1.0E-2, 1.0E-1, num=91)
    #
    # amplitudeArray1 = amplitudeArray[:-1]
    # amplitudeArray2 = amplitudeArray2[:-1]
    # amplitudeArray3 = amplitudeArray3[:-1]
    #
    # newArray = np.append(amplitudeArray, amplitudeArray2)
    # newArray2 = np.append(newArray, amplitudeArray3)
    # newArray3 = np.append(newArray2, amplitudeArray4)
    #
    # amplitudes = newArray3.tolist()
    #
    # amplitudes = np.linspace(1.0E-5,1.0E-4,num=91).tolist()
    # correction_times = [0.05,50.0]
    # low_dpi = True
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for alt_value in alt_values:
    #         for angle in angles:
    #             for correction_time in correction_times:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                   angle, amplitudes, correction_time, low_dpi)
    #
    #                 initial_guess_validation.plot_amplitude_effect()
    #                 # initial_guess_validation.plot_corrections_effect()
    #
    #                 del initial_guess_validation

    # lagrange_point_nrs = [1]
    # orbit_type = 'horizontal'
    # amplitudes = [0.0001,0.1]
    # angles = np.linspace(0,359,num=360)
    # alt_values = [0.1]
    # correction_times = [0.05,50.0]
    # low_dpi = True

    # for lagrange_point_nr in lagrange_point_nrs:
    #     for alt_value in alt_values:
    #         for amplitude in amplitudes:
    #             for correction_time in correction_times:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                   angles, amplitude, correction_time, low_dpi)
    #
    #                 initial_guess_validation.plot_angle_effect()
    #
    #                 del initial_guess_validation

    # amplitudeArray = np.linspace(1.0E-3,1.0E-2,num=91)
    # amplitudeArray1 = amplitudeArray[:-1]
    # amplitudeArray2 = np.linspace(1.0E-2, 1.0E-1, num=91)
    # newArray = np.append(amplitudeArray1, amplitudeArray2)
    # alt_values = newArray.tolist()
    # angles = [90]
    # amplitudes = [0.0001,0.1]
    # correction_times = [0.05,50.0]
    #
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for angle in angles:
    #         for amplitude in amplitudes:
    #             for correction_time in correction_times:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_values, \
    #                                                                   angle, amplitude, correction_time, low_dpi)
    #
    #                 initial_guess_validation.plot_acceleration_effect()
    #
    #                 del initial_guess_validation
    # #

    # lagrange_point_nrs = [1]
    # orbit_type = 'vertical'
    # alt_values = [0.0]
    # angles = [0.0]
    # amplitudes = np.linspace(1.0E-5, 1.0E-4, num=10).tolist()
    # numbers_of_corrections = 0
    # low_dpi = True
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for angle in angles:
    #         for alt_value in alt_values:
    #             initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                angle, amplitudes, numbers_of_corrections, low_dpi)
    #
    #             initial_guess_validation.plot_vertical_capability()
    #
    #             del initial_guess_validation



