
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

class initialGuessValidation:
    def __init__(self, lagrange_point_nr, orbit_type, acceleration_magnitude, alpha, amplitude, number_of_corrections, low_dpi):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.amplitude = amplitude
        self.numberOfCorrections = number_of_corrections

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
            self.numberOfSolutions = 6
            self.numberOfCorrectionPlots = len(self.numberOfCorrections)
            self.numberOfAxisTicks = 4
            self.numberOfAxisTicksOrtho = 5




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

        self.scaleDistanceXWide = self.scaleDistanceY * self.figureRatioWide

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

        indexPlotlist = np.linspace(0, len(self.amplitude) - 1, num=self.numberOfSolutions).tolist()
        Indexlist = 0

        # Compute the deviations and plot the desired intial guesses
        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
            + str("{:7.6f}".format(self.amplitude[i])) + '_' \
            + str(self.numberOfCorrections) + '_initialGuess.txt')



            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            deviation_list.append([self.amplitude[i], deltaR, deltaV])

            if i == indexPlotlist[Indexlist]:
                legendString = '$|A| = $' + str("{:2.1e}".format(self.amplitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1,label=legendString)
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
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks(xticks)

        ## Test for axes ratio
        #print(1 / ax1.get_data_ratio())

        # bodies_df = load_bodies_location()
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x'] - 0.14
        # yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        # zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        #
        #
        # ax1.contourf(xM, yM, zM, colors='black')

        lgd  = ax1.legend(frameon=True, loc='upper left',  bbox_to_anchor=(0, 1),prop={'size': 8})




        # Create the axes formats and legends for ax2
        ax2.set_xlim([min(deviation_df['amplitude']), max(deviation_df['amplitude'])])
        ax2.set_ylim([0, max(max(deviation_df['deltaV']),max(deviation_df['deltaR'])) * self.spacingFactor])

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
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
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_amplitude_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd, lgd2, suptitle), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
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

        indexPlotlist = np.linspace(0, 315, num=self.numberOfSolutions).tolist()
        Indexlist = 0

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
                + str(self.numberOfCorrections) + '_initialGuess.txt')


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
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAlphas)[i], linewidth=1, label= legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha[i])
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAlphas)[i], marker='x')

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
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                        '_angle_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd1,lgd2,suptitle), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
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

        indexPlotlist = np.linspace(0, len(self.accelerationMagnitude) - 1, num=self.numberOfSolutions).tolist()

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
            + str(self.numberOfCorrections) + '_initialGuess.txt')

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
                + str(self.numberOfCorrections) + '_initialGuess.txt')

            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])

            deviation_list.append([self.accelerationMagnitude[i], deltaR, deltaV])

            if i == indexPlotlist[Indexlist]:
                legendString = 'a$_{lt} = $' + str("{:2.1e}".format(self.accelerationMagnitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAccelerations)[i], linewidth=1, label= legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude[i], self.alpha)
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAccelerations)[i], marker='x')

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
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                        '_angle_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists =(lgd1, lgd2, suptitle), bbox_inches = 'tight')
        else:
            fig.savefig('../../data/figures/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + '_acceleration_effect.pdf', transparent=True)

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

        orbitIdsPlot = list(range(0, len(self.numberOfCorrections), 1))

        deviation_list = []

        indexPlot = [0,2,4,6,8,10]
        indexPlotlist = list(indexPlot)
        Indexlist = 0

        minimumX = 1000.0
        minimumY = 1000.0
        maximumX = 0.0
        maximumY = 0.0
        maximumDevR = 0.0
        maximumDevV = 0.0

        for i in orbitIdsPlot:
            df = load_orbit_augmented(
                '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str(self.numberOfCorrections[i]) + '_initialGuess.txt')

            maneuver_array = np.loadtxt('../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                + str("{:7.6f}".format(self.amplitude)) + '_' \
                + str(self.numberOfCorrections[i]) + '_Maneuvers.txt')
            print(self.numberOfCorrections[i])
            print(np.linalg.norm(maneuver_array))

            deltaVCORR = np.linalg.norm(maneuver_array)

            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])
            deltaVTOT  = np.sqrt(deltaVCORR ** 2 + deltaV**2 )


            deviation_list.append([self.numberOfCorrections[i], deltaR, deltaV, deltaVCORR, deltaVTOT])

            if i == indexPlotlist[Indexlist]:
                legendString = 'corr = ' + str("{:2.1e}".format(indexPlot[Indexlist]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfCorrectionPlots)[i], linewidth=1, label= legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfCorrectionPlots)[i], marker='x')

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
        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                      (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim(
            [Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%5f'))
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),
                              num=self.numberOfAxisTicks))
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

        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1f'))
        ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        xticks2 = (np.linspace(min(deviation_df['correction']), max(deviation_df['correction']), num=6))
        yticks2 = (np.linspace(0, max(max(deviation_df['deltaR']), max(deviation_df['deltaV']) * self.spacingFactor),
                               num=self.numberOfAxisTicksOrtho))
        ax2.xaxis.set_ticks(xticks2)
        ax2.yaxis.set_ticks(yticks2)

        lgd2 = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(0, 0.9), prop={'size': 7})

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
            + str(self.numberOfCorrections) + '_initialGuess.txt')



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




    pass

if __name__ == '__main__':

    lagrange_point_nrs = [1]
    orbit_type = 'horizontal'
    alt_values = [0.1]
    angles = [90.0]
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

    # amplitudes = np.linspace(1.0E-5,1.0E-4,num=91).tolist()
    #  amplitudeList = np.linspace(1.0E-5,1.0E-4,num=91).tolist()
    # amplitudeList2 = np.linspace(1.0E-4,1.0E-3,num=91).tolist()
    # amplitudeList3 = np.linspace(1.0E-3,1.0E-2,num=91).tolist()
    # amplitudeList4 = np.linspace(1.0E-2,1.0E-1,num=91).tolist()
    # amplitudes.append(amplitudeList)
    # amplitudes.append(amplitudeList2)
    # amplitudes.append(amplitudeList3)
    # amplitudes.append(amplitudeList4)
    numbers_of_points = [0]
    low_dpi = True

    for lagrange_point_nr in lagrange_point_nrs:
        for alt_value in alt_values:
            for angle in angles:
                for number_of_points in numbers_of_points:
                    initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
                                                                         angle, amplitudes, number_of_points, low_dpi)

                    initial_guess_validation.plot_amplitude_effect()


                    del initial_guess_validation

    # alt_values = [0.001,0.1]
    # angles = np.linspace(0, 359, num=360).tolist()
    # amplitudes = [0.0001]
    # numbers_of_points = [8]
    # low_dpi = True
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for alt_value in alt_values:
    #         for amplitude in amplitudes:
    #             for number_of_points in numbers_of_points:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                   angles, amplitude, number_of_points, low_dpi)
    #
    #                 initial_guess_validation.plot_angle_effect()
    #
    #                 del initial_guess_validation
    #
    #
    # alt_values = np.linspace(1.0E-2,1.0E-1,num=91).tolist()
    # angles = [180]
    # amplitudes = [0.0001]
    # numbers_of_points = [8]
    # low_dpi = True
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for angle in angles:
    #         for amplitude in amplitudes:
    #             for number_of_points in numbers_of_points:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_values, \
    #                                                                   angle, amplitude, number_of_points, low_dpi)
    #
    #                 initial_guess_validation.plot_acceleration_effect()
    #
    #                 del initial_guess_validation
    #
    # numbers_of_corrections = [0,2,3,4,5,6]
    # alt_values = [0.0]
    # angles = [0.0]
    # amplitudes = [1.0E-3]
    #
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for angle in angles:
    #         for amplitude in amplitudes:
    #             for alt_value in alt_values:
    #                 initial_guess_validation = initialGuessValidation(lagrange_point_nr, orbit_type, alt_value, \
    #                                                                   angle, amplitude, numbers_of_corrections, low_dpi)
    #
    #                 initial_guess_validation.plot_corrections_effect()
    #
    #                 del initial_guess_validation
    #
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



