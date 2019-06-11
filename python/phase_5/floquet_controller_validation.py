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

from load_data_augmented import load_orbit_augmented, load_lagrange_points_location_augmented

class floquetController:
    def __init__(self, orbit_type, lagrange_point_nr, acceleration_magnitude, alpha, amplitude, number_of_points, low_dpi):
        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self. accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.amplitude = amplitude
        self.numberOfPatchPoints = number_of_points

        if isinstance(self.amplitude, list):
            self.numberOfSolutions = 6
            self.numberOfAmplitudes = len(self.amplitude)
            self.numberOfAxisTicks = 4
        elif isinstance(self.alpha, list):
            self.numberOfSolutions = 8
            self.numberOfAlphas = len(self.alpha)
            self.numberOfAxisTicks = 9
            self.numberOfAxisTicksOrtho = 4
        else:
            self.numberOfSolutions = 6
            self.numberOfAccelerations = len(self.accelerationMagnitude)
            self.numberOfAxisTicks = 4

        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio =  (7 * (1 + np.sqrt(5)) / 2) / 7
        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               # 'lambda1': blues[40],
                               # 'lambda2': greens[50],
                               # 'lambda3': blues[90],
                               # 'lambda4': blues[90],
                               # 'lambda5': greens[70],
                               # 'lambda6': blues[60],
                               # 'singleLine': blues[80],
                               # 'doubleLine': [greens[50], blues[80]],
                               # 'tripleLine': [blues[40], greens[50], blues[80]],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}

        self.suptitleSize = 20
        self.dpi = 150

    def plot_offset_effect(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(2,2,1 )
        ax2 = fig.add_subplot(2,2,2 )
        ax3 = fig.add_subplot(2,2,3 )
        ax4 = fig.add_subplot(2,2,4 )

        # add libration point
        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude,self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:    
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                    color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                    color='black', marker='x')

        # bodies_df = load_bodies_location()
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # for body in ['Moon', 'Earth']:
        #     x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
        #     y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
        #     z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        #     ax1.contourf(x, y, z, colors='black')
        #     ax2.contourf(x, y, z, colors='black')

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('$|A|$ [-]')
        ax3.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('$|A|$ [-]')
        ax4.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax4.grid(True, which='both', ls=':')

        orbitIdsPlot = list(range(0, len(self.amplitude), 1))

        #deviation_df = pd.DataFrame({'Amplitude': [], 'DeltaR': [], 'DeltaV': []})
        deviation_list = []
        deviation_corrected_list = []

        indexPlotlist = np.linspace(0,len(self.amplitude)-1,num=self.numberOfSolutions).tolist()
        Indexlist = 0


        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
            + str("{:7.6f}".format(self.amplitude[i])) + '_' \
            + str(self.numberOfPatchPoints) + '_initialGuess.txt')



            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])



            deviation_list.append([self.amplitude[i], deltaR, deltaV])


            if i == indexPlotlist[Indexlist]:
                df_corrected = load_orbit_augmented(
                    '../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                    + str("{:7.6f}".format(self.amplitude[i])) + '_' \
                    + str(self.numberOfPatchPoints) + '_CorrectedGuess.txt')

                deviations_corrected = df_corrected.head(1).values[0] - df_corrected.tail(1).values[0]
                deltaR_corrected = np.linalg.norm(deviations_corrected[1:4])
                deltaV_corrected = np.linalg.norm(deviations_corrected[4:7])
                deviation_corrected_list.append([self.amplitude[i], deltaR_corrected, deltaV_corrected])

                legendString = '$A_{r} = $' + str("{:2.1e}".format(self.amplitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1, label= legendString )
                ax2.plot(df_corrected['x'], df_corrected['y'], color=sns.color_palette('viridis', self.numberOfAmplitudes)[i], linewidth=1, label=legendString )
                Indexlist = Indexlist + 1


        deviation_df=pd.DataFrame(deviation_list,columns=['amplitude','deltaR','deltaV'])
        deviation_corrected_df=pd.DataFrame(deviation_corrected_list,columns=['amplitude','deltaR','deltaV'])


        ax3.plot(deviation_df['amplitude'],deviation_df['deltaR'], color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax3.plot(deviation_df['amplitude'],deviation_df['deltaV'], color=sns.color_palette('viridis', 2)[1], linewidth=1)
        ax4.plot(deviation_corrected_df['amplitude'],deviation_corrected_df['deltaR'], color=sns.color_palette('viridis', 2)[0], linewidth=1, label='$| \\Delta R | [-]$ ')
        ax4.plot(deviation_corrected_df['amplitude'],deviation_corrected_df['deltaV'], color=sns.color_palette('viridis', 2)[1], linewidth=1, label='$| \\Delta V | [-]$ ')

        scaleDistance = max((max(df['x'])-min(df['x'])),(max(df['y'])-min(df['y'])), \
                            (max(df_corrected['x']) - min(df_corrected['x'])), (max(df_corrected['y']) - min(df_corrected['y'])))

        minimumX = min(min(df['x']),min(df_corrected['x']))
        minimumY = min(min(df['y']),min(df_corrected['y']))

        maximumX = max(max(df['x']), max(df_corrected['x']))
        maximumY = max(max(df['y']), max(df_corrected['y']))

        Xmiddle = minimumX + (maximumX-minimumX)/2
        Ymiddle = minimumY + (maximumY-minimumY)/2

        ax1.set_xlim([(Xmiddle - 0.5*scaleDistance*self.figureRatio*self.spacingFactor), (Xmiddle + 0.5*scaleDistance*self.figureRatio*self.spacingFactor) ])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance*self.spacingFactor, Ymiddle  + 0.5 * scaleDistance*self.spacingFactor])
        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])


        # ax1.set_xlim([(minimumX - scaleDistance*(self.spacingFactor-1.0)*self.figureRatio), (minimumX + scaleDistance*self.figureRatio*self.spacingFactor) ])
        # ax1.set_ylim([minimumY - scaleDistance*(self.spacingFactor - 1), minimumY  + scaleDistance*self.spacingFactor])
        # ax2.set_xlim([(minimumX - scaleDistance*(self.spacingFactor-1.0)*self.figureRatio), (minimumX + scaleDistance*self.figureRatio*self.spacingFactor) ])
        # ax2.set_ylim([minimumY - scaleDistance * (self.spacingFactor - 1), minimumY + scaleDistance * self.spacingFactor])





        ax3.set_xlim([ min(deviation_df['amplitude']), max(deviation_df['amplitude'])])
        ax3.set_ylim([0, max(deviation_df['deltaV'])*self.spacingFactor])
        ax4.set_xlim([ min(deviation_corrected_df['amplitude']), max(deviation_df['amplitude'])])
        ax4.set_ylim([0, max(deviation_corrected_df['deltaV'])*self.spacingFactor])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9,bottom= -0.1)

        ax1.set_title('Near-periodic solutions before correction')
        ax2.set_title('Corrected near-periodic solutions')
        ax3.set_title('Full-period deviations before correction')
        ax4.set_title('Full-period deviations after correction')

        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        xticks = (np.linspace((Xmiddle - 0.5*scaleDistance*self.figureRatio*self.spacingFactor), (Xmiddle + 0.5*scaleDistance*self.figureRatio*self.spacingFactor), num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks( xticks )
        ax2.xaxis.set_ticks( xticks )

        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        yticks = (np.linspace(minimumY- scaleDistance*(self.spacingFactor - 1), minimumY +scaleDistance * self.spacingFactor, num=self.numberOfAxisTicks))
        #ax1.yaxis.set_ticks( yticks )
        #ax2.yaxis.set_ticks( yticks )

        ax3.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))
        ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.1e'))

        xticks = (np.linspace(min(deviation_df['amplitude']), max(deviation_df['amplitude']), num=self.numberOfAxisTicks))
        ax3.xaxis.set_ticks( xticks )                                                                                                                                                       
        ax4.xaxis.set_ticks( xticks )

        lgd  = ax2.legend(frameon=True, loc='center left',  bbox_to_anchor=(1, 0.5))
        lgd2 = ax4.legend(frameon=True, loc='center left',  bbox_to_anchor=(1, 0.5))

        supttl = fig.suptitle('Initial guesses at L$_{'+ str(self.lagrangePointNr ) + '}$(a$_{lt}$ = '+ str("{:2.1e}".format(self.accelerationMagnitude)) \
        +', $\\alpha$ = ' + str("{:2.1f}".format(self.alpha)) + '$^{\\circ}$) after offset in $\\lambda_3$ direction', size=self.suptitleSize)

        if self.lowDpi:
            fig.savefig('../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_offset_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd, lgd2, supttl), bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_offset_effect.pdf', transparent=True)
        plt.close()
        pass

    def plot_alpha_effect(self):
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

        ax3.set_xlabel('$\\alpha$ [-]')
        ax3.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('$\\alpha$ [-]')
        ax4.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax4.grid(True, which='both', ls=':')

        orbitIdsPlot = list(range(0, len(self.alpha), 1))

        # deviation_df = pd.DataFrame({'Amplitude': [], 'DeltaR': [], 'DeltaV': []})
        deviation_list = []
        deviation_corrected_list = []

        indexPlotlist = np.linspace(0, 315, num=self.numberOfSolutions).tolist()
        Indexlist = 0

        minimumX = 0.0
        minimumY = 0.0
        maximumX = 0.0
        maximumY = 0.0
        maximumDevR = 0.0
        maximumDevV = 0.0

        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha[i])) + '_' \
            + str("{:7.6f}".format(self.amplitude)) + '_' \
            + str(self.numberOfPatchPoints) + '_initialGuess.txt')


            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])


            deviation_list.append([self.alpha[i], deltaR, deltaV])

            lagrange_points_df = load_lagrange_points_location_augmented(0.0, 0.0)
            if self.lagrangePointNr == 1:
                lagrange_point_nrs = ['L1']
            if self.lagrangePointNr == 2:
                lagrange_point_nrs = ['L2']

            for lagrange_point_nr in lagrange_point_nrs:
                ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            color='black', marker='x')
                ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            color='black', marker='x')

            if i == indexPlotlist[Indexlist]:
                df_corrected = load_orbit_augmented(
                    '../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha[i])) + '_' \
                    + str("{:7.6f}".format(self.amplitude)) + '_' \
                    + str(self.numberOfPatchPoints) + '_CorrectedGuess.txt')

                deviations_corrected = df_corrected.head(1).values[0] - df_corrected.tail(1).values[0]
                deltaR_corrected = np.linalg.norm(deviations_corrected[1:4])
                deltaV_corrected = np.linalg.norm(deviations_corrected[4:7])
                deviation_corrected_list.append([self.alpha[i], deltaR_corrected, deltaV_corrected])

                legendString = '$\\alpha = $' + str("{:4.1f}".format(self.alpha[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAlphas)[i], linewidth=1, label= legendString )
                ax2.plot(df_corrected['x'], df_corrected['y'], color=sns.color_palette('viridis', self.numberOfAlphas)[i], linewidth=1, label=legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha[i])
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAlphas)[i], marker='x')
                    ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAlphas)[i], marker='x')

                if Indexlist == 0.0:
                    minimumX = min(min(df['x']),min(df_corrected['x']))
                    minimumY = min(min(df['y']),min(df_corrected['y']))

                    maximumX = max(max(df['x']), max(df_corrected['x']))
                    maximumY = max(max(df['y']), max(df_corrected['y']))

                else:
                    minimumX_temp = min(min(df['x']), min(df_corrected['x']))
                    minimumY_temp = min(min(df['y']), min(df_corrected['y']))

                    maximumX_temp = max(max(df['x']), max(df_corrected['x']))
                    maximumY_temp = max(max(df['y']), max(df_corrected['y']))

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

        Xmiddle = minimumX + (maximumX - minimumX)/2.0
        Ymiddle = minimumY + (maximumY - minimumY)/2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])


        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor), (Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor), num=self.numberOfAxisTicksOrtho))
        ax1.xaxis.set_ticks( xticks )                                                                                                                                                       
        ax2.xaxis.set_ticks( xticks )

        deviation_df = pd.DataFrame(deviation_list, columns=['alpha', 'deltaR', 'deltaV'])
        deviation_corrected_df = pd.DataFrame(deviation_corrected_list, columns=['alpha', 'deltaR', 'deltaV'])

        ax3.plot(deviation_df['alpha'], deviation_df['deltaR'], color=sns.color_palette('viridis', 2)[0],
                     linewidth=1)
        ax3.plot(deviation_df['alpha'], deviation_df['deltaV'], color=sns.color_palette('viridis', 2)[1],
                     linewidth=1)
        ax4.plot(deviation_corrected_df['alpha'], deviation_corrected_df['deltaR'],
                     color=sns.color_palette('viridis', 2)[0], linewidth=1, label='$| \\Delta R | [-]$ ')
        ax4.plot(deviation_corrected_df['alpha'], deviation_corrected_df['deltaV'],
                     color=sns.color_palette('viridis', 2)[1], linewidth=1, label='$| \\Delta V | [-]$ ')

        ax3.set_xlim([min(deviation_df['alpha']), 360])
        ax3.set_ylim([0, max(deviation_df['deltaV'])*1.01])
        ax4.set_xlim([min(deviation_corrected_df['alpha']), 360])
        ax4.set_ylim([0, max(deviation_corrected_df['deltaV'])*1.01])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=-0.1)

        ax1.set_title('Near-periodic solutions before correction')
        ax2.set_title('Corrected near-periodic solutions')
        ax3.set_title('Full-period deviations before correction')
        ax4.set_title('Full-period deviations after correction')

        ax3.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%3.1f'))
        xticks = (np.linspace(0, 360, num=self.numberOfAxisTicks))
        ax3.xaxis.set_ticks(xticks)

        ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%3.1f'))
        xticks = ( np.linspace(0, 360, num=self.numberOfAxisTicks))
        ax4.xaxis.set_ticks(xticks)

        lgd = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
        lgd2 = ax4.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))


        supttl = fig.suptitle('Initial guesses at L$_{'+ str(self.lagrangePointNr ) + '}$(a$_{lt}$ = '+ str("{:2.1e}".format(self.accelerationMagnitude)) \
        +', $|$A$_{r}$$|$ = ' + str("{:2.1e}".format(self.amplitude)) + ') after offset in $\\lambda_3$ direction', size=self.suptitleSize)


        if self.lowDpi:
            fig.savefig(
                '../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                '_alpha_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd, lgd2, supttl), bbox_inches='tight')

        else:
            fig.savefig(
                '../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                '_alpha_effect.pdf', transparent=True)
        plt.close()

        pass

    def plot_accelerationEffect(self):
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

        ax3.set_xlabel('a$_{lt}$ [-]')
        ax3.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('a$_{lt}$ [-]')
        ax4.set_ylabel('$ |\\Delta R|$ [-], $|\\Delta V|$ [-]')
        ax4.grid(True, which='both', ls=':')

        orbitIdsPlot = list(range(0, len(self.accelerationMagnitude), 1))

        # deviation_df = pd.DataFrame({'Amplitude': [], 'DeltaR': [], 'DeltaV': []})
        deviation_list = []
        deviation_corrected_list = []

        indexPlotlist = np.linspace(0,len(self.accelerationMagnitude)-1,num=self.numberOfSolutions).tolist()

        Indexlist = 0

        minimumX = 0.0
        minimumY = 0.0
        maximumX = 0.0
        maximumY = 0.0
        maximumDevR = 0.0
        maximumDevV = 0.0

        for i in orbitIdsPlot:
            df = load_orbit_augmented('../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude[i])) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
            + str("{:7.6f}".format(self.amplitude)) + '_' \
            + str(self.numberOfPatchPoints) + '_initialGuess.txt')


            deviations = df.head(1).values[0] - df.tail(1).values[0]
            deltaR = np.linalg.norm(deviations[1:4])
            deltaV = np.linalg.norm(deviations[4:7])


            deviation_list.append([self.accelerationMagnitude[i], deltaR, deltaV])


            if i == indexPlotlist[Indexlist]:


                df_corrected = load_orbit_augmented(
                    '../../data/raw/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude[i])) + '_' + str("{:7.6f}".format(self.alpha)) + '_' \
                    + str("{:7.6f}".format(self.amplitude)) + '_' \
                    + str(self.numberOfPatchPoints) + '_CorrectedGuess.txt')

                deviations_corrected = df_corrected.head(1).values[0] - df_corrected.tail(1).values[0]
                deltaR_corrected = np.linalg.norm(deviations_corrected[1:4])
                deltaV_corrected = np.linalg.norm(deviations_corrected[4:7])
                deviation_corrected_list.append([self.accelerationMagnitude[i], deltaR_corrected, deltaV_corrected])

                legendString = 'a$_{lt} = $' + str("{:2.1e}".format(self.accelerationMagnitude[i]))
                ax1.plot(df['x'], df['y'], color=sns.color_palette('viridis', self.numberOfAccelerations)[i], linewidth=1, label= legendString )
                ax2.plot(df_corrected['x'], df_corrected['y'], color=sns.color_palette('viridis', self.numberOfAccelerations)[i], linewidth=1, label=legendString )

                lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude[i], self.alpha)
                if self.lagrangePointNr == 1:
                    lagrange_point_nrs = ['L1']
                if self.lagrangePointNr == 2:
                    lagrange_point_nrs = ['L2']

                for lagrange_point_nr in lagrange_point_nrs:
                    ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAccelerations)[i], marker='x')
                    ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                                color=sns.color_palette('viridis', self.numberOfAccelerations)[i], marker='x')

                if Indexlist == 0:
                    minimumX = min(min(df['x']),min(df_corrected['x']))
                    minimumY = min(min(df['y']),min(df_corrected['y']))

                    maximumX = max(max(df['x']), max(df_corrected['x']))
                    maximumY = max(max(df['y']), max(df_corrected['y']))


                else:
                    minimumX_temp = min(min(df['x']), min(df_corrected['x']))
                    minimumY_temp = min(min(df['y']), min(df_corrected['y']))

                    maximumX_temp = max(max(df['x']), max(df_corrected['x']))
                    maximumY_temp = max(max(df['y']), max(df_corrected['y']))

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

        lagrange_points_df = load_lagrange_points_location_augmented(0.0, 0.0)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])
        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5f'))
        xticks = (np.linspace((Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),num=self.numberOfAxisTicks))
        ax1.xaxis.set_ticks(xticks)
        ax2.xaxis.set_ticks(xticks)





        deviation_df = pd.DataFrame(deviation_list, columns=['acceleration', 'deltaR', 'deltaV'])
        deviation_corrected_df = pd.DataFrame(deviation_corrected_list, columns=['acceleration', 'deltaR', 'deltaV'])

        ax3.plot(deviation_df['acceleration'], deviation_df['deltaR'], color=sns.color_palette('viridis', 2)[0], linewidth=1)
        ax3.plot(deviation_df['acceleration'], deviation_df['deltaV'], color=sns.color_palette('viridis', 2)[1], linewidth=1)

        ax4.plot(deviation_corrected_df['acceleration'], deviation_corrected_df['deltaR'], color=sns.color_palette('viridis', 2)[0], linewidth=1, label='$| \\Delta R | [-]$ ')
        ax4.plot(deviation_corrected_df['acceleration'], deviation_corrected_df['deltaV'], color=sns.color_palette('viridis', 2)[1], linewidth=1, label='$| \\Delta V | [-]$ ')

        ax3.set_xlim([ min(deviation_df['acceleration']), max(deviation_df['acceleration'])])
        ax3.set_ylim([0, max(deviation_df['deltaV'])*self.spacingFactor])
        ax4.set_xlim([ min(deviation_corrected_df['acceleration']), max(deviation_df['acceleration'])])
        ax4.set_ylim([0, max(deviation_corrected_df['deltaV'])*self.spacingFactor])
        fig.tight_layout()
        fig.subplots_adjust(top=0.9,bottom= -0.1)

        ax1.set_title('Near-periodic solutions before correction')
        ax2.set_title('Corrected near-periodic solutions')
        ax3.set_title('Full-period deviations before correction')
        ax4.set_title('Full-period deviations after correction')

        lgd = ax2.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
        lgd2 = ax4.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))


        supttl = fig.suptitle('Initial guesses at L$_{'+ str(self.lagrangePointNr ) + '}$($\\alpha$ = '+ str("{:3.1f}".format(self.alpha)) \
        +', $|$A$_{r}$$|$ = ' + str("{:2.1e}".format(self.amplitude)) + ') after offset in $\\lambda_3$ direction', size=self.suptitleSize)


        if self.lowDpi:
            fig.savefig(
                '../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                '_acceleration_effect.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd, lgd2, supttl), bbox_inches='tight')

        else:
            fig.savefig(
                '../../data/figures/floquet_controller/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) + \
                '_acceleration_effect.pdf', transparent=True)








if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.01000]
    alphas = [180.0]
    amplitudes = np.linspace(1.0E-5,1.0E-4,num=91).tolist()
    numbers_of_points = [8]

    low_dpi = True


    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for alpha in alphas:
                    for number_of_points in numbers_of_points:
                        floquet_controller = floquetController(orbit_type, lagrange_point, acceleration_magnitude, \
                                        alpha, amplitudes, number_of_points, low_dpi)
                        floquet_controller.plot_offset_effect()
            del floquet_controller

    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.010000]
    alphas = np.linspace(0,359,num=360).tolist()
    amplitudes = [0.000100]
    numbers_of_points = [8]

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for amplitude in amplitudes:
                    for number_of_points in numbers_of_points:
                        floquet_controller = floquetController(orbit_type, lagrange_point, acceleration_magnitude, \
                                        alphas, amplitude, number_of_points, low_dpi)
                        floquet_controller.plot_alpha_effect()
            del floquet_controller


    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = np.linspace(1.0E-2,1.0E-1,num=91).tolist()
    alphas = [0.000000]
    amplitudes = [0.000100]
    numbers_of_points = [8]

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for alpha in alphas:
                for amplitude in amplitudes:
                    for number_of_points in numbers_of_points:
                        floquet_controller = floquetController(orbit_type, lagrange_point, acceleration_magnitudes, \
                                    alpha, amplitude, number_of_points, low_dpi)
                        floquet_controller.plot_accelerationEffect()
        del floquet_controller