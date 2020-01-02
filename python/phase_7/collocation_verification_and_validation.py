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
    load_states_continuation, load_initial_conditions_augmented_incl_M, load_patch_points, load_propagated_states

class DisplayCollocationProcedure:
    def __init__(self, low_dpi):


        # scale properties
        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1
        self.magnitudeFactor = 10

        # Define the marker sizes
        self.currentPatchStyle = 'o'
        self.patchSize = 2
        self.widthBar = 0.10

        # label properties
        self.numberOfXTicks = 5

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        self.figSizeLong = (7 * (1 + np.sqrt(5)) / 2, 7*1.5)


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
        self.plotAlpha = 1
        self.lineWidth = 0.5

    def plot_collocation_procedure(self):
        fig = plt.figure(figsize=self.figSizeWide)
        gs = matplotlib.gridspec.GridSpec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        #ax4 = fig.add_subplot(gs[1, :3])

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')
        ax1.set_title('Seed orbit $L_{1} (a_{lt} = 0.0, \\alpha =0.0, H_{lt}=-1.525)$')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')
        ax2.set_title('Input orbit')


        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')
        ax3.set_title('Converged orbit ')


        # ax4.set_xlabel('Collocation procedure stage [-]')
        # ax4.set_ylabel('Deviation Norms [-]')
        # ax4.grid(True, which='both', ls=':')
        # ax4.set_title('Deviations per stage')

        min_x = 1000
        min_y = 1000
        max_x = -1000
        max_y = -1000

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],color='black', marker='x')

        convergence_properties_list = []


        for i in range(3):
            print(i)
            if i == 2:
                pp = 43
            else:
                pp = 26
            orbit_df = load_orbit_augmented('../../data/raw/collocation/' + str(i) + '_stateHistory.txt')
            patch_points_df = load_patch_points('../../data/raw/collocation/' + str(i) + '_stateVectors.txt',pp)
            deviation = np.loadtxt('../../data/raw/collocation/' + str(i) + '_deviations.txt')
            propagated_states_df = load_propagated_states('../../data/raw/collocation/' + str(i) + '_propagatedStates.txt',pp-1)

            convergence_properties_list.append([i, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5], deviation[6]])

            if i == 0:
                print(orbit_df['x'])
                #ax1.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                ax1.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax1.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')
                ax1.scatter(patch_points_df['x'][0], patch_points_df['y'][0], color=self.plottingColors['lambda3'], edgecolor='face', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')


                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test1 = ax1.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda3'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'outgoing' + '}$')

                for row in propagated_states_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        state = row[1].values
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test2 = ax1.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda6'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'incoming' + '}$')

            if i == 1:
                print(orbit_df['x'])
                #ax2.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                ax2.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax2.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')
                ax2.scatter(patch_points_df['x'][0], patch_points_df['y'][0], color=self.plottingColors['lambda3'], edgecolor='face', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test1 = ax2.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda3'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'outgoing' + '}$')

                for row in propagated_states_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        state = row[1].values
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test2 = ax2.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda6'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'incoming' + '}$')

            if i == 2:
                #ax3.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                ax3.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax3.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')
                ax3.scatter(patch_points_df['x'][0], patch_points_df['y'][0], color=self.plottingColors['lambda3'], edgecolor='face', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test1 = ax3.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda3'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'outgoing' + '}$')

                for row in propagated_states_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        state = row[1].values
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        # test2 = ax3.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                        #                      arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                        #                                      color=self.plottingColors['lambda6'], shrinkA=0,
                        #                                      shrinkB=0), label='$V_{' + 'incoming' + '}$')

            if min(orbit_df['x']) < min_x:
                min_x = min(patch_points_df['x'])

            if min(orbit_df['y']) < min_y:
                min_y = min(orbit_df['y'])

            if max(orbit_df['x']) > max_x:
                max_x = max(orbit_df['x'])

            if max(orbit_df['y']) > max_y:
                max_y = max(orbit_df['y'])

            patch_points_df_previous = patch_points_df
            orbit_df_previous = orbit_df

        deviation_df = pd.DataFrame(convergence_properties_list, columns=['stage', 'deltaR', 'deltaV','deltaRint', 'deltaVint','deltaRext', 'deltaVext', 'deltaT'])

        print(deviation_df['deltaR'])
        print(deviation_df['deltaV'])

        print(str("{:1.14e}".format(deviation_df['deltaR'][0])))
        print(str("{:1.14e}".format(deviation_df['deltaV'][0])))
        print( str("{:1.14e}".format(deviation_df['deltaR'][1])))
        print( str("{:1.14e}".format(deviation_df['deltaV'][1])))
        print(str("{:1.14e}".format(deviation_df['deltaR'][2])))
        print(str("{:1.14e}".format(deviation_df['deltaV'][2])))


        Xmiddle = min_x + (max_x - min_x) / 2.0
        Ymiddle = min_y + (max_y - min_y) / 2.0
        scaleDistance = max((max_y - min_y), (max_x - min_x))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax3.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        # ind = np.arange(len(deviation_df['stage']))
        # ax4.bar(ind - 5*self.widthBar/2, deviation_df['deltaR'], self.widthBar, label='$|\Delta R|$',color=self.plottingColors['lambda6'])
        # ax4.bar(ind - 3*self.widthBar/2, deviation_df['deltaV'], self.widthBar, label='$|\Delta V|$',color=self.plottingColors['lambda3'])
        # ax4.bar(ind - 1* self.widthBar/2, deviation_df['deltaRint'], self.widthBar, label='$|\Delta R_{int}|$',color=self.plottingColors['lambda2'])
        # ax4.bar(ind + 1 * self.widthBar/2, deviation_df['deltaVint'], self.widthBar, label='$|\Delta V_{int}|$',color=self.plottingColors['lambda4'])
        # ax4.bar(ind + 3 * self.widthBar/2, deviation_df['deltaRext'], self.widthBar, label='$|\Delta R_{ext}|$',color=self.plottingColors['lambda1'])
        # ax4.bar(ind + 5 * self.widthBar/2, deviation_df['deltaVext'], self.widthBar, label='$|\Delta V_{ext}|$',color=self.plottingColors['lambda5'])
        # ax4.set_yscale('log')
        # ax4.set_ylim([1e-16, 1e-1])
        #
        # lgd1 = ax4.legend(frameon=True, loc='upper left', prop={'size': 9})



        plt.suptitle('$L_{1}$ ($a_{lt} = 0.05, \\alpha =0.0$ rad, $H_{lt}=-1.525$)  - Collocation procedure validation ', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.83)


        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/collocation_procedure.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/collocation_procedure.png', transparent=True, dpi=300)

    def plot_error_distribution(self):
        fig = plt.figure(figsize=self.figSizeLong)
        ax1 = fig.add_subplot(6, 3, 1)
        ax2 = fig.add_subplot(6, 3, 2)
        ax3 = fig.add_subplot(6, 3, 3)
        ax4 = fig.add_subplot(6, 3, 4)
        ax5 = fig.add_subplot(6, 3, 5)
        ax6 = fig.add_subplot(6, 3, 6)
        ax7 = fig.add_subplot(6, 3, 7)
        ax8 = fig.add_subplot(6, 3, 8)
        ax9 = fig.add_subplot(6, 3, 9)
        ax10 = fig.add_subplot(6, 3, 10)
        ax11 = fig.add_subplot(6, 3, 11)
        ax12 = fig.add_subplot(6, 3, 12)
        ax13 = fig.add_subplot(6, 3, 13)
        ax14 = fig.add_subplot(6, 3, 14)
        ax15 = fig.add_subplot(6, 3, 15)
        ax16 = fig.add_subplot(6, 3, 16)
        ax17 = fig.add_subplot(6, 3, 17)
        ax18 = fig.add_subplot(6, 3, 18)

        ax1.set_ylabel('$|e_{i}|$ [-]')
        ax1.grid(True, which='both', ls=':')
        ax4.set_ylabel('$|e_{i}|$ [-]')
        ax4.grid(True, which='both', ls=':')
        ax7.set_ylabel('$|e_{i}|$ [-]')
        ax7.grid(True, which='both', ls=':')
        ax10.set_ylabel('$|e_{i}|$ [-]')
        ax10.grid(True, which='both', ls=':')
        ax13.set_ylabel('$|e_{i}|$ [-]')
        ax13.grid(True, which='both', ls=':')
        ax16.set_ylabel('$|e_{i}|$ [-]')
        ax16.grid(True, which='both', ls=':')

        ax16.set_xlabel('$\\phi$ [-]')
        ax16.grid(True, which='both', ls=':')
        ax17.set_xlabel('$\\phi$ [-]')
        ax17.grid(True, which='both', ls=':')
        ax18.set_xlabel('$\\phi$ [-]')
        ax18.grid(True, which='both', ls=':')

        for i in range(1,19):
            ErrorArray = np.loadtxt(
                '../../data/raw/collocation/error_distribution/' + str(i) + '_ErrorDistribution.txt')
            TimeArray = np.loadtxt('../../data/raw/collocation/error_distribution/' + str(i) + '_TimeDistribution.txt')

            period = TimeArray[len(TimeArray) - 1]

            errorList = []
            positionList = []
            widthList = []
            checksum = 0
            widthFactor = 0.96


            for k in range(len(ErrorArray)):
                errorList.append(ErrorArray[k])
                positionList.append((TimeArray[k] + ((TimeArray[k + 1] - TimeArray[k]) / 2)) / period)
                widthList.append(((TimeArray[k + 1] - TimeArray[k]) / period) * widthFactor)
                checksum = checksum + (TimeArray[k + 1] - TimeArray[k]) / period

            Difference = max(errorList) - min(errorList)
            print(i)
            print(Difference)

            if i == 1:
                ax1.set_xlim([0, 1])
                ax1.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax1.set_yscale('log')
                ax1.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 4.594 \\cdot 10^{-8}$'
                ax1.set_title(stringTitle)
                xCors = np.linspace(0,1,num=len(errorList))
                ax1.plot(xCors,np.ones(len(errorList))*1.0e-9,linestyle='--',color='black')
            if i == 2:
                ax2.set_xlim([0, 1])
                ax2.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax2.set_yscale('log')
                ax2.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 1.680 \\cdot 10^{-8}$'
                ax2.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax2.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 3:
                ax3.set_xlim([0, 1])
                ax3.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax3.set_yscale('log')
                ax3.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 2.455 \\cdot 10^{-9}$'
                ax3.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax3.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 4:
                ax4.set_xlim([0, 1])
                ax4.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax4.set_yscale('log')
                ax4.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 2.016 \\cdot 10^{-9}$'
                ax4.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax4.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 5:
                ax5.set_xlim([0, 1])
                ax5.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax5.set_yscale('log')
                ax5.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 4.210 \\cdot 10^{-10}$'
                ax5.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax5.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 6:
                ax6.set_xlim([0, 1])
                ax6.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax6.set_yscale('log')
                ax6.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 3.114 \\cdot 10^{-10}$'
                ax6.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax6.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 7:
                ax7.set_xlim([0, 1])
                ax7.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax7.set_yscale('log')
                ax7.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 7.7663 \\cdot 10^{-11}$'
                ax7.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax7.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 8:
                ax8.set_xlim([0, 1])
                ax8.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax8.set_yscale('log')
                ax8.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 4.916 \\cdot 10^{-11}$'
                ax8.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax8.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 9:
                ax9.set_xlim([0, 1])
                ax9.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax9.set_yscale('log')
                ax9.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 1.7460 \\cdot 10^{-11}$'
                ax9.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax9.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 10:
                ax10.set_xlim([0, 1])
                ax10.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax10.set_yscale('log')
                ax10.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 6.4884 \\cdot 10^{-12}$'
                ax10.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax10.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 11:
                ax11.set_xlim([0, 1])
                ax11.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax11.set_yscale('log')
                ax11.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 3.8025 \\cdot 10^{-12}$'
                ax11.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax11.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 12:
                ax12.set_xlim([0, 1])
                ax12.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax12.set_yscale('log')
                ax12.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 7.9434 \\cdot 10^{-13}$'
                ax12.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax12.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 13:
                ax13.set_xlim([0, 1])
                ax13.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax13.set_yscale('log')
                ax13.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 3.9321 \\cdot 10^{-5}$'
                ax13.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax13.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 14:
                ax14.set_xlim([0, 1])
                ax14.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax14.set_yscale('log')
                ax14.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 1.7150 \\cdot 10^{-8}$'
                ax14.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax14.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 15:
                ax15.set_xlim([0, 1])
                ax15.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax15.set_yscale('log')
                ax15.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 1.6214 \\cdot 10^{-10}$'
                ax15.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax15.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 16:
                ax16.set_xlim([0, 1])
                ax16.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax16.set_yscale('log')
                ax16.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 2.7122 \\cdot 10^{-11}$'
                ax16.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax16.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 17:
                ax17.set_xlim([0, 1])
                ax17.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax17.set_yscale('log')
                ax17.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 4.7408 \\cdot 10^{-12}$'
                ax17.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax17.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')
            if i == 18:
                ax18.set_xlim([0, 1])
                ax18.bar(positionList, errorList, width=widthList, color=self.plottingColors['singleLine'], linewidth=0.1)
                ax18.set_yscale('log')
                ax18.set_ylim([1e-14, 1e-6])
                stringTitle = str(i) + '. $\Delta e_{i} = 3.6751 \\cdot 10^{-13}$'
                ax18.set_title(stringTitle)
                xCors = np.linspace(0, 1, num=len(errorList))
                ax18.plot(xCors, np.ones(len(errorList)) * 1.0e-9, linestyle='--', color='black')

        plt.suptitle('$L_{1}$ ($a_{lt} = 0.05$, $\\alpha =0.0$ rad, $H_{lt}=-1.525$)  - Mesh refinement process ',
                     size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)

        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/mesh_refinement_procedure.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/mesh_refinement_procedure.png', transparent=True, dpi=300)




if __name__ == '__main__':
    low_dpi = False
    display_collocation_procedure = DisplayCollocationProcedure(low_dpi)
    display_collocation_procedure.plot_collocation_procedure()
    display_collocation_procedure.plot_error_distribution()
    del display_collocation_procedure
