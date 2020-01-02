import numpy as np
import pandas as pd
import json
import matplotlib
from decimal import *
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
import time
import os.path

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
    load_patch_points, load_propagated_states, load_tlt_stage_properties

class TLTCorrectorValidation:
    def __init__(self, libration_point_nr, orbit_type, acceleration_magnitude, alpha, amplitude, number_of_patch_points, correction_time, number_of_cycles, low_dpi):

        self.lagrangePointNr = libration_point_nr
        self.orbitType = orbit_type
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.amplitude = amplitude
        self.numberOfPatchPoints = number_of_patch_points
        self.correctionTime = correction_time
        self.numberOfCycles = number_of_cycles


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

        # Bar properties
        self.widthBar = 0.35

        self.scaleDistanceXWide = self.scaleDistanceY * self.figureRatioWide

        # Define the marker sizes
        self.currentPatchStyle = 'o'
        self.previousPatchStyle = 'D'
        self.patchSize = 9

        #Define arrow properties
        self.magnitudeFactor = 10

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
        self.lineWidth = 1

    def plot_tlt_visualization(self):
        if self.numberOfCycles == 2:
            fig = plt.figure(figsize=self.figSize)

            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

        if self.numberOfCycles == 3:
            fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]*3/2))
            ax1 = fig.add_subplot(3, 2, 1)
            ax2 = fig.add_subplot(3, 2, 2)
            ax3 = fig.add_subplot(3, 2, 3)
            ax4 = fig.add_subplot(3, 2, 4)
            ax5 = fig.add_subplot(3, 2, 5)
            ax6 = fig.add_subplot(3, 2, 6)

            ax5.set_xlabel('x [-]')
            ax5.set_ylabel('y [-]')
            ax5.grid(True, which='both', ls=':')

            ax6.set_xlabel('x [-]')
            ax6.set_ylabel('y [-]')
            ax6.grid(True, which='both', ls=':')

        if self.numberOfCycles == 4:
            fig = plt.figure(figsize=(self.figSize[0], self.figSize[1] * 4 / 2))
            ax1 = fig.add_subplot(4, 2, 1)
            ax2 = fig.add_subplot(4, 2, 2)
            ax3 = fig.add_subplot(4, 2, 3)
            ax4 = fig.add_subplot(4, 2, 4)
            ax5 = fig.add_subplot(4, 2, 5)
            ax6 = fig.add_subplot(4, 2, 6)
            ax7 = fig.add_subplot(4, 2, 7)
            ax8 = fig.add_subplot(4, 2, 8)

            ax5.set_xlabel('x [-]')
            ax5.set_ylabel('y [-]')
            ax5.grid(True, which='both', ls=':')

            ax6.set_xlabel('x [-]')
            ax6.set_ylabel('y [-]')
            ax6.grid(True, which='both', ls=':')

            ax7.set_xlabel('x [-]')
            ax7.set_ylabel('y [-]')
            ax7.grid(True, which='both', ls=':')

            ax8.set_xlabel('x [-]')
            ax8.set_ylabel('y [-]')
            ax8.grid(True, which='both', ls=':')



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

        for lagrange_point_nr in lagrange_point_nrs:
            # ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            # ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            # ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            # ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            ax1.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
            ax2.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
            ax3.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
            ax4.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')

            if self.numberOfCycles > 2:
                # ax5.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                #           color='black', marker='x')
                # ax6.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                #             color='black', marker='x')

                ax5.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
                ax6.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')

            if self.numberOfCycles > 3:
                # ax7.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                #           color='black', marker='x')
                # ax8.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                #             color='black', marker='x')
                ax7.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
                ax8.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')




        patch_points_df = load_patch_points('../../data/raw/tlt_corrector/L' +str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                        + '_' + str(self.numberOfPatchPoints) + '_' +  str("{:7.6f}".format(self.correctionTime)) \
                        + '_0_0_stateVectors.txt', self.numberOfPatchPoints)

        orbit_df = load_orbit_augmented('../../data/raw/tlt_corrector/L' +str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                        + '_' + str(self.numberOfPatchPoints) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                        + '_0_0_stateHistory.txt')

        propagated_states_df = load_propagated_states(
            '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
            + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
            + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
            + '_' + str(self.numberOfPatchPoints) + '_' + str("{:7.6f}".format(self.correctionTime)) \
            + '_0_0_propagatedStates.txt', self.numberOfPatchPoints - 1)

        # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][0]])
        # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][1]])
        # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][2]])
        # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][3]])
        #
        #
        # print(orbit_df.iloc[69])
        # print(orbit_df.iloc[139])
        # print(orbit_df.iloc[209])
        # print(orbit_df.iloc[279])


        ax1.plot(orbit_df['x'].iloc[0:69],orbit_df['y'].iloc[0:69],color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Current guess')
        patch_points_df_previous = patch_points_df

        for row in patch_points_df.iterrows():
            state = row[1].values
            x_base = state[0]
            y_base = state[1]
            x_end = x_base + state[3]/self.magnitudeFactor
            y_end = y_base + state[4] /self.magnitudeFactor

           # ax1.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end), arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1', color='red', shrinkA=0, shrinkB=0))

        min_x = min(min(patch_points_df['x']),min(orbit_df['x']))
        min_y = min(min(patch_points_df['y']),min(orbit_df['y']))
        max_x = max(min(patch_points_df['x']),max(orbit_df['x']))
        max_y = max(min(patch_points_df['y']),max(orbit_df['y']))

        ax1.scatter(patch_points_df['x'], patch_points_df['y'],color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

        lgd1 = ax1.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

        ax1.plot(orbit_df['x'].iloc[70:139], orbit_df['y'].iloc[70:139], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)

        ax1.plot(orbit_df['x'].iloc[140:209], orbit_df['y'].iloc[140:209], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)
        #
        ax1.plot(orbit_df['x'].iloc[210:279], orbit_df['y'].iloc[210:279], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)

        for i in range(1,self.numberOfCycles+1):
            for j in range(1,3):
                if (i == self.numberOfCycles and j == 2):
                    # Condition to
                    break
                patch_points_df = load_patch_points(
                    '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                    + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                    + '_' + str(self.numberOfPatchPoints) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                    + '_'+ str(i) +'_' + str(j) + '_stateVectors.txt', self.numberOfPatchPoints)

                propagated_states_df = load_propagated_states(
                    '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                    + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                    + '_' + str(self.numberOfPatchPoints) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                    + '_' + str(i) + '_' + str(j) + '_propagatedStates.txt', self.numberOfPatchPoints - 1 )

                orbit_df = load_orbit_augmented('../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                                                + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                                                + str("{:7.6f}".format(self.alpha)) + '_' + str(
                    "{:7.6f}".format(self.amplitude)) \
                                                + '_' + str(self.numberOfPatchPoints) + '_' + str(
                    "{:7.6f}".format(self.correctionTime)) \
                                                    + '_' + str(i) + '_' + str(j) + '_stateHistory.txt')
                if i == 1 and j == 1:

                    ax2.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Current Guess')

                    for row in patch_points_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                            test1 = ax2.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0, shrinkB=0),label='$V_{'+ 'outgoing'+ '}$')

                    for row in propagated_states_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            state = row[1].values
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                            test2 = ax2.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                     arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                     color=self.plottingColors['lambda6'], shrinkA=0, shrinkB=0),label='$V_{'+ 'incoming'+ '}$')

                    ax2.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')
                    ax2.scatter( -5 ,-5, c=self.plottingColors['lambda3'],marker='$\\longrightarrow$',s=40, label='Outgoing velocity' )
                    ax2.scatter( -5 ,-5, c=self.plottingColors['lambda6'],marker='$\\longrightarrow$',s=40, label='Incoming velocity' )

                    lgd2 = ax2.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})



                if i == 1 and j == 2:
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][0]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][1]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][2]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][3]])

                    ax3.plot(orbit_df['x'].iloc[0:69], orbit_df['y'].iloc[0:69], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Current guess')
                    ax3.plot(orbit_df_previous['x'], orbit_df_previous['y'], color=self.plottingColors['doubleLine'][0],linewidth=self.lineWidth,linestyle='--', label='Previous guess')

                    ax3.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle, s=self.patchSize, label='Current patch points')
                    ax3.scatter(patch_points_df_previous['x'], patch_points_df_previous['y'], color='black', marker=self.previousPatchStyle, s=self.patchSize, label='Previous patch points')

                    lgd3 = ax3.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
                    ax3.plot(orbit_df['x'].iloc[70:139], orbit_df['y'].iloc[70:139], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)
                    ax3.plot(orbit_df['x'].iloc[140:209], orbit_df['y'].iloc[140:209], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)
                    ax3.plot(orbit_df['x'].iloc[210:279], orbit_df['y'].iloc[210:279], color=self.plottingColors['singleLine'],linewidth=self.lineWidth)


                if i == 2 and j == 1:
                    ax4.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth,label='Current guess')

                    for row in patch_points_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                            ax4.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                    arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                    color=self.plottingColors['lambda3'], shrinkA=0, shrinkB=0))

                    for row in propagated_states_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            state = row[1].values
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                            ax4.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                        arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                        color=self.plottingColors['lambda6'], shrinkA=0, shrinkB=0))

                    ax4.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize,label='Current patch points')
                    ax4.scatter(-5, -5, c=self.plottingColors['lambda3'], marker='$\\longrightarrow$', s=40,
                                    label='Outgoing velocity')
                    ax4.scatter(-5, -5, c=self.plottingColors['lambda6'], marker='$\\longrightarrow$', s=40,
                                    label='Incoming velocity')

                    lgd4 = ax4.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

                if i == 2 and j == 2:
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][0]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][1]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][2]])
                    # print(orbit_df.loc[orbit_df['x'] == propagated_states_df['x'][3]])

                    ax5.plot(orbit_df['x'].iloc[0:69], orbit_df['y'].iloc[0:69], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Current guess')

                    ax5.plot(orbit_df_previous['x'], orbit_df_previous['y'],
                            color=self.plottingColors['doubleLine'][0], linewidth=self.lineWidth, linestyle='--',
                            label='Previous guess')

                    ax5.scatter(patch_points_df['x'], patch_points_df['y'], color='black',
                                marker=self.currentPatchStyle, s=self.patchSize, label='Current patch points')
                    ax5.scatter(patch_points_df_previous['x'], patch_points_df_previous['y'], color='black',
                                    marker=self.previousPatchStyle, s=self.patchSize, label='Previous patch points')

                    lgd5 = ax5.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
                    ax5.plot(orbit_df['x'].iloc[70:139], orbit_df['y'].iloc[70:139],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)
                    ax5.plot(orbit_df['x'].iloc[140:209], orbit_df['y'].iloc[140:209],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)
                    ax5.plot(orbit_df['x'].iloc[210:279], orbit_df['y'].iloc[210:279],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)

                if i == 3 and j == 1:
                    ax6.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],
                            linewidth=self.lineWidth, label='Current guess')

                    for row in patch_points_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                        ax6.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                        arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0,
                                                             shrinkB=0))
                    ax6.scatter(patch_points_df['x'], patch_points_df['y'], color='black',
                                        marker=self.currentPatchStyle, s=self.patchSize, label='Current patch points')
                    ax6.scatter(-5, -5, c=self.plottingColors['lambda3'], marker='$\\longrightarrow$', s=40,
                                        label='Outgoing velocity')
                    ax6.scatter(-5, -5, c=self.plottingColors['lambda6'], marker='$\\longrightarrow$', s=40,
                                        label='Incoming velocity')

                    lgd6 = ax6.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

                if i == 3 and j == 2:
                    ax7.plot(orbit_df['x'].iloc[0:69], orbit_df['y'].iloc[0:69], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Current guess')

                    ax7.plot(orbit_df_previous['x'], orbit_df_previous['y'],
                            color=self.plottingColors['doubleLine'][0], linewidth=self.lineWidth, linestyle='--',
                            label='Previous guess')

                    ax7.scatter(patch_points_df['x'], patch_points_df['y'], color='black',
                                marker=self.currentPatchStyle, s=self.patchSize, label='Current patch points')
                    ax7.scatter(patch_points_df_previous['x'], patch_points_df_previous['y'], color='black',
                                    marker=self.previousPatchStyle, s=self.patchSize, label='Previous patch points')

                    lgd7 = ax7.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})
                    ax7.plot(orbit_df['x'].iloc[70:139], orbit_df['y'].iloc[70:139],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)
                    ax7.plot(orbit_df['x'].iloc[140:209], orbit_df['y'].iloc[140:209],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)
                    ax7.plot(orbit_df['x'].iloc[210:279], orbit_df['y'].iloc[210:279],
                             color=self.plottingColors['singleLine'], linewidth=self.lineWidth)

                if i == 4 and j == 1:
                    ax8.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],
                            linewidth=self.lineWidth, label='Current guess')

                    for row in patch_points_df.iterrows():
                        if row[0] > 0:
                            state = row[1].values
                            x_base = state[0]
                            y_base = state[1]
                            x_end = x_base + state[3] / self.magnitudeFactor
                            y_end = y_base + state[4] / self.magnitudeFactor

                        ax8.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                        arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0,
                                                             shrinkB=0))
                    ax8.scatter(patch_points_df['x'], patch_points_df['y'], color='black',
                                        marker=self.currentPatchStyle, s=self.patchSize, label='Current patch points')
                    ax8.scatter(-5, -5, c=self.plottingColors['lambda3'], marker='$\\longrightarrow$', s=40,
                                        label='Outgoing velocity')
                    ax8.scatter(-5, -5, c=self.plottingColors['lambda6'], marker='$\\longrightarrow$', s=40,
                                        label='Incoming velocity')

                    lgd6 = ax8.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 8})

                if min(patch_points_df['x']) < min_x:
                    min_x = min(patch_points_df['x'])

                if min(patch_points_df['y']) < min_y:
                    min_y = min(patch_points_df['y'])

                if max(patch_points_df['x']) > max_x:
                    max_x = max(patch_points_df['x'])

                if max(patch_points_df['y']) > max_y:
                    max_y = max(patch_points_df['y'])

                patch_points_df_previous = patch_points_df
                orbit_df_previous = orbit_df



        Xmiddle = min_x + (max_x - min_x) / 2.0
        Ymiddle = min_y + (max_y - min_y) / 2.0
        scaleDistance = max((max_y - min_y), (max_x - min_x))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim( [Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax3.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax4.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax4.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax1.set_title('Initial guess from Floquet controller')
        ax2.set_title('Cycle 1 - Level I ouput')
        ax3.set_title('Cycle 1 - Level II ouput')
        ax4.set_title('Cycle 2 - Level I ouput')

        if self.numberOfCycles > 2:
            ax5.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax5.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax6.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax6.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax5.set_title('Cycle 2 - Level II ouput')
            ax6.set_title('Cycle 3 - Level I ouput')

        if self.numberOfCycles > 3:
            ax7.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax7.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax8.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
            ax8.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor,Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

            ax7.set_title('Cycle 3 - Level II ouput')
            ax8.set_title('Cycle 4 - Level I ouput')

        # Add a subtitle and do a tight layout
        # suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = ' + str(
        #     "{:2.1e}".format(self.accelerationMagnitude)) + ' $, $\\alpha =' + str(
        #     "{:3.1f}".format(self.alpha)) + '$ $|A| = '+ str(
        #     "{:2.1e}".format(self.amplitude)) +' $) Correction procedure ', size=self.suptitleSize)

        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = 0.1$, $\\alpha =\\frac{1}{2}\\pi$ rad, $A_{x} = 8.0 \\cdot 10^{-3} '' $) - Correction procedure ', size=self.suptitleSize)

        fig.tight_layout()
        if self.numberOfCycles == 2:
            fig.subplots_adjust(top=0.9)
        elif self.numberOfCycles == 3:
            fig.subplots_adjust(top=0.93)
        elif self.numberOfCycles == 4:
            fig.subplots_adjust(top=0.93)


        if self.lowDpi:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) + \
                        '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.numberOfPatchPoints) + '_' \
                        + str(self.correctionTime) + '_' + str(self.numberOfCycles) + '_visualization.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) + \
                        '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.numberOfPatchPoints) + '_' \
                        + str(self.correctionTime) + '_' + str(self.numberOfCycles) +  \
                        '_visualization.png', transparent=True,dpi=300)

        plt.close()
        pass

    def plot_convergence_behaviour(self):
        fig = plt.figure(figsize=self.figSizeWide)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax3 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('Targeter Cycle [-]')
        ax1.set_ylabel('$|| \\Delta \\bar{R} ||$, $|| \\Delta \\bar{V} ||$ [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('Targeter Cycle  [-]')
        ax2.set_ylabel('Computational cost [s]')
        ax2.grid(True, which='both', ls=':')


        # Determine which files exist to compute the x-axis and extract the files
        numberOfCycles = 1
        fileExistence = True

        while fileExistence == True:
            fileExistence = os.path.isfile('../../data/raw/tlt_corrector/L' +str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                        + '_' + str(self.numberOfPatchPoints) + '_' +  str("{:7.6f}".format(self.correctionTime)) \
                        + '_' + str(numberOfCycles) + '_1_stateVectors.txt')

            if fileExistence == True:
                numberOfCycles = numberOfCycles + 1
            if fileExistence == False:
                numberOfCycles = numberOfCycles - 1


        # Create dataframe with all relevant data
        convergence_properties_list = []

        initDev =  np.loadtxt('../../data/raw/tlt_corrector/L' +str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                        + '_' + str(self.numberOfPatchPoints) + '_' +  str("{:7.6f}".format(self.correctionTime)) \
                        + '_0_0_deviations.txt')
        convergence_properties_list.append([0,0,initDev[0],initDev[1],initDev[2],initDev[3],initDev[4],initDev[5],initDev[6]])

        for i in range(1,numberOfCycles+1):
            for j in range(1,3):
                if (i == numberOfCycles and j == 2):
                    break
                stageDev = np.loadtxt('../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                           + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                           + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude)) \
                           + '_' + str(self.numberOfPatchPoints) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                           + '_' + str(i) +'_' + str(j) +'_deviations.txt')
                convergence_properties_list.append([i, j, stageDev[0], stageDev[1], stageDev[2], stageDev[3], stageDev[4], \
                                                    stageDev[5], stageDev[6]])

        deviation_df = pd.DataFrame(convergence_properties_list, \
                                    columns=['cycle', 'stage', 'deltaR','deltaV','deltaVint','deltaVext','deltaT','time','iterations'])


        # Create the xlabel tuple and indx
        xlabelTuple = ('Input', )
        loopIndex = 1
        indLabel = [0]
        indLabelSecond = [0,1,2,3,4,5,6,7,8,9,10,11]
        xlabelTupleSecond = ('','I','II','I','II','I','II','I','II','I','II','I')




        for i in range(1, numberOfCycles + 1):
            string = 'Cycle ' + str(i)
            xlabelTuple = xlabelTuple + (string,)

            if i == 1:
                indLabel = np.append(indLabel,[1.5])

            elif i == numberOfCycles:
                indLabel = np.append(indLabel, indLabel[len(indLabel) - 1] + 1.5)


            else:
                indLabel = np.append(indLabel,indLabel[len(indLabel)-1]+2)


        ind = np.arange(len(deviation_df['cycle']))


        ax1.bar(ind - self.widthBar/2, deviation_df['deltaR'], self.widthBar, label='$||\Delta \\bar{R}||$',color=self.plottingColors['lambda6'])
        ax1.bar(ind + self.widthBar/2, deviation_df['deltaV'], self.widthBar, label='$||\Delta \\bar{V}||$',color=self.plottingColors['lambda3'])
        ax1.set_yscale('log')
        ax1.hlines(1.0E-12,min(ind)-1,max(ind)+1,color='black',linestyle='--',label=' $||\Delta \\bar{R} ||$ Tolerance')
        ax1.hlines(5.0E-12,min(ind)-1,max(ind)+1,color='black',linestyle='-.',label=' $||\Delta \\bar{V} ||$ Tolerance')

        ax1.set_xlim(min(ind)-1,max(ind)+1)


        # ax2.set_ylim([0, 40])
        # ax2.set_yticks([0,10,20,30,40])
        # ax2.set_yticklabels(('0','10','20','30','40'))

        ax3 = ax1.twiny()
        ax3.set_xlim(min(ind)-1,max(ind)+1)
        ax3.grid(False)

        ax4 = ax2.twiny()
        ax4.set_xlim(min(ind) - 1, max(ind) + 1)
        ax4.grid(False)

        ax5 = ax2.twinx()
        ax5.set_xlim(min(ind) - 1, max(ind) + 1)
        ax2.set_xlim(min(ind) - 1, max(ind) + 1)

        ax5.grid(False)
        ax5.set_ylim([0,8])


        ax1.set_xticks(indLabel)
        ax1.set_xticklabels(xlabelTuple)

        ax3.set_xticks(indLabelSecond)
        ax3.set_xticklabels(xlabelTupleSecond)

        ax4.set_xticks(indLabelSecond)
        ax4.set_xticklabels(xlabelTupleSecond)

        ax2.bar(ind - self.widthBar / 2, deviation_df['time'], self.widthBar, label='Time',color=self.plottingColors['lambda6'])
        ax2.bar(-2, 2, self.widthBar, label='Iterations Level I',color=self.plottingColors['lambda3'])

        ax5.bar(ind + self.widthBar / 2, deviation_df['iterations'], self.widthBar, label='Iterations Level I',color=self.plottingColors['lambda3'])
        ax2.set_yscale('log')
        ax2.set_ylim([0,10e2])

        ax2.set_xticks(indLabel)
        ax2.set_xticklabels(xlabelTuple)

        lgd1 = ax1.legend(frameon=True, loc='upper right', prop={'size': 7})
        lgd2 = ax2.legend(frameon=True, loc='upper right', prop={'size': 7})


        # suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = ' + str(
        #     "{:2.1e}".format(self.accelerationMagnitude)) + ' $, $\\alpha =' + str(
        #     "{:3.1f}".format(self.alpha)) + '$ $|A| = ' + str(
        #     "{:2.1e}".format(self.amplitude)) + ' $) Convergence behaviour using ' + str(self.numberOfPatchPoints) + ' patch points ', size=self.suptitleSize)
        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = 0.1$, $\\alpha =\\frac{1}{2}\\pi$, $A_{x} = 8.0 \\cdot 10^{-3} '' $) - Convergence behaviour using ' + str(self.numberOfPatchPoints) + ' nodes ', size=self.suptitleSize)

        ax1.set_title('State defects', y=1.10)
        ax2.set_title('TLT computational cost and Level I behaviour', y=1.10)
        ax5.set_ylabel('Number of corrections [-]')

        fig.tight_layout()
        fig.subplots_adjust(top=0.77)



        if self.lowDpi:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) + \
                        '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.numberOfPatchPoints) + '_' \
                        + str(self.correctionTime) + '_convergence_behaviour.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) + \
                        '_' + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.numberOfPatchPoints) + '_' \
                        + str(self.correctionTime) +  \
                        '_convergence_behaviour.png',  transparent=True, dpi=300)
        pass

    def plot_sensitivity_analysis(self):
        fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]*3/2))
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')

        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')

        ax5.set_xlabel('x [-]')
        ax5.set_ylabel('y [-]')
        ax5.grid(True, which='both', ls=':')



        ax2.set_xlabel('Number of nodes [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        ax4.set_xlabel('Number of nodes [-]')
        ax4.set_ylabel('y [-]')
        ax4.grid(True, which='both', ls=':')

        ax6.set_xlabel('Number of nodes [-]')
        ax6.set_ylabel('y [-]')
        ax6.grid(True, which='both', ls=':')

        ax1.set_title('$A_{x} = 1.0 \\cdot 10^{-4}$ ')
        ax2.set_title('Convergence behaviour at $A_{x}=1.0 \\cdot 10^{-4}$')
        ax3.set_title('$A_{x} = 1.0 \\cdot 10^{-3} $')
        ax4.set_title('Convergence behaviour at $A_{x}=1.0 \\cdot 10^{-3}$')
        ax5.set_title('$A_{x} = 1.0 \\cdot 10^{-2} $')
        ax6.set_title('Convergence behaviour at $A_{x}=1.0 \\cdot 10^{-2}$')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax1.contourf(x, y, z, colors='black')
        ax3.contourf(x, y, z, colors='black')
        ax5.contourf(x, y, z, colors='black')

        # Create the orthogonal plots
        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)

        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            # ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            #
            # ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            # ax5.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')

            ax1.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
            ax3.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')
            ax5.scatter(0.83860821797119, 0.024289027459937, color='black', marker='x')


        min_x1 = 1000
        min_y1 = 1000
        max_x1 = -1000
        max_y1 = -1000

        min_x3 = 1000
        min_y3 = 1000
        max_x3 = -1000
        max_y3 = -1000

        min_x5 = 1000
        min_y5 = 1000
        max_x5 = -1000
        max_y5 = -1000

        for i in range(len(self.amplitude)):
            df = load_orbit_augmented(
                    '../../data/raw/initial_guess/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                        "{:7.6f}".format(self.alpha)) + '_' \
                    + str("{:7.6f}".format(self.amplitude[i])) + '_' \
                    + str("{:7.6f}".format(self.correctionTime)) + '_stateHistory.txt')
            if i == 0:
                ax1.plot(df['x'],df['y'],color='black',linewidth=self.lineWidth, linestyle='--', label='Input')

                if min(df['x']) < min_x1:
                    min_x1 = min(df['x'])
                if min(df['y']) < min_y1:
                    min_y1 = min(df['y'])
                if max(df['x']) > max_x1:
                    max_x1 = max(df['x'])
                if max(df['y']) > max_y1:
                    max_y1 = max(df['y'])

            if i == 1:
                ax3.plot(df['x'],df['y'],color='black',linewidth=self.lineWidth, linestyle='--', label='Input')

                if min(df['x']) < min_x3:
                    min_x3 = min(df['x'])
                if min(df['y']) < min_y3:
                    min_y3 = min(df['y'])
                if max(df['x']) > max_x3:
                    max_x3 = max(df['x'])
                if max(df['y']) > max_y3:
                    max_y3 = max(df['y'])
            if i == 2:
                ax5.plot(df['x'],df['y'],color='black',linewidth=self.lineWidth, linestyle='--', label='Input')

                if min(df['x']) < min_x5:
                    min_x5 = min(df['x'])
                if min(df['y']) < min_y5:
                    min_y5 = min(df['y'])
                if max(df['x']) > max_x5:
                    max_x5 = max(df['x'])
                if max(df['y']) > max_y5:
                    max_y5 = max(df['y'])


        # Create the patch point influence
        ind = np.arange(self.numberOfPatchPoints[0],self.numberOfPatchPoints[len(self.numberOfPatchPoints)-1]+1)
        max_cycles = 0
        for i in range(len(self.amplitude)):

            patch_point_influence_list = []
            for j in range(len(self.numberOfPatchPoints)):

                numberOfCycles = 1
                fileExistence = True

                while fileExistence == True:
                    fileExistence = os.path.isfile('../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                                                   + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                                                   + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude[i])) \
                                                   + '_' + str(self.numberOfPatchPoints[j]) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                                               + '_' + str(numberOfCycles) + '_1_stateVectors.txt')

                    if fileExistence == True:
                        numberOfCycles = numberOfCycles + 1
                    if fileExistence == False:
                        numberOfCycles = numberOfCycles - 1

                patch_point_influence_list.append([self.numberOfPatchPoints[j],numberOfCycles])

            point_influence_df = pd.DataFrame(patch_point_influence_list, \
                                        columns=['patchpoint', 'cycles'])

            if i == 0:
                ax2.bar(ind, point_influence_df['cycles'],self.widthBar, label='Number of cycles',color=self.plottingColors['lambda6'])

                if max(point_influence_df['cycles']) > max_cycles:
                    max_cycles = max(point_influence_df['cycles'])

                df_min = load_orbit_augmented(
                    '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str(
                        "{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                    + str("{:7.6f}".format(self.alpha)) + '_' + str(
                        "{:7.6f}".format(self.amplitude[0])) + '_' + str(
                        (point_influence_df['patchpoint'][0])) + '_' + str(
                        "{:7.6f}".format(self.correctionTime)) + '_' + str(
                        point_influence_df['cycles'][0]) + '_1_stateHistory.txt')

                ax1.plot(df_min['x'], df_min['y'], color=self.plottingColors['lambda6'], linewidth=self.lineWidth,
                         label=str((point_influence_df['patchpoint'][0])) + ' nodes')

                if min(df_min['x']) < min_x1:
                    min_x1 = min(df_min['x'])
                if min(df_min['y']) < min_y1:
                    min_y1 = min(df_min['y'])
                if max(df_min['x']) > max_x1:
                    max_x1 = max(df_min['x'])
                if max(df_min['y']) > max_y1:
                    max_y1 = max(df_min['y'])

                df_max = load_orbit_augmented(
                    '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str(
                        "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                        "{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude[0])) \
                    + '_' + str(point_influence_df['patchpoint'][len(point_influence_df) - 1]) + '_' + str(
                        "{:7.6f}".format(self.correctionTime)) \
                    + '_' + str(point_influence_df['cycles'][len(point_influence_df) - 1]) + '_1_stateHistory.txt')

                ax1.plot(df_max['x'], df_max['y'], color=self.plottingColors['lambda3'], linewidth=self.lineWidth,
                         label=str(
                             (point_influence_df['patchpoint'][len(point_influence_df) - 1])) + ' nodes')

                if min(df_max['x']) < min_x1:
                    min_x1 = min(df_max['x'])
                if min(df_max['y']) < min_y1:
                    min_y1 = min(df_max['y'])
                if max(df_max['x']) > max_x1:
                    max_x1 = max(df_max['x'])
                if max(df_max['y']) > max_y1:
                    max_y1 = max(df_max['y'])

            lgd1 = ax1.legend(frameon=True, loc='upper right', prop={'size': 9})

            if i == 1:
                ax4.bar(ind, point_influence_df['cycles'],self.widthBar, label='Number of cycles',color=self.plottingColors['lambda6'])

                if max(point_influence_df['cycles']) > max_cycles:
                    max_cycles = max(point_influence_df['cycles'])

                    df_min = load_orbit_augmented(
                        '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str(
                            "{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:7.6f}".format(self.alpha)) + '_' + str(
                            "{:7.6f}".format(self.amplitude[1])) + '_' + str(
                            (point_influence_df['patchpoint'][0])) + '_' + str(
                            "{:7.6f}".format(self.correctionTime)) + '_' + str(
                            point_influence_df['cycles'][0]) + '_1_stateHistory.txt')

                    ax3.plot(df_min['x'], df_min['y'], color=self.plottingColors['lambda6'], linewidth=self.lineWidth,
                             label=str((point_influence_df['patchpoint'][0])) + ' nodes')

                    if min(df_min['x']) < min_x3:
                        min_x3 = min(df_min['x'])
                    if min(df_min['y']) < min_y3:
                        min_y3 = min(df_min['y'])
                    if max(df_min['x']) > max_x3:
                        max_x3 = max(df_min['x'])
                    if max(df_min['y']) > max_y3:
                        max_y3 = max(df_min['y'])

                    df_max = load_orbit_augmented(
                        '../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str(
                            "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
                            "{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude[1])) \
                        + '_' + str(point_influence_df['patchpoint'][len(point_influence_df) - 1]) + '_' + str(
                            "{:7.6f}".format(self.correctionTime)) \
                        + '_' + str(point_influence_df['cycles'][len(point_influence_df) - 1]) + '_1_stateHistory.txt')

                    ax3.plot(df_max['x'], df_max['y'], color=self.plottingColors['lambda3'], linewidth=self.lineWidth,
                             label=str(
                                 (point_influence_df['patchpoint'][len(point_influence_df) - 1])) + ' nodes')

                    if min(df_max['x']) < min_x3:
                        min_x3 = min(df_max['x'])
                    if min(df_max['y']) < min_y3:
                        min_y3 = min(df_max['y'])
                    if max(df_max['x']) > max_x3:
                        max_x3 = max(df_max['x'])
                    if max(df_max['y']) > max_y3:
                        max_y3 = max(df_max['y'])

                lgd3 = ax3.legend(frameon=True, loc='upper right', prop={'size': 9})

            if i == 2:
                ax6.bar(ind, point_influence_df['cycles'],self.widthBar, label='Number of cycles',color=self.plottingColors['lambda6'])

                if max(point_influence_df['cycles']) > max_cycles:
                    max_cycles = max(point_influence_df['cycles'])

                df_min = load_orbit_augmented('../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                          + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude[2])) + '_' + str((point_influence_df['patchpoint'][0])) + '_' + str(
                    "{:7.6f}".format(self.correctionTime)) + '_' + str(point_influence_df['cycles'][0]) + '_1_stateHistory.txt')

                ax5.plot(df_min['x'], df_min['y'], color=self.plottingColors['lambda6'], linewidth=self.lineWidth,
                         label=str((point_influence_df['patchpoint'][0])) + ' nodes')

                if min(df_min['x']) < min_x5:
                    min_x5 = min(df_min['x'])
                if min(df_min['y']) < min_y5:
                    min_y5 = min(df_min['y'])
                if max(df_min['x']) > max_x5:
                    max_x5 = max(df_min['x'])
                if max(df_min['y']) > max_y5:
                    max_y5 = max(df_min['y'])

                df_max = load_orbit_augmented('../../data/raw/tlt_corrector/L' + str(self.lagrangePointNr) + '_' + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + '_' + str("{:7.6f}".format(self.amplitude[2])) \
                    + '_' + str(point_influence_df['patchpoint'][len(point_influence_df) - 1]) + '_' + str("{:7.6f}".format(self.correctionTime)) \
                    + '_' + str(point_influence_df['cycles'][len(point_influence_df) - 1]) + '_1_stateHistory.txt')

                ax5.plot(df_max['x'], df_max['y'], color=self.plottingColors['lambda3'], linewidth=self.lineWidth,
                         label=str((point_influence_df['patchpoint'][len(point_influence_df) - 1])) + ' nodes')

                if min(df_max['x']) < min_x5:
                    min_x5 = min(df_max['x'])
                if min(df_max['y']) < min_y5:
                    min_y5 = min(df_max['y'])
                if max(df_max['x']) > max_x5:
                    max_x5 = max(df_max['x'])
                if max(df_max['y']) > max_y5:
                    max_y5 = max(df_max['y'])

        lgd5 = ax5.legend(frameon=True, loc='upper right', prop={'size': 9})

        Xmiddle1 = min_x1 + (max_x1 - min_x1) / 2.0
        Ymiddle1 = min_y1 + (max_y1 - min_y1) / 2.0
        scaleDistance1 = max((max_y1 - min_y1), (max_x1 - min_x1))

        ax1.set_xlim([(Xmiddle1 - 0.5 * scaleDistance1 * self.figureRatio * self.spacingFactor),
                      (Xmiddle1 + 0.5 * scaleDistance1 * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle1 - 0.5 * scaleDistance1 * self.spacingFactor, Ymiddle1 + 0.5 * scaleDistance1 * self.spacingFactor])

        Xmiddle3 = min_x3 + (max_x3 - min_x3) / 2.0
        Ymiddle3 = min_y3 + (max_y3 - min_y3) / 2.0
        scaleDistance3 = max((max_y3 - min_y3), (max_x3 - min_x3))

        ax3.set_xlim([(Xmiddle3 - 0.5 * scaleDistance3 * self.figureRatio * self.spacingFactor),(Xmiddle3+ 0.5 * scaleDistance3 * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle3 - 0.5 * scaleDistance3 * self.spacingFactor, Ymiddle3 + 0.5 * scaleDistance3 * self.spacingFactor])

        Xmiddle5 = min_x5 + (max_x5 - min_x5) / 2.0
        Ymiddle5 = min_y5 + (max_y5 - min_y5) / 2.0
        scaleDistance5 = max((max_y5 - min_y5), (max_x5 - min_x5))

        ax5.set_xlim([(Xmiddle5 - 0.5 * scaleDistance5 * self.figureRatio * self.spacingFactor),(Xmiddle5 + 0.5 * scaleDistance5 * self.figureRatio * self.spacingFactor)])
        ax5.set_ylim([Ymiddle5 - 0.5 * scaleDistance5 * self.spacingFactor, Ymiddle5 + 0.5 * scaleDistance5 * self.spacingFactor])

        ax2.set_ylim(0,max_cycles+1)
        ax4.set_ylim(0,max_cycles+1)
        ax6.set_ylim(0,max_cycles+1)


        # suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = ' + str(
        #     "{:2.1e}".format(self.accelerationMagnitude)) + ' $, $\\alpha =' + str(
        #     "{:3.1f}".format(self.alpha)) +' $) Sensitivity analysis ', size=self.suptitleSize)

        suptitle = fig.suptitle('$L_{' + str(self.lagrangePointNr) + '}$ ($a_{lt} = 0.1  $, $\\alpha = \\frac{1}{2}\\pi$ rad) -  Influence of number of nodes', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lowDpi:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) \
                        + '_' + str(self.correctionTime) + '_sensitivity_analysis.pdf', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/tlt_corrector/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:7.6f}".format(self.accelerationMagnitude)) + str("{:7.6f}".format(self.alpha)) \
                        + '_' + str(self.correctionTime) + '_sensitivity_analysis.png', transparent=True, dpi=300)

        pass

if __name__ == '__main__':
    lagrange_point_nrs = [1]
    acceleration_magnitudes = [0.1]
    alphas = [90.0]
    orbit_type = 'horizontal'
    amplitudes = [0.008]
    numbers_of_patch_points = [5]
    correction_times = [0.05]
    numbers_of_cycles = [3]
    low_dpi = False

    for lagrange_point_nr in lagrange_point_nrs:
        for acceleration_magnitude in acceleration_magnitudes:
            for alpha in alphas:
                for amplitude in amplitudes:
                    for number_of_patch_points in numbers_of_patch_points:
                        for correction_time in correction_times:
                            for number_of_cycles in numbers_of_cycles:
                                tlt_corrector_validation = TLTCorrectorValidation(lagrange_point_nr, orbit_type, acceleration_magnitude, alpha, \
                                                                              amplitude, number_of_patch_points, correction_time, number_of_cycles, low_dpi)


                                tlt_corrector_validation.plot_tlt_visualization()

                                del tlt_corrector_validation

    # amplitudes = [0.008]
    # numbers_of_patch_points = [5]
    # correction_times = [0.05]
    # number_of_cycles = 4
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for acceleration_magnitude in acceleration_magnitudes:
    #         for alpha in alphas:
    #             for amplitude in amplitudes:
    #                 for number_of_patch_points in numbers_of_patch_points:
    #                     for correction_time in correction_times:
    #                         tlt_corrector_validation = TLTCorrectorValidation(lagrange_point_nr, orbit_type, acceleration_magnitude,
    #                                                                           alpha, amplitude, number_of_patch_points,correction_time, number_of_cycles,low_dpi)
    #
    #                         tlt_corrector_validation.plot_convergence_behaviour()
    #
    #                         del tlt_corrector_validation

    # amplitudes = [0.0001,0.001,0.01]
    # numbers_of_patch_points = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #
    # for lagrange_point_nr in lagrange_point_nrs:
    #     for acceleration_magnitude in acceleration_magnitudes:
    #         for alpha in alphas:
    #             for correction_time in correction_times:
    #                 tlt_corrector_validation = TLTCorrectorValidation(lagrange_point_nr, orbit_type, acceleration_magnitude,
    #                                                                    alpha, amplitudes, numbers_of_patch_points,correction_time, numbers_of_cycles,low_dpi)
    #
    #                 tlt_corrector_validation.plot_sensitivity_analysis()
    #
    #                 del tlt_corrector_validation