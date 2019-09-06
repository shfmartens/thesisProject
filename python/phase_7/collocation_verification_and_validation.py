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
        self.patchSize = 9
        self.widthBar = 0.10

        # label properties
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
        fig = plt.figure(figsize=self.figSize)
        gs = matplotlib.gridspec.GridSpec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :3])

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.grid(True, which='both', ls=':')
        ax1.set_title('Input guess')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')
        ax2.set_title('Disturbed guess')


        ax3.set_xlabel('x [-]')
        ax3.set_ylabel('y [-]')
        ax3.grid(True, which='both', ls=':')
        ax3.set_title('Converged guess')


        ax4.set_xlabel('Collocation procedure stage [-]')
        ax4.set_ylabel('Deviation Norms [-]')
        ax4.grid(True, which='both', ls=':')
        ax4.set_title('Deviations per stage')

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
            orbit_df = load_orbit_augmented('../../data/raw/collocation/' + str(i) + '_stateHistory.txt')
            patch_points_df = load_patch_points('../../data/raw/collocation/' + str(i) + '_stateVectors.txt',6)
            deviation = np.loadtxt('../../data/raw/collocation/' + str(i) + '_deviations.txt')
            propagated_states_df = load_propagated_states('../../data/raw/collocation/' + str(i) + '_propagatedStates.txt',6-1)

            convergence_properties_list.append([i, deviation[0], deviation[1], deviation[2], deviation[3], deviation[4], deviation[5], deviation[6]])

            if i == 0:
                ax1.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                #ax1.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax1.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        test1 = ax1.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'outgoing' + '}$')

                for row in propagated_states_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        state = row[1].values
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        test2 = ax1.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda6'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'incoming' + '}$')

            if i == 1:
                ax2.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                #ax2.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax2.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        test1 = ax2.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'outgoing' + '}$')

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
                                                             color=self.plottingColors['lambda6'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'incoming' + '}$')

            if i == 2:
                ax3.scatter(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],marker=self.currentPatchStyle,s=0.5, label='Initial Guess')
                #ax3.plot(orbit_df['x'], orbit_df['y'], color=self.plottingColors['singleLine'],linewidth=self.lineWidth, label='Initial Guess')
                ax3.scatter(patch_points_df['x'], patch_points_df['y'], color='black', marker=self.currentPatchStyle,s=self.patchSize, label='Current patch points')

                for row in patch_points_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        test1 = ax3.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda3'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'outgoing' + '}$')

                for row in propagated_states_df.iterrows():
                    if row[0] > 0:
                        state = row[1].values
                        x_base = state[0]
                        y_base = state[1]
                        state = row[1].values
                        x_end = x_base + state[3] / self.magnitudeFactor
                        y_end = y_base + state[4] / self.magnitudeFactor

                        test2 = ax3.annotate("", xy=(x_base, y_base), xytext=(x_end, y_end),
                                             arrowprops=dict(arrowstyle='<-, head_width=1e-1, head_length=2e-1',
                                                             color=self.plottingColors['lambda6'], shrinkA=0,
                                                             shrinkB=0), label='$V_{' + 'incoming' + '}$')

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

        print(deviation_df['deltaVint'])

        Xmiddle = min_x + (max_x - min_x) / 2.0
        Ymiddle = min_y + (max_y - min_y) / 2.0
        scaleDistance = max((max_y - min_y), (max_x - min_x))

        ax1.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax1.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax2.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax2.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ax3.set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        ax3.set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])

        ind = np.arange(len(deviation_df['stage']))
        ax4.bar(ind - 5*self.widthBar/2, deviation_df['deltaR'], self.widthBar, label='$|\Delta R|$',color=self.plottingColors['lambda6'])
        ax4.bar(ind - 3*self.widthBar/2, deviation_df['deltaV'], self.widthBar, label='$|\Delta V|$',color=self.plottingColors['lambda3'])
        ax4.bar(ind - 1* self.widthBar/2, deviation_df['deltaRint'], self.widthBar, label='$|\Delta R_{int}|$',color=self.plottingColors['lambda2'])
        ax4.bar(ind + 1 * self.widthBar/2, deviation_df['deltaVint'], self.widthBar, label='$|\Delta V_{int}|$',color=self.plottingColors['lambda4'])
        ax4.bar(ind + 3 * self.widthBar/2, deviation_df['deltaRext'], self.widthBar, label='$|\Delta R_{ext}|$',color=self.plottingColors['lambda1'])
        ax4.bar(ind + 5 * self.widthBar/2, deviation_df['deltaVext'], self.widthBar, label='$|\Delta V_{ext}|$',color=self.plottingColors['lambda5'])
        ax4.set_yscale('log')
        ax4.set_ylim([1e-16, 1e-2])

        lgd1 = ax4.legend(frameon=True, loc='upper left', prop={'size': 9})



        plt.suptitle('Collocation procedure overview', size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)


        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/collocation_procedure.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/collocation_procedure.pdf', transparent=True)




if __name__ == '__main__':
    low_dpi = True
    display_collocation_procedure = DisplayCollocationProcedure(low_dpi)
    display_collocation_procedure.plot_collocation_procedure()
    del display_collocation_procedure
