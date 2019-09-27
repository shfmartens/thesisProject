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

class DisplayMeshError:
    def __init__(self, amplitude, maximum_offset, number_of_patch_points, low_dpi):

        self.amplitude = amplitude
        self.maxOffset = maximum_offset
        self.patchPoints = number_of_patch_points
        self.lowDpi = low_dpi

        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1
        self.magnitudeFactor = 10

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        self.directoryName = '../../data/raw/collocation/mesh_effect/'

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



    def plot_error_distribution(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax0 = fig.add_subplot(1, 3, 1)
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)


        fileNameShooting = self.directoryName +str("{:7.6f}".format(self.amplitude)) + '_' + str(self.patchPoints) + \
                   '_' + str(self.maxOffset) + '_' + 'shootingDeviations.txt'
        fileNameCollocation = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.patchPoints) + \
                           '_' + str(self.maxOffset) + '_' + 'collocationDeviations.txt'
        fileNameCollocationError = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.patchPoints) + \
                              '_' + str(self.maxOffset) + '_' + 'collocationErrors.txt'

        shootingDeviations = np.loadtxt(fileNameShooting)
        collocationDeviations = np.loadtxt(fileNameCollocation)
        collocationErrors  = np.loadtxt(fileNameCollocationError)

        shootinglist = []
        collocationlist = []
        collocationErrorlist = []
        for i in range(self.patchPoints-1):
            shootinglist.append(np.sqrt(shootingDeviations[2*i] ** 2 + shootingDeviations[2*i+1] ** 2 ) )
            collocationlist.append(np.sqrt(collocationDeviations[2*i] ** 2 + collocationDeviations[2*i+1] ** 2) )
            collocationErrorlist.append(collocationErrors[i])

        segmentVector = np.arange(1,self.patchPoints,step=1)
        print(len(segmentVector))
        print(len(shootinglist))


        ax0.bar(segmentVector, shootinglist,color=self.plottingColors['singleLine'])
        ax0.set_yscale('log')
        ax0.set_ylim([1e-16, 1e-10])
        ax0.set_xlim([0,self.patchPoints])

        ax0.set_xlabel('segment [-]')
        ax0.set_ylabel('$|\\Delta F|$ [-]')
        ax0.grid(True, which='both', ls=':')
        ax0.set_title('Multiple shooting deviation norm per segment')

        ax1.bar(segmentVector, collocationlist,color=self.plottingColors['singleLine'])
        ax1.set_yscale('log')
        ax1.set_ylim([1e-16, 1e-10])
        ax1.set_xlim([0,self.patchPoints])

        ax1.set_xlabel('segment [-]')
        ax1.set_ylabel('$|\\Delta F|$ [-]')
        ax1.grid(True, which='both', ls=':')
        ax1.set_title('collocation deviation norm per segment')

        ax2.bar(segmentVector, collocationErrorlist, color=self.plottingColors['singleLine'])
        ax2.set_yscale('log')
        ax2.set_ylim([1e-16, 1e-5])
        ax2.set_xlim([0, self.patchPoints])

        ax2.set_xlabel('segment [-]')
        ax2.set_ylabel('$e_{i}$ [-]')
        ax2.grid(True, which='both', ls=':')
        ax2.set_title('collocation error  per segment')

        plt.suptitle('Error distribution ( $A$ =' + str("{:2.1e}".format(self.amplitude)) +', points = '+str(self.patchPoints) + \
                     ') max offset = ' + str(self.maxOffset), size=self.suptitleSize)

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)



        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/'+ str("{:7.6f}".format(self.amplitude)) + '_' +str(number_of_patch_points)\
                        + '_' + str(self.maxOffset) + '_error_distribution.png', transparent=True, dpi=self.dpi,
                        bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/' + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                number_of_patch_points) \
                        + '_' + str(self.maxOffset) + '_error_distribution.png', transparent=True)


        pass


    def plot_offset_effect(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax = fig.gca()

        shootinglist = []
        collocationlist = []
        fullPeriodList =[]

        widthBar = 0.2

        for i in range(len(self.maxOffset)):
            fileNameShooting = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.patchPoints) + \
                               '_' + str(self.maxOffset[i]) + '_' + 'shootingDeviations.txt'
            fileNameCollocation = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.patchPoints) + \
                                  '_' + str(self.maxOffset[i]) + '_' + 'collocationDeviations.txt'
            fileNamePeriod = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.patchPoints) + \
                             '_' + str(self.maxOffset[i]) + '_' + 'fullPeriodDeviations.txt'

            shootingDeviations = np.loadtxt(fileNameShooting)
            collocationDeviations = np.loadtxt(fileNameCollocation)
            periodDeviations = np.loadtxt(fileNamePeriod)

            shootinglist.append(np.linalg.norm(shootingDeviations))
            collocationlist.append(np.linalg.norm(collocationDeviations))
            fullPeriodList.append(np.linalg.norm(periodDeviations))

        segmentVector = segmentVector = np.arange(1,len(self.maxOffset)+1,step=1)

        xticks = (segmentVector)
        xtickslabels = ('0','-8','-5','-3')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1f'))
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xtickslabels)
        ax.bar(segmentVector-1.5*widthBar, shootinglist, widthBar , color=self.plottingColors['tripleLine'][0],label='shooting')
        ax.bar(segmentVector, collocationlist, widthBar , color=self.plottingColors['tripleLine'][1],label='collocation')
        ax.bar(segmentVector+1.5*widthBar, fullPeriodList, widthBar , color=self.plottingColors['tripleLine'][2],label='RK78')


        ax.set_yscale('log')
        ax.set_ylim([1e-16, 1e-8])
        ax.set_xlim([0, self.patchPoints])

        ax.set_xlabel('maxmimum offset noise vector [-]')
        ax.set_ylabel('$|\\Delta F|$ [-]')
        ax.grid(True, which='both', ls=':')
        ax.set_title('total deviation per method ( $A$ =' + str("{:2.1e}".format(self.amplitude)) +', points = '+str(self.patchPoints) + \
                     ')')

        ax.legend(frameon=True, loc='upper right', prop={'size': 9})



        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/' + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                number_of_patch_points) \
                     + '_offset_effect.png', transparent=True, dpi=self.dpi,
                        bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/' + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                number_of_patch_points) \
                        + '_offset_effect.png', transparent=True)

        pass

    def plot_patchpoint_effect(self):

        fig = plt.figure(figsize=self.figSizeWide)
        ax = fig.gca()

        shootinglist = []
        collocationlist = []
        collocationErrorlist = []
        fullPeriodList = []

        widthBar = 0.15

        for i in range(len(self.patchPoints)):
            fileNameShooting = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.patchPoints[i]) + \
                               '_' + str(self.maxOffset) + '_' + 'shootingDeviations.txt'
            fileNameCollocation = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.patchPoints[i]) + \
                                  '_' + str(self.maxOffset) + '_' + 'collocationDeviations.txt'

            fileNameCollocationErrors = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.patchPoints[i]) + \
                                  '_' + str(self.maxOffset) + '_' + 'collocationErrors.txt'

            fileNamePeriod = self.directoryName + str("{:7.6f}".format(self.amplitude)) + '_' + str(self.patchPoints[i]) + \
                             '_' + str(self.maxOffset) + '_' + 'fullPeriodDeviations.txt'

            shootingDeviations = np.loadtxt(fileNameShooting)
            collocationDeviations = np.loadtxt(fileNameCollocation)
            collocationErrors = np.loadtxt(fileNameCollocationErrors)
            periodDeviations = np.loadtxt(fileNamePeriod)

            shootinglist.append(np.linalg.norm(shootingDeviations))
            collocationErrorlist.append(np.linalg.norm(collocationErrors))
            collocationlist.append(np.linalg.norm(collocationDeviations))
            fullPeriodList.append(np.linalg.norm(periodDeviations))

        segmentVector = segmentVector = np.arange(1,len(self.patchPoints)+1,step=1)

        xticks = (segmentVector)
        xtickslabels = ('5','10','15')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1f'))
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xtickslabels)
        ax.bar(segmentVector - 1.5 * widthBar, shootinglist, widthBar, color=self.plottingColors['tripleLine'][0],
               label='shooting')
        ax.bar(segmentVector - 0.5 * widthBar, collocationErrorlist, widthBar, color='black',
               label='collocationError')
        ax.bar(segmentVector+0.5 * widthBar, collocationlist, widthBar, color=self.plottingColors['tripleLine'][1],
               label='collocation')
        ax.bar(segmentVector + 1.5 * widthBar, fullPeriodList, widthBar, color=self.plottingColors['tripleLine'][2],
               label='RK78')

        ax.set_yscale('log')
        ax.set_ylim([1e-16, 1e-4])
        ax.set_xlim([0, max(segmentVector)+1])

        ax.set_xlabel('number of patch points [-]')
        ax.set_ylabel('$|\\Delta F|$/error [-]')
        ax.grid(True, which='both', ls=':')
        ax.set_title('total deviation/error per method ( $A$ =' + str("{:2.1e}".format(self.amplitude)) + ', offset = ' + str(
            self.maxOffset) + ')' )

        ax.legend(frameon=True, loc='upper right', prop={'size': 9})

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)


        if self.lowDpi:
            fig.savefig('../../data/figures/collocation/' + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.maxOffset) \
                     + '_patchpoints_effect.png', transparent=True, dpi=self.dpi,
                        bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/collocation/' + str("{:7.6f}".format(self.amplitude)) + '_' + str(
                self.maxOffset) \
                        + '_patchpoints_effect.png', transparent=True)

        pass


    pass


if __name__ == '__main__':
    low_dpi = True
    amplitudes = [1.0e-4,1.0e-3,5.0e-3,9.0e-3]
    maximum_offsets = [0]
    numbers_of_patch_points = [5,10,15]

    for amplitude in amplitudes:
        for maximum_offset in maximum_offsets:
            for number_of_patch_points in numbers_of_patch_points:
                display_mesh_error = DisplayMeshError(amplitude, maximum_offset, number_of_patch_points,low_dpi)
                display_mesh_error.plot_error_distribution()
                del display_mesh_error

    # for amplitude in amplitudes:
    #     for number_of_patch_points in numbers_of_patch_points:
    #         display_mesh_error = DisplayMeshError(amplitude, maximum_offsets, number_of_patch_points, low_dpi)
    #         display_mesh_error.plot_offset_effect()
    #         del display_mesh_error


    for amplitude in amplitudes:
        for maximum_offset in maximum_offsets:
            display_mesh_error = DisplayMeshError(amplitude, maximum_offset, numbers_of_patch_points, low_dpi)
            display_mesh_error.plot_patchpoint_effect()
            del display_mesh_error









