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