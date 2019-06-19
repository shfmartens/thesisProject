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
    load_states_continuation

class DisplayPeriodicSolutions:
    def __init__(self,orbit_type, lagrange_point_nr,  acceleration_magnitude, alpha, beta, low_dpi):

        self.orbitType = orbit_type
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.beta = beta

        self.lowDPI = low_dpi

        print('=======================')
        print('L' + str(self.lagrangePointNr) + '_' + self.orbitType + ' (acc = ' + str(self.accelerationMagnitude) \
                + ', alpha = ' + str(self.alpha), 'beta = ' + str(self.beta) + ')' )
        print('=======================')

        self.orbitTypeForTitle = orbit_type.capitalize()
        if self.orbitTypeForTitle == 'Horizontal' or self.orbitTypeForTitle == 'Vertical':
            self.orbitTypeForTitle += ' Lyapunov'

        # Compute settings for obital families plots
        self.orbitID = []
        self.Hlt = []

        self.spacingFactor = 25

        # Plot lay-out settings
        self.plotAlpha = 1
        self.lineWidth = 0.5

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.spacingPlotFactor = 1.05
        self.figureRatio =  (7 * (1 + np.sqrt(5)) / 2) / 7

        generic_filepath = '../../data/raw/orbits/augmented/varying_hamiltonian'
        states_continuation_filepath = str(generic_filepath + '/L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
                                          + '_' +  str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
                                           str("{:14.13f}".format(self.alpha)) + '_' + \
                                           str("{:14.13f}".format(self.beta)) + '_states_continuation.txt' )

        differential_corrections_filepath = str(
            generic_filepath + '/L' + str(self.lagrangePointNr) + '_' + str(self.orbitType) \
            + '_' + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' + \
            str("{:14.13f}".format(self.alpha)) + '_' + \
            str("{:14.13f}".format(self.beta)) + '_differential_correction.txt')


        statesContinuation_df = load_states_continuation(states_continuation_filepath)
        differentialCorrections_df = load_differential_correction(differential_corrections_filepath)


        self.Hlt = []
        self.T = []
        self.orbitsId = []
        self.numberOfIterations = []

        for row in statesContinuation_df.iterrows():
            self.orbitsId.append(row[0])
            self.Hlt.append(row[1][1])

        for row in differentialCorrections_df.iterrows():
            self.T.append(row[1][2])
            self.numberOfIterations.append(row[1][0])

        # Determine heatmap for level of C
        self.numberOfPlotColorIndices = len(self.Hlt)
        self.plotColorIndexBasedOnHlt = []
        for hamiltonian in self.Hlt:
            self.plotColorIndexBasedOnHlt.append(int(np.round(
                (hamiltonian - min(self.Hlt)) / (max(self.Hlt) - min(self.Hlt)) * (self.numberOfPlotColorIndices - 1))))


        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        self.dpi = 150

        self.suptitleSize = 20

    def plot_families(self):
        hlt_normalized = [(value - min(self.Hlt)) / (max(self.Hlt) - min(self.Hlt)) for value in self.Hlt]



        colors = matplotlib.colors.ListedColormap(sns.color_palette("viridis_r"))(hlt_normalized)


        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis_r", len(self.Hlt))),
                                   norm=plt.Normalize(vmin=min(self.Hlt), vmax=max(self.Hlt)), )

        if self.orbitType == 'horizontal':
           fig = plt.figure(figsize=(self.figSize[0], self.figSize[1]/2))
           ax2 = fig.add_subplot(1, 2, 1, projection='3d')
           ax5 = fig.add_subplot(1, 2, 2)

        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax5.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))


        ax2.contourf(x, y, z, colors='black')
        ax5.contourf(x, y, z, colors='black')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.set_zlabel('z [-]')

        ax2.grid(True, which='both', ls=':')
        ax2.view_init(30, -120)

        ax5.set_xlabel('x [-]')
        ax5.set_ylabel('y [-]')

        scaleDistance = 0.13
        figureRatio = self.figSize[0]/ (self.figSize[1]/2) *0.5



        self.spacingFactor
        orbitIdsPlot = list(range(0, len(self.Hlt), self.spacingFactor))
        if orbitIdsPlot != len(self.Hlt):
            orbitIdsPlot.append(len(self.Hlt)-1)

        for i in orbitIdsPlot:
            plot_color = colors[self.plotColorIndexBasedOnHlt[i]]


            df = load_orbit('../../data/raw/orbits/augmented/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                            + str("{:14.13f}".format(self.accelerationMagnitude)) + '_' \
                            + str("{:14.13f}".format(self.alpha)) + '_' \
                            + str("{:14.13f}".format(self.beta)) + '_'  \
                            + str("{:14.13f}".format(self.Hlt[i])) + '_.txt')
            ax2.plot(df['x'], df['y'], df['z'], color=plot_color, alpha=self.plotAlpha, linewidth=self.lineWidth)
            ax5.plot(df['x'], df['y'], color=plot_color, alpha=self.plotAlpha, linewidth=self.lineWidth)


        ax2.set_xlim([0.8 - scaleDistance*figureRatio, 0.8 + scaleDistance*figureRatio])
        ax2.set_ylim([-scaleDistance, +scaleDistance])
        ax2.set_zlim([-scaleDistance, +scaleDistance])

        ax2.grid(True, which='both', ls=':')
        ax2.view_init(30, -120)



        ax5.set_xlim([0.9 - scaleDistance*figureRatio  , 0.9 + scaleDistance*figureRatio])
        ax5.set_ylim([-scaleDistance, + scaleDistance])

        fig.tight_layout()
        if self.orbitType == 'horizontal':
            fig.subplots_adjust(top=0.8, left=0.05)
        else:
            fig.subplots_adjust(top=0.8, left=0.05)

        sm.set_array([])

        cax, kw = matplotlib.colorbar.make_axes([ax2, ax5])

        ax2.set_aspect(1.0)
        ax5.set_aspect(1.0)




        cbar = plt.colorbar(sm, cax=cax, label='$H_{lt}$ [-]', **kw)

        print( 'ax5 data ratio: ' + str(ax5.get_data_ratio()))
        print( 'ax2 data ratio: ' + str(ax2.get_data_ratio()))

        print('ax5 aspect: ' + str(ax5.get_aspect()))
        print('ax2 aspect: ' + str(ax2.get_aspect()))

        print( 'figure ratio: ' + str(figureRatio))
        print( '1/figure ratio: ' + str(1/figureRatio))




        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ' + self.orbitTypeForTitle + ' ( $a_{lt} = '+ str("{:2.1e}".format(self.accelerationMagnitude)) + \
                     '$,$\\alpha = '+ str(self.alpha) + '$, $\\beta = '+ str(self.beta) + '$ ) ' + ' - Orthographic projection',
                     size=self.suptitleSize)

        if self.lowDPI:
            fig.savefig('../../data/figures/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_family_subplots.png', transparent=True, dpi=self.dpi, bbox_inches='tight')

        else:
            fig.savefig('../../data/figures/orbits/L' + str(self.lagrangePointNr) + '_' + self.orbitType + '_' \
                    + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' + str("{:7.6f}".format(self.alpha)) + \
                        '_offset_effect.pdf', transparent=True)
        plt.close()
        pass



        pass


if __name__ == '__main__':
    orbit_types = ['horizontal']
    lagrange_points = [1]
    acceleration_magnitudes = [0.001]
    alphas = [120.0]
    betas = [0.0]


    low_dpi = True

    for orbit_type in orbit_types:
        for lagrange_point in lagrange_points:
            for acceleration_magnitude in acceleration_magnitudes:
                for alpha in alphas:
                    for beta in betas:
                        display_periodic_solutions = DisplayPeriodicSolutions(orbit_type, lagrange_point, acceleration_magnitude, \
                                        alpha, beta, low_dpi)
                        display_periodic_solutions.plot_families()
            del display_periodic_solutions