import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
from scipy.interpolate import interp1d
sns.set_style("whitegrid")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/python/util')
from load_data import load_orbit, load_equilibria, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, cr3bp_velocity

class DisplayEquilibriaValidation:
    def __init__(self, lagrange_point_nr, acceleration_magnitude, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        print('========================')
        print('Equilibria around L' + str(lagrange_point_nr) + ' For an acceleration magnitude of ' + str(
            acceleration_magnitude))
        print('========================')

        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))

        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (
                MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.equilibriaDf = load_equilibria('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L'+ str(lagrange_point_nr) + '_' + str(format(acceleration_magnitude, '.6f')) + '_equilibria.txt')

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.suptitleSize = 20
        pass

    def plot_equilibria(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()
        c = list(self.equilibriaDf['alpha'].values)

        if self.lagrangePointNr == 1:
            equilibriaDf_second = load_equilibria(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L2_' + str(format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')
        else:
            equilibriaDf_second = load_equilibria(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L1_' + str(
                    format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')

        sc = ax.scatter(self.equilibriaDf['x'], self.equilibriaDf['y'], c=c, cmap='viridis', s=20)
        ax.scatter(equilibriaDf_second['x'], equilibriaDf_second['y'], c=c, cmap='viridis', s=20)
        cb = plt.colorbar(sc)
        cb.set_label('$\\alpha \enskip [rad]$')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1','L2']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                       color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.set_xlim([0.8,1.2])
        ax.set_ylim([-0.05,0.05])
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.title('$L_{1,2}$ ' + 'artificial equilibria at $a_{lt} =$ ' + str(
            format(self.accelerationMagnitude, '.1e')), size=self.suptitleSize)
        if self.lowDPI:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_total.png',transparent=True,dpi=self.dpi)
        else:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_total.png',transparent=True)

        pass

    def plot_equilibria_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()
        c = list(self.equilibriaDf['alpha'].values)


        sc = ax.scatter(self.equilibriaDf['x'], self.equilibriaDf['y'], c=c, cmap='viridis', s=20)
        cb = plt.colorbar(sc)
        cb.set_label('$\\alpha \enskip [rad]$')


        lagrange_points_df = load_lagrange_points_location()
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        else:
            lagrange_point_nrs = ['L2']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                       color='black', marker='x')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, y, z, colors='black', label='Moon')

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        if self.lagrangePointNr == 2:
            ax.set_xlim([1.0, 1.2])
            ax.set_ylim([-0.05, 0.05])
        else:
            ax.set_xlim([0.8, 1.0])
            ax.set_ylim([-0.05, 0.05])
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.title('$L_' + str(self.lagrangePointNr)+ '$ artificial equilibria at $a_{lt} =$ ' + str(format(self.accelerationMagnitude, '.1e')), size=self.suptitleSize)
        if self.lowDPI:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L'+ str(self.lagrangePointNr) + '_' +str(self.accelerationMagnitude) +'_equilibria_zoom.png', transparent=True,
                        dpi=self.dpi)
        else:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L'+ str(self.lagrangePointNr) + '_' +str(self.accelerationMagnitude) +'_equilibria_zoom.png', transparent=True)
        pass

if __name__ == '__main__':
    low_dpi = False
    lagrange_point_nrs = [1, 2]
    acceleration_magnitudes = [0.001000, 0.010000, 0.100000]

    for lagrange_point_nr in lagrange_point_nrs:
        for acceleration_magnitude in acceleration_magnitudes:
            display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nr, acceleration_magnitude, low_dpi=low_dpi)
            display_equilibria_validation.plot_equilibria()
            display_equilibria_validation.plot_equilibria_zoom()

            del display_equilibria_validation
