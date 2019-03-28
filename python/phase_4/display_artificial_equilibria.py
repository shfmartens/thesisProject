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
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, cr3bp_velocity, \
    compute_equilibrium_deviation, load_equilibria_stability

class DisplayEquilibriaValidation:
    def __init__(self, lagrange_point_nr, acceleration_magnitude, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        print('========================')
        print('Equilibria around L' + str(lagrange_point_nr) + ' for an acceleration magnitude of ' + str(
            acceleration_magnitude))
        print('========================')

        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))

        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (
                MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        # Compute Equilibria properties and verify offset
        self.equilibriaDf = load_equilibria('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L'+ str(lagrange_point_nr) + '_' + str(format(acceleration_magnitude, '.6f')) + '_equilibria.txt')
        self.numberOfAnglesPerAcceleration = len(set(self.equilibriaDf.index.get_level_values(0)))
        self.PosDeviationEquilibrium = []
        self.AngleArray = []
        self.NumberOfIterations = []
        self.StabilityType = []
        self.maxEigenvalueDeviation = 1.0e-3

        for row in self.equilibriaDf.iterrows():
            equilbriumPosition = row[1].values
            deviation_from_equilibrium = compute_equilibrium_deviation(equilbriumPosition[0],equilbriumPosition[1],equilbriumPosition[2],self.accelerationMagnitude)
            self.PosDeviationEquilibrium.append(deviation_from_equilibrium)
            self.AngleArray.append(equilbriumPosition[0])
            self.NumberOfIterations.append(equilbriumPosition[3])

        # Compute Equilibria stability and set-up Vectors containing the equilibria with specific stability
        self.maxEigenvalueDeviation = 1.0e-3
        self.equilibriaStabilityDf = load_equilibria_stability('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L'+ str(lagrange_point_nr) + '_' + str(format(acceleration_magnitude, '.6f')) + '_equilibria_stability.txt')
        self.equilibriaSaddleSaddle = pd.DataFrame(columns=['x', 'y','iterations'])
        self.equilibriaSaddleCenter = pd.DataFrame(columns=['x', 'y', 'iterations'])
        self.equilibriaCenterCenter = pd.DataFrame(columns=['x', 'y', 'iterations'])
        self.equilibriaMixedMixed = pd.DataFrame(columns=['x', 'y', 'iterations'])
        for row in self.equilibriaStabilityDf.iterrows():
            index = row[0]
            equilibriumRow = self.equilibriaDf.iloc[index]
            Mtranspose = np.matrix([list(row[1][1:7]), list(row[1][7:13]), list(row[1][13:19]), list(row[1][19:25]), list(row[1][25:31]), list(row[1][31:37])])
            M = Mtranspose.T
            eigenvalue = np.linalg.eigvals(M)

            countSaddle = 0
            countMixed = 0
            countCenter = 0
            countZero = 0

            # Find stability type by iterating over eigenvalue Vector
            for idx, l in enumerate(eigenvalue):
                if  abs(l.real) > self.maxEigenvalueDeviation and abs(l.imag) < self.maxEigenvalueDeviation:
                    countSaddle = countSaddle + 1
                elif  abs(l.real) < self.maxEigenvalueDeviation and abs(l.imag) > self.maxEigenvalueDeviation:
                    countCenter = countCenter + 1
                elif  abs(l.real) > self.maxEigenvalueDeviation and abs(l.imag) > self.maxEigenvalueDeviation:
                    countMixed = countMixed + 1
                else:
                    countZero = countZero + 1

            if countSaddle == 4:
                self.equilibriaSaddleSaddle = self.equilibriaSaddleSaddle.append(equilibriumRow, ignore_index=True)
            elif countSaddle == 2 and countCenter == 2:
                self.equilibriaSaddleCenter = self.equilibriaSaddleCenter = self.equilibriaSaddleCenter.append(equilibriumRow, ignore_index=True)
            elif countMixed == 4:
                self.equilibriaMixedMixed = self.equilibriaMixedMixed.append(equilibriumRow, ignore_index=True)
            elif countCenter == 4:
                self.equilibriaCenterCenter = self.equilibriaCenterCenter.append(equilibriumRow, ignore_index=True)


        # Plotting lay-out properties
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        n_colors = 3
        n_colors_l = 6
        n_colors_s = 4
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'SxS': sns.color_palette("viridis", n_colors_s)[0],
                               'SxC': sns.color_palette("viridis", n_colors_s)[1],
                               'CxC': sns.color_palette("viridis", n_colors_s)[2],
                               'MxM': sns.color_palette("viridis", n_colors_s)[3],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}
        self.line_width = 1
        self.suptitleSize = 20
        pass

    def plot_equilibria(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()
        c  = list(self.equilibriaDf['alpha'].values)

        if self.lagrangePointNr == 1:
            equilibriaDf_second = load_equilibria(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L2_' + str(format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')
        else:
            equilibriaDf_second = load_equilibria(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L1_' + str(
                    format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')

        sc = ax.scatter(self.equilibriaDf['x'], self.equilibriaDf['y'], c=c, cmap='viridis', s=20)
        ax.scatter(equilibriaDf_second['x'], equilibriaDf_second['y'], c=c, cmap='viridis', s=20)
        cb = plt.colorbar(sc, ticks = [0, np.pi, 2*np.pi])
        cb.set_ticklabels(['0','$\pi$','$2\pi$'])
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
        ax.set_ylim([-0.2,0.2])
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
            ax.set_ylim([-0.1, 0.1])
        else:
            ax.set_xlim([0.8, 1.0])
            ax.set_ylim([-0.1, 0.1])
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

    def plot_equilibria_validation(self):
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)

        ax0.semilogy(self.AngleArray,self.PosDeviationEquilibrium,linewidth=self.line_width, c=self.plottingColors['singleLine'], label='$|\sqrt{\Delta x^2 + \Delta y^2}|$')
        ax0.set_xlabel('$\\alpha$ [rad]')
        ax0.set_ylabel('$\\Delta r$ [-]')
        ax0.set_xlim([0.0, 2*np.pi])
        ax0.set_ylim([10e-19, 10e-14])
        ax0.legend(frameon=True, loc = 'upper right')
        ax0.grid(True, which='both', ls=':')
        ax0.set_title('Position deviation from equilibrium')

        ax1.plot(self.AngleArray,self.NumberOfIterations,linewidth=3, c=self.plottingColors['singleLine'], label='Number of iterations')
        ax1.set_xlabel('$\\alpha$ [rad]')
        ax1.set_ylabel('Number of iterations [-]')
        ax1.set_xlim([0.0, 2 * np.pi])
        ax1.set_ylim([0, max(self.NumberOfIterations)+10])
        ax1.set_ylim(bottom=0)
        ax1.legend(frameon=True, loc='upper right')
        ax1.grid(True, which='both', ls=':')
        ax1.set_title('Position deviation from equilibrium')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        fig.suptitle('Equilibrium verification at $a_{lt} = $' + str(
            format(self.accelerationMagnitude, '.1e')), size=self.suptitleSize)

        if self.lowDPI:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L'+ str(self.lagrangePointNr) + '_' +str(self.accelerationMagnitude) +'_equilibria_validation.png', transparent=True,
                        dpi=self.dpi)
        else:
            fig.savefig('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L'+ str(self.lagrangePointNr) + '_' +str(self.accelerationMagnitude) +'_equilibria_validation.png', transparent=True)
        pass

    def plot_equilibria_stability(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        if self.lagrangePointNr == 1:
            equilibriaDf_second = load_equilibria(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L2_' + str(format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')
            equilibriaStabilityDf_second = load_equilibria_stability('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L2_' + str(format(acceleration_magnitude, '.6f')) + '_equilibria_stability.txt')


        else:
            equilibriaDf_second = load_equilibria('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L1_' + str(format(self.accelerationMagnitude, '.6f')) + '_equilibria.txt')
            equilibriaStabilityDf_second = load_equilibria_stability('/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/raw/equilibria/L1_' + str(format(acceleration_magnitude, '.6f')) + '_equilibria_stability.txt')

        equilibriaSaddleSaddle_second = pd.DataFrame(columns=['x', 'y', 'iterations'])
        equilibriaSaddleCenter_second = pd.DataFrame(columns=['x', 'y', 'iterations'])
        equilibriaCenterCenter_second = pd.DataFrame(columns=['x', 'y', 'iterations'])
        equilibriaMixedMixed_second = pd.DataFrame(columns=['x', 'y', 'iterations'])

        for row in equilibriaStabilityDf_second.iterrows():
            index = row[0]
            equilibriumRow = equilibriaDf_second.iloc[index]
            Mtranspose = np.matrix([list(row[1][1:7]), list(row[1][7:13]), list(row[1][13:19]), list(row[1][19:25]), list(row[1][25:31]), list(row[1][31:37])])
            M = Mtranspose.T
            eigenvalue = np.linalg.eigvals(M)

            countSaddle = 0
            countMixed = 0
            countCenter = 0
            countZero = 0

            # Find stability type by iterating over eigenvalue Vector
            for idx, l in enumerate(eigenvalue):
                if  abs(l.real) > self.maxEigenvalueDeviation and abs(l.imag) < self.maxEigenvalueDeviation:
                    countSaddle = countSaddle + 1
                elif  abs(l.real) < self.maxEigenvalueDeviation and abs(l.imag) > self.maxEigenvalueDeviation:
                    countCenter = countCenter + 1
                elif  abs(l.real) > self.maxEigenvalueDeviation and abs(l.imag) > self.maxEigenvalueDeviation:
                    countMixed = countMixed + 1
                else:
                    countZero = countZero + 1

            if countSaddle == 4:
                equilibriaSaddleSaddle_second = equilibriaSaddleSaddle_second.append(equilibriumRow, ignore_index=True)
            elif countSaddle == 2 and countCenter == 2:
                equilibriaSaddleCenter_second = equilibriaSaddleCenter_second.append(equilibriumRow, ignore_index=True)
            elif countMixed == 4:
                equilibriaMixedMixed_second = equilibriaMixedMixed_second.append(equilibriumRow, ignore_index=True)
            elif countCenter == 4:
                equilibriaCenterCenter_second = equilibriaCenterCenter_second.append(equilibriumRow, ignore_index=True)

        ax.scatter(self.equilibriaSaddleCenter['x'], self.equilibriaSaddleCenter['y'], color=self.plottingColors['SxC'],label='$S$ $X$ $C$')
        ax.scatter(self.equilibriaSaddleSaddle['x'], self.equilibriaSaddleSaddle['y'], color=self.plottingColors['SxS'],label='$S$  $X$ $S$')
        ax.scatter(self.equilibriaMixedMixed['x'], self.equilibriaMixedMixed['y'], color=self.plottingColors['MxM'],label='$M$ $X$ $M$')
        ax.scatter(self.equilibriaCenterCenter['x'], self.equilibriaCenterCenter['y'], color=self.plottingColors['CxC'],label='$C$ $X$ $C$')
        ax.legend(frameon=True, loc='upper right')
        ax.grid(True, which='both', ls=':')
        ax.scatter(equilibriaSaddleCenter_second['x'], equilibriaSaddleCenter_second['y'], color=self.plottingColors['SxC'],label='$S$ $X$ $C$')
        ax.scatter(equilibriaSaddleSaddle_second['x'], equilibriaSaddleSaddle_second['y'], color=self.plottingColors['SxS'],label='$S$  $X$ $S$')
        ax.scatter(equilibriaMixedMixed_second['x'], equilibriaMixedMixed_second['y'], color=self.plottingColors['MxM'],label='$M$ $X$ $M$')
        ax.scatter(equilibriaCenterCenter_second['x'], equilibriaCenterCenter_second['y'], color=self.plottingColors['CxC'],label='$C$ $X$ $C$')





        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
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
        ax.set_xlim([0.8, 1.2])
        ax.set_ylim([-0.2, 0.2])
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.title('$L_{1,2}$ ' + 'artificial equilibria at $a_{lt} =$ ' + str(
            format(self.accelerationMagnitude, '.1e')), size=self.suptitleSize)
        if self.lowDPI:
            fig.savefig(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(
                    self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_stability.png',
                transparent=True,
                dpi=self.dpi)
        else:
            fig.savefig(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(
                    self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_stability.png',
                transparent=True)

    def plot_equilibria_stability_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.scatter(self.equilibriaSaddleCenter['x'], self.equilibriaSaddleCenter['y'], color=self.plottingColors['SxC'], label='$S$ $X$ $C$')
        ax.scatter(self.equilibriaSaddleSaddle['x'], self.equilibriaSaddleSaddle['y'], color=self.plottingColors['SxS'], label='$S$  $X$ $S$')
        ax.scatter(self.equilibriaMixedMixed['x'], self.equilibriaMixedMixed['y'], color=self.plottingColors['MxM'], label='$M$ $X$ $M$')
        ax.scatter(self.equilibriaCenterCenter['x'], self.equilibriaCenterCenter['y'], color=self.plottingColors['CxC'], label='$C$ $X$ $C$')
        ax.legend(frameon=True, loc='upper right')
        ax.grid(True, which='both', ls=':')

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
            ax.set_ylim([-0.1, 0.1])
        else:
            ax.set_xlim([0.8, 1.0])
            ax.set_ylim([-0.1, 0.1])
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.title('$L_' + str(self.lagrangePointNr) + '$ Artificial equilibria stability at $a_{lt} =$ ' + str(
            format(self.accelerationMagnitude, '.1e')), size=self.suptitleSize)
        if self.lowDPI:
            fig.savefig(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(
                    self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_stability_zoom.png',
                transparent=True,
                dpi=self.dpi)
        else:
            fig.savefig(
                '/Users/Sjors/Documents/thesisSoftware/tudatBundle/tudatApplications/thesisProject/data/figures/equilibria/L' + str(
                    self.lagrangePointNr) + '_' + str(self.accelerationMagnitude) + '_equilibria_stability_zoom.png',
                transparent=True)
        pass


if __name__ == '__main__':
    low_dpi = False
    lagrange_point_nrs = [1,2]
    acceleration_magnitudes = [0.200000]

    for lagrange_point_nr in lagrange_point_nrs:
        for acceleration_magnitude in acceleration_magnitudes:
            display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nr, acceleration_magnitude, low_dpi=low_dpi)
            display_equilibria_validation.plot_equilibria()
            #display_equilibria_validation.plot_equilibria_zoom()
            #display_equilibria_validation.plot_equilibria_validation()
            #display_equilibria_validation.plot_equilibria_stability()
            #display_equilibria_validation.plot_equilibria_stability_zoom()

            del display_equilibria_validation
