import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
sys.path.append('../util')
from load_data import load_lagrange_points_location, load_bodies_location
from load_data_augmented import load_equilibria_acceleration, load_equilibria_alpha, compute_stability_type, load_stability_data, \
cr3bplt_velocity, load_lagrange_points_location_augmented, potential_deviation, compute_eigenvalue_contour, load_eigenvalue_data



class DisplayDynamicalBehaviour:
    def __init__(self, motion_of_interest,low_dpi):

        self.motionOfInterest = motion_of_interest
        self.lowDPI = low_dpi
        self.arrayLength = 2000
        self.threshold = 6

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figRatio = self.figSize[0] / self.figSize[1]
        self.dpi = 150
        self.spacingPlotFactor = 1.05
        self.scaleDistanceY = 2.5
        self.scaleDistanceX = self.scaleDistanceY * self.figRatio

        n_colors = 4
        n_colors_l = 10
        self.plottingColors = {'SXC': sns.color_palette("Greys", n_colors_l)[0],
                               'CXC': sns.color_palette("Greys", n_colors_l)[2],
                               'MXM': sns.color_palette("Greys", n_colors_l)[4],
                               'SXS': sns.color_palette("Greys", n_colors_l)[9],
                               }

    def plot_global_eigenvalues(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')


        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        X = np.linspace(-self.scaleDistanceX / 2, self.scaleDistanceX / 2, self.arrayLength)
        Y = np.linspace(-self.scaleDistanceY / 2, self.scaleDistanceY / 2, self.arrayLength)

        type2 = load_stability_data('../../data/raw/equilibria/stability_2_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_2000.txt')

        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right',markerscale=15)


        if self.motionOfInterest == 'saddle':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 1, self.threshold)
        if self.motionOfInterest == 'center':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 2, self.threshold)

        lambdaColumn = eigenValue_df['maxLambda']
        lambdaColumn2 = eigenValue_df['minLambda']
        stabIndex     = eigenValue_df['stabIndex']
        print(lambdaColumn)
        print('test')
        print(stabIndex)


        # Create colorbar next to of plots
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(lambdaColumn))),
            norm=plt.Normalize(vmin=0, vmax=max(lambdaColumn)))

        ax.scatter(eigenValue_df['x'], eigenValue_df['y'], c=lambdaColumn, cmap="viridis", s=0.1)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.contourf(xM, yM, zM, colors='black')
        ax.contourf(xE, yE, zE, colors='black')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$\\lambda$ [-]')

        ax.set_title('Overview of the ' + str(self.motionOfInterest) +  ' eigenvalue behaviour')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)


        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/globalEigenvalues_' + str(self.motionOfInterest) + '.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/globalEigenvalues_' + str(self.motionOfInterest) + '.pdf', transparent=True)
        pass

    def plot_global_eigenvalues_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([1.0-self.scaleDistanceX / 12.5, 1.0+self.scaleDistanceX / 12.5])
        ax.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax.set_title('Overview of the ' + str(self.motionOfInterest) + ' eigenvalue behaviour')

        X = np.linspace(1-self.scaleDistanceX / 12.5,1+ self.scaleDistanceX / 12.5, self.arrayLength)
        Y = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, self.arrayLength)

        type2 = load_stability_data('../../data/raw/equilibria/stability_2_zoom.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_zoom.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_zoom.txt')

        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right', markerscale=15)

        if self.motionOfInterest == 'saddle':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 1, self.threshold)
        if self.motionOfInterest == 'center':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 2, self.threshold)

        lambdaColumn = eigenValue_df['maxLambda']


        # Create colorbar next to of plots
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(lambdaColumn))),
            norm=plt.Normalize(vmin=0, vmax=max(lambdaColumn)))

        ax.scatter(eigenValue_df['x'], eigenValue_df['y'], c=lambdaColumn, cmap="viridis", s=0.1)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.contourf(xM, yM, zM, colors='black')
        ax.contourf(xE, yE, zE, colors='black')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$\\lambda$ [-]')

        ax.set_title('Overview of the ' + str(self.motionOfInterest) +  ' eigenvalue behaviour')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/globalEigenvalues_' + str(self.motionOfInterest) + '_zoom_.png',
                        transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/globalEigenvalues_' + str(self.motionOfInterest) + '_zoom_.pdf',
                        transparent=True)
        pass

    def plot_global_dynamics(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        X = np.linspace(-self.scaleDistanceX / 2, self.scaleDistanceX / 2, self.arrayLength)
        Y = np.linspace(-self.scaleDistanceY / 2, self.scaleDistanceY / 2, self.arrayLength)

        type2 = load_stability_data('../../data/raw/equilibria/stability_2_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_2000.txt')

        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right', markerscale=15)

        if self.motionOfInterest == 'saddle':
            #eigenValue_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Saddle.txt')
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 1, self.threshold)
        if self.motionOfInterest == 'center':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 2, self.threshold)

        lambdaColumn = eigenValue_df['maxLambda']

        dataDynamics = []

        for row in eigenValue_df.iterrows():
            eigenValue = row[1].values[2]
            eigenVector = row[1].values[3:7]
            linearTransformation = np.linalg.norm(eigenValue*eigenVector)
            dataDynamics.append(linearTransformation)




        # Create colorbar next to of plots
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(dataDynamics))),
            norm=plt.Normalize(vmin=0, vmax=max(dataDynamics)))

        ax.scatter(eigenValue_df['x'], eigenValue_df['y'], c=dataDynamics, cmap="viridis", s=0.1)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.contourf(xM, yM, zM, colors='black')
        ax.contourf(xE, yE, zE, colors='black')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$| \\lambda \\cdot \\nu |$ [-]')

        ax.set_title('Overview of the ' + str(self.motionOfInterest) + ' eigenvalue behaviour')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '.png',
                        transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '.pdf',
                        transparent=True)
        pass

        pass

    def plot_global_dynamics_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([1-self.scaleDistanceX / 12.5, 1+self.scaleDistanceX / 12.5])
        ax.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        X = np.linspace(1-self.scaleDistanceX / 12.5, 1+self.scaleDistanceX / 12.5, self.arrayLength)
        Y = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, self.arrayLength)

        type2 = load_stability_data('../../data/raw/equilibria/stability_2_zoom.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_zoom.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_zoom.txt')

        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right', markerscale=15)

        if self.motionOfInterest == 'saddle':
            #eigenValue_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Saddle.txt')
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 1, self.threshold)
        if self.motionOfInterest == 'center':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 2, self.threshold)

        lambdaColumn = eigenValue_df['maxLambda']

        dataDynamics = []

        for row in eigenValue_df.iterrows():
            eigenValue = row[1].values[2]
            eigenVector = row[1].values[3:7]
            linearTransformation = np.linalg.norm(eigenValue * eigenVector)
            dataDynamics.append(linearTransformation)

        # Create colorbar next to of plots
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(dataDynamics))),
            norm=plt.Normalize(vmin=0, vmax=max(dataDynamics)))

        ax.scatter(eigenValue_df['x'], eigenValue_df['y'], c=dataDynamics, cmap="viridis", s=0.1)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.contourf(xM, yM, zM, colors='black')
        ax.contourf(xE, yE, zE, colors='black')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$| \\lambda \\cdot \\nu |$ [-]')

        ax.set_title('Overview of the ' + str(self.motionOfInterest) + ' eigenvalue behaviour')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '_zoom_.png',transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '_zoom_.pdf',transparent=True)

        pass

if __name__ == '__main__':
    motions_of_interest = ['saddle']

    low_dpi = True

    for motion_of_interest in motions_of_interest:
        display_dynamical_behaviour = DisplayDynamicalBehaviour(motion_of_interest, low_dpi)

        display_dynamical_behaviour.plot_global_eigenvalues()
        # display_dynamical_behaviour.plot_global_eigenvalues_zoom()
        # display_dynamical_behaviour.plot_global_dynamics()
        # display_dynamical_behaviour.plot_global_dynamics_zoom()


    del display_dynamical_behaviour

