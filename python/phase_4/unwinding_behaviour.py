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
        self.threshold = 10

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)

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
            #eigenValue_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Saddle.txt', 1)
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 1, self.threshold)
        if self.motionOfInterest == 'center':
            eigenValue_df = compute_eigenvalue_contour(X, Y, 1, 2, self.threshold)

        lambdaColumn = eigenValue_df['maxLambda']
        lambdaColumn2 = eigenValue_df['minLambda']
        stabIndex     = eigenValue_df['stabIndex']

        plotting_list = []
        print('start filtering threshold data')
        for row in eigenValue_df.iterrows():
           x = row[1][0]
           y = row[1][1]
           maxLambda = row[1][2]
           minLambda = row[1][3]
           stabilityIndex = row[1][4]

           if stabilityIndex < self.threshold:
               plotting_list.append([x, y, maxLambda, minLambda, stabilityIndex])
        print(' filtering threshold complete data')


        plotting_df = pd.DataFrame(plotting_list, columns=['x','y','maxSaddle','minSaddle','stabilityIndex'])

        # Create colorbar next to of plots
        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(plotting_df['maxSaddle']))),
            norm=plt.Normalize(vmin=0, vmax=max(plotting_df['maxSaddle'])))

        ax.scatter(plotting_df['x'], plotting_df['y'], c=plotting_df['maxSaddle'], cmap="viridis", s=0.1)

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
        fig.subplots_adjust(right=0.95)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '_zoom_.png',transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/globalDynamics_' + str(self.motionOfInterest) + '_zoom_.pdf',transparent=True)

        pass

    def plot_eigenvalue_spm_behaviour(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        #ax2 = fig.add_subplot(2, 2, 3)
        #ax3 = fig.add_subplot(2, 2, 4)

        #fig.suptitle('Eigenvalue behaviour')
        #ax0.set_title('$\\lambda_{saddle}$')
        #ax1.set_title('$\\lambda_{saddle}$ zoom')
        #ax2.set_title('$\\lambda_{center}$')
        #ax3.set_title('$\\lambda_{center}$ zoom')

        ax0.set_xlabel('$x$ [-]')
        ax0.set_ylabel('$y$ [-]')

        ax0.set_xlim([-self.scaleDistanceX / 2.2, self.scaleDistanceX / 2.2])
        ax0.set_ylim([-self.scaleDistanceY / 2.2, self.scaleDistanceY / 2.2])

        ax1.set_xlabel('$x$ [-]')
        ax1.set_ylabel('$y$ [-]')

        ax1.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax1.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        # ax2.set_xlabel('$x$ [-]')
        # ax2.set_ylabel('$y$ [-]')
        #
        # ax2.set_xlim([-self.scaleDistanceX / 2.2, self.scaleDistanceX / 2.2])
        # ax2.set_ylim([-self.scaleDistanceY / 2.2, self.scaleDistanceY / 2.2])
        #
        # ax3.set_xlabel('$x$ [-]')
        # ax3.set_ylabel('$y$ [-]')
        #
        # ax3.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        # ax3.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        type1ZOOM = load_stability_data('../../data/raw/equilibria/stability_1_ZOOM_2000.txt')
        type2ZOOM = load_stability_data('../../data/raw/equilibria/stability_2_ZOOM_2000.txt')
        type3ZOOM = load_stability_data('../../data/raw/equilibria/stability_3_ZOOM_2000.txt')
        type4ZOOM = load_stability_data('../../data/raw/equilibria/stability_4_ZOOM_2000.txt')

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_2000.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_2000.txt')


        ax0.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax0.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax0.legend(frameon=True, loc='lower left', markerscale=15)
        ax0.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.04, label='CxC')


        ax1.scatter(type1ZOOM['x'], type1ZOOM['y'], color=self.plottingColors['SXC'], s=0.04, label='CxC')
        ax1.scatter(type2ZOOM['x'], type2ZOOM['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax1.scatter(type3ZOOM['x'], type3ZOOM['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')

        # ax2.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        # ax2.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')
        # ax2.legend(frameon=True, loc='lower right', markerscale=15)
        #
        #
        # ax3.scatter(type3ZOOM['x'], type3ZOOM['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        # ax3.scatter(type4ZOOM['x'], type4ZOOM['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')


        ###### EIGENVALUE LOAD DATA FRAMES ######
        eigenvalue_df_saddle = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Saddle_incSxS.txt',1)
        eigenvalue_df_saddle_zoom = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Saddle_incSxS_ZOOM.txt',1)
        #eigenvalue_df_center = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Center_incCxC.txt', 2)
        #eigenvalue_df_center_zoom = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Center_incCxC_ZOOM.txt', 2)

        print('appending saddle_list')
        plotting_list_saddle = []
        for row in eigenvalue_df_saddle.iterrows():
            x = row[1][0]
            y = row[1][1]
            maxLambda = row[1][2]
            minLambda = row[1][3]
            stabilityIndex = row[1][4]

            if stabilityIndex < self.threshold:
                plotting_list_saddle.append([x, y, maxLambda, minLambda, stabilityIndex])

        plotting_df_saddle = pd.DataFrame(plotting_list_saddle, columns=['x', 'y', 'maxSaddle', 'minSaddle', 'stabilityIndex'])

        print('appending saddle_list_zoom')
        plotting_list_saddle_zoom = []
        for row in eigenvalue_df_saddle_zoom.iterrows():
            x = row[1][0]
            y = row[1][1]
            maxLambda = row[1][2]
            minLambda = row[1][3]
            stabilityIndex = row[1][4]

            if stabilityIndex < self.threshold:
                plotting_list_saddle_zoom.append([x, y, maxLambda, minLambda, stabilityIndex])

        plotting_df_saddle_zoom = pd.DataFrame(plotting_list_saddle_zoom, columns=['x', 'y', 'maxSaddle', 'minSaddle', 'stabilityIndex'])

        # plotting_list_center = []
        # for row in eigenvalue_df_center.iterrows():
        #     x = row[1][0]
        #     y = row[1][1]
        #     maxLambda = row[1][3]
        #     minLambda = row[1][5]
        #     stabilityIndex = row[1][6]
        #
        #     if stabilityIndex < self.threshold:
        #         plotting_list_center.append([x, y, maxLambda, minLambda, stabilityIndex])
        #
        # plotting_df_center = pd.DataFrame(plotting_list_center,
        #                                   columns=['x', 'y', 'maxSaddle', 'minSaddle', 'stabilityIndex'])
        #
        # plotting_list_center_zoom = []
        # for row in eigenvalue_df_center_zoom.iterrows():
        #     x = row[1][0]
        #     y = row[1][1]
        #     maxLambda = row[1][3]
        #     minLambda = row[1][5]
        #     stabilityIndex = row[1][6]
        #
        #     if stabilityIndex < self.threshold:
        #         plotting_list_center_zoom.append([x, y, maxLambda, minLambda, stabilityIndex])

        # plotting_df_center_zoom = pd.DataFrame(plotting_list_center_zoom,
        #                                        columns=['x', 'y', 'maxSaddle', 'minSaddle', 'stabilityIndex'])

        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(eigenvalue_df_saddle['maxLambda']))),
            norm=plt.Normalize(vmin=0.0, vmax=self.threshold))

        print('start_plotting')


        ax0.scatter(plotting_df_saddle['x'], plotting_df_saddle['y'], c=plotting_df_saddle['maxSaddle'], cmap="viridis", s=0.1)
        ax1.scatter(plotting_df_saddle_zoom['x'], plotting_df_saddle_zoom['y'], c=plotting_df_saddle_zoom['maxSaddle'], cmap="viridis",s=0.1)
        # ax2.scatter(plotting_df_center['x'], plotting_df_center['y'], c=plotting_df_center['maxSaddle'], cmap="viridis",s=0.1)
        # ax3.scatter(plotting_df_center_zoom['x'], plotting_df_center_zoom['y'], c=plotting_df_center_zoom['maxSaddle'], cmap="viridis",s=0.1)

        ## Add the moon and earth bodies
        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        ax0.contourf(xM, yM, zM, colors='black')
        ax0.contourf(xE, yE, zE, colors='black')

        ax1.contourf(xM, yM, zM, colors='black')
        ax1.contourf(xE, yE, zE, colors='black')

        # ax2.contourf(xM, yM, zM, colors='black')
        # ax2.contourf(xE, yE, zE, colors='black')
        #
        # ax3.contourf(xM, yM, zM, colors='black')
        # ax3.contourf(xE, yE, zE, colors='black')

        ## Add Natural collinear points
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2','L3','L4','L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            # ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')
            # ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
            #             color='black', marker='x')

        ## Set aspect ratio equal to 1
        ax0.set_aspect(1.0)
        ax1.set_aspect(1.0)
        # ax2.set_aspect(1.0)
        # ax3.set_aspect(1.0)


        # Create a colourbar right from the plots with equal height as two subplots
        # upperRightPosition = ax1.get_position()
        # lowerRightPosition = ax3.get_position()
        # upperRightPoints = upperRightPosition.get_points()
        # lowerRightPoints = lowerRightPosition.get_points()
        #
        # cb_x0 = upperRightPoints[1][0] + 0.02
        # cb_y0 = lowerRightPoints[0][1]
        #
        # width_colourbar = (upperRightPoints[1][0] - upperRightPoints[0][0]) / 25
        # height_colourbar = upperRightPoints[1][1] - lowerRightPoints[0][1]
        # axes_colourbar = [cb_x0, cb_y0, width_colourbar, height_colourbar]
        #cax = plt.axes(axes_colourbar)


        sm.set_array([])

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar0 = plt.colorbar(sm, cax=cax, label='$|| \\lambda ||$ [-]')

        #fig.subplots_adjust(hspace=0.4)
        fig.tight_layout()

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/final_plot_eigenvalue_behaviour.png',transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/final_plot_eigenvalue_behaviour.png',transparent=True, dpi=300)

        pass






if __name__ == '__main__':
    motions_of_interest = ['center']

    low_dpi = False

    for motion_of_interest in motions_of_interest:
        display_dynamical_behaviour = DisplayDynamicalBehaviour(motion_of_interest, low_dpi)

        #display_dynamical_behaviour.plot_global_eigenvalues()
        # display_dynamical_behaviour.plot_global_eigenvalues_zoom()
        # display_dynamical_behaviour.plot_global_dynamics()
        # display_dynamical_behaviour.plot_global_dynamics_zoom()
        display_dynamical_behaviour.plot_eigenvalue_spm_behaviour()


    del display_dynamical_behaviour

