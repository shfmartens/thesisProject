import numpy as np
import pandas as pd
import json
import math
import matplotlib
from decimal import *
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib2tikz import save as tikz_save
from matplotlib.transforms import Bbox

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
compute_stability_type, load_stability_data, compute_eigenvalue_contour, load_eigenvalue_data

class initialGuessConstruction:
    def __init__ (self, lagrange_point_nr, acceleration_magnitude, alpha, low_dpi):

        self.lowDPI = low_dpi
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha

        self.threshold = 3
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figRatio = self.figSize[0] / self.figSize[1]
        self.dpi = 150
        self.spacingPlotFactor = 1.05
        self.scaleDistanceY = 2.5
        self.scaleDistanceX = self.scaleDistanceY * self.figRatio

        self.visualizationThreshold = 6
        self.visualizationMultiplier = 1.5
        n_colors = 4
        n_colors_l = 10
        self.plottingColors = {'SXC': sns.color_palette("Greys", n_colors_l)[0],
                               'CXC': sns.color_palette("Greys", n_colors_l)[2],
                               'MXM': sns.color_palette("Greys", n_colors_l)[4],
                               'SXS': sns.color_palette("Greys", n_colors_l)[9],
                               }

    def plot_center_eigenvalues(self):
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        ## Add labels and titles
        ax0.set_xlabel('$x$ [-]')
        ax0.set_ylabel('$y$ [-]')

        ax0.set_xlim([1-self.scaleDistanceX / 12.5, 1+self.scaleDistanceX / 12.5])
        ax0.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax1.set_xlabel('$x$ [-]')
        ax1.set_ylabel('$y$ [-]')

        ax1.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax1.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax2.set_xlabel('$x$ [-]')
        ax2.set_ylabel('$y$ [-]')

        ax2.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax2.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax3.set_xlabel('$x$ [-]')
        ax3.set_ylabel('$y$ [-]')

        ax3.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax3.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax0.set_title('Modulus of $|\\lambda|_{center}$')
        ax1.set_title('$\\lambda_{center}*|\\nu_{center}|$')
        ax2.set_title('Real part of $|\\lambda|_{center}$')
        ax3.set_title('Imaginary part of $|\\lambda|_{center}$')

        ## Add grey scale for the stability regions
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_zoom.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_zoom.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_zoom.txt')

        ax0.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax0.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax0.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax1.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax1.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax1.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax2.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax2.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax2.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax3.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax3.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax3.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ## Plot the center eigenvalue properties
        x_loc = np.linspace(1-self.scaleDistanceX / 12.5, 1+self.scaleDistanceX / 12.5, 1000)
        y_loc = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, 1000)

        #eigenvalues_df = compute_eigenvalue_contour(x_loc, y_loc, 1, 2, self.threshold)
        eigenvalues_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Center_zoom.txt',2)
        lambdaModList = []
        lambdaRealList = []
        lambdaImagList = []
        dynamicList = []
        for index, row in eigenvalues_df.iterrows():
            lambdaMod = np.sqrt( row['maxLambdaImag'] ** 2 + row['maxLambdaReal'] ** 2 )
            lambdaReal = np.abs(row['maxLambdaReal'])
            lambdaImag = row['maxLambdaImag']
            eigenVector = np.array([row['maxV1Real'],row['maxV2Real'],row['maxV3Real'],row['maxV4Real']])

            nu1 = complex(row['maxV1Real'], row['maxV1Imag'])
            nu2 = complex(row['maxV2Real'], row['maxV2Imag'])
            nu3 = complex(row['maxV3Real'], row['maxV3Imag'])
            nu4 = complex(row['maxV4Real'], row['maxV4Imag'])

            nuMod = np.sqrt(abs(nu1) ** 2 + abs(nu2) ** 2 + abs(nu3) ** 2 + abs(nu4) ** 2)

            dynamic = lambdaImag*nuMod


            if lambdaMod > self.visualizationThreshold:
                lambdaMod = self.visualizationThreshold * self.visualizationMultiplier

            if lambdaReal > self.visualizationThreshold:
                lambdaReal = self.visualizationThreshold * self.visualizationMultiplier

            if lambdaImag > self.visualizationThreshold:
                lambdaImag = self.visualizationThreshold * self.visualizationMultiplier

            if dynamic > self.visualizationThreshold:
                dynamic = self.visualizationThreshold * self.visualizationMultiplier


            lambdaModList.append(lambdaMod)
            lambdaRealList.append(lambdaReal)
            lambdaImagList.append(lambdaImag)
            dynamicList.append(dynamic)



        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(lambdaModList))),
            norm=plt.Normalize(vmin=0.0, vmax=self.visualizationThreshold * self.visualizationMultiplier))

        ax0.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=lambdaModList, cmap="viridis", s=0.1)
        ax1.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=dynamicList, cmap="viridis", s=0.1)
        ax2.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=lambdaRealList, cmap="viridis", s=0.1)
        ax3.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=lambdaImagList, cmap="viridis", s=0.1)






        ## Add Natural collinear points
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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

        ax2.contourf(xM, yM, zM, colors='black')
        ax2.contourf(xE, yE, zE, colors='black')

        ax3.contourf(xM, yM, zM, colors='black')
        ax3.contourf(xE, yE, zE, colors='black')

        fig.suptitle('Center dynamics in the SxC region')

        fig.tight_layout()
        fig.subplots_adjust(top=0.90, right = 0.98)

        ## Set aspect ratio equal to 1
        ax0.set_aspect(1.0)
        ax1.set_aspect(1.0)
        ax2.set_aspect(1.0)
        ax3.set_aspect(1.0)

        # Create a colourbar right from the plots with equal height as two subplots
        upperRightPosition = ax1.get_position()
        lowerRightPosition = ax3.get_position()
        upperRightPoints = upperRightPosition.get_points()
        lowerRightPoints = lowerRightPosition.get_points()

        cb_x0 = upperRightPoints[1][0] - 0.015
        cb_y0 = lowerRightPoints[0][1]

        width_colourbar = ( upperRightPoints[1][0] - upperRightPoints[0][0] ) / 25
        height_colourbar = upperRightPoints[1][1]- lowerRightPoints[0][1]
        axes_colourbar = [cb_x0, cb_y0,width_colourbar, height_colourbar]

        sm.set_array([])

        cax = plt.axes(axes_colourbar)
        cbar0 = plt.colorbar(sm, cax=cax)

        fig.tight_layout()
        fig.subplots_adjust(top=0.90 )


        if self.lowDPI:
            fig.savefig('../../data/figures/initial_guess/eigenvalue_analysis.png', transparent=True,dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/initial_guess/eigenvalue_analysis.pdf', transparent=True)
        pass


    def plot_center_eigenvectors(self):
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        ## Add labels and titles
        ax0.set_xlabel('$x$ [-]')
        ax0.set_ylabel('$y$ [-]')

        ax0.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax0.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax1.set_xlabel('$x$ [-]')
        ax1.set_ylabel('$y$ [-]')

        ax1.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax1.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax2.set_xlabel('$x$ [-]')
        ax2.set_ylabel('$y$ [-]')

        ax2.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax2.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax3.set_xlabel('$x$ [-]')
        ax3.set_ylabel('$y$ [-]')

        ax3.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax3.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])


        ax0.set_title('Modulus of $|\\nu|_{center}$')
        ax1.set_title('$\\lambda_{center}*|\\nu_{center}|$')
        ax2.set_title('Modulus of $|\\nu|_{center_{real}}$')
        ax3.set_title('Modulus of $|\\nu|_{center_{imag}}$')

        ## Add grey scale for the stability regions
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_zoom.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_zoom.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_zoom.txt')

        ax0.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax0.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax0.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax1.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax1.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax1.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax2.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax2.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax2.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax3.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax3.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax3.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ## Plot the center eigenvalue properties
        x_loc = np.linspace(1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5, 1000)
        y_loc = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, 1000)

        # eigenvalues_df = compute_eigenvalue_contour(x_loc, y_loc, 1, 2, self.threshold)
        eigenvalues_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Center_zoom.txt', 2)
        nuModList = []
        nuRealList = []
        nuImagList = []
        dynamicList = []
        for index, row in eigenvalues_df.iterrows():

            nu1 = complex(row['maxV1Real'],row['maxV1Imag'])
            nu2 = complex(row['maxV2Real'], row['maxV2Imag'])
            nu3 = complex(row['maxV3Real'], row['maxV3Imag'])
            nu4 = complex(row['maxV4Real'], row['maxV4Imag'])

            nuMod = np.sqrt( abs(nu1) ** 2 + abs(nu2) ** 2 + abs(nu3) ** 2 + abs(nu4) ** 2 )


            eigenVectorReal = np.array([row['maxV1Real'], row['maxV2Real'], row['maxV3Real'], row['maxV4Real']])
            eigenVectorImag = np.array([row['maxV1Imag'], row['maxV2Imag'], row['maxV3Imag'], row['maxV4Imag']])

            nuReal = np.linalg.norm(eigenVectorReal)
            nuImag = np.linalg.norm(eigenVectorImag)

            lambdaMag= row['maxLambdaImag']
            dynamic = lambdaMag * nuMod

            if nuMod > self.visualizationThreshold:
                nuMod = self.visualizationThreshold * self.visualizationMultiplier

            if nuMod > 0.999 and nuMod < 1.0001:
                nuMod = 1.0

            if nuReal > self.visualizationThreshold:
                nuReal = self.visualizationThreshold * self.visualizationMultiplier

            if nuImag > self.visualizationThreshold:
                nuImag = self.visualizationThreshold * self.visualizationMultiplier

            if dynamic > self.visualizationThreshold:
                dynamic = self.visualizationThreshold * self.visualizationMultiplier

            nuRealList.append(nuReal)
            nuImagList.append(nuImag)
            nuModList.append(nuMod)
            dynamicList.append(dynamic)

        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(nuRealList))),
            norm=plt.Normalize(vmin=0.0, vmax=self.visualizationThreshold * self.visualizationMultiplier))

        print(min(nuModList))
        print(max(nuModList))

        ax0.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuModList, cmap="viridis", s=0.1)
        ax1.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=dynamicList, cmap="viridis", s=0.1)
        ax2.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuRealList, cmap="viridis", s=0.1)
        ax3.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuImagList, cmap="viridis", s=0.1)

        ## Add Natural collinear points
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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

        ax2.contourf(xM, yM, zM, colors='black')
        ax2.contourf(xE, yE, zE, colors='black')

        ax3.contourf(xM, yM, zM, colors='black')
        ax3.contourf(xE, yE, zE, colors='black')

        fig.suptitle('Center dynamics in the SxC region')

        fig.tight_layout()
        fig.subplots_adjust(top=0.90, right=0.98)

        ## Set aspect ratio equal to 1
        ax0.set_aspect(1.0)
        ax1.set_aspect(1.0)
        ax2.set_aspect(1.0)
        ax3.set_aspect(1.0)

        # Create a colourbar right from the plots with equal height as two subplots
        upperRightPosition = ax1.get_position()
        lowerRightPosition = ax3.get_position()
        upperRightPoints = upperRightPosition.get_points()
        lowerRightPoints = lowerRightPosition.get_points()

        cb_x0 = upperRightPoints[1][0] - 0.015
        cb_y0 = lowerRightPoints[0][1]

        width_colourbar = (upperRightPoints[1][0] - upperRightPoints[0][0]) / 25
        height_colourbar = upperRightPoints[1][1] - lowerRightPoints[0][1]
        axes_colourbar = [cb_x0, cb_y0, width_colourbar, height_colourbar]

        sm.set_array([])

        cax = plt.axes(axes_colourbar)
        cbar0 = plt.colorbar(sm, cax=cax)

        fig.tight_layout()
        fig.subplots_adjust(top=0.90)

        if self.lowDPI:
            fig.savefig('../../data/figures/initial_guess/eigenvector_analysis.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/initial_guess/eigenvector_analysis.pdf', transparent=True)
        pass

    def plot_center_eigenvectors_orientation(self):
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        ## Add labels and titles
        ax0.set_xlabel('$x$ [-]')
        ax0.set_ylabel('$y$ [-]')

        ax0.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax0.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax1.set_xlabel('$x$ [-]')
        ax1.set_ylabel('$y$ [-]')

        ax1.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax1.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax2.set_xlabel('$x$ [-]')
        ax2.set_ylabel('$y$ [-]')

        ax2.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax2.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax3.set_xlabel('$x$ [-]')
        ax3.set_ylabel('$y$ [-]')

        ax3.set_xlim([1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5])
        ax3.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        ax0.set_title('$|\\nu|_{center_{mod}}$ position orientation')
        ax1.set_title('$|\\nu|_{center_{real}}$ position orientation')
        ax2.set_title('$|\\nu|_{center_{mod}}$ velocity orientation')
        ax3.set_title('$|\\nu|_{center_{real}}$ velocity orientation')

        ## Add grey scale for the stability regions
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_zoom.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_zoom.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_zoom.txt')

        ax0.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax0.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax0.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax1.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax1.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax1.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax2.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax2.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax2.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax3.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax3.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax3.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ## Plot the center eigenvalue properties
        #x_loc = np.linspace(1 - self.scaleDistanceX / 12.5, 1 + self.scaleDistanceX / 12.5, 100)
        #y_loc = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, 100)
        #eigenvalues_df = compute_eigenvalue_contour(x_loc, y_loc, 1, 2, self.threshold)

        eigenvalues_df = load_eigenvalue_data('../../data/raw/equilibria/eigenvalueSxC_Center_zoom.txt', 2)
        nuModPosList = []
        nuModVelList = []
        nuRealPosList = []
        nuRealVelList = []

        dynamicList = []
        for index, row in eigenvalues_df.iterrows():

            nu1 = complex(row['maxV1Real'], row['maxV1Imag'])
            nu2 = complex(row['maxV2Real'], row['maxV2Imag'])
            nu3 = complex(row['maxV3Real'], row['maxV3Imag'])
            nu4 = complex(row['maxV4Real'], row['maxV4Imag'])

            eigenVectorReal = np.array([row['maxV1Real'], row['maxV2Real'], row['maxV3Real'], row['maxV4Real']])
            eigenVectorImag = np.array([row['maxV1Imag'], row['maxV2Imag'], row['maxV3Imag'], row['maxV4Imag']])

            nuModPos = math.atan2(abs(nu2),abs(nu1))
            nuModVel = math.atan2(abs(nu4),abs(nu3))
            nuRealPos = math.atan2(row['maxV2Real'],row['maxV1Real'])
            nuRealVel = math.atan2(row['maxV4Real'],row['maxV3Real'])

            if nuModPos < 0.0:
                nuModPos = nuModPos + 2*np.pi
            if nuModVel < 0.0:
                nuModVel = nuModVel + 2*np.pi
            if nuRealPos < 0.0:
                nuRealPos = nuRealPos + 2*np.pi
            if nuRealVel < 0.0:
                nuRealVel = nuRealVel + 2*np.pi

            if nuModPos > self.visualizationThreshold:
                nuModPos = self.visualizationThreshold * self.visualizationMultiplier

            if nuModVel > self.visualizationThreshold:
                nuModVel = self.visualizationThreshold * self.visualizationMultiplier

            if nuRealPos > self.visualizationThreshold:
                nuRealPos = self.visualizationThreshold * self.visualizationMultiplier

            if nuRealVel > self.visualizationThreshold:
                nuRealVel = self.visualizationThreshold * self.visualizationMultiplier

            nuModPosList.append(nuModPos)
            nuModVelList.append(nuModVel)
            nuRealPosList.append(nuRealPos)
            nuRealVelList.append(nuRealVel)

        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(nuRealPosList))),
            norm=plt.Normalize(vmin=0.0, vmax=2*np.pi))


        ax0.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuModPosList, cmap="viridis", s=0.1)
        ax1.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuRealPosList, cmap="viridis", s=0.1)
        ax2.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuModVelList, cmap="viridis", s=0.1)
        ax3.scatter(eigenvalues_df['x'], eigenvalues_df['y'], c=nuRealVelList, cmap="viridis", s=0.1)

        ## Add Natural collinear points
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

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

        ax2.contourf(xM, yM, zM, colors='black')
        ax2.contourf(xE, yE, zE, colors='black')

        ax3.contourf(xM, yM, zM, colors='black')
        ax3.contourf(xE, yE, zE, colors='black')

        fig.suptitle('Orientation of the $|\\nu_{center}|$  components')

        fig.tight_layout()
        fig.subplots_adjust(top=0.90, right=0.95)

        ## Set aspect ratio equal to 1
        ax0.set_aspect(1.0)
        ax1.set_aspect(1.0)
        ax2.set_aspect(1.0)
        ax3.set_aspect(1.0)

        # Create a colourbar right from the plots with equal height as two subplots
        upperRightPosition = ax1.get_position()
        lowerRightPosition = ax3.get_position()
        upperRightPoints = upperRightPosition.get_points()
        lowerRightPoints = lowerRightPosition.get_points()

        cb_x0 = upperRightPoints[1][0] - 0.015
        cb_y0 = lowerRightPoints[0][1]

        width_colourbar = (upperRightPoints[1][0] - upperRightPoints[0][0]) / 25
        height_colourbar = upperRightPoints[1][1] - lowerRightPoints[0][1]
        axes_colourbar = [cb_x0, cb_y0, width_colourbar, height_colourbar]

        sm.set_array([])

        cax = plt.axes(axes_colourbar)
        cbar0 = plt.colorbar(sm, cax=cax, label='$\\alpha$ [-]')

        #fig.tight_layout()
        #fig.subplots_adjust(top=0.90)

        if self.lowDPI:
            fig.savefig('../../data/figures/initial_guess/eigenvector_orientation_analysis.png', transparent=True, dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/initial_guess/eigenvector_orientation_analysis.pdf', transparent=True)
        pass




if __name__ == '__main__':
    low_dpi = True
    lagrange_point_nrs = [1]
    acceleration_magnitudes = [0.07]
    alphas = [0.0]
    print(np.linspace(0.01.1,num=3))

    display_initial_guess_construction = initialGuessConstruction(lagrange_point_nrs, acceleration_magnitudes, alphas, low_dpi=low_dpi)
    #display_initial_guess_construction.plot_center_eigenvalues()
    #display_initial_guess_construction.plot_center_eigenvectors()
    display_initial_guess_construction.plot_center_eigenvectors_orientation()



    del display_initial_guess_construction