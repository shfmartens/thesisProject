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
cr3bplt_velocity, load_lagrange_points_location_augmented, potential_deviation, compute_eigenvalue_contour, load_equilibria_acceleration_deviation


class DisplayEquilibriaValidation:
    def __init__(self, lagrange_point_nrs, acceleration_magnitude, alphas, seeds, continuations, low_dpi):

        self.lagrangePointNrs = lagrange_point_nrs
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alphas
        self.seeds = seeds
        self.continuations = continuations
        self.lowDPI = low_dpi

        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)

        self.figRatio = self.figSize[0]/self.figSize[1]
        self.dpi = 150
        self.spacingPlotFactor = 1.05
        self.scaleDistanceY = 2.5
        self.scaleDistanceX = self.scaleDistanceY * self.figRatio

        self.cbarTicksAcc = ([0, 0.5* np.pi ,np.pi, 1.5*np.pi, 2 * np.pi])
        self.cbarTicksAccLabels = (['0', '$\\frac{1}{2}\pi$','$\pi$','$\\frac{3}{2}\pi$', '$2\pi$'])

        self.cbarTicksAngle = ([0.0, 0.05, 0.1])
        self.cbarTicksAngleLabels = (['0', '0.05', '0.1'])

        n_colors = 4
        n_colors_l = 10
        self.plottingColors = {'SXC': sns.color_palette("Greys", n_colors_l)[0],
                               'CXC': sns.color_palette("Greys", n_colors_l)[2],
                               'MXM': sns.color_palette("Greys", n_colors_l)[4],
                               'SXS': sns.color_palette("Greys", n_colors_l)[9],
                               }


    def plot_global_stability(self):
        fig = plt.figure(figsize=self.figSize)
        fig2 = plt.figure(figsize=self.figSize)

        ax = fig.gca()

        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        X = np.linspace(-self.scaleDistanceX / 2, self.scaleDistanceX / 2, 1000)
        Y = np.linspace(-self.scaleDistanceY / 2, self.scaleDistanceY / 2, 1000)

        X = np.linspace(1.0 - self.scaleDistanceX / 12.5, 1.0 + self.scaleDistanceX / 12.5, 2000)
        Y = np.linspace(-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5, 2000)

        type1 = compute_stability_type(X, Y, 1)
        type2 = compute_stability_type(X, Y, 2)
        type3 = compute_stability_type(X, Y, 3)
        type4 = compute_stability_type(X, Y, 4)

        type1 = load_stability_data('../../data/raw/equilibria/stability_1.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4.txt')


        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.04, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')
        ax.legend(frameon=True, loc='upper right',bbox_to_anchor=(1.12, 1.00),markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')


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


        fig.suptitle('Linear modes of linear dynamics about static planar states')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        ax.set_aspect(1.0)



        if self.lowDPI:
            fig.savefig(
                '../../data/figures/equilibria/stabilityPlot.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig(
                '../../data/figures/equilibria/stabilityPlot.pdf', transparent=True)
        pass

    def plot_global_eigenvalues(self):
        fig = plt.figure(figsize=self.figSize)
        fig2 = plt.figure(figsize=self.figSize)

        ax = fig.gca()

        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        X = np.linspace(-self.scaleDistanceX / 2, self.scaleDistanceX / 2, 100)
        Y = np.linspace(-self.scaleDistanceY / 2, self.scaleDistanceY / 2, 100)

        type1 = compute_eigenvalue_contour(X, Y, 1, 2)
        type2= load_stability_data('../../data/raw/equilibria/stability_2.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4.txt')



        # Create colormap and scalar mappable:)

        evMap = type1['maxEV']

        print(evMap)
        print(min(evMap))
        print(max(evMap))


        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(evMap))),
            norm=plt.Normalize(vmin=min(evMap), vmax=max(evMap)))

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$|\\lambda|$ [-]')
        #cbar.set_ticklabels(self.cbarTicksAccLabels)


        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['SXC'], s=0.04, label='SxC')
        ax.scatter(type1['x'], type1['y'], c=evMap, cmap="viridis", s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')
        ax.legend(frameon=True, loc='upper right',bbox_to_anchor=(1.12, 1.00),markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')


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


        fig.suptitle('maximum eigenvalue plot at each location')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        ax.set_aspect(1.0)



        if self.lowDPI:
            fig.savefig(
                '../../data/figures/equilibria/eigenvaluePlot.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig(
                '../../data/figures/equilibria/eigenvaluePlot.pdf', transparent=True)
        pass

    def plot_equilibria_acceleration(self):
        fig = plt.figure(figsize=self.figSize)
        fig2 = plt.figure(figsize=self.figSize)

        ax = fig.gca()
        #ax2 = fig2.gca()


        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')


        ax.set_xlim([-self.scaleDistanceX/2, self.scaleDistanceX/2])
        ax.set_ylim([-self.scaleDistanceY/2, self.scaleDistanceY/2])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.04, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right',markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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

        lagrange_point_nrs = ['L1','L2','L3','L4','L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        for seed in self.seeds:
            for continuation in self.continuations:
                for lagrangePointNr in self.lagrangePointNrs:
                    equilibria_df = load_equilibria_acceleration('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                         + '_acceleration_'  \
                                                         + str("{:7.6f}".format(self.accelerationMagnitude)) + '_' \
                                                         + str("{:7.6f}".format(seed)) + '_' + continuation +'_equilibria.txt')



                    if len(equilibria_df['alpha']) > 1:
                        alpha = equilibria_df['alpha']


                        # Create colorbar next to of plots
                        sm = plt.cm.ScalarMappable(
                            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(alpha))),
                            norm=plt.Normalize(vmin=0.0, vmax=2*np.pi ))

                        ax.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)
                        #type = str(self.accelerationMagnitude)
                        #ax.text(equilibria_df['x'][1]+0.1, equilibria_df['y'][1]+0.1, type, fontsize=9)


                        #ax2.plot(equilibria_df.index,equilibria_df['alpha'])
                        #ax2.plot(equilibria_df.index,equilibria_df['x'])
                        #ax2.plot(equilibria_df.index,equilibria_df['y'])





        fig.suptitle('Artificial equilibria at $a_{lt} = ' + str(self.accelerationMagnitude) + '$')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.15)

        cbar = plt.colorbar(sm, cax=cax, label='$\\alpha$ [-]', ticks = self.cbarTicksAcc )
        cbar.set_ticklabels(self.cbarTicksAccLabels)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.accelerationMagnitude) +'_equilibria_acceleration_effect.png', transparent=True,dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.accelerationMagnitude) +'_equilibria_acceleration_effect.pdf', transparent=True)

        pass

    def plot_equilibria_acceleration_total(self):
        fig = plt.figure(figsize=self.figSize)
        fig2 = plt.figure(figsize=self.figSize)

        ax = fig.gca()
        #ax2 = fig2.gca()


        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        #ax.grid(True, which='both', ls=':')

        ax.set_xlim([-self.scaleDistanceX / 2.2, self.scaleDistanceX / 2.2])
        ax.set_ylim([-self.scaleDistanceY / 2.2, self.scaleDistanceY / 2.2])

        ax.set_xlim([-1.05,-0.95])
        ax.set_ylim([-0.5, 0.5])
        type1 = load_stability_data('../../data/raw/equilibria/stability_1_L3.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_L3.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_L3.txt')
        #type4 = load_stability_data('../../data/raw/equilibria/stability_4_L3.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')
        #ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.09, label='SxS')

        ax.legend(frameon=True, loc='lower right', markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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



        lagrange_point_nrs = ['L1','L2','L3','L4','L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        for accMag in self.accelerationMagnitude:
            for seed in self.seeds:
                for continuation in self.continuations:
                    for lagrangePointNr in self.lagrangePointNrs:
                        equilibria_df = load_equilibria_acceleration('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                             + '_acceleration_'  \
                                                             + str("{:7.6f}".format(accMag)) + '_' \
                                                             + str("{:7.6f}".format(seed)) + '_' + continuation +'_equilibria.txt')



                        if len(equilibria_df['alpha']) > 1:
                            alpha = equilibria_df['alpha']


                            # Create colorbar next to of plots
                            sm = plt.cm.ScalarMappable(
                                cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(alpha))),
                                norm=plt.Normalize(vmin=0.0, vmax=2*np.pi ))

                            ax.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)
                            #type = str(self.accelerationMagnitude)
                            #ax.text(equilibria_df['x'][1]+0.1, equilibria_df['y'][1]+0.1, type, fontsize=9)


                            #ax2.plot(equilibria_df.index,equilibria_df['alpha'])
                            #ax2.plot(equilibria_df.index,equilibria_df['x'])
                            #ax2.plot(equilibria_df.index,equilibria_df['y'])

        ax.text(-0.963,0.278,'0.003', fontsize=8,rotation=10,rotation_mode='anchor') # for L3
        ax.text(-0.962, 0.276, '0.0105', fontsize=8, rotation=10, rotation_mode='anchor')  # for L3

        #ax.text(-0.95,0.3,'0.003', fontsize=8,rotation=70,rotation_mode='anchor')
        # ax.text(0.575,0.795,'0.003', fontsize=8,rotation=320,rotation_mode='anchor')
        # ax.text(0.59,-0.82,'0.003', fontsize=8,rotation=40,rotation_mode='anchor')

        ax.text(0.91, 0.285, '0.1', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.91, -0.305, '0.1', fontsize=8, rotation=0, rotation_mode='anchor')



        ax.text(0.87,0.17,'0.25', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(0.87,-0.2,'0.25', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(-1.12,-0.0,'0.25', fontsize=8,rotation=90,rotation_mode='anchor')

        #fig.suptitle('Artificial equilibria contours at various acceleration magnitudes')
        fig.suptitle('$L_3$ contour at various acceleration magnitudes')


        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        #ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar = plt.colorbar(sm, cax=cax, label='$\\alpha$ [-]', ticks = self.cbarTicksAcc )
        cbar.set_ticklabels(self.cbarTicksAccLabels)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_equilibria_acceleration_contour.png', transparent=True,dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_equilibria_acceleration_contour.pdf', transparent=True)
        pass

    def plot_equilibria_acceleration_total_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        fig2 = plt.figure(figsize=self.figSize)

        ax = fig.gca()
        #ax2 = fig2.gca()


        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([1.0-self.scaleDistanceX / 12.5, 1.0+self.scaleDistanceX / 12.5])
        ax.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_ZOOM_2000.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_ZOOM_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_ZOOM_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_ZOOM_2000.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.04, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')

        ax.legend(frameon=True, loc='lower right', markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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

        ax.set_xlim([1.0-self.scaleDistanceX / 12.5, 1.0+self.scaleDistanceX / 12.5])
        ax.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])



        lagrange_point_nrs = ['L1','L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        for accMag in self.accelerationMagnitude:
            for seed in self.seeds:
                for continuation in self.continuations:
                    for lagrangePointNr in self.lagrangePointNrs:
                        equilibria_df = load_equilibria_acceleration('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                             + '_acceleration_'  \
                                                             + str("{:7.6f}".format(accMag)) + '_' \
                                                             + str("{:7.6f}".format(seed)) + '_' + continuation +'_equilibria.txt')



                        if len(equilibria_df['alpha']) > 1:
                            alpha = equilibria_df['alpha']


                            # Create colorbar next to of plots
                            sm = plt.cm.ScalarMappable(
                                cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(alpha))),
                                norm=plt.Normalize(vmin=0.0, vmax=2*np.pi ))

                            ax.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)
                            #type = str(self.accelerationMagnitude)
                            #ax.text(equilibria_df['x'][1]+0.1, equilibria_df['y'][1]+0.1, type, fontsize=9)


                            #ax2.plot(equilibria_df.index,equilibria_df['alpha'])
                            #ax2.plot(equilibria_df.index,equilibria_df['x'])
                            #ax2.plot(equilibria_df.index,equilibria_df['y'])

        ax.text(0.83,0.02,'0.07', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(0.83,0.039,'0.15', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(0.833,0.055,'0.2', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(0.835, 0.071, '0.25', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(1.148,0.037,'0.07', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(1.14,0.075,'0.15', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(1.13,0.11,'0.2', fontsize=8,rotation=0,rotation_mode='anchor')
        ax.text(1.105,0.162, '0.25', fontsize=8, rotation=0, rotation_mode='anchor')







        #fig.suptitle('Artificial equilibria contoursa a $L_{' + str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '}$ at different acceleration magnitudes')
        fig.suptitle('Artificial equilibria contours around $L_1$ and $L_{2}$ at various acceleration magnitudes')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar = plt.colorbar(sm, cax=cax, label='$\\alpha$ [-]', ticks = self.cbarTicksAcc )
        cbar.set_ticklabels(self.cbarTicksAccLabels)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_equilibria_acceleration_contour_ZOOM.png', transparent=True,dpi=self.dpi)
            #fig2.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.accelerationMagnitude) +'_COLORBAR.png', transparent=True,dpi=self.dpi)

        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_equilibria_acceleration_contour_ZOOM.pdf')
            #fig2.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.accelerationMagnitude) +'_COLORBAR.png', transparent=True)
        pass


    def plot_equilibria_alpha(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_2000.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_2000.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.09, label='SxS')


        ax.legend(frameon=True, loc='lower right', markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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

        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')


        for lagrangePointNr in self.lagrangePointNrs:
                equilibria_df = load_equilibria_alpha('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                         + '_angle_'  \
                                                         + str("{:7.6f}".format(alpha)) + '_' + '0.000000_forward_equilibria.txt')

                acc = equilibria_df['acc']

                if len(equilibria_df['acc']) > 1:
                    acc = equilibria_df['acc']

                    # Create colorbar next to of plots
                    sm = plt.cm.ScalarMappable(
                        cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(acc))),
                        norm=plt.Normalize(vmin=0.0, vmax=0.1))


                    ax.scatter(equilibria_df['x'], equilibria_df['y'], c=acc, cmap="viridis", s=0.1)


        fig.suptitle('Artificial equilibria at $\\alpha = ' + str(self.alpha ) + '$$^{\\circ}$')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar = plt.colorbar(sm, cax=cax, label='$a_{lt}$ [-]', ticks=self.cbarTicksAngle)
        cbar.set_ticklabels(self.cbarTicksAngleLabels)



        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_effect.png', transparent=True,dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_effect.pdf')
        pass

    def plot_equilibria_alpha_total(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([-self.scaleDistanceX / 2.2, self.scaleDistanceX / 2.2])
        ax.set_ylim([-self.scaleDistanceY / 2.2, self.scaleDistanceY / 2.2])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_2000.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_2000.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.09, label='SxS')


        ax.legend(frameon=True, loc='lower right', markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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

        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        for angle in self.alpha:
            for lagrangePointNr in self.lagrangePointNrs:
                equilibria_df = load_equilibria_alpha('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                         + '_angle_'  \
                                                         + str("{:7.6f}".format(angle)) + '_' + '0.000000_forward_equilibria.txt')

                acc = equilibria_df['acc']

                if len(equilibria_df['acc']) > 1:
                    acc = equilibria_df['acc']

                    # Create colorbar next to of plots
                    sm = plt.cm.ScalarMappable(
                        cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(acc))),
                        norm=plt.Normalize(vmin=0.0, vmax=0.1))


                    ax.scatter(equilibria_df['x'], equilibria_df['y'], c=acc, cmap="viridis", s=0.1)

        ax.text(-1.1, -0.01, '0$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(-0.012, 0.95, '0$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(-0.012, -0.98, '0$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(-0.12, 0.92, '90$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.905, 0.295, '90$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.025, -1.07, '90$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(-0.95, -0.01, '180$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.905, 0.42, '180$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.905, -0.45, '180$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(-0.12,-0.94 , '270$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.905, -0.315, '270$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(0.06, 1.04, '270$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')




        #fig.suptitle('Artificial equilibria at $L_{' + str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '}$ at different angles')
        fig.suptitle('Artificial equilibria contours at various acceleration orientations')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar = plt.colorbar(sm, cax=cax, label='$a_{lt}$ [-]', ticks=self.cbarTicksAngle)
        cbar.set_ticklabels(self.cbarTicksAngleLabels)



        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_CONTOUR.png', transparent=True,dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_CONTOUR.pdf')
        pass

    def plot_equilibria_alpha_total_zoom(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([1.0-self.scaleDistanceX / 12.5, 1.0+self.scaleDistanceX / 12.5])
        ax.set_ylim([-self.scaleDistanceY / 12.5, self.scaleDistanceY / 12.5])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_ZOOM_2000.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_ZOOM_2000.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_ZOOM_2000.txt')
        type4 = load_stability_data('../../data/raw/equilibria/stability_4_ZOOM_2000.txt')

        ax.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.04, label='SxC')
        ax.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.04, label='CxC')
        ax.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.04, label='MxM')
        ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.04, label='SxS')


        ax.legend(frameon=True, loc='lower right', markerscale=15)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

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

        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        for angle in self.alpha:
            for lagrangePointNr in self.lagrangePointNrs:
                equilibria_df = load_equilibria_alpha('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                         + '_angle_'  \
                                                         + str("{:7.6f}".format(angle)) + '_' + '0.000000_forward_equilibria.txt')

                acc = equilibria_df['acc']

                if len(equilibria_df['acc']) > 1:
                    acc = equilibria_df['acc']

                    # Create colorbar next to of plots
                    sm = plt.cm.ScalarMappable(
                        cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", len(acc))),
                        norm=plt.Normalize(vmin=0.0, vmax=0.1))


                    ax.scatter(equilibria_df['x'], equilibria_df['y'], c=acc, cmap="viridis", s=1)

        ax.text(0.815, -0.002, '0$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.135, -0.002, '0$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(0.823, 0.025, '60$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.135, 0.033, '60$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(0.84, 0.023, '120$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.15, 0.05, '120$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(0.85, -0.002, '180$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.175, -0.002, '180$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(0.84, -0.027, '240$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.15, -0.057, '240$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        ax.text(0.823, -0.029, '300$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')
        ax.text(1.135, -0.041, '300$^{\\circ}$', fontsize=8, rotation=0, rotation_mode='anchor')

        fig.suptitle('Artificial equilibria at $L_{' + str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '}$ at different angles')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93,right=0.95)

        ax.set_aspect(1.0)

        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        cbar = plt.colorbar(sm, cax=cax, label='$a_{lt}$ [-]', ticks=self.cbarTicksAngle)
        cbar.set_ticklabels(self.cbarTicksAngleLabels)



        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_CONTOURZOOM.png', transparent=True,dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/L'+ str(", ".join( repr(e) for e in self.lagrangePointNrs )) + '_' +str(self.alpha) +'_equilibria_alpha_CONTOURZOOM.pdf')
        pass



    def plot_forbidden_region_test(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        H_lt = -1.562
        alpha = 0
        acc = 0.07

        ax.set_xlabel('$x$ [-]')
        ax.set_ylabel('$y$ [-]')
        ax.grid(True, which='both', ls=':')

        ax.set_xlim([-self.scaleDistanceX / 2, self.scaleDistanceX / 2])
        ax.set_ylim([-self.scaleDistanceY / 2, self.scaleDistanceY / 2])

        x_range = np.arange(-self.scaleDistanceX / 2, self.scaleDistanceX / 2,0.001)
        y_range = np.arange(-self.scaleDistanceX / 2, self.scaleDistanceX / 2,0.001)
        x_mesh, y_mesh = np.meshgrid(x_range,y_range)
        z_mesh = cr3bplt_velocity(x_mesh, y_mesh, acc, alpha, H_lt)

        if z_mesh.min() < 0:
            ax.contourf(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)


        lagrange_points_df = load_lagrange_points_location_augmented(acc, alpha)

        lagrange_point_nrs = ['L1', 'L2', 'L3','L4','L5']
        for lagrange_point_nr in lagrange_point_nrs:
                if (lagrange_points_df[lagrange_point_nr]['x']) != 0.0:
                    ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                    color='black', marker='x')

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

        fig.suptitle('Forbidden Regions in the CR3BP at $H_{lt}$ =' + str(H_lt) + ',$a_{lt}$ = ' + str(acc) + ', $\\alpha$ = ' + str(alpha) )

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)

        ax.set_aspect(1.0)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/ForbiddenRegion_'+'_'+str(H_lt)+'_'+str(acc)+'_'+str(alpha)+'.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/ForbiddenRegion_'+'_'+str(H_lt)+'_'+str(acc)+'_'+str(alpha)+'.pdf', transparent=True)
        pass



        pass

    def plot_equilibria_validation(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(2,2,  1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)


        ax1.set_xlabel('$\\alpha$ [-]')
        ax1.set_ylabel('$||\\Delta F||$')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$\\alpha$ [-]')
        ax2.set_ylabel('$||\\Delta F||$')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('$\\alpha$ [-]')
        ax3.set_ylabel('$||\\Delta F||$')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('$\\alpha$ [-]')
        ax4.set_ylabel('$||\\Delta F ||$')
        ax4.grid(True, which='both', ls=':')

        ylim = [1.0e-14,1.0e-12]
        xlim = [0,2*np.pi]
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)

        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax3.set_xlim(xlim)
        ax4.set_xlim(xlim)



        counter = 0
        plotFrequency = 10
        for accMagnitude in self.accelerationMagnitude:
            counter = counter + 1
            if accMagnitude < 0.01:
                for lagrangePointNr in self.lagrangePointNrs:
                    equilibria_df = load_equilibria_alpha('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                      + '_acceleration_' \
                                                      + str("{:7.6f}".format(accMagnitude)) + '_' + '0.000000_forward_equilibria.txt')

                    deviationList = []
                    alpha = []
                    listCounter = 0
                    for row in equilibria_df.iterrows():
                        if listCounter % plotFrequency == 0:
                            alpha.append(row[1][0])
                            potentialDeviation = potential_deviation(accMagnitude, row[1][0],row[1][1],row[1][2])
                            deviationList.append(potentialDeviation)
                        listCounter = listCounter + 1

                    if counter == 1:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax1.semilogy(alpha,deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax1.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')
                    if counter == 2:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax2.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax2.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')
                    if counter == 3:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax3.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax3.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')
                    if counter == 4:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax4.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax4.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')

            if accMagnitude > 0.01:
                for lagrangePointNr in self.lagrangePointNrs:
                    if lagrangePointNr < 3:

                        equilibria_df = load_equilibria_alpha('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                              + '_acceleration_' \
                                                              + str(
                            "{:7.6f}".format(accMagnitude)) + '_' + '0.000000_forward_equilibria.txt')

                        deviationList = []
                        alpha = []
                        listCounter = 0
                        for row in equilibria_df.iterrows():
                            if listCounter % plotFrequency == 0:
                                alpha.append(row[1][0])
                                potentialDeviation = potential_deviation(accMagnitude, row[1][0], row[1][1], row[1][2])
                                deviationList.append(potentialDeviation)
                            listCounter = listCounter + 1

                    if lagrangePointNr == 3:

                        dfCounter = 0
                        customLagrangeNrs = [3,4,5]

                        for customLagrangeNr in customLagrangeNrs:
                            for seed in self.seeds:
                                for continuation in self.continuations:

                                    if dfCounter == 0:
                                        total_df = load_equilibria_acceleration(
                                            '../../data/raw/equilibria/L' + str(customLagrangeNr) \
                                            + '_acceleration_' \
                                            + str("{:7.6f}".format(accMagnitude)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + continuation + '_equilibria.txt')
                                    if dfCounter > 0:

                                        equilibria_df = load_equilibria_acceleration('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                             + '_acceleration_'  \
                                                             + str("{:7.6f}".format(accMagnitude)) + '_' \
                                                             + str("{:7.6f}".format(seed)) + '_' + continuation +'_equilibria.txt')
                                        total_df = total_df.append(equilibria_df,ignore_index = True)

                                    dfCounter = dfCounter+1

                        total_df.sort_values('alpha',ascending=True)

                        deviationList = []
                        alpha = []
                        listCounter = 0
                        for row in equilibria_df.iterrows():
                            if listCounter % 12 * plotFrequency == 0:
                                alpha.append(row[1][0])
                                potentialDeviation = potential_deviation(accMagnitude, row[1][0], row[1][1], row[1][2])
                                deviationList.append(potentialDeviation)
                            listCounter = listCounter + 1

                    if counter == 1:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax1.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax1.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')

                    if counter == 2:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax2.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax2.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')
                    if counter == 3:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax3.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax3.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')
                    if counter == 4:
                        labelString = '$E_' + str(lagrangePointNr) + '$'
                        ax4.semilogy(alpha, deviationList, label=labelString, color=sns.color_palette('viridis', 5)[lagrangePointNr-1])
                        ax4.semilogy(alpha, 1e-13 * np.ones(len(alpha)), color='black', linewidth=1,
                                     linestyle='--')



        titleString = '$a_{lt} = $' + str(self.accelerationMagnitude[0])
        ax1.set_title(titleString)
        titleString = '$a_{lt} = $' + str(self.accelerationMagnitude[1])
        ax2.set_title(titleString)
        titleString = '$a_{lt} = $' + str(self.accelerationMagnitude[2])
        ax3.set_title(titleString)
        titleString = '$a_{lt} = $' + str(self.accelerationMagnitude[3])
        ax4.set_title(titleString)

        # Major ticks
        ticksLocators = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
        labels = ('0', '$\\frac{1}{2}\\pi$', '\\pi', '$\\frac{3}{2}\\pi$', '2\\pi')
        ax1.set_xticks(ticksLocators, minor=False)
        ax1.set_xticklabels(labels, fontdict=None, minor=False)

        ax2.set_xticks(ticksLocators, minor=False)
        ax2.set_xticklabels(labels, fontdict=None, minor=False)

        ax3.set_xticks(ticksLocators, minor=False)
        ax3.set_xticklabels(labels, fontdict=None, minor=False)

        ax4.set_xticks(ticksLocators, minor=False)
        ax4.set_xticklabels(labels, fontdict=None, minor=False)

        lgd = ax2.legend(frameon=True, loc='upper right',bbox_to_anchor=(1.22, 1.04))



        # load all files and put in a dataframe and sort them
        # Compute deviation for each point in the frame

        # Plot deviations over the whole alpha range
        # Plot number of iterations over the whole alpha

        fig.suptitle('Defect vector verification of $\\alpha$-varying equilibria contours',size=20)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if self.lowDPI:
            print('test')
            fig.savefig('../../data/figures/equilibria/contourVerification.png', transparent=True, dpi=self.dpi, bbox_extra_artists=(lgd), bbox_inches='tight')
        else:
            fig.savefig('../../data/figures/equilibria/contourVerification.pdf', transparent=True)

        pass

    def plot_equilibria_validation_propagation(self):
        fig = plt.figure(figsize=self.figSizeWide)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.set_xlabel('$\\alpha$ [-]')
        ax1.set_ylabel('$||\\Delta R||$')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('$\\alpha$ [-]')
        ax2.set_ylabel('$||\\Delta V||$')
        ax2.grid(True, which='both', ls=':')

        counter = 0
        plotFrequency = 1
        for lagrangePointNr in self.lagrangePointNrs:

             if lagrangePointNr == 1:
                 labelString = '$E_' + str(lagrangePointNr) + '$'
                 equilibria_df = load_equilibria_acceleration_deviation(
                     '../../data/raw/equilibria/equilibria_val/L' + str(lagrangePointNr) + '_acceleration_' + str(
                         "{:7.6f}".format(self.accelerationMagnitude)) + '_' + '0.000000_backward_equilibria_deviation.txt')
             if lagrangePointNr == 2:
                 labelString = '$E_' + str(lagrangePointNr) + '$'
                 equilibria_df = load_equilibria_acceleration_deviation('../../data/raw/equilibria/equilibria_val/L' + str(lagrangePointNr) \
                                                + '_acceleration_' \
                                                + str(
                "{:7.6f}".format(self.accelerationMagnitude)) + '_' + '0.000000_backward_equilibria_deviation.txt')

             if lagrangePointNr == 3:
                 labelString = '$E_' + str(lagrangePointNr) + '$'
                 equilibria_df = load_equilibria_acceleration_deviation(
                     '../../data/raw/equilibria/equilibria_val/L' + str(lagrangePointNr) \
                     + '_acceleration_' \
                     + str(
                         "{:7.6f}".format(self.accelerationMagnitude)) + '_' + '0.000000_backward_equilibria_deviation.txt')

             if lagrangePointNr == 4:
                 labelString = '$E_' + str(lagrangePointNr) + '$'
                 equilibria_df = load_equilibria_acceleration_deviation(
                     '../../data/raw/equilibria/equilibria_val/L' + str(lagrangePointNr) \
                     + '_acceleration_' \
                     + str(
                         "{:7.6f}".format(self.accelerationMagnitude)) + '_' + '180.000000_backward_equilibria_deviation.txt')

             if lagrangePointNr == 5:
                 labelString = '$E_' + str(lagrangePointNr) + '$'
                 equilibria_df = load_equilibria_acceleration_deviation(
                     '../../data/raw/equilibria/equilibria_val/L' + str(lagrangePointNr) \
                     + '_acceleration_' \
                     + str(
                         "{:7.6f}".format(self.accelerationMagnitude)) + '_' + '180.000000_backward_equilibria_deviation.txt')

             deviationPositionList = []
             deviationVelocityList = []
             alpha = []
             listCounter = 0
             for row in equilibria_df.iterrows():
                if listCounter % plotFrequency == 0:
                    alpha.append(row[1][0])
                    deviationPosition = np.sqrt(row[1][1] ** 2 + row[1][2] ** 2 + row[1][3] ** 2)
                    if deviationPosition < 1.0e-20:
                        deviationPosition = 1.0E-20
                    deviationVelocity = np.sqrt(row[1][4] ** 2 + row[1][5] ** 2 + row[1][6] ** 2)
                    if deviationVelocity < 1.0e-20:
                        deviationVelocity = 1.0E-20
                    deviationPositionList.append(deviationPosition)
                    deviationVelocityList.append(deviationVelocity)

                listCounter = listCounter + 1


             labelString = '$E_' + str(lagrangePointNr) + '$'
             ax1.semilogy(alpha, deviationPositionList, label=labelString,
             color=sns.color_palette('viridis', 5)[lagrangePointNr - 1])
             ax2.semilogy(alpha, deviationVelocityList, label=labelString,
             color=sns.color_palette('viridis', 5)[lagrangePointNr - 1])

        ylim = [1.0e-21, 1.0e-11]
        xlim = [0, 2 * np.pi]
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)


        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

        titleString = 'Position deviation after a synodic half period'
        ax1.set_title(titleString)
        titleString = 'Velocity deviation after a synodic half period'
        ax2.set_title(titleString)


        # Major ticks
        ticksLocators = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
        labels = ('0', '$\\frac{1}{2}\\pi$', '\\pi', '$\\frac{3}{2}\\pi$', '2\\pi')
        ax1.set_xticks(ticksLocators, minor=False)
        ax1.set_xticklabels(labels, fontdict=None, minor=False)
        ax1.semilogy(np.linspace(0,2*np.pi,num=100), 1e-12 * np.ones(100), color='black', linewidth=1, linestyle='--')


        ax2.set_xticks(ticksLocators, minor=False)
        ax2.set_xticklabels(labels, fontdict=None, minor=False)
        ax2.semilogy(np.linspace(0,2*np.pi,num=100), 1e-12 * np.ones(100), color='black', linewidth=1, linestyle='--')


        lgd = ax2.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.22, 1.04))



        suptitleString = 'Deviation validation of $\\alpha$-varying equilibria contours ($a_{lt} = $' + str(self.accelerationMagnitude) + ')'
        fig.suptitle(suptitleString, size=20)

        fig.tight_layout()
        fig.subplots_adjust(top=0.8,right=0.9)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/contourValidation.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/contourValidation.pdf', transparent=True)

        pass

    def plot_L3_phenomenon(self):
        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        fig.suptitle('Eigenvalue behaviour')
        ax0.set_title('$a_{lt} = 0.003$')
        ax1.set_title('$a_{lt} = 0.0105$')
        ax2.set_title('$a_{lt} = 0.0106$')
        ax3.set_title('$a_{lt} = 0.0107$')

        ax0.set_xlim([-1.05, -0.95])
        ax0.set_ylim([-0.5, 0.5])
        ax1.set_xlim([-1.05, -0.95])
        ax1.set_ylim([-0.5, 0.5])
        ax2.set_xlim([-1.05, -0.95])
        ax2.set_ylim([-0.5, 0.5])
        ax3.set_xlim([-1.05, -0.95])
        ax3.set_ylim([-0.5, 0.5])

        type1 = load_stability_data('../../data/raw/equilibria/stability_1_L3.txt')
        type2 = load_stability_data('../../data/raw/equilibria/stability_2_L3.txt')
        type3 = load_stability_data('../../data/raw/equilibria/stability_3_L3.txt')
        # type4 = load_stability_data('../../data/raw/equilibria/stability_4_L3.txt')

        ax0.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax0.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax0.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')

        ax1.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax1.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax1.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')

        ax2.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax2.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax2.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')

        ax3.scatter(type1['x'], type1['y'], color=self.plottingColors['SXC'], s=0.09, label='SxC')
        ax3.scatter(type2['x'], type2['y'], color=self.plottingColors['CXC'], s=0.09, label='CxC')
        ax3.scatter(type3['x'], type3['y'], color=self.plottingColors['MXM'], s=0.09, label='MxM')
        # ax.scatter(type4['x'], type4['y'], color=self.plottingColors['SXS'], s=0.09, label='SxS')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = []
        for equilibrium in self.lagrangePointNrs:
            if equilibrium == 1:
                lagrange_point_nrs.append('L1')
            if equilibrium == 2:
                lagrange_point_nrs.append('L2')
            if equilibrium == 3:
                lagrange_point_nrs.append('L3')
            if equilibrium == 4:
                lagrange_point_nrs.append('L4')
            if equilibrium == 5:
                lagrange_point_nrs.append('L5')

        lagrange_point_nrs = ['L1', 'L2', 'L3', 'L4', 'L5']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')

        ax0.legend(frameon=True, loc='lower left', markerscale=13)
        ax1.legend(frameon=True, loc='lower left', markerscale=13)
        ax2.legend(frameon=True, loc='lower left', markerscale=13)
        ax3.legend(frameon=True, loc='lower left', markerscale=13)




        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.ListedColormap(sns.color_palette("viridis", 100000 )),
            norm=plt.Normalize(vmin=0.0, vmax=2*np.pi))

        ax0.set_xlabel('$x$ [-]')
        ax0.set_ylabel('$y$ [-]')

        ax1.set_xlabel('$x$ [-]')
        ax1.set_ylabel('$y$ [-]')

        ax2.set_xlabel('$x$ [-]')
        ax2.set_ylabel('$y$ [-]')

        ax3.set_xlabel('$x$ [-]')
        ax3.set_ylabel('$y$ [-]')

        for accMag in self.accelerationMagnitude:
            for seed in self.seeds:
                for continuation in self.continuations:
                    for lagrangePointNr in self.lagrangePointNrs:
                        equilibria_df = load_equilibria_acceleration('../../data/raw/equilibria/L' + str(lagrangePointNr) \
                                                             + '_acceleration_'  \
                                                             + str("{:7.6f}".format(accMag)) + '_' \
                                                             + str("{:7.6f}".format(seed)) + '_' + continuation +'_equilibria.txt')


                        if len(equilibria_df['alpha']) > 1:
                            alpha = equilibria_df['alpha']

                        if accMag < 0.004:
                            ax0.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)

                        if accMag > 0.004 and accMag < 0.01055:
                            ax1.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)

                        if accMag > 0.01055 and accMag < 0.01065:
                            ax2.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)

                        if accMag > 0.01065 and accMag < 0.01075:
                            ax3.scatter(equilibria_df['x'], equilibria_df['y'], c=alpha, cmap="viridis",  s=0.1)



        suptitle = fig.suptitle('Switching phenomenon of equilibria contour at $L_3$')

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.90, right=0.98)

        # Create a colourbar right from the plots with equal height as two subplots
        upperRightPosition = ax1.get_position()
        lowerRightPosition = ax3.get_position()
        upperRightPoints = upperRightPosition.get_points()
        lowerRightPoints = lowerRightPosition.get_points()

        cb_x0 = upperRightPoints[1][0] + 0.015
        cb_y0 = lowerRightPoints[0][1]


        width_colourbar = (upperRightPoints[1][0] - upperRightPoints[0][0]) / 25
        height_colourbar = upperRightPoints[1][1] - lowerRightPoints[0][1]
        axes_colourbar = [cb_x0, cb_y0, width_colourbar, height_colourbar]

        sm.set_array([])

        cax = plt.axes(axes_colourbar)
        cbar0 = plt.colorbar(sm, cax=cax, label='$\\alpha$ [-]', ticks=self.cbarTicksAcc)
        cbar0.set_ticklabels(self.cbarTicksAccLabels)

        fig.subplots_adjust(hspace=0.4)
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.90)

        if self.lowDPI:
            fig.savefig('../../data/figures/equilibria/L3_phenomenon.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/equilibria/L3_phenomenon.pdf', transparent=True)



if __name__ == '__main__':


    ### Contours

    lagrange_point_nrs = [3]
    acceleration_magnitudes = [0.003, 0.0105, 0.0106, 0.0107]
    seeds = [0.0,180.0]
    continuations = ['backward','forward']
    alphas = [0.0]
    low_dpi = True

    # display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitudes, alphas,
    #                                                               seeds, continuations, low_dpi=low_dpi)
    # display_equilibria_validation.plot_L3_phenomenon()
    #
    # del display_equilibria_validation


    # lagrange_point_nrs = [1,2]
    # acceleration_magnitudes = [0.07,0.15, 0.2,0.25]
    #
    # display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitudes, alphas,
    #                                                             seeds, continuations, low_dpi=low_dpi)
    # display_equilibria_validation.plot_equilibria_acceleration_total_zoom()
    #
    # plt.close('all')
    #
    #
    # del display_equilibria_validation












    # low_dpi = True
    # lagrange_point_nrs = [2]
    # acceleration_magnitudes = [0.003, 0.00873, 0.07, 0.1, 0.15, 0.2, 0.25]
    # seeds = [0.0,180.0]
    # continuations = ['backward','forward']
    # alphas = [0.0]



    #acceleration_magnitudes = [0.003,0.1,0.25]


    #display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitudes,alphas, seeds, continuations, low_dpi=low_dpi)
    #display_equilibria_validation.plot_global_stability()
    #display_equilibria_validation.plot_global_eigenvalues()
    #display_equilibria_validation.plot_forbidden_region_test()
    #display_equilibria_validation.plot_equilibria_validation()

    #plt.close('all')


    #del display_equilibria_validation
     #lagrange_point_nrs = [1,2,3,4,5]
    # acceleration_magnitudes = [0.1]
    #
    # for acceleration_magnitude in acceleration_magnitudes:
    #      display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitude,alphas, seeds, continuations, low_dpi=low_dpi)
    #      display_equilibria_validation.plot_equilibria_validation_propagation()
    # #     display_equilibria_validation.plot_equilibria_acceleration()
    # #
    #      plt.close('all')
    #
    #      del display_equilibria_validation



    lagrange_point_nrs = [1, 2, 3, 4, 5]
    seeds = [0.0]
    continuations = ['forward']
    # alphas = [0, 60, 90, 120, 180, 240, 270, 300]
    #
    #
    #
    # for alpha in alphas:
    #     print(alpha)
    #     display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitudes,alpha, seeds, continuations, low_dpi=low_dpi)
    #     display_equilibria_validation.plot_equilibria_alpha()
    #
    #     plt.close('all')
    #
    #     del display_equilibria_validation
    #
    alphas = [0, 90, 180, 270]


    display_equilibria_validation = DisplayEquilibriaValidation(lagrange_point_nrs, acceleration_magnitudes, alphas,
                                                                seeds, continuations, low_dpi=low_dpi)
    display_equilibria_validation.plot_equilibria_alpha_total()
    #display_equilibria_validation.plot_equilibria_alpha_total_zoom()

    #plt.close('all')

    #del display_equilibria_validation






