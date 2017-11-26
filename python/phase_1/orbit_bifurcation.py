import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
import time
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, load_initial_conditions_incl_M, load_manifold


class OrbitBifurcation:
    def __init__(self):
        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        n_colors = 3
        n_colors_l = 6
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               # 'lambda1': blues[40],
                               # 'lambda2': greens[50],
                               # 'lambda3': blues[90],
                               # 'lambda4': blues[90],
                               # 'lambda5': greens[70],
                               # 'lambda6': blues[60],
                               # 'singleLine': blues[80],
                               # 'doubleLine': [greens[50], blues[80]],
                               # 'tripleLine': [blues[40], greens[50], blues[80]],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}
        self.suptitleSize = 20
        self.horizontalBifurcations = [181, 938, 1233]
        self.axialMaxId = 937
        self.haloMaxId = 2443
        self.verticalBifurcations = [2271]
        pass

    def plot_horizontal_to_halo(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        horizontal_color = self.plottingColors['tripleLine'][2]
        halo_color = self.plottingColors['tripleLine'][0]

        # Plot bifurcations
        line_width = 2
        plot_alpha = 1
        df = load_orbit('../../data/raw/orbits/L1_horizontal_' + str(self.horizontalBifurcations[0]) + '.txt')
        l1, = ax1.plot(df['x'], df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=1, label='Horizontal Lyapunov')
        ax1.plot(df['x'], df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax2.plot(df['x'], df['y'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax3.plot(df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax4.plot(df['x'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)

        number_of_orbits = 10
        line_width = 1
        plot_alpha = 0.5
        for i in [int(i) for i in (np.linspace(1, self.haloMaxId, number_of_orbits))]:
            df = load_orbit('../../data/raw/orbits/L1_halo_' + str(i) + '.txt')
            l2, = ax1.plot(df['x'], df['y'], df['z'], color=halo_color, alpha=plot_alpha, linewidth=line_width, label='Halo')
            ax2.plot(df['x'], df['y'], color=halo_color, alpha=plot_alpha, linewidth=line_width)
            ax3.plot(df['y'], df['z'], color=halo_color, alpha=plot_alpha, linewidth=line_width)
            ax4.plot(df['x'], df['z'], color=halo_color, alpha=plot_alpha, linewidth=line_width)

            # Lagrange points and bodies
            lagrange_points_df = load_lagrange_points_location()
            lagrange_point_nrs = ['L1', 'L2']
            for lagrange_point_nr in lagrange_point_nrs:
                ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
                ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                            color='black', marker='x')
                ax3.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')
                ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                            color='black', marker='x')

            bodies_df = load_bodies_location()
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            for body in ['Moon']:
                x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
                y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
                z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
                ax1.plot_surface(x, y, z, color='black')
                ax2.contourf(x, y, z, colors='black')
                ax3.contourf(y, z, x, colors='black')
                ax4.contourf(x, z, y, colors='black')

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.set_zlabel('z [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('y [-]')
        ax3.set_ylabel('z [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('x [-]')
        ax4.set_ylabel('z [-]')
        ax4.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        ax1.legend(frameon=True, handles=[l1, l2])
        plt.suptitle('$L_1$ Bifurcation - Halo family connecting to horizontal Lyapunov orbits', size=self.suptitleSize)
        plt.savefig('../../data/figures/orbits/L1_bifurcation_halo.pdf', transparent=True)
        plt.close()
        pass

    def plot_horizontal_to_axial_to_vertical(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        horizontal_color = self.plottingColors['tripleLine'][2]
        axial_color = self.plottingColors['tripleLine'][0]
        vertical_color = self.plottingColors['tripleLine'][1]

        # Plot bifurcations
        line_width = 2
        plot_alpha = 1
        df = load_orbit('../../data/raw/orbits/L1_horizontal_' + str(self.horizontalBifurcations[1]) + '.txt')
        l1, = ax1.plot(df['x'], df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=1, label='Horizontal Lyapunov')
        ax1.plot(df['x'], df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax2.plot(df['x'], df['y'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax3.plot(df['y'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)
        ax4.plot(df['x'], df['z'], color=horizontal_color, alpha=plot_alpha, linewidth=line_width)

        number_of_orbits = 10
        line_width = 1
        plot_alpha = 0.5
        for i in [int(i) for i in (np.linspace(100, self.axialMaxId, number_of_orbits))]:
            df = load_orbit('../../data/raw/orbits/L1_axial_' + str(i) + '.txt')
            l2, = ax1.plot(df['x'], df['y'], df['z'], color=axial_color, alpha=plot_alpha, linewidth=line_width, label='Axial')
            ax2.plot(df['x'], df['y'], color=axial_color, alpha=plot_alpha, linewidth=line_width)
            ax3.plot(df['y'], df['z'], color=axial_color, alpha=plot_alpha, linewidth=line_width)
            ax4.plot(df['x'], df['z'], color=axial_color, alpha=plot_alpha, linewidth=line_width)

        # Plot bifurcations
        line_width = 2
        plot_alpha = 1
        df = load_orbit('../../data/raw/orbits/L1_vertical_' + str(self.verticalBifurcations[0]) + '.txt')
        l3, = ax1.plot(df['x'], df['y'], df['z'], color=vertical_color, alpha=plot_alpha, linewidth=1, label='Vertical Lyapunov')
        ax1.plot(df['x'], df['y'], df['z'], color=vertical_color, alpha=plot_alpha, linewidth=line_width)
        ax2.plot(df['x'], df['y'], color=vertical_color, alpha=plot_alpha, linewidth=line_width)
        ax3.plot(df['y'], df['z'], color=vertical_color, alpha=plot_alpha, linewidth=line_width)
        ax4.plot(df['x'], df['z'], color=vertical_color, alpha=plot_alpha, linewidth=line_width)

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        color='black', marker='x')
            ax3.scatter(lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')
            ax4.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for body in ['Moon']:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_surface(x, y, z, color='black')
            ax2.contourf(x, y, z, colors='black')
            ax3.contourf(y, z, x, colors='black')
            ax4.contourf(x, z, y, colors='black')

        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('y [-]')
        ax1.set_zlabel('z [-]')
        ax1.grid(True, which='both', ls=':')

        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('y [-]')
        ax2.grid(True, which='both', ls=':')

        ax3.set_xlabel('y [-]')
        ax3.set_ylabel('z [-]')
        ax3.grid(True, which='both', ls=':')

        ax4.set_xlabel('x [-]')
        ax4.set_ylabel('z [-]')
        ax4.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        ax1.legend(frameon=True, handles=[l1, l2, l3])
        plt.suptitle('$L_1$ Bifurcation - Axial family connecting horizontal and vertical Lyapunov orbits', size=self.suptitleSize)
        plt.savefig('../../data/figures/orbits/L1_bifurcation_axial.pdf', transparent=True)
        plt.close()
        pass


if __name__ == '__main__':
    orbit_bifurcation = OrbitBifurcation()
    orbit_bifurcation.plot_horizontal_to_halo()
    orbit_bifurcation.plot_horizontal_to_axial_to_vertical()
    # plt.show()
