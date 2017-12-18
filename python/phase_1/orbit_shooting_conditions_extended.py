import numpy as np
import pandas as pd
import json
import matplotlib
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
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, load_initial_conditions_incl_M, load_manifold


class OrbitShootingConditions:
    def __init__(self, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150

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
        pass

    def plot_2d_shooting_conditions(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca()

        orbit_types = ['horizontal', 'vertical', 'halo']
        lagrange_point_nrs = [1, 2]

        x = []
        z = []
        ydot = []
        c = []
        for idx, orbit_type in enumerate(orbit_types):
            for lagrange_point_nr in lagrange_point_nrs:
                if orbit_type == 'vertical':
                    initial_conditions_file_path = '../../data/raw/orbits/extended/L' + str(
                        lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
                else:
                    initial_conditions_file_path = '../../data/raw/orbits/L' + str(
                        lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'

                initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

                plot_label = orbit_type.capitalize()
                if plot_label == 'Horizontal' or plot_label == 'Vertical':
                    plot_label += ' Lyapunov'

                x.extend(list(initial_conditions_incl_m_df[2].values))
                z.extend(list(initial_conditions_incl_m_df[4].values))
                c.extend(list(initial_conditions_incl_m_df[0].values))

        sc = ax.scatter(x, z, c=c, cmap='viridis', s=20)
        cb = plt.colorbar(sc)
        cb.set_label('$C \enskip [-]$')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.contourf(x, z, y, colors='black')

        ax.legend(frameon=True, loc='upper right')
        ax.set_xlabel('x [-]')
        ax.set_ylabel('z [-]')
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle('$L_1, L_2$ - Shooting conditions for H-L, halo, and V-L', size=self.suptitleSize)
        if self.lowDPI:
            fig.savefig('../../data/figures/orbits/extended/orbit_shooting_conditions_2d.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/orbits/extended/orbit_shooting_conditions_2d.pdf', transparent=True)
        pass

    def plot_3d_shooting_conditions(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca(projection='3d')

        orbit_types = ['horizontal', 'vertical', 'halo']
        lagrange_point_nrs = [1, 2]

        lines = []
        linewidth = 2
        for lagrange_point_nr in lagrange_point_nrs:
            for idx, orbit_type in enumerate(orbit_types):
                initial_conditions_file_path = '../../data/raw/orbits/L' + str(
                    lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
                initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

                plot_label = orbit_type.capitalize()
                if plot_label == 'Horizontal' or plot_label == 'Vertical':
                    plot_label += ' Lyapunov'

                line, = ax.plot(initial_conditions_incl_m_df[2].values, initial_conditions_incl_m_df[6].values,
                                initial_conditions_incl_m_df[4].values, label=plot_label, linewidth=linewidth,
                                color=self.plottingColors['tripleLine'][idx])
                lines.append(line)

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2', 'L3']
        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                       color='black', marker='x')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='black')


        # print(ax.elev)
        # print(ax.azim)
        # ax.view_init(20, -75)
        plt.plot(ax.get_xlim(), [0, 0], 'black', linewidth=0.5)
        ax.legend(frameon=True, loc='upper right', handles=lines[:3])
        ax.set_xlabel('x [-]')
        ax.set_ylabel('$\dot{y}$ [-]')
        ax.set_zlabel('z [-]')
        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle('$L_1, L_2$ - Shooting conditions', size=self.suptitleSize)
        fig.savefig('../../data/figures/orbits/orbit_shooting_conditions_3d.pdf', transparent=True)
        pass

    def plot_2d_3d_shooting_conditions(self):
        fig = plt.figure(figsize=self.figSize)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)

        orbit_types = ['horizontal', 'vertical', 'halo']
        lagrange_point_nrs = [1, 2]

        x = []
        z = []
        c = []
        lines = []
        linewidth = 2
        for lagrange_point_nr in lagrange_point_nrs:
            for idx, orbit_type in enumerate(orbit_types):
                initial_conditions_file_path = '../../data/raw/orbits/L' + str(
                    lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
                initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

                plot_label = orbit_type.capitalize()
                if plot_label == 'Horizontal' or plot_label == 'Vertical':
                    plot_label += ' Lyapunov'

                line, = ax1.plot(initial_conditions_incl_m_df[2].values, initial_conditions_incl_m_df[6].values,
                                 initial_conditions_incl_m_df[4].values, label=plot_label, linewidth=linewidth,
                                 color=self.plottingColors['tripleLine'][idx])
                # ax2.plot(initial_conditions_incl_m_df[2].values, initial_conditions_incl_m_df[4].values,
                #          linewidth=linewidth, color=self.plottingColors['tripleLine'][idx])
                lines.append(line)
                x.extend(list(initial_conditions_incl_m_df[2].values))
                z.extend(list(initial_conditions_incl_m_df[4].values))
                c.extend(list(initial_conditions_incl_m_df[0].values))

        sc = ax2.scatter(x, z, c=c, cmap='viridis', s=10)
        cb = plt.colorbar(sc)
        cb.set_label('$C \enskip [-]$')

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax1.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')
            ax2.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['z'],
                        color='black', marker='x')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_surface(x, y, z, color='black')
            ax2.contourf(x, z, y, colors='black')

        # print(ax1.elev)
        # print(ax1.azim)
        # ax1.view_init(10, -40)
        # plt.plot(ax1.get_xlim(), [0, 0], 'black', linewidth=0.5)
        # plt.plot(ax2.get_xlim(), [0, 0], 'black', linewidth=0.5)
        ax1.legend(frameon=True, loc='upper right', handles=lines[:3])
        ax1.set_xlabel('x [-]')
        ax1.set_ylabel('$\dot{y}$ [-]')
        ax1.set_zlabel('z [-]')

        ax1.grid(True, which='both', ls=':')
        ax2.set_xlabel('x [-]')
        ax2.set_ylabel('z [-]')
        ax2.grid(True, which='both', ls=':')
        ax2.set_ylim([-0.4, 0.8])

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle('$L_1, L_2$ - Shooting conditions', size=self.suptitleSize)
        fig.savefig('../../data/figures/orbits/orbit_shooting_conditions_2d_3d.pdf', transparent=True)
        pass


if __name__ == '__main__':
    low_dpi = False

    orbit_shooting_conditions = OrbitShootingConditions(low_dpi=low_dpi)
    orbit_shooting_conditions.plot_2d_shooting_conditions()
    # orbit_shooting_conditions.plot_3d_shooting_conditions()
    # orbit_shooting_conditions.plot_2d_3d_shooting_conditions()
    # plt.show()
