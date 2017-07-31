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
import time

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, cr3bp_velocity


class DisplayFamiliesOfEqualEnergy:
    def __init__(self):
        pass

    def plot(self):
        colors = sns.color_palette("Blues", n_colors=6)

        # Plot: 3d overview
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']

        # Lagrange points and bodies
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       lagrange_points_df[lagrange_point_nr]['z'], color='grey')

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        y = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='black')
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='black')

        C = 3.1
        x_range = np.arange(0.7, 1.3, 0.001)
        y_range = np.arange(-0.3, 0.3, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)

        if Z.min() < 0:
            plt.contourf(X, Y, Z, 0, colors='black', alpha=0.05, zorder=1000)


        df = load_orbit('../../data/raw_equal_energy/horizontal_L1_577.txt')
        ax.plot(df['x'], df['y'], df['z'], color=colors[2], alpha=0.75)
        df = load_orbit('../../data/raw_equal_energy/horizontal_L2_760.txt')
        ax.plot(df['x'], df['y'], df['z'], color=colors[2], alpha=0.75)
        df = load_orbit('../../data/raw_equal_energy/vertical_L1_1163.txt')
        # ax.plot(df[df['z'] > 0]['x'], df[df['z'] > 0]['y'], df[df['z'] > 0]['z'], color=colors[5], alpha=1)
        ax.plot(df['x'], df['y'], df['z'], color=colors[5], alpha=0.75)
        df = load_orbit('../../data/raw_equal_energy/vertical_L2_1299.txt')
        ax.plot(df['x'], df['y'], df['z'], color=colors[5], alpha=0.75)
        df = load_orbit('../../data/raw_equal_energy/halo_L1_799.txt')
        ax.plot(df['x'], df['y'], df['z'], color=colors[4], alpha=0.75)
        df = load_orbit('../../data/raw_equal_energy/halo_L2_651.txt')
        ax.plot(df['x'], df['y'], df['z'], color=colors[4], alpha=0.75)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.set_zlabel('z [-]')
        ax.grid(True, which='both', ls=':')
        # ax.view_init(25, -60)

        ax.set_xlim([0.7, 1.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([-0.3, 0.3])

        # plt.show()
        fig.savefig('../../data/figures/family_of_equal_energy.png')
        tikz_save('../../data/figures/family_of_equal_energy.tex')
        plt.close()
        pass

if __name__ == '__main__':
    display_families_of_equal_energy = DisplayFamiliesOfEqualEnergy()
    display_families_of_equal_energy.plot()
