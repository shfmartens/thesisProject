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
#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util/')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, load_initial_conditions_incl_M, load_manifold, cr3bp_velocity


class DisplayFamiliesOfEqualEnergy:
    def __init__(self, low_dpi=False):
        self.lowDPI = low_dpi
        self.dpi = 150
        pass

    def plot(self):
        colors = sns.color_palette("Blues", n_colors=6)

        # Plot: 3d overview
        fig = plt.figure(figsize=(7 * (1 + np.sqrt(5)) / 2, 7))
        ax = fig.gca(projection='3d')

        # Plot both primaries
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='black')

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x',
                         s=50)


        # ax.annotate('Moon', xy=(-0.002,0.004),
        #             xytext=(-0.002, 0.04), fontsize=20, ha = 'center', va = 'top',
        #             arrowprops=dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'),
        #             )
        # ax.annotate('$L_1$', xy=(-0.023, 0.012),
        #             xytext=(-0.023, 0.04), fontsize=20, ha='center', va='top',
        #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        #             )
        # ax.annotate('$L_2$', xy=(0.023, -0.004),
        #             xytext=(0.023, 0.04), fontsize=20, ha='center', va='top',
        #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        #             )
        # params = {'legend.fontsize': 16}
        # plt.rcParams.update(params)

        C = 3.1
        # x_range = np.arange(0.7, 1.3, 0.001)
        # y_range = np.arange(-0.3, 0.3, 0.001)
        x_range = np.arange(0.8, 1.2, 0.001)
        y_range = np.arange(-0.2, 0.2, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)

        if Z.min() < 0:
            plt.contourf(X, Y, Z, 0, colors='black', alpha=0.05, zorder=1000)

        linewidth=2
        df = load_orbit('../../data/raw_equal_energy/horizontal_L1_577.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[0], alpha=0.75, linestyle='-', label='Horizontal Lyapunov', linewidth=linewidth)

        df = load_orbit('../../data/raw_equal_energy/halo_L1_799.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[2], alpha=0.75, linestyle='-', label='Halo', linewidth=linewidth)

        df = load_orbit('../../data/raw_equal_energy/vertical_L1_1163.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[1], alpha=0.75, linestyle='-', label='Vertical Lyapunov', linewidth=linewidth)

        ax.legend(frameon=True, loc='lower right')

        df = load_orbit('../../data/raw_equal_energy/horizontal_L2_760.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[0], alpha=0.75, linestyle='-', linewidth=linewidth)
        df = load_orbit('../../data/raw_equal_energy/vertical_L2_1299.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[1], alpha=0.75, linestyle='-', linewidth=linewidth)
        df = load_orbit('../../data/raw_equal_energy/halo_L2_651.txt')
        ax.plot(df['x'], df['y'], df['z'], color=sns.color_palette("viridis", 3)[2], alpha=0.75, linestyle='-', linewidth=linewidth)

        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.set_zlabel('z [-]')
        ax.grid(True, which='both', ls=':')
        # ax.view_init(25, -60)
        ax.view_init(20, -60)
        # ax.set_xlim([0.7, 1.3])
        # ax.set_ylim([-0.3, 0.3])
        # ax.set_zlim([-0.3, 0.3])
        ax.set_xlim([0.8, 1.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([-0.2, 0.2])
        plt.tight_layout()


        # plt.show()
        # fig.savefig('../../../data/figures/family_of_equal_energy.png')
        if self.lowDPI:
            fig.savefig('../../data/figures/new_family_of_equal_energy.png', transparent=True, dpi=self.dpi)
        else:
            fig.savefig('../../data/figures/new_family_of_equal_energy.pdf', transparent=True)
        # tikz_save('../../../data/figures/family_of_equal_energy.tex')
        plt.close()
        pass


if __name__ == '__main__':
    low_dpi = True

    display_families_of_equal_energy = DisplayFamiliesOfEqualEnergy(low_dpi=low_dpi)
    display_families_of_equal_energy.plot()
