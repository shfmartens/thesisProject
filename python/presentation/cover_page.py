import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
from multiprocessing import Pool
import time
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored, \
    cr3bp_velocity


class CoverPage:
    def __init__(self, number_of_orbits_per_manifold=100, max_position_dev=1e-3, orbit_type='vertical'):
        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        self.suptitleSize = 20
        # self.figSize = (4, 4 * np.sqrt(2))
        self.figSize = (10, 7 * np.sqrt(2))
        # self.figSize = (10, 20)
        self.orbitType = orbit_type
        self.orbitTypeForTitle = orbit_type.capitalize()
        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.numberOfOrbitsPerManifold = number_of_orbits_per_manifold
        self.maximumPositionDeviation = max_position_dev

        df = pd.read_table('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
        df.columns = ['theta', 'taus', 'ts', 'xs', 'ys', 'zs', 'xdots', 'ydots', 'zdots', 'tauu', 'tu', 'xu', 'yu', 'zu', 'xdotu', 'ydotu', 'zdotu']
        df['dr'] = np.sqrt((df['xs'] - df['xu']) ** 2 + (df['ys'] - df['yu']) ** 2 + (df['zs'] - df['zu']) ** 2)
        df['dv'] = np.sqrt((df['xdots'] - df['xdotu']) ** 2 + (df['ydots'] - df['ydotu']) ** 2 + (df['zdots'] - df['zdotu']) ** 2)
        df = df.set_index('theta')

        self.thetaRangeList = sorted(list(df.index))
        self.minimumImpulse = df[df['dv'] != 0]
        self.numberOfPlotColorIndices = len(self.thetaRangeList)

        self.plotColorIndexBasedOnTheta = {theta: int(np.round(
            (theta - min(self.thetaRangeList)) / (max(self.thetaRangeList) - min(self.thetaRangeList)) * (
                    self.numberOfPlotColorIndices - 1))) for theta in self.thetaRangeList}

        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        self.colorPaletteStable = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        self.colorPaletteUnstable = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[0], sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[int((n_colors-1)/2)]],
                               'limit': 'black'}
        pass

    def plot_image_trajectories(self, theta):
        line_width_near_heteroclinic = 3

        print('Theta:')
        # for theta in self.thetaRangeList:
        print(theta)

        df_s = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(int(theta)) + '_full.txt')
        df_u = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(int(theta)) + '_full.txt')
        print(df_s)

        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(111, projection='3d')

        plot_alpha = 1

        # Plot near-heteroclinic connection
        index_near_heteroclinic_s = int(self.minimumImpulse.loc[theta]['taus'] * self.numberOfOrbitsPerManifold)
        index_near_heteroclinic_u = int(self.minimumImpulse.loc[theta]['tauu'] * self.numberOfOrbitsPerManifold)

        # for i in range(0, 5000, 50):
        #     ax0.plot(df_s.xs(i)['x'], df_s.xs(i)['y'],
        #              df_s.xs(i)['z'], color='g', alpha=plot_alpha,
        #              linewidth=0.5)
        #     ax0.plot(df_u.xs(i)['x'], df_u.xs(i)['y'],
        #              df_u.xs(i)['z'], color='r', alpha=plot_alpha,
        #              linewidth=0.5)

        # (T1)
        ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)

        # # (T4)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        #
        # # (T3)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'],
        #          -df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'],
        #          -df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # # (T2)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'],
        #          -df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'],
        #          -df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        # for body in ['Earth', 'Moon']:
        for body in ['Moon']:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax0.plot_surface(x, y, z, color='black')

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([max(df_s['x'].max(), df_u['x'].max()) - min(df_s['x'].min(), df_u['x'].min()),
                              max(df_s['y'].max(), df_u['y'].max()) - min(df_s['y'].min(), df_u['y'].min()),
                              max(df_s['z'].max(), df_u['z'].max()) - min(df_s['z'].min(), df_u['z'].min())]).max()

        Xb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.2 * (max(df_s['x'].max(), df_u['x'].max()) + min(df_s['x'].min(), df_u['x'].min()))
        Yb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(df_s['y'].max(), df_u['y'].max()) + min(df_s['y'].min(), df_u['y'].min()))
        Zb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(df_s['z'].max(), df_u['z'].max()) + min(df_s['z'].min(), df_u['z'].min()))
        # Comment or uncomment following both lines to test the fake bounding box:
        if self.orbitType != 'halo':
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax0.plot([xb], [yb], [zb], 'w')

        # azim=150
        # elev=25
        print(ax0.azim)
        print(ax0.elev)
        ax0.view_init(elev=15, azim=220)  # Single line
        # ax0.view_init(elev=20, azim=240)  # Manifold lines
        plt.tight_layout()
        plt.axis('off')
        # plt.show()

        plt.savefig('../../data/figures/cover_page.pdf', transparent=True)
        plt.close()
        pass

    def plot_image_trajectories_with_angle(self, theta):
        line_width_near_heteroclinic = 2

        print('Theta:')
        # for theta in self.thetaRangeList:
        print(theta)

        df_s = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(int(theta)) + '_full.txt')
        df_u = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(int(theta)) + '_full.txt')

        fig = plt.figure(figsize=self.figSize)
        ax0 = fig.add_subplot(111, projection='3d')

        plot_alpha = 1

        # Plot near-heteroclinic connection
        index_near_heteroclinic_s = int(self.minimumImpulse.loc[theta]['taus'] * self.numberOfOrbitsPerManifold)
        index_near_heteroclinic_u = int(self.minimumImpulse.loc[theta]['tauu'] * self.numberOfOrbitsPerManifold)

        portrait_s = []
        portrait_u = []
        for i in range(0, 5000, 10):
            portrait_s.append(df_s.xs(i).head(1))
            portrait_u.append(df_u.xs(i).tail(1))
            pass

        portrait_s = pd.concat(portrait_s)
        portrait_u = pd.concat(portrait_u)
        plt.plot(portrait_s['x'], portrait_s['y'], portrait_s['z'], color='g', linewidth=0.5)
        plt.plot(portrait_u['x'], portrait_u['y'], portrait_u['z'], color='r', linewidth=0.5)

        # for i in range(0, 5000, 10):
        #     ax0.scatter(df_s.xs(i).head(1)['x'], df_s.xs(i).head(1)['y'],
        #                 df_s.xs(i).head(1)['z'], color='g', s=0.1)
        #     ax0.scatter(df_u.xs(i).tail(1)['x'], df_u.xs(i).tail(1)['y'],
        #                 df_u.xs(i).tail(1)['z'], color='r', s=0.1)

        # (T1)
        ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)

        # # (T4)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
        #
        # # (T3)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'],
        #          -df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'],
        #          -df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # # (T2)
        # ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'],
        #          -df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)
        # ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'],
        #          -df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha,
        #          linewidth=line_width_near_heteroclinic)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        # for body in ['Moon']:
        for body in ['Earth', 'Moon']:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax0.plot_surface(x, y, z, color='black')

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax0.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                        lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([max(df_s['x'].max(), df_u['x'].max()) - min(df_s['x'].min(), df_u['x'].min()),
                              max(df_s['y'].max(), df_u['y'].max()) - min(df_s['y'].min(), df_u['y'].min()),
                              max(df_s['z'].max(), df_u['z'].max()) - min(df_s['z'].min(), df_u['z'].min())]).max()

        Xb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.2 * (max(df_s['x'].max(), df_u['x'].max()) + min(df_s['x'].min(), df_u['x'].min()))
        Yb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(df_s['y'].max(), df_u['y'].max()) + min(df_s['y'].min(), df_u['y'].min()))
        Zb = 0.3 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(df_s['z'].max(), df_u['z'].max()) + min(df_s['z'].min(), df_u['z'].min()))
        # Comment or uncomment following both lines to test the fake bounding box:
        if self.orbitType != 'halo':
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax0.plot([xb], [yb], [zb], 'w')

        arc_length = 0.2
        surface_alpha = 0.1
        surface_color = 'black'

        x_range = np.arange(1 - self.massParameter - arc_length * 0.75, 1 - self.massParameter, 0.001)
        z_range = np.arange(-arc_length * 0.75, arc_length * 0.75, 0.001)
        x_mesh, z_mesh = np.meshgrid(x_range, z_range)
        y_mesh = -abs(x_mesh-(1-self.massParameter))/np.tan(35*np.pi/180)

        # Plot poincare plane
        ax0.plot_surface(x_mesh, y_mesh, z_mesh, alpha=surface_alpha, color=surface_color)
        ax0.plot_wireframe(x_mesh, y_mesh, z_mesh, color=surface_color, rstride=500, cstride=500, linewidth=1)

        # Plot line at angle
        plt.plot([1 - self.massParameter, np.min(x_mesh)], [0, np.min(y_mesh)], [0, 0], color='k', linestyle=':',
                 linewidth=1)

        # Plot line orthogonal to collinear points
        plt.plot([1 - self.massParameter, 1 - self.massParameter], [0, np.min(y_mesh)], [0, 0], color='k',
                 linestyle=':', linewidth=1)

        # Plot line from primary to L2
        plt.plot([-self.massParameter, lagrange_points_df['L2']['x']], [0, 0], [0, 0], color='k', linestyle=':',
                 linewidth=1)

        # Indicate connection on plane
        ax0.scatter(df_u.xs(index_near_heteroclinic_u).tail(1)['x'], df_u.xs(index_near_heteroclinic_u).tail(1)['y'],
                    df_u.xs(index_near_heteroclinic_u).tail(1)['z'], s=20,
                    linewidth=line_width_near_heteroclinic, facecolors='none', edgecolors='r')

        # Set viewpoint
        ax0.set_xlim(lagrange_points_df['L1']['x'], lagrange_points_df['L2']['x'])
        ax0.set_ylim(-0.15, 0.05)
        ax0.set_zlim(-0.1, 0.1)
        print(ax0.azim)
        print(ax0.elev)
        # ax0.view_init(elev=15, azim=220)  # Single line
        ax0.view_init(elev=5, azim=220)  # View from Earth
        # ax0.view_init(elev=5, azim=340)  # View from outside
        plt.tight_layout()
        fig.subplots_adjust(right=1.1)
        plt.axis('off')
        # plt.show()
        plt.savefig('../../data/figures/cover_page_angle.pdf', transparent=True)
        plt.close()
        pass


if __name__ == '__main__':
    max_position_deviation = {100: 1e-3, 1000: 1e-4, 5000: 1e-4}

    orbit_type = 'vertical'

    number_of_orbits_per_manifold = 5000
    cover_page = CoverPage(number_of_orbits_per_manifold=number_of_orbits_per_manifold,
                           max_position_dev=max_position_deviation[number_of_orbits_per_manifold],
                           orbit_type=orbit_type)

    # cover_page.plot_image_trajectories(-125.0)
    cover_page.plot_image_trajectories_with_angle(-125.0)
