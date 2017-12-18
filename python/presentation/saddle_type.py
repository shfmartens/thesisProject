import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pylab
sys.path.append('../util')
from load_data import load_orbit, load_manifold_refactored, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
sns.set_style("whitegrid")
params = {'text.latex.preamble': [r"\usepackage{lmodern}"],
          'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          'figure.figsize': (20 * (1 + np.sqrt(5)) / 2, 20),
          'axes.labelsize': 33,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22}
plt.rcParams.update(params)
plt.rcParams['animation.ffmpeg_path'] = '../../ffmpeg-git-20170607-64bit-static/ffmpeg'
import multiprocessing

class SaddleType:
    def __init__(self, c_level):
        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (
                MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
        self.suptitleSize = 20
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        # self.xlim = [-self.massParameter, 1.5]
        self.xlim = [0.6, 1-self.massParameter]
        self.ylim = [-0.25, 0.25]
        self.zlim = [-0.25, 0.25]
        self.cLevel = c_level
        self.lines = []
        self.timeText = ''  # Will become a plt.text-object
        print('C = ' + str(self.cLevel))
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[0],
                                              sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)]],
                               'limit': 'black'}
        pass

    def plot_saddle(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.add_subplot(111, projection='3d')
        # azim=150
        # elev=25
        # print(ax.azim)
        # ax.view_init(elev=elev, azim=azim)
        # ax.view_init(elev=elev)

        # Plot zero velocity surface
        x_range = np.arange(self.xlim[0], self.xlim[1], 0.01)
        y_range = np.arange(self.ylim[0], self.ylim[1], 0.01)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.cLevel)
        self.zlim = [-0.5, -min([min(i) for i in z_mesh])]
        print(z_mesh)
        threshold_z = 0.3
        for idx1, row in enumerate(z_mesh):
            for idx2, item in enumerate(row):
                if item > threshold_z:
                    z_mesh[idx1][idx2] = threshold_z
        # print(max([max(i) for i in z_mesh]))
        # print(min([min(i) for i in z_mesh]))

        if z_mesh.min() < 0:
            # plt.contour(x_mesh, y_mesh, z_mesh, [z_mesh.min(), 0], colors='black', alpha=0.3)
            # ax.contour(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r',
            #                 alpha=0.5)
            ax.plot_surface(x_mesh, y_mesh, -z_mesh, alpha=0.75, linewidth=0, cmap='viridis', zorder=-1,
                            vmin=self.zlim[0], vmax=self.zlim[1])
            # ax.plot_wireframe(x_mesh, y_mesh, -z_mesh, alpha=1, linewidth=0.5, color='black', rstride=50, cstride=50)

        # Lagrange points and bodies
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for idx, lagrange_point_nr in enumerate(lagrange_point_nrs):
            ax.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x', zorder=idx)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for idx, body in enumerate(bodies_df):
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='black', zorder=2+idx)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_zlim(self.zlim)

        # ax.set_title(str(azim))

        plt.axis('off')
        plt.show()
        pass


if __name__ == '__main__':
    c_levels_times_100 = list(range(200, 305, 5))
    c_levels = [i / 100 for i in c_levels_times_100]
    c_levels = [3.15]
    for c_level in c_levels:
        saddle_type = SaddleType(c_level)
        saddle_type.plot_saddle()