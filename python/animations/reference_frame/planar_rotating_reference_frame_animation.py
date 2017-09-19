import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sys.path.append('../../util')
from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
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


class SpatialOrbitsRotatingAnimation:
    def __init__(self, c_level):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.xLim = [-1, 1]
        self.yLim = self.xLim
        self.zLim = self.yLim
        self.orbitAlpha = 0.8
        self.orbitLinewidth = 5
        self.lagrangePointMarkerSize = 100
        self.orbitalBodiesEnlargementFactor = 2
        self.orbitColor = 'navy'
        self.cLevel = c_level
        self.t = []
        self.initialElevation = 0
        self.initialAzimuth = 0

        self.lines = []
        self.horizontalLyapunov = []
        self.verticalLyapunov = []
        self.halo = []
        self.timeText = ''  # Will become a plt.text-object

        print('C = ' + str(c_level))
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        current_time = self.t[i]
        print(current_time)
        self.timeText.set_text('$\|t\| \\approx$ {:.2f}'.format(round(abs(current_time), 2)))
        self.ax.view_init(elev=self.initialElevation, azim=self.initialAzimuth - current_time % 1 * 360)
        # for j, line in enumerate(self.lines):
        #     if j < 2:
        #         x = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['x'].tolist()
        #         y = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['y'].tolist()
        #         z = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['z'].tolist()
        #         pass
        #     if 2 <= j < 4:
        #         x = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['x'].tolist()
        #         y = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['y'].tolist()
        #         z = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['z'].tolist()
        #         pass
        #     if 4 <= j < 6:
        #         x = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['x'].tolist()
        #         y = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['y'].tolist()
        #         z = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['z'].tolist()
        #         pass
        #     line.set_data(x, y)
        #     line.set_3d_properties(z)
        #     pass
        return self.lines

    def animate(self):
        fig = plt.figure()
        self.ax = fig.gca()
        self.lines = [plt.plot([], [], color=self.orbitColor, alpha=self.orbitAlpha, marker='o', markevery=[-1])[0] for idx in range(6)]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = self.ax.text2D(0.05, 0.95, s='$\|t\| \\approx 0$', transform=self.ax.transAxes, size=self.timeTextSize)

        # Plot zero velocity surface
        x_range = np.arange(-5, 5, 0.001)
        y_range = np.arange(-5, 5, 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.cLevel)
        # self.ax.plot_surface(x_mesh, y_mesh, -z_mesh, alpha=0.1, linewidth=0,
        #                      cmap=matplotlib.colors.ListedColormap(sns.color_palette("Blues", n_colors=100)),
        #                      vmin=self.zLim[0], vmax=-z_mesh.min(), rstride=50, cstride=50)
        self.ax.plot_surface(x_mesh, y_mesh, -z_mesh, alpha=0.2, linewidth=0, color='black')
        self.ax.plot_wireframe(x_mesh, y_mesh, -z_mesh, alpha=1, linewidth=0.5, color='black', rstride=50, cstride=50)

        # Plot both primaries
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = self.orbitalBodiesEnlargementFactor * bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = self.orbitalBodiesEnlargementFactor * bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = self.orbitalBodiesEnlargementFactor * bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color='black', zorder=3)

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            self.ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                              lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x', s=self.lagrangePointMarkerSize, zorder=1)

        title = 'Rotating reference frame - Spatial overview at C = ' + str(self.cLevel)

        self.ax.set_xlim3d(self.xLim)
        self.ax.set_ylim3d(self.yLim)
        self.ax.set_zlim3d(self.zLim)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle(title, size=self.suptitleSize)

        # self.ax.elev = 60
        self.initialElevation = self.ax.elev
        self.initialAzimuth = self.ax.azim
        self.ax.set_aspect('equal')
        # plt.show()
        plt.axis('off')

        # Determine the maximum value of t
        t_max = 1
        print('Maximum value for t = ' + str(t_max) + ', animation t: = ')

        # Introduce a new time-vector for linearly spaced time throughout the animation
        # self.t = np.linspace(0, t_max, np.round(t_max / 0.05) + 1)
        self.t = np.linspace(0, t_max, np.round(t_max / 0.005) + 1)

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=len(self.t), interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        file_name = '../../../data/animations/reference_frame/spatial_rotating_reference_frame_' + str(self.cLevel) + '.mp4'
        animation_function.save(file_name, writer=animation_writer)


def worker(arg):
    # Wrapper function to execute the function 'animate' in instance of class 'SpatialOrbitsRotatingAnimation'
    obj = arg
    return obj.animate()


if __name__ == '__main__':
    maximum_number_of_processes = min(14, multiprocessing.cpu_count()-1)  # 14 processes is the maximum on the server, but perform a safety check on the available number of threads

    c_levels_times_100 = list(range(200, 305, 5))
    c_levels = [i / 100 for i in c_levels_times_100]

    list_of_objects = [SpatialOrbitsRotatingAnimation(c_level) for c_level in c_levels]  # Instantiate the classes
    pool = multiprocessing.Pool(min(maximum_number_of_processes, len(list_of_objects)))  # Start a pool of workers
    pool.imap_unordered(worker, (obj for obj in list_of_objects))  # Execute the tasks in parallel, regardless of order
    pool.close()  # No more tasks can be submitted to the pool
    pool.join()  # Wait for processes in pool to exit
