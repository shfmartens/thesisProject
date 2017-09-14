import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation
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


class SpatialManifoldsRotatingAnimation:
    def __init__(self, orbit_type, c_level, orbit_ids):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.xLim = [-4, 4]
        self.yLim = [-4, 4]
        self.zLim = self.yLim
        self.orbitAlpha = 0.8
        self.orbitLinewidth = 2
        self.orbitColor = 'navy'

        self.orbitType = orbit_type
        self.cLevel = c_level
        self.orbitIds = [orbit_ids[orbit_type][1][c_level], orbit_ids[orbit_type][2][c_level]]
        self.lines = []
        self.W_S_plus = []
        self.W_S_min = []
        self.W_U_plus = []
        self.W_U_min = []
        self.t = []
        self.numberOfOrbitsPerManifold = 0
        self.lagrangePointMarkerSize = 100
        self.timeText = ''  # Will become a plt.text-object

        self.orbitTypeForTitle = orbit_type.capitalize()
        if self.orbitTypeForTitle == ('Horizontal' or 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        print(self.orbitTypeForTitle + ' at C = ' + str(c_level))
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
        for j, line in enumerate(self.lines):
            if j < self.numberOfOrbitsPerManifold:
                temp_df = self.W_U_min[0].xs(j)
                pass
            if self.numberOfOrbitsPerManifold <= j < self.numberOfOrbitsPerManifold * 2:
                temp_df = self.W_S_plus[0].xs(j - self.numberOfOrbitsPerManifold)
                pass
            if self.numberOfOrbitsPerManifold * 2 <= j < self.numberOfOrbitsPerManifold * 3:
                temp_df = self.W_U_plus[0].xs(j - self.numberOfOrbitsPerManifold * 2)
                pass
            if self.numberOfOrbitsPerManifold * 3 <= j < self.numberOfOrbitsPerManifold * 4:
                temp_df = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold * 3)
                pass
            if self.numberOfOrbitsPerManifold * 4 <= j < self.numberOfOrbitsPerManifold * 5:
                temp_df = self.W_U_min[1].xs(j - self.numberOfOrbitsPerManifold * 4)
                pass
            if self.numberOfOrbitsPerManifold * 5 <= j < self.numberOfOrbitsPerManifold * 6:
                temp_df = self.W_S_plus[1].xs(j - self.numberOfOrbitsPerManifold * 5)
                pass
            if self.numberOfOrbitsPerManifold * 6 <= j < self.numberOfOrbitsPerManifold * 7:
                temp_df = self.W_S_min[1].xs(j - self.numberOfOrbitsPerManifold * 6)
                pass
            if self.numberOfOrbitsPerManifold * 7 <= j < self.numberOfOrbitsPerManifold * 8:
                temp_df = self.W_U_plus[1].xs(j - self.numberOfOrbitsPerManifold * 7)
                pass
            x = temp_df[abs(temp_df.index) <= current_time]['x'].tolist()
            y = temp_df[abs(temp_df.index) <= current_time]['y'].tolist()
            z = temp_df[abs(temp_df.index) <= current_time]['z'].tolist()
            line.set_data(x, y)
            line.set_3d_properties(z)
        return self.lines

    def animate(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        self.W_S_plus = [load_manifold('../../../data/raw/manifolds/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitIds[0]) + '_W_S_plus.txt'),
                         load_manifold('../../../data/raw/manifolds/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitIds[1]) + '_W_S_plus.txt')]
        self.W_S_min = [load_manifold('../../../data/raw/manifolds/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitIds[0]) + '_W_S_min.txt'),
                        load_manifold('../../../data/raw/manifolds/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitIds[1]) + '_W_S_min.txt')]
        self.W_U_plus = [load_manifold('../../../data/raw/manifolds/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitIds[0]) + '_W_U_plus.txt'),
                         load_manifold('../../../data/raw/manifolds/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitIds[1]) + '_W_U_plus.txt')]
        self.W_U_min = [load_manifold('../../../data/raw/manifolds/L' + str(1) + '_' + self.orbitType + '_' + str(self.orbitIds[0]) + '_W_U_min.txt'),
                        load_manifold('../../../data/raw/manifolds/L' + str(2) + '_' + self.orbitType + '_' + str(self.orbitIds[1]) + '_W_U_min.txt')]

        self.numberOfOrbitsPerManifold = len(set(self.W_S_plus[0].index.get_level_values(0)))
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        self.lines = [plt.plot([], [], color=color_palette_red[idx], alpha=self.orbitAlpha)[0] for idx in
                      range(self.numberOfOrbitsPerManifold)]
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx], alpha=self.orbitAlpha)[0] for idx in
                           range(self.numberOfOrbitsPerManifold)])

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = self.ax.text2D(0.05, 0.05, s='$\|t\| \\approx 0$', transform=self.ax.transAxes, size=self.timeTextSize)

        # Plot zero velocity surface
        x_range = np.arange(self.xLim[0], self.xLim[1], 0.001)
        y_range = np.arange(self.yLim[0], self.yLim[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, c_level)
        if z_mesh.min() < 0:
            plt.contour(x_mesh, y_mesh, z_mesh, [z_mesh.min(), 0], colors='black', alpha=0.3)

        # Plot both orbits
        for k in range(2):
            orbit_df = load_orbit('../../../data/raw/orbits/L' + str(k+1) + '_' + self.orbitType + '_' + str(self.orbitIds[k]) + '.txt')
            plt.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColor, alpha=self.orbitAlpha, linewidth=self.orbitLinewidth)

        # Plot both primaries
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color='black')

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            self.ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                              lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x', s=self.lagrangePointMarkerSize)

        title = self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(c_level)

        self.ax.set_xlim3d(self.xLim)
        self.ax.set_ylim3d(self.yLim)
        self.ax.set_zlim3d(self.zLim)
        self.ax.set_xlabel('x [-]')
        self.ax.set_ylabel('y [-]')
        self.ax.set_zlabel('z [-]')

        self.ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle(title, size=self.suptitleSize)

        # Fix overlap between labels and ticks
        self.ax.xaxis._axinfo['label']['space_factor'] = 4.0
        self.ax.yaxis._axinfo['label']['space_factor'] = 4.0
        self.ax.zaxis._axinfo['label']['space_factor'] = 4.0

        self.initialElevation = self.ax.elev
        self.initialAzimuth = self.ax.azim

        # Determine the maximum value of t
        t_max = 0
        for lagrange_point_idx in [0, 1]:
            for index in range(self.numberOfOrbitsPerManifold):
                t_max = max(t_max, abs(self.W_S_plus[lagrange_point_idx].xs(index).tail(1).index.values[0]))
                t_max = max(t_max, abs(self.W_S_min[lagrange_point_idx].xs(index).tail(1).index.values[0]))
                t_max = max(t_max, abs(self.W_U_plus[lagrange_point_idx].xs(index).tail(1).index.values[0]))
                t_max = max(t_max, abs(self.W_U_min[lagrange_point_idx].xs(index).tail(1).index.values[0]))
        print('Maximum value for t = ' + str(t_max) + ', animation t: = ')

        # Introduce a new time-vector for linearly spaced time throughout the animation
        self.t = np.linspace(0, t_max, np.round(t_max / 0.01) + 1)

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=len(self.t), interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        file_name = '../../../data/animations/manifolds/spatial_manifolds_rotating_' + orbit_type + '_' + str(c_level) + '.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':
    orbit_types = ['horizontal', 'vertical', 'halo']
    c_levels = [3.05, 3.1, 3.15]
    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for orbit_type in orbit_types:
        for c_level in c_levels:
            spatial_manifolds_rotating_animation = SpatialManifoldsRotatingAnimation(orbit_type, c_level, orbit_ids)
            spatial_manifolds_rotating_animation.animate()
