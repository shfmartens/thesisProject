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


class PlanarManifoldAnimation:
    def __init__(self, orbit_type, c_level, orbit_ids):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.xLim = [-5, 2]
        self.yLim = [-4, 4]
        self.zLim = self.yLim

        self.orbitType = orbit_type
        self.cLevel = c_level
        self.orbitIds = [orbit_ids[orbit_type][1][c_level], orbit_ids[orbit_type][2][c_level]]
        self.lines = []
        self.W_S_plus = []
        self.W_S_min = []
        self.W_U_plus = []
        self.W_U_min = []
        self.numberOfOrbitsPerManifold = 0
        self.timeText = plt.text(0, 0, 's')

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
        for j, line in enumerate(self.lines):
            if j < self.numberOfOrbitsPerManifold:
                x = self.W_S_plus[0].xs(j)['x'].tolist()
                y = self.W_S_plus[0].xs(j)['y'].tolist()
                z = self.W_S_plus[0].xs(j)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold <= j < self.numberOfOrbitsPerManifold * 2:
                x = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold)['x'].tolist()
                y = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold)['y'].tolist()
                z = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 2 <= j < self.numberOfOrbitsPerManifold * 3:
                x = self.W_U_plus[0].xs(j - self.numberOfOrbitsPerManifold * 2)['x'].tolist()
                y = self.W_U_plus[0].xs(j - self.numberOfOrbitsPerManifold * 2)['y'].tolist()
                z = self.W_U_plus[0].xs(j - self.numberOfOrbitsPerManifold * 2)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 3 <= j < self.numberOfOrbitsPerManifold * 4:
                x = self.W_U_min[0].xs(j - self.numberOfOrbitsPerManifold * 3)['x'].tolist()
                y = self.W_U_min[0].xs(j - self.numberOfOrbitsPerManifold * 3)['y'].tolist()
                z = self.W_U_min[0].xs(j - self.numberOfOrbitsPerManifold * 3)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 4 <= j < self.numberOfOrbitsPerManifold * 5:
                x = self.W_S_plus[1].xs(j - self.numberOfOrbitsPerManifold * 4)['x'].tolist()
                y = self.W_S_plus[1].xs(j - self.numberOfOrbitsPerManifold * 4)['y'].tolist()
                z = self.W_S_plus[1].xs(j - self.numberOfOrbitsPerManifold * 4)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 5 <= j < self.numberOfOrbitsPerManifold * 6:
                x = self.W_S_min[1].xs(j - self.numberOfOrbitsPerManifold * 5)['x'].tolist()
                y = self.W_S_min[1].xs(j - self.numberOfOrbitsPerManifold * 5)['y'].tolist()
                z = self.W_S_min[1].xs(j - self.numberOfOrbitsPerManifold * 5)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 6 <= j < self.numberOfOrbitsPerManifold * 7:
                x = self.W_U_plus[1].xs(j - self.numberOfOrbitsPerManifold * 6)['x'].tolist()
                y = self.W_U_plus[1].xs(j - self.numberOfOrbitsPerManifold * 6)['y'].tolist()
                z = self.W_U_plus[1].xs(j - self.numberOfOrbitsPerManifold * 6)['z'].tolist()
                pass
            if self.numberOfOrbitsPerManifold * 7 <= j < self.numberOfOrbitsPerManifold * 8:
                x = self.W_U_min[1].xs(j - self.numberOfOrbitsPerManifold * 7)['x'].tolist()
                y = self.W_U_min[1].xs(j - self.numberOfOrbitsPerManifold * 7)['y'].tolist()
                z = self.W_U_min[1].xs(j - self.numberOfOrbitsPerManifold * 7)['z'].tolist()
                pass
            line.set_data(x[:i], y[:i])
            line.set_3d_properties(z[:i])
        try:
            t = self.W_U_plus[1].xs(1).index.values[i]
            self.timeText.set_text('$\|t\| \\approx$ {:.2f}'.format(round(abs(t), 2)))
        except IndexError:
            pass

        return self.lines

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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

        self.lines = [plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)]
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])
        self.lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(self.numberOfOrbitsPerManifold)])

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = ax.text(0.9, 0.02, self.zLim[1], s='', transform=ax.transAxes, size=self.timeTextSize)

        # Plot zero velocity surface
        x_range = np.arange(-4.0, 2.0, 0.001)
        y_range = np.arange(-3.0, 3.0, 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, c_level)
        if z_mesh.min() < 0:
            plt.contourf(x_mesh, y_mesh, z_mesh, [z_mesh.min(), 0], colors='black', alpha=0.1)

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
            ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x')

        title = self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(c_level)

        ax.set_xlim3d(self.xLim)
        ax.set_ylim3d(self.yLim)
        ax.set_zlim3d(self.zLim)
        ax.set_xlabel('x [-]')
        ax.set_ylabel('y [-]')
        ax.set_zlabel('z [-]')

        ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle(title, size=self.suptitleSize)

        # Determine the maximum number of frames
        number_of_frames = 0
        for lagrange_point_idx in [0, 1]:
            for index in range(self.numberOfOrbitsPerManifold):
                number_of_frames = max(number_of_frames, len(self.W_S_plus[lagrange_point_idx].xs(index)['x']))
                number_of_frames = max(number_of_frames, len(self.W_S_min[lagrange_point_idx].xs(index)['x']))
                number_of_frames = max(number_of_frames, len(self.W_U_plus[lagrange_point_idx].xs(index)['x']))
                number_of_frames = max(number_of_frames, len(self.W_U_min[lagrange_point_idx].xs(index)['x']))

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=int(number_of_frames), interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        file_name = '../../../data/animations/manifolds/' + orbit_type + '_' + str(c_level) + '_3d.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':
    orbit_types = ['horizontal', 'vertical', 'halo']
    c_levels = [3.05, 3.1, 3.15]
    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for orbit_type in orbit_types:
        for c_level in c_levels:
            planar_manifold_animation = PlanarManifoldAnimation(orbit_type, c_level, orbit_ids)
            planar_manifold_animation.animate()
