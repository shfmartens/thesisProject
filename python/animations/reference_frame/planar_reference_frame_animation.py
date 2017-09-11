import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
sys.path.append('../../util')
from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
import seaborn as sns
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


class PlanarReferenceFrameAnimation:
    def __init__(self):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.orbitAlpha = 0.8
        self.orbitLinewidth = 2
        self.orbitColor = 'navy'
        self.xLim = [-1, 1]
        self.yLim = [-1, 1]
        self.cLevel = 3  # TODO change

        self.lines = []
        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
        self.timeText = plt.text(0, 0, 's')
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        for j, line in enumerate(self.lines):
            if j == 0:
                x = self.W_S_plus[0].xs(j)['x'].tolist()
                y = self.W_S_plus[0].xs(j)['y'].tolist()
                pass
            if j == 1:
                x = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold)['x'].tolist()
                y = self.W_S_min[0].xs(j - self.numberOfOrbitsPerManifold)['y'].tolist()
                pass
            line.set_data(x[:i], y[:i])
        try:
            t = self.W_U_plus[1].xs(1).index.values[i]
            self.timeText.set_text('$\|t\| \\approx$ {:.2f}'.format(round(abs(t), 2)))
        except IndexError:
            pass

        return self.lines

    def animate(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(self.xLim[0], self.xLim[1]), ylim=(self.yLim[0], self.yLim[1]))

        # self.lines = [plt.plot([], [], color=color_palette_green[idx], alpha=self.orbitAlpha)[0] for idx in range(self.numberOfOrbitsPerManifold)]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = ax.text(0.04, 0.02, s='', transform=ax.transAxes, size=self.timeTextSize)

        # # Plot zero velocity surface
        # x_range = np.arange(self.xLim[0], self.xLim[1], 0.001)
        # y_range = np.arange(self.yLim[0], self.yLim[1], 0.001)
        # x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        # z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.cLevel)
        # if z_mesh.min() < 0:
        #     plt.contour(x_mesh, y_mesh, z_mesh, [z_mesh.min(), 0], colors='black', alpha=0.3)

        # Plot trajectories both primaries
        bodies_df = load_bodies_location()
        trajectory_primary_1 = plt.Circle((0, 0), self.massParameter, color='black', fill=False, linestyle=':')
        trajectory_primary_2 = plt.Circle((0, 0), 1-self.massParameter, color='black', fill=False, linestyle=':')

        ax.add_artist(trajectory_primary_1)
        ax.add_artist(trajectory_primary_2)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for body in bodies_df:
            x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']- 0.5
            y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))- 0.5
            z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v)) - 0.5
            plt.contourf(x, y, z, colors='black')

        # Plot Lagrange points 1 and 2
        # lagrange_points_df = load_lagrange_points_location()
        # lagrange_point_nrs = ['L1', 'L2']
        # for lagrange_point_nr in lagrange_point_nrs:
        #     plt.scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'], color='black', marker='x')

        # Plot both orbits
        # for k in range(2):
        #     orbit_df = load_orbit('../../../data/raw/orbits/L' + str(k + 1) + '_' + self.orbitType + '_' + str(self.orbitIds[k]) + '.txt')
        #     plt.plot(orbit_df['x'], orbit_df['y'], color=self.orbitColor, alpha=self.orbitAlpha, linewidth=self.orbitLinewidth)

        # title = self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Spatial overview at C = ' + str(c_level)

        plt.xlim(self.xLim)
        plt.ylim(self.yLim)
        plt.xlabel('x [-]')
        plt.ylabel('y [-]')
        plt.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
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
        file_name = '../../../data/animations/rotating_reference_frame.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':

    planar_reference_frame_animation = PlanarReferenceFrameAnimation()
    planar_reference_frame_animation.animate()
