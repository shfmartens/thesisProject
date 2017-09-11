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


class SpatialOrbitsZoomAnimation:
    '''
    Zoomed spatial animation of the horizontal/vertical Lyapunov and halo families in the EM-system.
    '''
    def __init__(self, c_level, orbit_ids):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.xLim = [0.7, 1.3]
        self.yLim = [-0.3, 0.3]
        self.zLim = self.yLim
        self.orbitAlpha = 0.8
        self.orbitLinewidth = 2
        self.lagrangePointMarkerSize = 300
        self.orbitColor = 'navy'
        self.cLevel = c_level
        self.orbitIds = orbit_ids
        self.t = []
        self.lines = []
        self.horizontalLyapunov = []
        self.verticalLyapunov = []
        self.halo = []
        self.timeText = ''  # Will become a plt.text-object
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        current_time = self.t[i]
        print(current_time)
        self.timeText.set_text('$\|t\| \\approx$ {:.2f}'.format(round(abs(current_time), 2)))
        for j, line in enumerate(self.lines):
            if j < 2:
                x = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['x'].tolist()
                y = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['y'].tolist()
                z = self.horizontalLyapunov[j][self.horizontalLyapunov[j]['time'] <= current_time]['z'].tolist()
                pass
            if 2 <= j < 4:
                x = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['x'].tolist()
                y = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['y'].tolist()
                z = self.verticalLyapunov[j - 2][self.verticalLyapunov[j - 2]['time'] <= current_time]['z'].tolist()
                pass
            if 4 <= j < 6:
                x = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['x'].tolist()
                y = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['y'].tolist()
                z = self.halo[j - 4][self.halo[j - 4]['time'] <= current_time]['z'].tolist()
                pass
            line.set_data(x, y)
            line.set_3d_properties(z)
            pass
        return self.lines

    def animate(self):
        print('Producing SpatialOrbitsZoomAnimation at C = ' + str(c_level) + '\n')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.horizontalLyapunov = [load_orbit('../../../data/raw/orbits/L' + str(1) + '_horizontal_' + str(self.orbitIds['horizontal'][1][self.cLevel]) + '_100.txt'),
                                   load_orbit('../../../data/raw/orbits/L' + str(2) + '_horizontal_' + str(self.orbitIds['horizontal'][2][self.cLevel]) + '_100.txt')]

        self.verticalLyapunov = [load_orbit('../../../data/raw/orbits/L' + str(1) + '_vertical_' + str(self.orbitIds['vertical'][1][self.cLevel]) + '_100.txt'),
                                 load_orbit('../../../data/raw/orbits/L' + str(2) + '_vertical_' + str(self.orbitIds['vertical'][2][self.cLevel]) + '_100.txt')]

        self.halo = [load_orbit('../../../data/raw/orbits/L' + str(1) + '_halo_' + str(self.orbitIds['halo'][1][self.cLevel]) + '_100.txt'),
                     load_orbit('../../../data/raw/orbits/L' + str(2) + '_halo_' + str(self.orbitIds['halo'][2][self.cLevel]) + '_100.txt')]

        self.lines = [plt.plot([], [], color=self.orbitColor, alpha=self.orbitAlpha, marker='o', markevery=[-1])[0] for idx in range(6)]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = ax.text2D(0.05, 0.05, s='$\|t\| \\approx 0$', transform=ax.transAxes, size=self.timeTextSize)

        # Plot both orbits
        for k in range(2):
            plt.plot(self.horizontalLyapunov[k]['x'], self.horizontalLyapunov[k]['y'], self.horizontalLyapunov[k]['z'],
                     color=self.orbitColor, alpha=self.orbitAlpha, linewidth=self.orbitLinewidth, linestyle=':')
            plt.plot(self.verticalLyapunov[k]['x'], self.verticalLyapunov[k]['y'], self.verticalLyapunov[k]['z'],
                     color=self.orbitColor, alpha=self.orbitAlpha, linewidth=self.orbitLinewidth, linestyle=':')
            plt.plot(self.halo[k]['x'], self.halo[k]['y'], self.halo[k]['z'],
                     color=self.orbitColor, alpha=self.orbitAlpha, linewidth=self.orbitLinewidth, linestyle=':')

        # Plot both primaries
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        body = 'Moon'
        x = bodies_df[body]['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df[body]['x']
        y = bodies_df[body]['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df[body]['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='black')

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x', s=self.lagrangePointMarkerSize)

        title = 'Types of periodic libration point motion - Spatial zoom at C = ' + str(c_level)

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

        # Fix overlap between labels and ticks
        ax.xaxis._axinfo['label']['space_factor'] = 2.0
        ax.yaxis._axinfo['label']['space_factor'] = 2.0
        ax.zaxis._axinfo['label']['space_factor'] = 2.0

        # Determine the maximum value of t
        t_max = 0
        for lagrange_point_idx in [0, 1]:
            t_max = max(t_max, self.horizontalLyapunov[lagrange_point_idx].tail(1)['time'].values[0])
            t_max = max(t_max, self.verticalLyapunov[lagrange_point_idx].tail(1)['time'].values[0])
            t_max = max(t_max, self.halo[lagrange_point_idx].tail(1)['time'].values[0])
        print('Maximum value for t = ' + str(t_max) + ', animation t: = ')

        # Introduce a new time-vector for linearly spaced time throughout the animation
        self.t = np.linspace(0, t_max, np.round(t_max/0.005)+1)

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=len(self.t), interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=60, metadata=dict(artist='Koen Langemeijer'))
        file_name = '../../../data/animations/orbits/spatial_orbits_zoom_' + str(c_level) + '.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':
    c_levels = [3.05, 3.1, 3.15]
    orbit_ids = {'horizontal':  {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo':  {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    for c_level in c_levels:
        spatial_orbits_zoom_animation = SpatialOrbitsZoomAnimation(c_level, orbit_ids)
        spatial_orbits_zoom_animation.animate()
