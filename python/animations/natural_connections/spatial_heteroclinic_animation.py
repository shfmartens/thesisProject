import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import multiprocessing as mp
sys.path.append('../../util')
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


class SpatialHeteroclinicAnimation:
    def __init__(self, orbit_type, theta, number_of_orbits_per_manifold, orbit_ids):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.xLim = [0.7, 1.3]
        self.yLim = [-0.3, 0.3]
        self.zLim = self.yLim
        self.orbitAlpha = 0.8
        self.orbitLinewidth = 3
        self.orbitColor = 'navy'

        self.orbitType = orbit_type
        self.theta = theta
        self.orbitIds = orbit_ids
        self.lines = []
        self.W_S_plus = []
        self.W_S_min = []
        self.W_U_plus = []
        self.W_U_min = []
        self.t = []
        self.numberOfOrbitsPerManifold = number_of_orbits_per_manifold
        self.lagrangePointMarkerSize = 100
        self.timeText = ''  # Will become a plt.text-object
        self.cLevel = 3.1
        self.orbitTypeForTitle = orbit_type.capitalize()
        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        print(self.orbitTypeForTitle + ' at theta = ' + str(theta))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.tight_layout()

        df = pd.read_table('../../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
        df.columns = ['theta', 'taus', 'ts', 'xs', 'ys', 'zs', 'xdots', 'ydots', 'zdots', 'tauu', 'tu', 'xu', 'yu', 'zu', 'xdotu', 'ydotu', 'zdotu']
        df['dr'] = np.sqrt((df['xs'] - df['xu']) ** 2 + (df['ys'] - df['yu']) ** 2 + (df['zs'] - df['zu']) ** 2)
        df['dv'] = np.sqrt((df['xdots'] - df['xdotu']) ** 2 + (df['ydots'] - df['ydotu']) ** 2 + (df['zdots'] - df['zdotu']) ** 2)
        self.minimumImpulse = df.set_index('theta')

        index_near_heteroclinic_s = int(self.minimumImpulse.loc[theta]['taus'] * self.numberOfOrbitsPerManifold)
        index_near_heteroclinic_u = int(self.minimumImpulse.loc[theta]['tauu'] * self.numberOfOrbitsPerManifold)

        self.W_S_min = load_manifold_refactored('../../../data/raw/poincare_sections/' + str(
            self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(
            int(theta)) + '_full.txt').xs(index_near_heteroclinic_s)

        self.W_U_plus = load_manifold_refactored('../../../data/raw/poincare_sections/' + str(
            self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(
            int(theta)) + '_full.txt').xs(index_near_heteroclinic_u)
        self.firstTimeStable = True
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        current_time = self.t[i]
        print(current_time)
        self.timeText.set_text('$\|t\| \\approx$ {:.2f}'.format(round(abs(current_time), 2)))

        if current_time < max(abs(self.W_U_plus.index)):
            x = self.W_U_plus[abs(self.W_U_plus.index) <= current_time]['x'].tolist()
            y = self.W_U_plus[abs(self.W_U_plus.index) <= current_time]['y'].tolist()
            z = self.W_U_plus[abs(self.W_U_plus.index) <= current_time]['z'].tolist()
            self.lines[0].set_data(x, y)
            self.lines[0].set_3d_properties(z)
            pass
        else:
            if self.firstTimeStable == True:
                x = self.W_U_plus['x'].tolist()
                y = self.W_U_plus['y'].tolist()
                z = self.W_U_plus['z'].tolist()
                self.lines[0].set_data(x, y)
                self.lines[0].set_3d_properties(z)
                self.firstTimeStable = False
                pass

            current_time_2 = current_time - max(abs(self.W_U_plus.index)) - max(abs(self.W_S_min.index))
            xs = self.W_S_min[(self.W_S_min.index) <= current_time_2]['x'].tolist()
            ys = self.W_S_min[(self.W_S_min.index) <= current_time_2]['y'].tolist()
            zs = self.W_S_min[(self.W_S_min.index) <= current_time_2]['z'].tolist()
            self.lines[1].set_data(xs, ys)
            self.lines[1].set_3d_properties(zs)
            pass

        return self.lines

    def animate(self):

        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        self.lines = [self.ax.plot([], [], color='red', linewidth=self.orbitLinewidth, alpha=self.orbitAlpha)[0],
                      self.ax.plot([], [], color='green', linewidth=self.orbitLinewidth, alpha=self.orbitAlpha)[0]]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.timeText = self.ax.text2D(0.05, 0.05, s='$\|t\| \\approx 0$', transform=self.ax.transAxes, size=self.timeTextSize)

        # Plot zero velocity surface
        x_range = np.arange(self.xLim[0], self.xLim[1], 0.001)
        y_range = np.arange(self.yLim[0], self.yLim[1], 0.001)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = cr3bp_velocity(x_mesh, y_mesh, self.cLevel)
        if z_mesh.min() < 0:
            # plt.contour(x_mesh, y_mesh, z_mesh, [z_mesh.min(), 0], colors='black', alpha=0.3)
            self.ax.contour(x_mesh, y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)

        # Plot both orbits
        for k in range(2):
            orbit_df = load_orbit('../../../data/raw/orbits/refined_for_c/L' + str(k+1) + '_' + self.orbitType + '_' + str(self.orbitIds[self.orbitType][k+1][self.cLevel]) + '.txt')
            self.ax.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColor, alpha=self.orbitAlpha, linewidth=2, linestyle=':')

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

        title = self.orbitTypeForTitle + ' $\{ \mathcal{W}^{S \pm}, \mathcal{W}^{U \pm} \}$ - Orthographic projection (C = ' + str(self.cLevel) + ')'

        self.ax.set_xlim(self.xLim)
        self.ax.set_ylim(self.yLim)
        self.ax.set_zlim(self.zLim)
        self.ax.set_xlabel('x [-]')
        self.ax.set_ylabel('y [-]')
        self.ax.set_zlabel('z [-]')

        self.ax.grid(True, which='both', ls=':')
        # self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)
        self.fig.suptitle(title, size=self.suptitleSize)

        # Fix overlap between labels and ticks
        self.ax.xaxis._axinfo['label']['space_factor'] = 6.0
        self.ax.yaxis._axinfo['label']['space_factor'] = 6.0
        self.ax.zaxis._axinfo['label']['space_factor'] = 6.0

        # Determine the maximum value of t
        t_max = max(abs(self.W_U_plus.index)) + max(abs(self.W_S_min.index))
        print('Maximum value for unstable = ' + str(max(abs(self.W_U_plus.index))))
        print('Maximum value for stable = ' + str(max(abs(self.W_S_min.index))))
        print('Maximum value for t = ' + str(t_max) + ', animation t: = ')

        # Introduce a new time-vector for linearly spaced time throughout the animation
        self.t = np.linspace(0, t_max, 150)
        
        self.animation_function = animation.FuncAnimation(self.fig, self.update_lines, init_func=self.initiate_lines,
                                                          frames=len(self.t), interval=1, blit=True)

        self.empty_writer_object = animation.writers['ffmpeg']
        self.animation_writer = self.empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        self.file_name = '../../../data/animations/natural_connections/spatial_heteroclinic_' + self.orbitType + '_' + str(int(self.theta)) + '.mp4'
        self.animation_function.save(self.file_name, writer=self.animation_writer)


if __name__ == '__main__':
    orbit_types = ['horizontal', 'vertical', 'halo']
    orbit_types = ['vertical']
    orbit_ids = {'horizontal': {1: {3.05: 808, 3.1: 577, 3.15: 330}, 2: {3.05: 1066, 3.1: 760, 3.15: 373}},
                 'halo': {1: {3.05: 1235, 3.1: 836, 3.15: 358}, 2: {3.05: 1093, 3.1: 651, 3.15: 0}},
                 'vertical': {1: {3.05: 1664, 3.1: 1159, 3.15: 600}, 2: {3.05: 1878, 3.1: 1275, 3.15: 513}}}

    theta = -125.0

    for orbit_type in orbit_types:
        spatial_heteroclinic_animation = SpatialHeteroclinicAnimation(orbit_type, theta, 5000, orbit_ids)
        spatial_heteroclinic_animation.animate()
