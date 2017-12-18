import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sys.path.append('../../util')
from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity, load_initial_conditions_incl_M
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


class HorizontalLyapunovBifurcationToHaloAnimation:
    def __init__(self, libration_point_nr, include_surface_hill=False):
        self.suptitleSize = 44
        self.timeTextSize = 33
        self.setContourLastTime = True
        self.includeSurfaceHill = include_surface_hill
        self.xLim = [0.8, 1.2]
        self.yLim = [-0.15, 0.15]
        self.zLim = [-0.3, 0.3]

        self.orbitAlpha = 0.8
        self.orbitLinewidth = 2
        self.lagrangePointMarkerSize = 300
        # self.orbitColor = 'gray'
        self.librationPointNr = libration_point_nr
        self.t = []
        self.lines = []
        self.horizontalLyapunov = []
        self.halo = []
        self.jacobiEnergyText = ''  # Will become a plt.text-object
        self.jacobiEnergyHorizontalLyapunov = []
        self.jacobiEnergyHalo = []
        self.jacobiEnergyHaloN = []
        self.T = []
        self.orderOfLinearInstability = []
        self.orbitIdBifurcations = []

        # Include reverse halo orbit continuation to horizontal Lyapunov tangent bifurcation

        initial_conditions_file_path = '../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_n_initial_conditions.txt'
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)[::-1]
        self.numberOfHaloExtensionOrbits = len(initial_conditions_incl_m_df)
        for row in initial_conditions_incl_m_df.iterrows():
            self.jacobiEnergyHaloN.append(row[1][0])

        initial_conditions_incl_m_df = load_initial_conditions_incl_M('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_initial_conditions.txt')
        for row in initial_conditions_incl_m_df.iterrows():
            self.jacobiEnergyHalo.append(row[1][0])

        # Determine the index for bifurcation
        initial_conditions_incl_m_df = load_initial_conditions_incl_M('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_horizontal_initial_conditions.txt')
        for row in initial_conditions_incl_m_df.iterrows():
            self.jacobiEnergyHorizontalLyapunov.append(row[1][0])
            self.T.append(row[1][1])

            M = np.matrix([list(row[1][8:14]), list(row[1][14:20]), list(row[1][20:26]), list(row[1][26:32]), list(row[1][32:38]), list(row[1][38:44])])
            eigenvalue = np.linalg.eigvals(M)

            # Determine order of linear instability
            reduction = 0
            for i in range(6):
                if (abs(eigenvalue[i]) - 1.0) < 1e-2:
                    reduction += 1

            if len(self.orderOfLinearInstability) > 0:
                # Check for a bifurcation, when the order of linear instability changes
                if (6 - reduction) != self.orderOfLinearInstability[-1]:
                    self.orbitIdBifurcations.append(row[0])
            self.orderOfLinearInstability.append(6 - reduction)

        print('Index for bifurcations from horizontal Lyapunov family: ')
        print(self.orbitIdBifurcations)

        # Select the indices to be plotted for the horizontal Lyapunov family (please remember that the bifurcation to the halo family corresponds to the first bifurcation of the horizontal Lyapunov family)
        self.horizontalLyapunovIndices = list(range(0, self.orbitIdBifurcations[0]))
        self.horizontalLyapunovIndices.append(self.orbitIdBifurcations[0])

        # Determine the indices for the halo family
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(
            '../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_initial_conditions.txt')
        self.haloIndices = list(range(0, initial_conditions_incl_m_df.index.max()))
        self.haloIndices.append(initial_conditions_incl_m_df.index.max())

        self.orbitColors = {'horizontal': sns.color_palette("viridis", 3)[0],
                            'halo': sns.color_palette("viridis", 3)[2],
                            'vertical': sns.color_palette("viridis", 3)[1]}
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        if i % 10 == 0:
            print(i)

        if i <= self.orbitIdBifurcations[0]:
            jacobi_energy = self.jacobiEnergyHorizontalLyapunov[i]
            self.jacobiEnergyText.set_text('Horizontal Lyapunov family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
            orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_horizontal_' + str(i) + '.txt')
        elif self.orbitIdBifurcations[0] < i <= self.orbitIdBifurcations[0] + self.numberOfHaloExtensionOrbits:
            index_for_halo = self.numberOfHaloExtensionOrbits - (i - self.orbitIdBifurcations[0])
            jacobi_energy = self.jacobiEnergyHaloN[(i - self.orbitIdBifurcations[0] - 1)]
            self.jacobiEnergyText.set_text('Southern halo family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
            orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_n_' + str(index_for_halo) + '.txt')
            self.lines[0].set_color(self.orbitColors['halo'])
            pass
        else:
            index_for_halo = i - self.orbitIdBifurcations[0] - self.numberOfHaloExtensionOrbits + 1
            jacobi_energy = self.jacobiEnergyHalo[index_for_halo]
            self.jacobiEnergyText.set_text('Southern halo family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
            orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_' + str(index_for_halo) + '.txt')
            self.lines[0].set_color(self.orbitColors['halo'])
            pass

        if i == self.orbitIdBifurcations[0]:
            self.ax.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColors['horizontal'], linewidth=2)

        x = orbit_df['x'].tolist()
        y = orbit_df['y'].tolist()
        z = orbit_df['z'].tolist()

        self.lines[0].set_data(x, y)
        self.lines[0].set_3d_properties(z)

        if self.includeSurfaceHill:
            z_mesh = cr3bp_velocity(self.x_mesh, self.y_mesh, jacobi_energy)
            if self.setContourLastTime:
                for coll in self.contour.collections:
                    self.ax.collections.remove(coll)
            if z_mesh.min() < 0:
                self.contour = self.ax.contour(self.x_mesh, self.y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)),
                                               cmap='gist_gray_r', alpha=0.5)
                self.setContourLastTime = True
            else:
                self.setContourLastTime = False

        if i % 100 == 0 and i != 0:
            if i <= self.orbitIdBifurcations[0]:
                self.ax.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColors['horizontal'], linewidth=self.orbitLinewidth)
            else:
                self.ax.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColors['halo'], linewidth=self.orbitLinewidth)
        return self.lines

    def animate(self):
        print('\nProducing a HorizontalLyapunovBifurcationToHaloAnimation at L' + str(self.librationPointNr) + '\n')

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        if self.includeSurfaceHill:
            x_range = np.arange(self.xLim[0], self.xLim[1], 0.001)
            y_range = np.arange(self.yLim[0], self.yLim[1], 0.001)
            self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)
            z_mesh = cr3bp_velocity(self.x_mesh, self.y_mesh, self.jacobiEnergyHorizontalLyapunov[0])
            self.contour = self.ax.contour(self.x_mesh, self.y_mesh, z_mesh, list(np.linspace(z_mesh.min(), 0, 10)), cmap='gist_gray_r', alpha=0.5)
        self.lines = [self.ax.plot([], [], color=self.orbitColors['horizontal'], alpha=self.orbitAlpha, linewidth=self.orbitLinewidth)[0]]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.jacobiEnergyText = self.ax.text2D(0.05, 0.05, s='Horizontal Lyapunov family \n $C \\approx$ {:.4f}'.format(round(self.jacobiEnergyHorizontalLyapunov[0], 4)),
                                               transform=self.ax.transAxes, size=self.timeTextSize)

        # Plot the first orbit
        orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_horizontal_' + str(0) + '.txt')
        self.ax.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color=self.orbitColors['horizontal'], alpha=self.orbitAlpha, linewidth=self.orbitLinewidth)

        # Plot the Moon
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x, y, z, color='black')

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            self.ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x',
                         s=self.lagrangePointMarkerSize)

        title = 'Bifurcation from horizontal Lyapunov to southern halo family at $L_' + str(self.librationPointNr) + '$'

        self.ax.set_xlim3d(self.xLim)
        self.ax.set_ylim3d(self.yLim)
        self.ax.set_zlim3d(self.zLim)
        self.ax.set_xlabel('x [-]')
        self.ax.set_ylabel('y [-]')
        self.ax.set_zlabel('z [-]')

        self.ax.grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.suptitle(title, size=self.suptitleSize)

        # Fix overlap between labels and ticks
        self.ax.xaxis._axinfo['label']['space_factor'] = 2.0
        self.ax.yaxis._axinfo['label']['space_factor'] = 2.0
        self.ax.zaxis._axinfo['label']['space_factor'] = 2.0

        # Manually set camera angles for best display of the halo bifurcation
        self.ax.elev = 10
        self.ax.azim = -70

        # Determine number of frames
        number_of_frames = len(self.horizontalLyapunovIndices) + self.numberOfHaloExtensionOrbits + len(self.haloIndices) - 2
        print('Total number of frames: ' + str(number_of_frames) + '\nFrame number:')

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=number_of_frames, interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        if self.includeSurfaceHill:
            file_name = '../../../data/animations/bifurcations/L' + str(
                self.librationPointNr) + '_horizontal_lyapunov_bifurcation_to_halo_family_hill.mp4'
        else:
            file_name = '../../../data/animations/bifurcations/L' + str(
                self.librationPointNr) + '_horizontal_lyapunov_bifurcation_to_halo_family.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':
    libration_point_nrs = [1, 2]
    include_hills = [True, False]

    for libration_point_nr in libration_point_nrs:
        for include_hill in include_hills:
            horizontal_lyapunov_bifurcation_to_halo_animation = HorizontalLyapunovBifurcationToHaloAnimation(libration_point_nr, include_hill)
            horizontal_lyapunov_bifurcation_to_halo_animation.animate()
