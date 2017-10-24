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
    def __init__(self, libration_point_nr):
        self.suptitleSize = 44
        self.timeTextSize = 33

        self.xLim = [0.8, 1.2]
        self.yLim = [-0.15, 0.15]
        self.zLim = [-0.3, 0.3]

        self.orbitAlpha = 0.8
        self.orbitLinewidth = 2
        self.lagrangePointMarkerSize = 300
        self.orbitColor = 'gray'
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
        pass

    def initiate_lines(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def update_lines(self, i):
        if i % 10 == 0:
            print(i)

        for j, line in enumerate(self.lines):
            if i <= self.orbitIdBifurcations[0]:
                jacobi_energy = self.jacobiEnergyHorizontalLyapunov[i]
                self.jacobiEnergyText.set_text('Horizontal Lyapunov family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
                orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_horizontal_' + str(i) + '.txt')
            elif self.orbitIdBifurcations[0] < i <= self.orbitIdBifurcations[0] + self.numberOfHaloExtensionOrbits:
                index_for_halo = self.numberOfHaloExtensionOrbits - (i - self.orbitIdBifurcations[0])
                jacobi_energy = self.jacobiEnergyHaloN[(i - self.orbitIdBifurcations[0] - 1)]
                self.jacobiEnergyText.set_text('Southern halo family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
                orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_n_' + str(index_for_halo) + '.txt')
                pass
            else:
                index_for_halo = i - self.orbitIdBifurcations[0] - self.numberOfHaloExtensionOrbits + 1
                jacobi_energy = self.jacobiEnergyHalo[index_for_halo]
                self.jacobiEnergyText.set_text('Southern halo family \n $C \\approx$ {:.4f}'.format(round(jacobi_energy, 4)))
                orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_halo_' + str(index_for_halo) + '.txt')
                pass

            if i == self.orbitIdBifurcations[0]:
                plt.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color='navy', linewidth=2)

            x = orbit_df['x'].tolist()
            y = orbit_df['y'].tolist()
            z = orbit_df['z'].tolist()

            line.set_data(x, y)
            line.set_3d_properties(z)
            pass

        if i % 100 == 0 and i != 0:
            plt.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color='gray', linewidth=1)


        return self.lines

    def animate(self):
        print('\nProducing a HorizontalLyapunovBifurcationToHaloAnimation at L' + str(self.librationPointNr) + '\n')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.lines = [plt.plot([], [], color=self.orbitColor, alpha=self.orbitAlpha)[0]]

        # Text object to display absolute normalized time of trajectories within the manifolds
        self.jacobiEnergyText = ax.text2D(0.05, 0.05, s='Horizontal Lyapunov family \n $C \\approx$ {:.4f}'.format(round(self.jacobiEnergyHorizontalLyapunov[0], 4)), transform=ax.transAxes, size=self.timeTextSize)

        # Plot the first orbit
        orbit_df = load_orbit('../../../data/raw/orbits/L' + str(self.librationPointNr) + '_horizontal_' + str(0) + '.txt')
        plt.plot(orbit_df['x'], orbit_df['y'], orbit_df['z'], color='orange', alpha=self.orbitAlpha, linewidth=self.orbitLinewidth)

        # Plot the Moon
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        bodies_df = load_bodies_location()
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='black')

        # Plot Lagrange points 1 and 2
        lagrange_points_df = load_lagrange_points_location()
        lagrange_point_nrs = ['L1', 'L2']
        for lagrange_point_nr in lagrange_point_nrs:
            ax.scatter3D(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                         lagrange_points_df[lagrange_point_nr]['z'], color='black', marker='x',
                         s=self.lagrangePointMarkerSize)

        title = 'Bifurcation from horizontal Lyapunov to southern halo family at $L_' + str(self.librationPointNr) + '$'

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

        # Manually set camera angles for best display of the halo bifurcation
        ax.elev = 10
        ax.azim = -70

        # Determine number of frames
        number_of_frames = len(self.horizontalLyapunovIndices) + self.numberOfHaloExtensionOrbits + len(self.haloIndices) - 2
        print('Total number of frames: ' + str(number_of_frames) + '\nFrame number:')

        animation_function = animation.FuncAnimation(fig, self.update_lines, init_func=self.initiate_lines,
                                                     frames=number_of_frames, interval=1, blit=True)

        empty_writer_object = animation.writers['ffmpeg']
        animation_writer = empty_writer_object(fps=30, metadata=dict(artist='Koen Langemeijer'))
        file_name = '../../../data/animations/bifurcations/L' + str(self.librationPointNr) + '_horizontal_lyapunov_bifurcation_to_halo_family.mp4'
        animation_function.save(file_name, writer=animation_writer)


if __name__ == '__main__':
    libration_point_nrs = [1, 2]

    for libration_point_nr in libration_point_nrs:
        horizontal_lyapunov_bifurcation_to_halo_animation = HorizontalLyapunovBifurcationToHaloAnimation(libration_point_nr)
        horizontal_lyapunov_bifurcation_to_halo_animation.animate()
