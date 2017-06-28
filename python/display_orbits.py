import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, cr3bp_velocity


class DisplayOrbits:

    def __init__(self, config, orbit_type):
        self.orbit = []
        self.config = config
        self.orbitType = orbit_type
        self.massParameter = 0.0121505810173
        self.figSize = (40, 40)
        self.titleSize = 20
        self.suptitleSize = 30

        orbit_ids = []
        for orbit_id in list(self.config[self.orbitType].keys()):
            ls = orbit_id.split('_')
            orbit_ids.append(int(ls[2]))
        orbit_ids = [self.orbitType + '_' + str(idx) for idx in sorted(orbit_ids)]

        for orbit_id in orbit_ids:
            self.orbit.append(load_orbit('../data/raw/' + orbit_id + '_final_orbit.txt'))

        self.lagrangePoints = load_lagrange_points_location()
        self.bodies = load_bodies_location()

        sns.set_style("whitegrid")
        pass

    def show_2d_subplots(self, axis_1, axis_2):
        nr_rows = int(np.ceil(np.sqrt(len(self.orbit))))
        nr_columns = int(np.ceil(len(self.orbit)/nr_rows))

        f, axarr = plt.subplots(nr_rows, nr_columns, sharex=True, sharey=True, figsize=self.figSize)
        for idx, orbit in enumerate(self.orbit):
            column = idx % nr_columns
            row = int((idx - idx % nr_columns) / nr_columns)

            # Orbit
            axarr[row, column].plot(orbit[axis_1], orbit[axis_2], color='darkblue')

            # Lagrange points and bodies
            for lagrange_point in self.lagrangePoints:
                axarr[row, column].scatter(self.lagrangePoints[lagrange_point][axis_1],
                           self.lagrangePoints[lagrange_point][axis_2],
                           color='grey')

            phi = np.linspace(0, 2 * np.pi, 100)
            theta = np.linspace(0, np.pi, 100)
            for body in self.bodies:
                x = self.bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + self.bodies[body][axis_1]
                y = self.bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + self.bodies[body][axis_2]
                axarr[row, column].plot(x, y, color='black')

            C = float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['C'])

            if axis_1 == 'x' and axis_2 == 'y':
                x = np.arange(-2.0, 2.0, 0.1)
                y = np.arange(-2.0, 2.0, 0.1)
                X, Y = np.meshgrid(x, y)
                Z = cr3bp_velocity(X, Y, C)
                axarr[row, column].contourf(X, Y, Z, levels=[-1, 0], colors='grey')

            title = 'C = ' + str(round(C, 3)) + \
                    ', T = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['T']), 3))
            axarr[row, column].set_title(title, size=self.titleSize)

        # axarr[0, 0].set_aspect('equal')
        # axarr[0, 0].set_ylim(axarr[0, 0].get_zlim())
        f.suptitle(self.orbitType + ' subplots 2D', size=self.suptitleSize)
        plt.savefig('../data/figures/orbit_' + self.orbitType + '_2d_' + axis_1 + '_' + axis_2 + '.png')
        pass

    def show_3d_subplots(self):
        nr_rows = int(np.ceil(np.sqrt(len(self.orbit))))
        nr_columns = int(np.ceil(len(self.orbit) / nr_rows))

        f, axarr = plt.subplots(nr_rows, nr_columns, figsize=self.figSize, subplot_kw={'projection': '3d'})

        for idx, orbit in enumerate(self.orbit):
            column = idx % nr_columns
            row = int((idx - idx % nr_columns) / nr_columns)

            # Orbit
            axarr[row, column].plot(orbit['x'], orbit['y'], orbit['z'], color='darkblue')

            # Lagrange points and bodies
            for lagrange_point in self.lagrangePoints:
                axarr[row, column].scatter(self.lagrangePoints[lagrange_point]['x'],
                                           self.lagrangePoints[lagrange_point]['y'],
                                           self.lagrangePoints[lagrange_point]['z'],
                                           color='grey')

            phi = np.linspace(0, 2 * np.pi, 100)
            theta = np.linspace(0, np.pi, 100)
            for body in self.bodies:
                x = self.bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + self.bodies[body]['x']
                y = self.bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + self.bodies[body]['y']
                z = self.bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + self.bodies[body]['z']
                axarr[row, column].plot_surface(x, y, z, color='black')

            title = 'C = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['C']), 3)) + \
                    ', T = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['T']), 3))
            axarr[row, column].set_title(title, size=self.titleSize)
            axarr[row, column].set_xlim([0.8, 1.2])
            axarr[row, column].set_ylim([-0.2, 0.2])
            axarr[row, column].set_zlim([-0.2, 0.2])

        f.suptitle(self.orbitType + ' subplots 3D', size=self.suptitleSize)
        # axarr[0, 0].set_ylim(axarr[0, 0].get_xlim())
        # axarr[0, 0].set_zlim(axarr[0, 0].get_xlim())
        plt.savefig('../data/figures/orbit_' + self.orbitType + "_3d_subplots.png")
        pass

    def show_3d_plot(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca(projection='3d')
        colors = sns.color_palette("Blues", n_colors=len(self.orbit))

        for idx, orbit in enumerate(self.orbit):
            label = 'C = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['C']), 3)) + \
                    ', T = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['T']), 3))
            if idx not in range(40):
                ax.plot(orbit['x'], orbit['y'], orbit['z'], color=colors[idx], label=label)

        # Lagrange points and bodies
        for lagrange_point in self.lagrangePoints:
            ax.scatter(self.lagrangePoints[lagrange_point]['x'],
                       self.lagrangePoints[lagrange_point]['y'],
                       self.lagrangePoints[lagrange_point]['z'],
                       color='grey')

        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        for body in self.bodies:
            x = self.bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + self.bodies[body]['x']
            y = self.bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + self.bodies[body]['y']
            z = self.bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + self.bodies[body]['z']
            ax.plot_surface(x, y, z, color='black')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_ylim(ax.get_zlim())
        ax.set_xlim([1, 1.2])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.1, 0.1])
        ax.set_title(self.orbitType + ' plot 3D', size=self.suptitleSize)
        plt.legend()
        plt.savefig('../data/figures/orbit_' + self.orbitType + "_3d_plot.png")
        pass


if __name__ == '__main__':
    with open('../config/config.json') as data_file:
        config = json.load(data_file)

    for orbit_type in config.keys():
        orbits_display = DisplayOrbits(config, orbit_type)
        orbits_display.show_2d_subplots('x', 'y')
        orbits_display.show_2d_subplots('y', 'z')
        orbits_display.show_2d_subplots('x', 'z')
        orbits_display.show_3d_subplots()
        orbits_display.show_3d_plot()

    # plt.show()
# plt.savefig('../data/figures/' + self.orbitType + "_3d_subplots.eps", format='eps')