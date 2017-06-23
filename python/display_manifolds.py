import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location


class DisplayManifold:

    def __init__(self, config, orbit_type):
        self.orbit = []
        self.manifold_S_plus, self.manifold_S_min, self.manifold_U_plus, self.manifold_U_min = [], [], [], []
        self.config = config
        self.orbitType = orbit_type
        self.massParameter = 0.0121505810173
        self.numberOfManifolds = 100
        self.figSize = (40, 40)
        self.titleSize = 20
        self.suptitleSize = 30
        orbit_names = sorted(list(self.config[self.orbitType].keys()))

        for orbit_name in orbit_names:
            self.orbit.append(load_orbit('../data/raw/' + orbit_name + '_final_orbit.txt'))
            self.manifold_S_plus.append(load_manifold('../data/raw/' + orbit_name + '_W_S_plus.txt'))
            self.manifold_S_min.append(load_manifold('../data/raw/' + orbit_name + '_W_S_min.txt'))
            self.manifold_U_plus.append(load_manifold('../data/raw/' + orbit_name + '_W_U_plus.txt'))
            self.manifold_U_min.append(load_manifold('../data/raw/' + orbit_name + '_W_U_min.txt'))

        self.lagrangePoints = load_lagrange_points_location()
        self.bodies = load_bodies_location()

        sns.set_style("whitegrid")
        pass

    def cr3bp_velocity(self, x, y, C):
        r_1 = np.sqrt((x + self.massParameter) ** 2 + y ** 2)
        r_2 = np.sqrt((x - 1 + self.massParameter) ** 2 + y ** 2)
        V = x ** 2 + y ** 2 + 2 * (1 - self.massParameter) / r_1 + 2 * self.massParameter / r_2 - C
        return V

    def show_2d_subplots(self, axis_1, axis_2):
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfManifolds)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfManifolds)

        nr_rows = int(np.ceil(np.sqrt(len(self.orbit))))
        nr_columns = int(np.ceil(len(self.orbit)/nr_rows))

        f, axarr = plt.subplots(nr_rows, nr_columns, sharex=True, sharey=True, figsize=self.figSize)
        for idx, orbit in enumerate(self.orbit):
            column = idx % nr_columns
            row = int((idx - idx % nr_columns) / nr_columns)

            # Manifold
            for manifold_orbit_number in range(1, self.numberOfManifolds+1):
                axarr[row, column].plot(self.manifold_S_plus[idx].xs(manifold_orbit_number)[axis_1],
                                        self.manifold_S_plus[idx].xs(manifold_orbit_number)[axis_2],
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_S_min[idx].xs(manifold_orbit_number)[axis_1],
                                        self.manifold_S_min[idx].xs(manifold_orbit_number)[axis_2],
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_plus[idx].xs(manifold_orbit_number)[axis_1],
                                        self.manifold_U_plus[idx].xs(manifold_orbit_number)[axis_2],
                                        color=color_palette_red[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_min[idx].xs(manifold_orbit_number)[axis_1],
                                        self.manifold_U_min[idx].xs(manifold_orbit_number)[axis_2],
                                        color=color_palette_red[manifold_orbit_number - 1])

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
            axarr[row, column].set_aspect('equal')

            C = float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['C'])

            if axis_1 == 'x' and axis_2 == 'y':
                x = np.arange(-2.0, 2.0, 0.1)
                y = np.arange(-2.0, 2.0, 0.1)
                X, Y = np.meshgrid(x, y)
                Z = self.cr3bp_velocity(X, Y, C)
                axarr[row, column].contourf(X, Y, Z, levels=[-1, 0], colors='grey')

            title = 'C = ' + str(round(C, 3)) + \
                    ', T = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['T']), 3))
            axarr[row, column].set_title(title, size=self.titleSize)

        axarr[0, 0].set_aspect('equal')
        # axarr[0, 0].set_ylim(axarr[0, 0].get_zlim())
        f.suptitle(self.orbitType + ' subplots 2D', size=self.suptitleSize)
        plt.savefig('../data/figures/manifold_' + self.orbitType + '_2d_' + axis_1 + '_' + axis_2 + '.png')
        pass

    def show_3d_subplots(self):
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfManifolds)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfManifolds)

        nr_rows = int(np.ceil(np.sqrt(len(self.orbit))))
        nr_columns = int(np.ceil(len(self.orbit) / nr_rows))

        f, axarr = plt.subplots(nr_rows, nr_columns, figsize=self.figSize, subplot_kw={'projection': '3d'})

        for idx, orbit in enumerate(self.orbit):
            column = idx % nr_columns
            row = int((idx - idx % nr_columns) / nr_columns)

            # Manifold
            for manifold_orbit_number in range(1, self.numberOfManifolds+1):
                axarr[row, column].plot(self.manifold_S_plus[idx].xs(manifold_orbit_number)['x'],
                                        self.manifold_S_plus[idx].xs(manifold_orbit_number)['y'],
                                        self.manifold_S_plus[idx].xs(manifold_orbit_number)['z'],
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_S_min[idx].xs(manifold_orbit_number)['x'],
                                        self.manifold_S_min[idx].xs(manifold_orbit_number)['y'],
                                        self.manifold_S_min[idx].xs(manifold_orbit_number)['z'],
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_plus[idx].xs(manifold_orbit_number)['x'],
                                        self.manifold_U_plus[idx].xs(manifold_orbit_number)['y'],
                                        self.manifold_U_plus[idx].xs(manifold_orbit_number)['z'],
                                        color=color_palette_red[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_min[idx].xs(manifold_orbit_number)['x'],
                                        self.manifold_U_min[idx].xs(manifold_orbit_number)['y'],
                                        self.manifold_U_min[idx].xs(manifold_orbit_number)['z'],
                                        color=color_palette_red[manifold_orbit_number - 1])

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
        plt.savefig('../data/figures/manifold_' + self.orbitType + "_3d_subplots.png")
        pass


    def show_3d_subplots_head(self, end_orbits):
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfManifolds)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfManifolds)

        nr_rows = int(np.ceil(np.sqrt(len(self.orbit))))
        nr_columns = int(np.ceil(len(self.orbit) / nr_rows))

        f, axarr = plt.subplots(nr_rows, nr_columns, figsize=self.figSize, subplot_kw={'projection': '3d'})

        for idx, orbit in enumerate(self.orbit):
            column = idx % nr_columns
            row = int((idx - idx % nr_columns) / nr_columns)

            # Manifold
            for manifold_orbit_number in range(1, self.numberOfManifolds+1):
                axarr[row, column].plot(self.manifold_S_plus[idx].xs(manifold_orbit_number)['x'].head(end_orbits),
                                        self.manifold_S_plus[idx].xs(manifold_orbit_number)['y'].head(end_orbits),
                                        self.manifold_S_plus[idx].xs(manifold_orbit_number)['z'].head(end_orbits),
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_S_min[idx].xs(manifold_orbit_number)['x'].head(end_orbits),
                                        self.manifold_S_min[idx].xs(manifold_orbit_number)['y'].head(end_orbits),
                                        self.manifold_S_min[idx].xs(manifold_orbit_number)['z'].head(end_orbits),
                                        color=color_palette_green[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_plus[idx].xs(manifold_orbit_number)['x'].head(end_orbits),
                                        self.manifold_U_plus[idx].xs(manifold_orbit_number)['y'].head(end_orbits),
                                        self.manifold_U_plus[idx].xs(manifold_orbit_number)['z'].head(end_orbits),
                                        color=color_palette_red[manifold_orbit_number - 1])
                axarr[row, column].plot(self.manifold_U_min[idx].xs(manifold_orbit_number)['x'].head(end_orbits),
                                        self.manifold_U_min[idx].xs(manifold_orbit_number)['y'].head(end_orbits),
                                        self.manifold_U_min[idx].xs(manifold_orbit_number)['z'].head(end_orbits),
                                        color=color_palette_red[manifold_orbit_number - 1])

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
        plt.savefig('../data/figures/manifold_' + self.orbitType + "_3d_subplots_" + str(end_orbits) + ".png")
        pass

    def show_3d_plot(self):
        fig = plt.figure(figsize=self.figSize)
        ax = fig.gca(projection='3d')
        colors = sns.color_palette("Blues", n_colors=len(self.orbit))

        for idx, orbit in enumerate(self.orbit):
            label = 'C = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['C']), 3)) + \
                    ', T = ' + str(round(float(config[self.orbitType][self.orbitType + '_' + str(idx + 1)]['T']), 3))
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
        plt.savefig('../data/figures/manifold_' + self.orbitType + "_3d_plot.png")
        pass


if __name__ == '__main__':
    with open('../config/config.json') as data_file:
        config = json.load(data_file)

    for orbit_type in config.keys():
        manifold_display = DisplayManifold(config, orbit_type)
        manifold_display.show_2d_subplots('x', 'y')
        manifold_display.show_2d_subplots('y', 'z')
        manifold_display.show_2d_subplots('x', 'z')
        manifold_display.show_3d_subplots()
        manifold_display.show_3d_subplots_head(100)
        manifold_display.show_3d_plot()

    # plt.show()
# plt.savefig('../data/figures/' + self.orbitType + "_3d_subplots.eps", format='eps')