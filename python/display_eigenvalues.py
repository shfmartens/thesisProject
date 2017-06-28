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


class DisplayOrbitsEigenvalues:

    def __init__(self, config, orbit_type):
        self.orbit = []
        self.config = config
        self.orbitType = orbit_type
        self.massParameter = 0.0121505810173
        self.figSize = (20, 20)
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

    def show_2d_subplots(self):
        df = pd.DataFrame.from_dict(self.config[orbit_type]).T

        # colors = sns.color_palette("Blues", n_colors=6)

        for orbit_id, row in df.iterrows():
            f, axarr = plt.subplots(2, 1, figsize=self.figSize)
            orbit = load_orbit('../data/raw/' + orbit_id + '_final_orbit.txt')
            label = '$x_0$ = ' + str(np.round(row['x'], 3)) \
                    + '\n$y_0$ = ' + str(np.round(row['y'], 3)) \
                    + '\n$z_0$ = ' + str(np.round(row['z'], 3)) \
                    + '\n$\dot{x}_0$ = ' + str(np.round(row['x_dot'], 3)) \
                    + '\n$\dot{y}_0$ = ' + str(np.round(row['y_dot'], 3)) \
                    + '\n$\dot{z}_0$ = ' + str(np.round(row['z_dot'], 3))

            axarr[0].plot(orbit['y'].values, orbit['z'].values, color='darkblue', label=label)

            circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
            axarr[1].add_patch(circ)

            for j in range(1, 7):
                x = row['l_' + str(j) + '_re']
                y = row['l_' + str(j) + '_im']
                axarr[1].scatter(x, y, color='darkblue', label='$\lambda_' + str(j) + '$ = (' + str(np.round(x, 2))
                                                               + ', ' + str(np.round(y, 2)) + ')')

            axarr[0].set_title('T = ' + str(np.round(row['T'], 2))
                               + ', C = ' + str(np.round(row['C'], 2)))

            for k in range(2):
                # Shrink current axis
                box = axarr[k].get_position()
                axarr[k].set_position([box.x0, box.y0, box.width * 0.75, box.height])

                # Put a legend to the right of the current axis
                axarr[k].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            axarr[1].set_xlim([-10, 10])
            axarr[1].set_ylim([-5, 5])

            plt.suptitle(orbit_id, size=self.suptitleSize)
            plt.savefig('../data/figures/eigenvalues_' + orbit_id + '_2d.png')
            plt.close()
        pass

    def show_3d_subplots(self):
        df = pd.DataFrame.from_dict(self.config[orbit_type]).T

        # colors = sns.color_palette("Blues", n_colors=6)

        for orbit_id, row in df.iterrows():
            fig = plt.figure(figsize=self.figSize)
            ax = fig.add_subplot(2, 1, 1, projection='3d')

            orbit = load_orbit('../data/raw/' + orbit_id + '_final_orbit.txt')
            label = '$x_0$ = ' + str(np.round(row['x'], 3)) \
                    + '\n$y_0$ = ' + str(np.round(row['y'], 3)) \
                    + '\n$z_0$ = ' + str(np.round(row['z'], 3)) \
                    + '\n$\dot{x}_0$ = ' + str(np.round(row['x_dot'], 3)) \
                    + '\n$\dot{y}_0$ = ' + str(np.round(row['y_dot'], 3)) \
                    + '\n$\dot{z}_0$ = ' + str(np.round(row['z_dot'], 3))

            ax.plot(orbit['x'].values, orbit['y'].values, orbit['z'].values, color='darkblue', label=label)

            # Shrink current axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.set_xlim([0.8, 1.1])
            ax.set_ylim([-0.15, 0.15])
            ax.set_zlim([-0.15, 0.15])

            ax = fig.add_subplot(2, 1, 2)
            circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
            ax.add_patch(circ)

            for j in range(1, 7):
                x = row['l_' + str(j) + '_re']
                y = row['l_' + str(j) + '_im']
                ax.scatter(x, y, color='darkblue', label='$\lambda_' + str(j) + '$ = (' + str(np.round(x, 2))
                                                         + ', ' + str(np.round(y, 2)) + ')')
            ax.set_title('T = ' + str(np.round(row['T'], 2)) + ', C = ' + str(np.round(row['C'], 2)))

            # Shrink current axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.set_xlim([-10, 10])
            ax.set_ylim([-5, 5])
            plt.suptitle(orbit_id, size=self.suptitleSize)
            plt.savefig('../data/figures/eigenvalues_' + orbit_id + '_3d.png')
            plt.close()
        pass


if __name__ == '__main__':
    with open('../config/config.json') as data_file:
        config = json.load(data_file)

    for orbit_type in config.keys():
        orbits_eigenvalues_display = DisplayOrbitsEigenvalues(config, orbit_type)
        # orbits_eigenvalues_display.show_2d_subplots()
        orbits_eigenvalues_display.show_3d_subplots()
