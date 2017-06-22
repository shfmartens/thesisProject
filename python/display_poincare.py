import json
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import radviz
import numpy as np

from load_data import load_manifold, cr3bp_velocity


with open('../config/config.json') as data_file:
    config = json.load(data_file)


for orbit_type in config.keys():
    for orbit_name in config[orbit_type].keys():
        print(orbit_name)
        manifold_S_plus = load_manifold('../data/raw/' + orbit_name + '_W_S_plus.txt')
        manifold_S_min = load_manifold('../data/raw/' + orbit_name + '_W_S_min.txt')
        manifold_U_plus = load_manifold('../data/raw/' + orbit_name + '_W_U_plus.txt')
        manifold_U_min = load_manifold('../data/raw/' + orbit_name + '_W_U_min.txt')

        ls_S_plus = []
        ls_S_min = []
        ls_U_plus = []
        ls_U_min = []

        for i in range(1, 101):
            ls_S_plus.append(manifold_S_plus.xs(i).tail(1))
            ls_S_min.append(manifold_S_min.xs(i).tail(1))
            ls_U_plus.append(manifold_U_plus.xs(i).tail(1))
            ls_U_min.append(manifold_U_min.xs(i).tail(1))
        data_S_plus = pd.concat(ls_S_plus).reset_index(drop=True)
        data_S_min = pd.concat(ls_S_min).reset_index(drop=True)
        data_U_plus = pd.concat(ls_U_plus).reset_index(drop=True)
        data_U_min = pd.concat(ls_U_min).reset_index(drop=True)
        data_S_plus['manifold'] = 'halo_1_W_S_p'
        data_S_min['manifold'] = 'halo_1_W_S_m'
        data_U_plus['manifold'] = 'halo_1_W_U_p'
        data_U_min['manifold'] = 'halo_1_W_U_m'

        data = pd.concat([data_S_plus, data_S_min, data_U_plus, data_U_min])

        # radviz(data, 'manifold', color=['g', 'r'], alpha=0.1)

        axes = ['x', 'y', 'z']
        fig, axarr = plt.subplots(2, 3, figsize=(20, 10))

        C = float(config[orbit_type][orbit_name]['C'])
        x_range = np.arange(-4.0, 2.0, 0.001)
        y_range = np.arange(-3.0, 3.0, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)

        if Z.min() < 0:
            X_contour = X[abs(Z) < 1e-3]
            Y_contour = Y[abs(Z) < 1e-3]

            if abs(Y_contour).min() < 1e-3:
                U_1_x_min = max(X_contour[abs(Y_contour) < 1e-3])
                U_4_x_max = min(X_contour[abs(Y_contour) < 1e-3])

                U_1_S_plus = data_S_plus[data_S_plus['x'] > U_1_x_min]
                U_1_S_min = data_S_min[data_S_min['x'] > U_1_x_min]
                U_1_U_plus = data_U_plus[data_U_plus['x'] > U_1_x_min]
                U_1_U_min = data_U_min[data_U_min['x'] > U_1_x_min]

                U_4_S_plus = data_S_plus[data_S_plus['x'] < U_4_x_max]
                U_4_S_min = data_S_min[data_S_min['x'] < U_4_x_max]
                U_4_U_plus = data_U_plus[data_U_plus['x'] < U_4_x_max]
                U_4_U_min = data_U_min[data_U_min['x'] < U_4_x_max]

                for idx, axis in enumerate(axes):
                    axarr[0, idx].scatter(U_1_S_plus[axis], U_1_S_plus[axis + 'dot'], color='g', alpha=0.5)
                    axarr[0, idx].scatter(U_1_S_min[axis], U_1_S_min[axis + 'dot'], color='g', alpha=0.5)
                    axarr[0, idx].scatter(U_1_U_plus[axis], U_1_U_plus[axis + 'dot'], color='r', alpha=0.5)
                    axarr[0, idx].scatter(U_1_U_min[axis], U_1_U_min[axis + 'dot'], color='r', alpha=0.5)

                    axarr[1, idx].scatter(U_4_S_plus[axis], U_4_S_plus[axis + 'dot'], color='g', alpha=0.5)
                    axarr[1, idx].scatter(U_4_S_min[axis], U_4_S_min[axis + 'dot'], color='g', alpha=0.5)
                    axarr[1, idx].scatter(U_4_U_plus[axis], U_4_U_plus[axis + 'dot'], color='r', alpha=0.5)
                    axarr[1, idx].scatter(U_4_U_min[axis], U_4_U_min[axis + 'dot'], color='r', alpha=0.5)

                    axarr[0, idx].set_xlabel(axis)
                    axarr[0, idx].set_ylabel('$\dot{' + axis + '}$')
                    axarr[0, idx].grid()
                    axarr[0, idx].set_aspect('equal', 'datalim')

                    axarr[1, idx].set_xlabel(axis)
                    axarr[1, idx].set_ylabel('$\dot{' + axis + '}$')
                    axarr[1, idx].grid()
                    axarr[1, idx].set_aspect('equal', 'datalim')

                axarr[0, 1].set_title('$U_1$', size=16)
                axarr[1, 1].set_title('$U_4$', size=16)
                plt.suptitle(orbit_name, size=22)
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                plt.savefig('../data/figures/' + orbit_name + "_poincare.png")
                # plt.show()
