import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
# import seaborn as sns

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, cr3bp_velocity

mpl.rcParams['axes.linewidth'] = 5

orbit = load_orbit('../../data/verification/ver_halo_1_final_orbit.txt')
orbit['x'] = (orbit['x'] - 1) * 384400e-4
orbit['y'] = orbit['y'] * 384400e-4

eigenvectors_S = pd.read_table('../../data/verification/ver_halo_1_W_S_plus.txt', delim_whitespace=True, header=None).filter(list(range(6)))
eigenvectors_U = pd.read_table('../../data/verification/ver_halo_1_W_U_plus.txt', delim_whitespace=True, header=None).filter(list(range(6)))
eigenvectors_S.columns = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
eigenvectors_U.columns = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
eigenvectors_indices = [np.floor(i * len(orbit) / 20) for i in range(20)]

plt.figure(figsize=(30, 40))
plt.plot(orbit['x'], orbit['y'], color='blue', linewidth=4)
plt.scatter((load_lagrange_points_location()['L2']['x']-1) * 384400e-4,
            load_lagrange_points_location()['L2']['y'] * 384400e-4, color='grey', s=100)
plt.scatter(0, 0, color='grey', s=1000)

v_U = [0.46173686071464,
 -0.12104697597009]
v_S=[0.048496125209463,
 0.38642078136333]


for idx, eigenvectors_index in enumerate(eigenvectors_indices):

    x_S = [orbit.xs(eigenvectors_index)['x'] - 1 * eigenvectors_S.xs(idx)['x'],
           orbit.xs(eigenvectors_index)['x'] + 1 * eigenvectors_S.xs(idx)['x']]
    y_S = [orbit.xs(eigenvectors_index)['y'] - 1 * eigenvectors_S.xs(idx)['y'],
           orbit.xs(eigenvectors_index)['y'] + 1 * eigenvectors_S.xs(idx)['y']]

    x_U = [orbit.xs(eigenvectors_index)['x'] - 1 * eigenvectors_U.xs(idx)['x'],
           orbit.xs(eigenvectors_index)['x'] + 1 * eigenvectors_U.xs(idx)['x']]
    y_U = [orbit.xs(eigenvectors_index)['y'] - 1 * eigenvectors_U.xs(idx)['y'],
           orbit.xs(eigenvectors_index)['y'] + 1 * eigenvectors_U.xs(idx)['y']]
    # x_S = [orbit.xs(eigenvectors_index)['x'] - 1 * v_S[0],
    #        orbit.xs(eigenvectors_index)['x'] + 1 * v_S[0]]
    # y_S = [orbit.xs(eigenvectors_index)['y'] - 1 * v_S[1],
    #        orbit.xs(eigenvectors_index)['y'] + 1 * v_S[1]]
    #
    # x_U = [orbit.xs(eigenvectors_index)['x'] - 1 * v_U[0],
    #        orbit.xs(eigenvectors_index)['x'] + 1 * v_U[0]]
    # y_U = [orbit.xs(eigenvectors_index)['y'] - 1 * v_U[1],
    #        orbit.xs(eigenvectors_index)['y'] + 1 * v_U[1]]
    plt.plot(x_S, y_S, color='green', linewidth=7)
    plt.plot(x_U, y_U, color='red', linewidth=7)


    # plt.scatter(orbit.xs(eigenvectors_index)['x'], orbit.xs(eigenvectors_index)['y'], marker='x')
plt.xlim([-2, 10])
plt.ylim([-8, 8])
ax = plt.gca()
ax.xaxis.set_tick_params(size=30, width=5, direction='in', pad=25)
ax.yaxis.set_tick_params(size=30, width=5, direction='in', pad=25)
plt.xticks(list(range(0, 10, 2)), size=70)
plt.yticks(list(range(-6, 8, 2)), size=70)
plt.xlabel('$ x (\\times 10^4 $ km)', size=100)
plt.ylabel('$ y (\\times 10^4 $ km)', size=100)
plt.annotate('$\mathrm{Moon}$', xy=(-0.7, -0.7), size=100)
plt.annotate('$L_2$', xy=((load_lagrange_points_location()['L2']['x']-1) * 384400e-4 - 0.3, -0.7), size=100)
plt.savefig('../../data/verification/variable_eigenvectors.png')
# plt.show()
