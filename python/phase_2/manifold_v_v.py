import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, cr3bp_velocity

# mpl.rcParams['axes.linewidth'] = 5

orbit = load_orbit('../../data/raw/L1_horizontal_577.txt')

eigenvector_location_S = pd.read_table('../../data/raw/L1_horizontal_577_W_S_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
# eigenvector_location_U = pd.read_table('../../data/raw/L1_horizontal_577_W_U_plus_eigenvector_location.txt', delim_whitespace=True, header=None).filter(list(range(6)))
eigenvector_S = pd.read_table('../../data/raw/L1_horizontal_577_W_S_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))
# eigenvector_U = pd.read_table('../../data/raw/L1_horizontal_577_W_U_plus_eigenvector.txt', delim_whitespace=True, header=None).filter(list(range(6)))

# eigenvectors_indices = [np.floor(i * len(orbit) / len(eigenvector_S)) for i in range(len(eigenvector_S))]

# plt.figure(figsize=(30, 40))
plt.plot(orbit['x'], orbit['y'], color='blue')
# plt.scatter((load_lagrange_points_location()['L2']['x']-1) * 384400e-4,
#             load_lagrange_points_location()['L2']['y'] * 384400e-4, color='grey', s=100)
# plt.scatter(0, 0, color='grey', s=1000)
print(len(orbit))

eigenvector_offset = 1e-2
for idx in range(len(eigenvector_S)):
    if idx%2==0:
        x_S = [eigenvector_location_S.xs(idx)[0] - eigenvector_offset * eigenvector_S.xs(idx)[0],
               eigenvector_location_S.xs(idx)[0] + eigenvector_offset * eigenvector_S.xs(idx)[0]]
        y_S = [eigenvector_location_S.xs(idx)[1] - eigenvector_offset * eigenvector_S.xs(idx)[1],
               eigenvector_location_S.xs(idx)[1] + eigenvector_offset * eigenvector_S.xs(idx)[1]]

        # x_U = [eigenvector_location_U.xs(idx)[0] - eigenvector_offset * eigenvector_U.xs(idx)[0],
        #        eigenvector_location_U.xs(idx)[0] + eigenvector_offset * eigenvector_U.xs(idx)[0]]
        # y_U = [eigenvector_location_U.xs(idx)[1] - eigenvector_offset * eigenvector_U.xs(idx)[1],
        #        eigenvector_location_U.xs(idx)[1] + eigenvector_offset * eigenvector_U.xs(idx)[1]]

        plt.plot(x_S, y_S, color='green')
        # plt.plot(x_U, y_U, color='red')
        pass
    pass



    # plt.scatter(orbit.xs(eigenvectors_index)['x'], orbit.xs(eigenvectors_index)['y'], marker='x')
# plt.xlim([-2, 10])
# plt.ylim([-8, 8])
# ax = plt.gca()
# ax.xaxis.set_tick_params(size=30, width=5, direction='in', pad=25)
# ax.yaxis.set_tick_params(size=30, width=5, direction='in', pad=25)
# plt.xticks(list(range(0, 10, 2)), size=70)
# plt.yticks(list(range(-6, 8, 2)), size=70)
# plt.xlabel('$ x (\\times 10^4 $ km)', size=100)
# plt.ylabel('$ y (\\times 10^4 $ km)', size=100)
# plt.annotate('$\mathrm{Moon}$', xy=(-0.7, -0.7), size=100)
# plt.annotate('$L_2$', xy=((load_lagrange_points_location()['L2']['x']-1) * 384400e-4 - 0.3, -0.7), size=100)
# plt.savefig('../../data/verification/variable_eigenvectors.png')
plt.show()
