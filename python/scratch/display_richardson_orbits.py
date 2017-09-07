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

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(0, 1000, 50):
#     try:
#         df = load_orbit('../data/raw/horizontal_L1_' + str(i) + '.txt')
#         ax.plot(df['x'], df['y'], df['z'])
#     except FileNotFoundError:
#         pass
#     try:
#         df = load_orbit('../data/raw/horizontal_L2_' + str(i) + '.txt')
#         ax.plot(df['x'], df['y'], df['z'])
#     except FileNotFoundError:
#         pass
#     try:
#         df = load_orbit('../data/raw/halo_L1_' + str(i) + '.txt')
#         ax.plot(df['x'], df['y'], df['z'])
#     except FileNotFoundError:
#         pass
#     try:
#         df = load_orbit('../data/raw/halo_L2_' + str(i) + '.txt')
#         ax.plot(df['x'], df['y'], df['z'])
#     except FileNotFoundError:
#         pass

orbit_types = ['horizontal', 'halo']
lagrange_point_nrs = [1, 2]

for orbit_type in orbit_types:
    if orbit_type == 'horizontal':
            continue

    for lagrange_point_nr in lagrange_point_nrs:

        if orbit_type == 'halo' and lagrange_point_nr == 1:
            continue

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(2000, 3000, 50):
            try:
                df = load_orbit('../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_' + str(i) + '.txt')
                ax.plot(df['x'], df['y'], df['z'])
            except FileNotFoundError:
                pass

        plt.savefig('../data/figures/family_' + orbit_type + '_L' + str(lagrange_point_nr) + '.png')
        plt.close()

