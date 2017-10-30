import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, load_initial_conditions_incl_M, load_manifold


def findIdxCorrespondingToC(c_level):
    # Find index of orbit which is closest to the desired Jacobi energy C

    orbit_types = ['horizontal', 'halo', 'vertical']
    lagrange_point_nrs = [1, 2]

    print('Index for orbit closest to C = ' + str(c_level) + '\n')

    for idx, orbit_type in enumerate(orbit_types):
        for lagrange_point_nr in lagrange_point_nrs:
            initial_conditions_file_path = '../../data/raw/orbits/L' + str(
                lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
            initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

            if (initial_conditions_incl_m_df[initial_conditions_incl_m_df[0] == c_level - min(abs(initial_conditions_incl_m_df[0] - c_level))]).empty:
                row = initial_conditions_incl_m_df[initial_conditions_incl_m_df[0] == c_level + min(abs(initial_conditions_incl_m_df[0] - c_level))]
            else:
                row = initial_conditions_incl_m_df[initial_conditions_incl_m_df[0] == c_level - min(abs(initial_conditions_incl_m_df[0] - c_level))]

            print('L' + str(lagrange_point_nr) + ' ' + orbit_type + ' at index: ' + str(row.index[0])
                  + ' (dC = |' + str(abs(row[0].values[0] - c_level)) + '|)')
    pass


if __name__ == '__main__':
    c_level = 3.1
    findIdxCorrespondingToC(c_level)
