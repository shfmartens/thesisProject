import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
import time
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


orbit_types = ['horizontal', 'halo', 'vertical']
lagrange_point_nrs = [1, 2]

blues = sns.color_palette('Blues', 100)
greens = sns.color_palette('BuGn', 100)
n_colors = 3
plottingColors = {'lambda1': blues[40],
                   'lambda2': greens[50],
                   'lambda3': blues[90],
                   'lambda4': blues[90],
                   'lambda5': greens[70],
                   'lambda6': blues[60],
                   'singleLine': blues[80],
                   'doubleLine': [greens[50], blues[80]],
                   # 'tripleLine': [blues[80], greens[50], blues[40]],
                   'tripleLine': [sns.color_palette("viridis", n_colors)[0], sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[int((n_colors-1)/2)]],
                   'limit': 'black'}

fig = plt.figure(figsize=(7 * (1 + np.sqrt(5)) / 2, 7))
ax = fig.gca()
for lagrange_point_nr in lagrange_point_nrs:
    for idx, orbit_type in enumerate(orbit_types):
        if orbit_type == 'vertical':
            initial_conditions_file_path = '../../data/raw/orbits/extended/L' + str(
                lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
        else:
            initial_conditions_file_path = '../../data/raw/orbits/L' + str(
                lagrange_point_nr) + '_' + orbit_type + '_initial_conditions.txt'
        initial_conditions_incl_m_df = load_initial_conditions_incl_M(initial_conditions_file_path)

        plot_label = orbit_type.capitalize()
        if plot_label == 'Horizontal' or plot_label == 'Vertical':
            plot_label += ' Lyapunov'

        if lagrange_point_nr == 1:
            linestyle = '-'
        else:
            linestyle = '--'

        ax.plot(initial_conditions_incl_m_df[1].values, initial_conditions_incl_m_df[0].values,
                label='$L_' + str(lagrange_point_nr) + '$ ' + plot_label,
                linestyle=linestyle, color=plottingColors['tripleLine'][idx])

ax.legend(frameon=True, loc='upper right')
ax.set_xlabel('T [-]')
ax.set_ylabel('C [-]')
ax.grid(True, which='both', ls=':')

fig.tight_layout()
fig.subplots_adjust(top=0.8)
fig.suptitle('Families overview - Orbital energy and period', size=20)
plt.show()
fig.savefig('../../data/figures/orbits/extended/overview_families_orbital_energy_period.pdf', transparent=True)
fig.savefig('../../data/figures/orbits/extended/overview_families_orbital_energy_period.png', transparent=True, dpi=150)
