import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from load_data import load_orbit


orbit_type = 'halo'
source_directory = '../src/verification/'
source_name = 'halo_verification_l1'
plot_suptitle = 'Halo verification L1 (Howell)'

with open(source_directory + source_name + '.json') as data_file:
    config = json.load(data_file)

df = pd.DataFrame.from_dict(config[orbit_type]).T

f, axarr = plt.subplots(6, 2, figsize=(10, 15))
colors = sns.color_palette("Blues", n_colors=6)

row_nr = 0
for idx, row in df.iterrows():

    orbit = load_orbit('../src/verification/' + idx + '_l1.txt')
    label = '$x_0$ = ' + str(row['x']) + '\n$y_0$ = ' + str(row['y']) + '\n$z_0$ = ' + str(row['z']) \
            + '\n$\dot{x}_0$ = ' + str(row['x_dot']) + '\n$\dot{y}_0$ = ' + str(row['y_dot']) + '\n$\dot{z}_0$ = ' \
            + str(row['z_dot'])
    axarr[row_nr, 0].plot(orbit['x'], orbit['y'], color='darkblue', label=label)

    circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
    axarr[row_nr, 1].add_patch(circ)

    for j in range(1, 7):
        x = row['l_' + str(j) + '_re']
        y = row['l_' + str(j) + '_im']
        axarr[row_nr, 1].scatter(x, y, color=colors[j-1],
                                         label='$\lambda_' + str(j) + '$ = (' + str(np.round(x, 2)) + ', ' + str(np.round(y, 2)) + ')')

    axarr[row_nr, 1].set_title('T/2 = ' + str(np.round(row['T']/2, 2)) + ', C = ' + str(np.round(row['C'], 2)))

    for k in range(2):
        # Shrink current axis
        box = axarr[row_nr, k].get_position()
        axarr[row_nr, k].set_position([box.x0, box.y0, box.width * 0.5, box.height])

        # Put a legend to the right of the current axis
        axarr[row_nr, k].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axarr[row_nr, 0].set_xlim([0.55, 1.15])
    axarr[row_nr, 0].set_ylim([-0.3, 0.3])
    axarr[row_nr, 1].set_xlim([-2, 2])
    axarr[row_nr, 1].set_ylim([-2, 2])
    axarr[row_nr, 1].set_xticks([])
    axarr[row_nr, 1].set_yticks([])
    row_nr += 1

plt.suptitle(plot_suptitle, size=30)
plt.savefig(source_directory + source_name + '.png')
# plt.show()
