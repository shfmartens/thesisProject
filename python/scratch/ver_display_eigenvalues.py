import json
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


orbit_type = 'halo'
source_directory = '../src/verification/'
source_name = 'halo_verification_l1'
plot_suptitle = 'Halo verification L1 (Howell)'

with open(source_directory + source_name + '.json') as data_file:
    config = json.load(data_file)

df = pd.DataFrame.from_dict(config[orbit_type]).T

plt.plot(figsize=(20, 10))
ax = plt.gca()
circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
ax.add_patch(circ)

colors = sns.color_palette("Blues", n_colors=6)

row_nr = 0
for idx, row in df.iterrows():
    label = '$C$ = ' + str(np.round(row['C'], 3))
    x = []
    y = []

    for j in range(1, 7):
        x.append(row['l_' + str(j) + '_re'])
        y.append(row['l_' + str(j) + '_im'])
    ax.scatter(x, y, color=colors[row_nr], label=label)
    row_nr += 1

# Shrink current axis
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.95])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
plt.suptitle(plot_suptitle, size=20)
plt.savefig(source_directory + 'eigenvalues.png')
# plt.show()
