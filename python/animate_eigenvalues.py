import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
import json
import pandas as pd


def init():
    lines[0].set_data([], [])
    lines[0].set_3d_properties([])
    lines[1].set_offsets([])
    return lines


def animate(i):
    df1 = dfT['near_vertical_' + str(i + 1)]
    df2 = df1.T
    label = '$x_0$ = ' + str(np.round(df2['x'], 3)) \
            + '\n$y_0$ = ' + str(np.round(df2['y'], 3)) \
            + '\n$z_0$ = ' + str(np.round(df2['z'], 3)) \
            + '\n$\dot{x}_0$ = ' + str(np.round(df2['x_dot'], 3)) \
            + '\n$\dot{y}_0$ = ' + str(np.round(df2['y_dot'], 3)) \
            + '\n$\dot{z}_0$ = ' + str(np.round(df2['z_dot'], 3))

    lines[0].set_data(orbit[i]['x'].values, orbit[i]['y'].values)
    lines[0].set_3d_properties(orbit[i]['z'].values)
    lines[0].set_label(label)
    fig.suptitle('C = ' + str(np.round(df2['C'], 2)) + ', T = ' + str(np.round(df2['T'], 2)), size=30)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    x = []
    y = []
    for j in range(1, 7):
        x.append(df2['l_' + str(j) + '_re'])
        y.append(df2['l_' + str(j) + '_im'])

    lines[1].set_offsets([x, y])
    return lines


with open("../config/config.json") as data_file:
    config = json.load(data_file)


for orbit_type in config.keys():
    df = pd.DataFrame.from_dict(config[orbit_type]).T

    orbit_names = sorted(list(config[orbit_type].keys()))
    orbit = []
    for orbit_name in orbit_names:
        orbit.append(load_orbit('../data/raw/' + orbit_name + '_final_orbit.txt'))

    dfT = df.T
    df1 = dfT['near_vertical_1']
    df2 = df1.T
    label = '$x_0$ = ' + str(np.round(df2['x'], 3)) \
            + '\n$y_0$ = ' + str(np.round(df2['y'], 3)) \
            + '\n$z_0$ = ' + str(np.round(df2['z'], 3)) \
            + '\n$\dot{x}_0$ = ' + str(np.round(df2['x_dot'], 3)) \
            + '\n$\dot{y}_0$ = ' + str(np.round(df2['y_dot'], 3)) \
            + '\n$\dot{z}_0$ = ' + str(np.round(df2['z_dot'], 3))

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    line1 = ax1.plot(orbit[0]['x'].values, orbit[0]['y'].values, orbit[0]['z'].values, color='darkblue', label=label)

    # Shrink current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.set_xlim([0.8, 1.1])
    ax1.set_ylim([-0.15, 0.15])
    ax1.set_zlim([-0.15, 0.15])
    fig.suptitle('C = ' + str(np.round(df2['C'], 2)) + ', T = ' + str(np.round(df2['T'], 2)), size=30)

    ax2 = fig.add_subplot(2, 1, 2)
    line2 = ax2.scatter(1, 1)

    lines = [line1[0], line2]
    circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
    ax2.add_patch(circ)

    # Shrink current axis
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-5, 5])


    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(len(orbit)), interval=30000, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Koen Langemeijer'))
    anim.save(('../data/animations/eigenvalues.mp4'), writer=writer)
