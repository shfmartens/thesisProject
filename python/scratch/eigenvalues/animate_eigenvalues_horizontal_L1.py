import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import numpy as np
from load_data import load_orbit, load_manifold, load_initial_conditions, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
import json
import pandas as pd


def init():
    lines[0].set_data([], [])
    lines[0].set_3d_properties([])
    lines[1].set_data([], [])
    for k in range(2, 8):
        lines[k].set_offsets([])
    return lines


def animate(i):

    label_orbit = '$x_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['x']), 3)) \
                  + '\n$y_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['y']), 3)) \
                  + '\n$z_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['z']), 3)) \
                  + '\n$\dot{x}_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['xdot']), 3)) \
                  + '\n$\dot{y}_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['ydot']), 3)) \
                  + '\n$\dot{z}_0$ = ' + str(np.round(float(initial_conditions_df.iloc[i]['zdot']), 3))

    lines[0].set_data(orbit[i]['x'].values, orbit[i]['y'].values)
    lines[0].set_3d_properties(orbit[i]['z'].values)
    lines[0].set_label(label_orbit)
    fig.suptitle('C = ' + str(np.round(initial_conditions_df.iloc[i]['C'], 2)) +
                 ', T = ' + str(np.round(initial_conditions_df.iloc[i]['T'], 2)), size=30)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot 2
    if orbit_type == 'horizontal':
        lines[1].set_data(orbit[i]['x'].values, orbit[i]['y'].values)
        pass
    if orbit_type == 'vertical':
        lines[1].set_data(orbit[i]['y'].values, orbit[i]['z'].values)
        pass

    for j in range(1, 7):
        x = float(initial_conditions_df.iloc[i]['lambda' + str(j) + 'real'])
        y = float(initial_conditions_df.iloc[i]['lambda' + str(j) + 'imag'])

        lines[j+1].set_offsets([x, y])
        lines[j+1].set_label('$\lambda_' + str(j) + '$ = (' + str(np.round(x, 2)) + ', ' + str(np.round(y, 2)) + ')')

    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return lines


orbit_type = 'horizontal'
lagrange_point_nr = 1

# Load data
initial_conditions_file_path = '../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_initial_conditions.txt'
initial_conditions_df = load_initial_conditions(initial_conditions_file_path)

orbit_ids = list(range(len(initial_conditions_df)))

orbit = []
for orbit_id in orbit_ids:
    orbit.append(load_orbit('../data/raw/' + orbit_type + '_L' + str(lagrange_point_nr) + '_' + str(orbit_id) + '.txt'))

label_orbit = '$x_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['x'], 3)) \
              + '\n$y_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['y'], 3)) \
              + '\n$z_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['z'], 3)) \
              + '\n$\dot{x}_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['xdot'], 3)) \
              + '\n$\dot{y}_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['ydot'], 3)) \
              + '\n$\dot{z}_0$ = ' + str(np.round(initial_conditions_df.iloc[0]['zdot'], 3))

# Create plot
fig = plt.figure(figsize=(20, 20))
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg-git-20170607-64bit-static/ffmpeg'
gs = GridSpec(2, 2)
ax1 = plt.subplot(gs[0, :-1], projection='3d')
ax2 = plt.subplot(gs[0, -1])
ax3 = plt.subplot(gs[1, :])

# Plot 1: 3D-view of orbit
line1 = ax1.plot(orbit[0]['x'].values, orbit[0]['y'].values, orbit[0]['z'].values, color='darkblue', label=label_orbit)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # Shrink current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Put a legend to the right of the current axis
ax1.set_xlim([0.8, 1.1])
ax1.set_ylim([-0.15, 0.15])
ax1.set_zlim([-0.15, 0.15])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D view', size=20)

phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
bodies = load_bodies_location()
for body in bodies:
    x_body = bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + bodies[body]['x']
    y_body = bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies[body]['y']
    z_body = bodies[body]['r'] * np.cos(theta) + bodies[body]['z']
    ax1.plot_surface(x_body, y_body, z_body, color='black')

lagrange_points = load_lagrange_points_location()
for lagrange_point in ['L1', 'L2']:
    ax1.scatter3D(lagrange_points[lagrange_point]['x'],
                  lagrange_points[lagrange_point]['y'],
                  lagrange_points[lagrange_point]['z'], color='grey', marker='d', alpha=0.75)
    ax1.text(lagrange_points[lagrange_point]['x'],
             lagrange_points[lagrange_point]['y'],
             lagrange_points[lagrange_point]['z'], lagrange_point, size=16)


# Plot 2: 2D-view of orbit
if orbit_type == 'horizontal':
    line2 = ax2.plot(orbit[0]['x'].values, orbit[0]['y'].values, color='darkblue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Plot location of Moon and Lagrange points
    ax2.scatter(bodies['Earth']['x'], bodies['Earth']['y'], color='black')
    ax2.scatter(bodies['Moon']['x'], bodies['Moon']['y'], color='black')
    for lagrange_point in ['L1', 'L2']:
        ax2.scatter(lagrange_points[lagrange_point]['x'],
                    lagrange_points[lagrange_point]['y'], color='grey', marker='d', alpha=0.75)
    ylim_max = (ax2.get_xlim()[1]-ax2.get_xlim()[0])/2
    ax2.set_ylim([-ylim_max, ylim_max])
    pass
if orbit_type == 'vertical':
    line2 = ax2.plot(orbit[0]['y'].values, orbit[0]['z'].values, color='darkblue')
    ax2.set_xlabel('y')
    ax2.set_ylabel('z')
    pass

ax2.set_title('2D view', size=20)
lines = [line1[0], line2[0]]

# Plot 3: eigenvalues
circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None', linestyle=':')
ax3.add_patch(circ)

for j in range(1, 7):
    x = float(initial_conditions_df.iloc[0]['lambda' + str(j) + 'real'])
    y = float(initial_conditions_df.iloc[0]['lambda' + str(j) + 'imag'])
    label_eigenvalues = '$\lambda_' + str(j) + '$ = (' + str(np.round(x, 2)) + ', ' + str(np.round(y, 2))
    lines.append(ax3.scatter(x, y, color='darkblue', label=label_eigenvalues))

box = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # Shrink current axis
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Put a legend to the right of the current axis
ax3.set_xlim([-10, 10])
ax3.set_ylim([-5, 5])
ax3.set_xlabel('Re')
ax3.set_ylabel('Im')
ax3.set_title('Eigenvalues', size=20)

fig.suptitle('C = ' + str(np.round(initial_conditions_df.iloc[0]['C'], 2)) +
             ', T = ' + str(np.round(initial_conditions_df.iloc[0]['T'], 2)), size=30)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(len(orbit)), interval=5000, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Koen Langemeijer'))
anim.save(('../data/animations/' + orbit_type + '_L' + str(lagrange_point_nr) + '_eigenvalues.mp4'), writer=writer)
