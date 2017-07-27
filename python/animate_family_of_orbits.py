import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import numpy as np
from load_data import load_orbit, load_manifold, load_initial_conditions_incl_M, load_bodies_location, load_lagrange_points_location, cr3bp_velocity
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
import json
import pandas as pd


def init():
    for k in range(6):
        lines[k].set_data([], [])
        lines[k].set_3d_properties([])
    return lines


def animate(i):

    lines[0].set_data(horizontal_L1[i]['x'].values, horizontal_L1[i]['y'].values)
    lines[0].set_3d_properties(horizontal_L1[i]['z'].values)
    lines[1].set_data(horizontal_L2[i]['x'].values, horizontal_L2[i]['y'].values)
    lines[1].set_3d_properties(horizontal_L2[i]['z'].values)
    lines[2].set_data(vertical_L1[i]['x'].values, vertical_L1[i]['y'].values)
    lines[2].set_3d_properties(vertical_L1[i]['z'].values)
    lines[3].set_data(vertical_L2[i]['x'].values, vertical_L2[i]['y'].values)
    lines[3].set_3d_properties(vertical_L2[i]['z'].values)
    lines[4].set_data(halo_L1[i]['x'].values, halo_L1[i]['y'].values)
    lines[4].set_3d_properties(halo_L1[i]['z'].values)
    lines[5].set_data(halo_L2[i]['x'].values, halo_L2[i]['y'].values)
    lines[5].set_3d_properties(halo_L2[i]['z'].values)
    # fig.suptitle('C = ' + str(np.round(initial_conditions_vertical_L1.iloc[i][1], 2)), size=30)
    return lines


orbit_type = 'halo'

# Load data
initial_conditions_horizontal_L1 = load_initial_conditions_incl_M('../data/raw/horizontal_L1_initial_conditions.txt')
initial_conditions_horizontal_L2 = load_initial_conditions_incl_M('../data/raw/horizontal_L2_initial_conditions.txt')
initial_conditions_vertical_L1 = load_initial_conditions_incl_M('../data/raw/vertical_L1_initial_conditions.txt')
initial_conditions_vertical_L2 = load_initial_conditions_incl_M('../data/raw/vertical_L2_initial_conditions.txt')
initial_conditions_halo_L1 = load_initial_conditions_incl_M('../data/raw/halo_L1_initial_conditions.txt')
initial_conditions_halo_L2 = load_initial_conditions_incl_M('../data/raw/halo_L2_initial_conditions.txt')

horizontal_L1 = []
horizontal_L2 = []
vertical_L1 = []
vertical_L2 = []
halo_L1 = []
halo_L2 = []

for orbit_id in list(range(len(initial_conditions_horizontal_L1))):
    horizontal_L1.append(load_orbit('../data/raw/horizontal_L1_' + str(orbit_id) + '.txt'))
for orbit_id in list(range(len(initial_conditions_horizontal_L2))):
    horizontal_L2.append(load_orbit('../data/raw/horizontal_L2_' + str(orbit_id) + '.txt'))
for orbit_id in list(range(len(initial_conditions_vertical_L1))):
    vertical_L1.append(load_orbit('../data/raw/vertical_L1_' + str(orbit_id) + '.txt'))
for orbit_id in list(range(len(initial_conditions_vertical_L2))):
    vertical_L2.append(load_orbit('../data/raw/vertical_L2_' + str(orbit_id) + '.txt'))
for orbit_id in list(range(len(initial_conditions_halo_L1))):
    halo_L1.append(load_orbit('../data/raw/halo_L1_' + str(orbit_id) + '.txt'))
for orbit_id in list(range(len(initial_conditions_halo_L2))):
    halo_L2.append(load_orbit('../data/raw/halo_L2_' + str(orbit_id) + '.txt'))

# Create plot
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg-git-20170607-64bit-static/ffmpeg'
fig = plt.figure(figsize=(40, 20))
ax = fig.gca(projection='3d')

# Plot 1: 3D-view of orbit
line1 = ax.plot(horizontal_L1[0]['x'].values, horizontal_L1[0]['y'].values, horizontal_L1[0]['z'].values, color='darkblue')
line2 = ax.plot(horizontal_L2[0]['x'].values, horizontal_L2[0]['y'].values, horizontal_L2[0]['z'].values, color='darkblue')
line3 = ax.plot(vertical_L1[0]['x'].values, vertical_L1[0]['y'].values, vertical_L1[0]['z'].values, color='darkblue')
line4 = ax.plot(vertical_L2[0]['x'].values, vertical_L2[0]['y'].values, vertical_L2[0]['z'].values, color='darkblue')
line5 = ax.plot(halo_L1[0]['x'].values, halo_L1[0]['y'].values, halo_L1[0]['z'].values, color='darkblue')
line6 = ax.plot(halo_L2[0]['x'].values, halo_L2[0]['y'].values, halo_L2[0]['z'].values, color='darkblue')

lines = [line1[0], line2[0], line3[0], line4[0], line5[0], line6[0]]
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([0.3, 1.5])
ax.set_ylim([-0.6, 0.6])
ax.set_zlim([-0.6, 0.6])

phi = np.linspace(0, 2*np.pi, 100)
theta = np.linspace(0, np.pi, 100)
bodies = load_bodies_location()

x_body = bodies['Moon']['r'] * np.outer(np.cos(phi), np.sin(theta)) + bodies['Moon']['x']
y_body = bodies['Moon']['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies['Moon']['y']
z_body = bodies['Moon']['r'] * np.cos(theta) + bodies['Moon']['z']
ax.plot_surface(x_body, y_body, z_body, color='black')

lagrange_points = load_lagrange_points_location()
for lagrange_point in ['L1', 'L2']:
    ax.scatter3D(lagrange_points[lagrange_point]['x'],
                 lagrange_points[lagrange_point]['y'],
                 lagrange_points[lagrange_point]['z'], color='grey', marker='d', alpha=0.75)
    ax.text(lagrange_points[lagrange_point]['x'],
            lagrange_points[lagrange_point]['y'],
            lagrange_points[lagrange_point]['z'], lagrange_point, size=16)

# fig.suptitle('C = ' + str(np.round(initial_conditions_vertical_L1.iloc[0][1], 2)), size=30)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(len(vertical_L1)), interval=5000, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Koen Langemeijer'))
anim.save(('../data/animations/family_of_all_orbits.mp4'), writer=writer)
