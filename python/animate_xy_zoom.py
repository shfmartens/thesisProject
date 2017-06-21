import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from load_data import load_orbit, load_manifold, load_bodies_location, load_lagrange_points_location
import seaborn as sns
import json


def init():
    for line in lines:
        line.set_data([], [])
    return lines


def animate(i):
    for j, line in enumerate(lines):
        if j < numberOfOrbitsPerManifolds:
            x = manifold_S_plus.xs(j + 1)['x'].tolist()
            y = manifold_S_plus.xs(j + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds <= j < numberOfOrbitsPerManifolds * 2:
            x = manifold_S_min.xs(j - numberOfOrbitsPerManifolds + 1)['x'].tolist()
            y = manifold_S_min.xs(j - numberOfOrbitsPerManifolds + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds * 2 <= j < numberOfOrbitsPerManifolds * 3:
            x = manifold_U_plus.xs(j - numberOfOrbitsPerManifolds * 2 + 1)['x'].tolist()
            y = manifold_U_plus.xs(j - numberOfOrbitsPerManifolds * 2 + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds * 3 <= j:
            x = manifold_U_min.xs(j - numberOfOrbitsPerManifolds * 3 + 1)['x'].tolist()
            y = manifold_U_min.xs(j - numberOfOrbitsPerManifolds * 3 + 1)['y'].tolist()
        line.set_data(x[:i], y[:i])
    try:
        t = manifold_U_min.xs(1).index.values[i]
        time_text.set_text('t = {:.2f}'.format(round(abs(t), 2)))
    except IndexError:
        pass

    return lines


def cr3bp_velocity(x_loc, y_loc, c):
    r_1 = np.sqrt((x_loc + massParameter) ** 2 + y_loc ** 2)
    r_2 = np.sqrt((x_loc - 1 + massParameter) ** 2 + y_loc ** 2)
    v = x_loc ** 2 + y_loc ** 2 + 2 * (1 - massParameter) / r_1 + 2 * massParameter / r_2 - c
    return v


with open("../config/config.json") as data_file:
    config = json.load(data_file)


for orbit_type in config.keys():
    for orbit_name in config[orbit_type].keys():
        print(orbit_name)

        numberOfOrbitsPerManifolds = 100
        xlim = [0.5, 1.5]
        ylim = [-0.5, 0.5]

        fig = plt.figure(figsize=(20, 20))
        # plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg_sources/ffmpeg/ffmpeg'
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        # plt.rcParams['animation.codec'] = 'libx264'

        # matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
        ax = plt.axes(xlim=(0, 2), ylim=(0, 100))
        numberOfOrbits = numberOfOrbitsPerManifolds * 4
        color_palette_green = sns.dark_palette('green', n_colors=numberOfOrbitsPerManifolds)
        color_palette_red = sns.dark_palette('red', n_colors=numberOfOrbitsPerManifolds)
        lines = [plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(numberOfOrbitsPerManifolds)]
        lines.extend([plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])
        lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])
        lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])

        manifold_S_plus = load_manifold('../data/raw/' + orbit_name + '_W_S_plus.txt')
        manifold_S_min = load_manifold('../data/raw/' + orbit_name + '_W_S_min.txt')
        manifold_U_plus = load_manifold('../data/raw/' + orbit_name + '_W_U_plus.txt')
        manifold_U_min = load_manifold('../data/raw/' + orbit_name + '_W_U_min.txt')

        plt.xlim(xlim)
        plt.ylim(ylim)
        time_text = ax.text(0.01, 0.01, s='', transform=ax.transAxes, size=22)

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

        C = float(config[orbit_type][orbit_name]['C'])
        x_range = np.arange(0.5, 1.5, 0.001)
        y_range = np.arange(-0.5, 0.5, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)
        if Z.min() < 0:
            plt.contourf(X, Y, Z, [Z.min(), 0], colors='black', alpha=0.05)

        title = 'C = ' + str(round(C, 3)) + \
                ', T = ' + str(round(float(config[orbit_type][orbit_name]['T']), 3))
        plt.title(title, size=30)

        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        bodies = load_bodies_location()
        for body in bodies:
            x_body = bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + bodies[body]['x']
            y_body = bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies[body]['y']
            plt.plot(x_body, y_body, color='black')

        lagrange_points = load_lagrange_points_location()
        for lagrange_point in ['L1', 'L2']:
            ax.scatter(lagrange_points[lagrange_point]['x'],
                       lagrange_points[lagrange_point]['y'], color='grey', marker='d', alpha=0.75)
            ax.text(lagrange_points[lagrange_point]['x'],
                    lagrange_points[lagrange_point]['y'], lagrange_point, size=16)

        numberOfFrames = 0
        for index in range(1, numberOfOrbitsPerManifolds + 1):
            numberOfFrames = max(numberOfFrames, len(manifold_S_plus.xs(index)['x']))
            numberOfFrames = max(numberOfFrames, len(manifold_S_min.xs(index)['x']))
            numberOfFrames = max(numberOfFrames, len(manifold_U_plus.xs(index)['x']))
            numberOfFrames = max(numberOfFrames, len(manifold_U_min.xs(index)['x']))

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=int(numberOfFrames), interval=1, blit=True)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Koen Langemeijer'))
        # writer = animation.FFMpegWriter(fps=30)
        anim.save(('../data/animations/' + orbit_name + '_xy_zoom.mp4'), writer=writer)

# plt.show()
