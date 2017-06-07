from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from load_data import load_orbit, load_manifold, load_bodies_location
import seaborn as sns
import json


def init():
    for line in lines:
        line.set_data([], [])
    return lines


def animate(i):
    for j, line in enumerate(lines):
        if j < numberOfOrbitsPerManifolds:
            x = manifold_S_plus.xs(j +  1)['x'].tolist()
            y = manifold_S_plus.xs(j + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds <= j < numberOfOrbitsPerManifolds * 2:
            x = manifold_S_min.xs(j - numberOfOrbitsPerManifolds + 1)['x'].tolist()
            y = manifold_S_min.xs(j - numberOfOrbitsPerManifolds + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds * 2 <= j < numberOfOrbitsPerManifolds * 2:
            x = manifold_U_plus.xs(j - numberOfOrbitsPerManifolds * 2 + 1)['x'].tolist()
            y = manifold_U_plus.xs(j - numberOfOrbitsPerManifolds * 2 + 1)['y'].tolist()
        if numberOfOrbitsPerManifolds * 3 <= j:
            x = manifold_U_min.xs(j - numberOfOrbitsPerManifolds * 3 + 1)['x'].tolist()
            y = manifold_U_min.xs(j - numberOfOrbitsPerManifolds * 3 + 1)['y'].tolist()
        line.set_data(x[:i], y[:i])
    try:
        t = manifold_U_min.xs(1).index.values[i]
        plt.title('T = {:.2f}'.format(round(t, 2)), size=22)
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
        plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg-git-20170605-64bit-static/ffmpeg'
        ax = plt.axes(xlim=(0, 2), ylim=(0, 100))
        numberOfOrbits = numberOfOrbitsPerManifolds * 4
        color_palette_green = sns.dark_palette('green', n_colors=numberOfOrbitsPerManifolds)
        color_palette_red = sns.dark_palette('red', n_colors=numberOfOrbitsPerManifolds)
        lines = [plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(numberOfOrbitsPerManifolds)]
        lines.extend([plt.plot([], [], color=color_palette_green[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])
        lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])
        lines.extend([plt.plot([], [], color=color_palette_red[idx])[0] for idx in range(numberOfOrbitsPerManifolds)])

        manifold_S_plus = load_manifold('../data/' + orbit_name + '_W_S_plus.txt')
        manifold_S_min = load_manifold('../data/' + orbit_name + '_W_S_min.txt')
        manifold_U_plus = load_manifold('../data/' + orbit_name + '_W_U_plus.txt')
        manifold_U_min = load_manifold('../data/' + orbit_name + '_W_U_min.txt')

        plt.xlim(xlim)
        plt.ylim(ylim)

        massParameter = 0.0121505810173
        C = float(config[orbit_type][orbit_name]['C'])
        x_range = np.arange(-2.0, 2.0, 0.001)
        y_range = np.arange(-2.0, 2.0, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)
        plt.contourf(X, Y, Z, levels=[-1, 0], colors='grey')

        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        bodies = load_bodies_location()
        for body in bodies:
            x_body = bodies[body]['r'] * np.outer(np.cos(phi), np.sin(theta)) + bodies[body]['x']
            y_body = bodies[body]['r'] * np.outer(np.sin(phi), np.sin(theta)) + bodies[body]['y']
            plt.plot(x_body, y_body, color='black')

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=int(len(manifold_U_min.xs(1)['x'])*2), interval=1, blit=True)


        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Koen Langemeijer'))#, bitrate=5000)
        anim.save(('../data/animations/' + orbit_name + '_xy_zoom.mp4'), writer=writer)
        # plt.show()
