import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
import matplotlib.gridspec as gridspec
#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

from load_data import load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity


class DisplayPoincarePlanar:

    def __init__(self, w_s, w_u, u_section):
        self.WS = load_manifold('../../data/raw/' + w_s + '.txt')
        self.WU = load_manifold('../../data/raw/' + w_u + '.txt')
        self.numberOfOrbitsPerManifold = len(set(self.WS.index.get_level_values(0)))
        self.U_section = u_section

        # Select last entry of manifolds
        ls_s = []
        ls_u = []

        for i in range(100):
            ls_s.append(self.WS.xs(i).tail(1))
            ls_u.append(self.WU.xs(i).tail(1))

        self.poincareWS = pd.concat(ls_s).reset_index(drop=True)
        self.poincareWU = pd.concat(ls_u).reset_index(drop=True)

        self.figSize = (40, 40)
        self.titleSize = 20
        self.suptitleSize = 30

        EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
        SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
        MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
        self.massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)
        pass

    def plot_manifolds(self):
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        fig = plt.figure()
        ax = fig.gca()
        for i in range(self.numberOfOrbitsPerManifold):
            plt.plot(self.WS.xs(i)['x'], self.WS.xs(i)['y'], color=color_palette_green[i])
            plt.plot(self.WU.xs(i)['x'], self.WU.xs(i)['y'], color=color_palette_red[i])

        C = 3.15
        x_range = np.arange(0.7, 1.3, 0.001)
        y_range = np.arange(-0.3, 0.3, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)

        if Z.min() < 0:
            plt.contourf(X, Y, Z, [Z.min(), 0], colors='black', alpha=0.2)

        if self.U_section == (2 or 3):
            ax.axvline(1 - self.massParameter, color='black')

            bodies_df = load_bodies_location()
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
            y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
            plt.scatter(x, y, color='black')
        pass

    def plot_poincare_sections(self):
        axes = ['x', 'y', 'z']
        fig, axarr = plt.subplots(1, 3)
        for idx, axis in enumerate(axes):
            axarr[idx].scatter(self.poincareWS[axis].values, self.poincareWS[axis + 'dot'].values, color='g', alpha=0.5)
            axarr[idx].scatter(self.poincareWU[axis].values, self.poincareWU[axis + 'dot'].values, color='r', alpha=0.5)

            axarr[idx].set_xlabel(axis)
            axarr[idx].set_ylabel('$\dot{' + axis + '}$')
            axarr[idx].grid()
            # axarr[idx].set_aspect('equal', 'datalim')

        title = '$U_' + str(self.U_section) + '$'
        axarr[1].set_title(title)
        pass

    def plot_result(self):
        color_palette_green = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        color_palette_red = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)

        C = 3.15
        x_range = np.arange(0.8, 1.2, 0.001)
        y_range = np.arange(-0.15, 0.15, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, C)

        plt.figure(figsize=(5*(1+np.sqrt(5))/2, 5))
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])

        # Subplot 1: manifolds
        for i in range(self.numberOfOrbitsPerManifold):
            ax1.plot(self.WS.xs(i)['x'], self.WS.xs(i)['y'], color=color_palette_green[i], alpha=0.5)
            ax1.plot(self.WU.xs(i)['x'], self.WU.xs(i)['y'], color=color_palette_red[i], alpha=0.5)
        ax1.plot(self.WS.xs(46)['x'], self.WS.xs(46)['y'], color='black')
        ax1.plot(self.WS.xs(37)['x'], self.WS.xs(37)['y'], color='black')
        ax1.plot(self.WU.xs(65)['x'], self.WU.xs(65)['y'], color='black')
        ax1.plot(self.WU.xs(77)['x'], self.WU.xs(77)['y'], color='black')

        if Z.min() < 0:
            ax1.contourf(X, Y, Z, [Z.min(), 0], colors='black', alpha=0.2)

        if self.U_section == 2:
            # ax1.axvline(1 - self.massParameter, color='black')
            ax1.plot([1 - self.massParameter, 1 - self.massParameter], [-0.11, 0.01], color='black', linewidth=3)
            ax1.text(s='$\sum_2$', x=(1-0.5*self.massParameter), y=0.01)

        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))
        # ax1.scatter(x, y, color='black', s=1)
        ax1.contourf(x, y, z, [z.min(), 0], colors='black')
        ax1.grid(True, which='both', ls=':')

        # Subplot 2: poincare
        v_max = 2
        ax2.plot(self.poincareWS[abs(self.poincareWS['ydot']) < v_max]['y'].values, self.poincareWS[abs(self.poincareWS['ydot']) < v_max]['ydot'].values, color='g')
        ax2.plot(self.poincareWU[abs(self.poincareWU['ydot']) < v_max]['y'].values, self.poincareWU[abs(self.poincareWU['ydot']) < v_max]['ydot'].values, color='r')
        ax2.scatter(self.poincareWS['y'][46], self.poincareWS['ydot'][46], color='black')
        ax2.scatter(self.poincareWU['y'][65], self.poincareWU['ydot'][65], color='black')
        ax2.scatter(self.poincareWS['y'][37], self.poincareWS['ydot'][37], color='black')
        ax2.scatter(self.poincareWU['y'][77], self.poincareWU['ydot'][77], color='black')
        #     y_1   y_2
        # WS   46    41
        # WU   65    83
        # x_intersection = -0.0491777
        # print(self.poincareWS[self.poincareWS['y'] == x_intersection + abs(self.poincareWS['y'] - x_intersection).min()])
        # print(self.poincareWS[self.poincareWS['y'] == x_intersection - abs(self.poincareWS['y'] - x_intersection).min()])
        # print(self.poincareWU[self.poincareWU['y'] == x_intersection + abs(self.poincareWU['y'] - x_intersection).min()])
        # print(self.poincareWU[self.poincareWU['y'] == x_intersection - abs(self.poincareWU['y'] - x_intersection).min()])

        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$\dot{y}$')
        ax2.set_ylim([-1.5, 1.5])
        ax2.grid(True, which='both', ls=':')

        title = '$\sum_' + str(self.U_section) + '$'
        ax2.set_title(title)

        # Subplot 3: x error
        ax3.axhline(abs(self.poincareWS['y'][46] - self.poincareWU['y'][65]))
        ax3.axhline(abs(self.poincareWS['y'][37] - self.poincareWU['y'][77]))
        ax3.axhline(abs(self.poincareWS['ydot'][46] - self.poincareWU['ydot'][65]))
        ax3.axhline(abs(self.poincareWS['ydot'][37] - self.poincareWU['ydot'][77]))
        ax3.semilogy((1 - self.massParameter) - self.poincareWS['x'].values)
        ax3.set_ylabel('$\| x - (1-\mu) \|$')
        ax3.set_xlabel('orbitId [-]')
        ax3.grid(True, which='both', ls=':')
        plt.tight_layout()

        # plt.savefig('../../data/figures/heteroclinic_connection.pdf')
        pass


if __name__ == '__main__':

    display_poincare_planar = DisplayPoincarePlanar(w_s='L1_horizontal_330_W_U_plus', w_u='L2_horizontal_373_W_S_plus', u_section=2)
    # display_poincare_planar.plot_manifolds()
    # display_poincare_planar.plot_poincare_sections()
    display_poincare_planar.plot_result()
    plt.show()
