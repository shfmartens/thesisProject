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
from itertools import product
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

from load_data import load_manifold, load_bodies_location, load_lagrange_points_location, cr3bp_velocity, load_initial_conditions_incl_M


class DisplayPoincarePlanar:

    def __init__(self, w_s, w_u, u_section, round_order_connection):

        # orbit_idx = []
        # for l_point, orbit_type in [(l_point_1, orbit_type_1), (l_point_2, orbit_type_2)]
        #     df = load_initial_conditions_incl_M('../../data/raw/' + l_point + '_' + orbit_type + '_initial_conditions.txt')
        #     orbit_idx.append(df[abs(df[0] - C) == (abs(df[0] - C)).min()].index)

        l_point, orbit_type, orbit_id, w, s, sign = w_s.split('_')
        self.C = [load_initial_conditions_incl_M('../../data/raw/' + l_point + '_' + orbit_type + '_initial_conditions.txt').xs(int(orbit_id))[0]]
        l_point, orbit_type, orbit_id, w, s, sign = w_u.split('_')
        self.C.append(load_initial_conditions_incl_M('../../data/raw/' + l_point + '_' + orbit_type + '_initial_conditions.txt').xs(int(orbit_id))[0])

        self.WS = load_manifold('../../data/raw/' + w_s + '.txt')
        self.WU = load_manifold('../../data/raw/' + w_u + '.txt')
        self.numberOfOrbitsPerManifold = len(set(self.WS.index.get_level_values(0)))
        self.U_section = u_section
        self.roundOrderConnection = round_order_connection
        # Select last entry of manifolds
        ls_s = []
        ls_u = []

        for i in range(len(set(self.WS.index.get_level_values(0)))):
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

        plt.figure(figsize=(5*(1+np.sqrt(5))/2, 5))
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])

        # Subplot 1: manifolds
        for i in range(self.numberOfOrbitsPerManifold):
            ax1.plot(self.WS.xs(i)['x'], self.WS.xs(i)['y'], color=color_palette_green[i], alpha=0.5)
            ax1.plot(self.WU.xs(i)['x'], self.WU.xs(i)['y'], color=color_palette_red[i], alpha=0.5)

        x_range = np.arange(ax1.get_xlim()[0], ax1.get_xlim()[1], 0.001)
        y_range = np.arange(ax1.get_ylim()[0]*1.2, ax1.get_ylim()[1]*1.2, 0.001)
        X, Y = np.meshgrid(x_range, y_range)
        Z = cr3bp_velocity(X, Y, self.C[0])

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
        ax1.set_title('$C_1 = ' + str(self.C[0]) + ', C_2 = ' + str(self.C[1]) + '$')
        ax1.set_aspect('equal', adjustable='box')

        # Subplot 2: poincare
        v_max = 2
        if self.U_section in set([2, 3]):
            variable_axis = 'y'
        elif self.U_section in set([3, 4]):
            variable_axis = 'x'

        ax2.plot(self.poincareWS[abs(self.poincareWS[variable_axis + 'dot']) < v_max][variable_axis].values, self.poincareWS[abs(self.poincareWS[variable_axis + 'dot']) < v_max][variable_axis + 'dot'].values, color='g')
        ax2.plot(self.poincareWU[abs(self.poincareWU[variable_axis + 'dot']) < v_max][variable_axis].values, self.poincareWU[abs(self.poincareWU[variable_axis + 'dot']) < v_max][variable_axis + 'dot'].values, color='r')

        # Find entry of lowest difference in derivative of variable_axis (which is the variable with maximum spread in phase plane)
        s = set(np.round(self.poincareWS[variable_axis + 'dot'], self.roundOrderConnection))
        intersections = list(s.intersection(np.round(self.poincareWU[variable_axis + 'dot'], self.roundOrderConnection)))
        poincare_w_s_in_v = self.poincareWS[np.round(self.poincareWS[variable_axis + 'dot'], self.roundOrderConnection).isin(intersections)]
        poincare_w_u_in_v = self.poincareWU[np.round(self.poincareWU[variable_axis + 'dot'], self.roundOrderConnection).isin(intersections)]

        # Use tuples to bind pairs of position and velocity
        subset = poincare_w_s_in_v[[variable_axis, variable_axis + 'dot']]
        tuples_w_s_in_v = [tuple(np.round(x, self.roundOrderConnection)) for x in subset.values]
        subset = poincare_w_u_in_v[[variable_axis, variable_axis + 'dot']]
        tuples_w_u_in_v = [tuple(np.round(x, self.roundOrderConnection)) for x in subset.values]

        s = set(tuples_w_s_in_v)
        intersections = list(s.intersection(tuples_w_u_in_v))

        for intersection in intersections:
            poincare_temp = poincare_w_s_in_v[np.round(poincare_w_s_in_v[variable_axis], self.roundOrderConnection) == intersection[0]]
            poincare_temp = poincare_temp[np.round(poincare_temp[variable_axis + 'dot'], self.roundOrderConnection) == intersection[1]]
            idx_s = int(poincare_temp.index.values)
            poincare_temp = poincare_w_u_in_v[np.round(poincare_w_u_in_v[variable_axis], self.roundOrderConnection) == intersection[0]]
            poincare_temp = poincare_temp[np.round(poincare_temp[variable_axis + 'dot'], self.roundOrderConnection) == intersection[1]]
            idx_u = int(poincare_temp.index.values)

            ax2.scatter(self.poincareWS[variable_axis][idx_s], self.poincareWS[variable_axis + 'dot'][idx_s], color='black')
            ax2.scatter(self.poincareWU[variable_axis][idx_u], self.poincareWU[variable_axis + 'dot'][idx_u], color='black')

            ax1.plot(self.WS.xs(idx_s)['x'], self.WS.xs(idx_s)['y'], color='black')
            ax1.plot(self.WU.xs(idx_u)['x'], self.WU.xs(idx_u)['y'], color='black')

        ax2.set_xlabel('$' + variable_axis + '$')
        ax2.set_ylabel('$\dot{' + variable_axis + '}$')
        ax2.set_ylim([-1.5, 1.5])
        ax2.grid(True, which='both', ls=':')

        title = '$\sum_' + str(self.U_section) + '$'
        ax2.set_title(title)

        # Subplot 3: x error
        # ax3.axhline(abs(self.poincareWS['y'][46] - self.poincareWU['y'][65]))
        # ax3.axhline(abs(self.poincareWS['y'][37] - self.poincareWU['y'][77]))
        # ax3.axhline(abs(self.poincareWS['ydot'][46] - self.poincareWU['ydot'][65]))
        # ax3.axhline(abs(self.poincareWS['ydot'][37] - self.poincareWU['ydot'][77]))
        if self.U_section == (2 or 3):
            ax3.semilogy(abs((1 - self.massParameter) - self.poincareWS['x'].values), color='g')
            ax3.semilogy(abs((1 - self.massParameter) - self.poincareWU['x'].values), color='r')
            ax3.set_ylabel('$\| x - (1-\mu) \|$')
        elif self.U_section == (1 or 4):
            ax3.semilogy(abs(self.poincareWS['y'].values), color='g')
            ax3.semilogy(abs(self.poincareWU['y'].values), color='r')
            ax3.set_ylabel('$\| y \|$')

        ax3.set_xlabel('orbitId [-]')
        ax3.grid(True, which='both', ls=':')
        plt.tight_layout()

        # plt.savefig('../../data/figures/heteroclinic_connection.pdf')
        # plt.savefig('../../data/figures/homoclinic_connection.pdf')
        pass


if __name__ == '__main__':
    # display_poincare_planar = DisplayPoincarePlanar(w_s='L2_horizontal_373_W_S_plus', w_u='L1_horizontal_330_W_U_plus', u_section=2, round_order_connection=3)
    # display_poincare_planar = DisplayPoincarePlanar(w_s='L1_horizontal_808_W_S_plus', w_u='L1_horizontal_808_W_U_plus', u_section=1, round_order_connection=2)
    display_poincare_planar = DisplayPoincarePlanar(w_s='L2_horizontal_373_W_S_min', w_u='L2_horizontal_373_W_U_plus', u_section=4, round_order_connection=2)
    # display_poincare_planar = DisplayPoincarePlanar(w_s='L2_horizontal_1066_W_S_plus', w_u='L2_horizontal_1066_W_U_min', u_section=4, round_order_connection=2)
    # display_poincare_planar.plot_manifolds()
    # display_poincare_planar.plot_poincare_sections()
    display_poincare_planar.plot_result()
    plt.show()
