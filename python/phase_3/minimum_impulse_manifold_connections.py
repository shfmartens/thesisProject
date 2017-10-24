import numpy as np
import pandas as pd
import json
import matplotlib
# matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
import time
sys.path.append('../util')
from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm


class MinimumImpulseManifoldConnections:
    def __init__(self, number_of_orbits_per_manifold=100, max_position_dev=1e-3):
        self.suptitleSize = 20
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)

        self.numberOfOrbitsPerManifold = number_of_orbits_per_manifold
        self.maximumPositionDeviation = max_position_dev
        # TODO fix proper constants
        self.positionDimensionFactor = 384400
        self.velocityDimensionFactor = 384400e3 / (27.3 * 24 * 3600)

        df = pd.read_table('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/vertical_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
        df.columns = ['theta', 'tau1', 't1', 'x1', 'y1', 'z1', 'xdot1', 'ydot1', 'zdot1', 'tau2', 't2', 'x2', 'y2', 'z2', 'xdot2', 'ydot2', 'zdot2']
        df['dr'] = np.sqrt((df['x1'] - df['x2']) ** 2 + (df['y1'] - df['y2']) ** 2 + (df['z1'] - df['z2']) ** 2)
        df['dv'] = np.sqrt((df['xdot1'] - df['xdot2']) ** 2 + (df['ydot1'] - df['ydot2']) ** 2 + (df['zdot1'] - df['zdot2']) ** 2)
        df = df.set_index('theta')

        df['dr'] = df['dr'] * self.positionDimensionFactor
        df['dv'] = df['dv'] * self.velocityDimensionFactor
        self.thetaRangeList = sorted(list(df.index))
        self.minimumImpulse = df[df['dv'] != 0]
        self.numberOfPlotColorIndices = len(self.thetaRangeList)

        self.plotColorIndexBasedOnTheta = {theta: int(np.round(
            (theta - min(self.thetaRangeList)) / (max(self.thetaRangeList) - min(self.thetaRangeList)) * (
            self.numberOfPlotColorIndices - 1))) for theta in self.thetaRangeList}

        blues = sns.color_palette('Blues', 100)
        greens = sns.color_palette('BuGn', 100)
        self.colorPaletteStable = sns.dark_palette('green', n_colors=self.numberOfOrbitsPerManifold)
        self.colorPaletteUnstable = sns.dark_palette('red', n_colors=self.numberOfOrbitsPerManifold)
        self.plottingColors = {'lambda1': blues[40],
                               'lambda2': greens[50],
                               'lambda3': blues[90],
                               'lambda4': blues[90],
                               'lambda5': greens[70],
                               'lambda6': blues[60],
                               'singleLine': blues[80],
                               'doubleLine': [greens[50], blues[80]],
                               'tripleLine': [blues[40], greens[50], blues[80]],
                               'limit': 'black'}
        pass

    def plot_impulse_angle(self):
        fig, axarr = plt.subplots(2, 2, figsize=self.figSize)

        axarr[0, 0].plot(self.minimumImpulse['dv'], color='navy')
        axarr[0, 0].set_xlabel('$ \\theta [^\circ] $')
        axarr[0, 0].set_ylabel('$ \Delta \mathbf{V} [m/s] $')
        axarr[0, 0].set_title('Velocity discrepancy')

        axarr[0, 1].plot(self.minimumImpulse['dr'], color='navy')
        axarr[0, 1].axhline(self.maximumPositionDeviation * self.positionDimensionFactor, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        axarr[0, 1].set_xlabel('$ \\theta [^\circ]$')
        axarr[0, 1].set_ylabel('$ \Delta \mathbf{r} [km] $')
        axarr[0, 1].set_title('Position discrepancy')

        theta_normalized = [(theta-min(list(self.minimumImpulse.index)))/(max(list(self.minimumImpulse.index))-min(list(self.minimumImpulse.index))) for theta in self.thetaRangeList]
        colors = matplotlib.colors.ListedColormap(sns.color_palette("YlGnBu_r"))(theta_normalized)

        sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("YlGnBu_r")),
                                   norm=plt.Normalize(vmin=min(list(self.minimumImpulse.index)), vmax=max(list(self.minimumImpulse.index))))
        # clean the array of the scalar mappable
        sm._A = []

        for theta in self.thetaRangeList:
            plot_color = colors[self.plotColorIndexBasedOnTheta[theta]]
            try:
                axarr[1, 0].scatter(self.minimumImpulse.loc[theta]['tau1'], self.minimumImpulse.loc[theta]['tau2'], color=plot_color, alpha=1)
            except KeyError:
                pass

        axarr[1, 0].set_xlabel('$ \\tau_1 [-] $')
        axarr[1, 0].set_ylabel('$ \\tau_2 [-] $')
        # axarr[1, 0].set_xlim([0.577, 0.581])
        # axarr[1, 0].set_ylim([0.577, 0.581])
        # axarr[1, 0].set_xlim(max(axarr[1, 0].get_xlim(), axarr[1, 0].get_ylim()))
        # axarr[1, 0].set_ylim(max(axarr[1, 0].get_xlim(), axarr[1, 0].get_ylim()))
        axarr[1, 0].set_title('Phases on departure/target orbit')

        for i in range(2):
            for j in range(2):
                axarr[i, j].grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        cax, kw = matplotlib.colorbar.make_axes([axarr[1, 0]])
        plt.colorbar(sm, cax=cax, label='$\\theta [^\circ]$', **kw)

        plt.suptitle('Near-heteroclinic connection for $min(\Delta V) \enskip \\forall \Delta r < 10^{-3}$ (at $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$, C = 3.1, \#' +
                     str(self.numberOfOrbitsPerManifold) + ')', size=self.suptitleSize)

        plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/vertical_3.1_' + str(self.numberOfOrbitsPerManifold) + '_heteroclinic_connection_validation.pdf')
        plt.close()
        pass

    def plot_manifolds(self):
        line_width_near_heteroclinic = 2
        color_near_heteroclinic = 'k'

        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)

            df_s = load_manifold('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L2_vertical_W_S_min_3.1_' + str(int(theta)) + '_full.txt')
            df_u = load_manifold('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L1_vertical_W_U_plus_3.1_' + str(int(theta)) + '_full.txt')

            fig = plt.figure(figsize=self.figSize)
            ax0 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax2 = fig.add_subplot(2, 2, 4)

            # ax0.set_aspect('equal')
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # ax3.set_aspect('equal')

            if self.numberOfOrbitsPerManifold > 100:
                # Plot at max 100 lines
                range_step_size = int(self.numberOfOrbitsPerManifold/100)
            else:
                range_step_size = 1

            plot_alpha = 1
            line_width = 0.5

            for i in range(0, self.numberOfOrbitsPerManifold, range_step_size):
                # Plot manifolds
                ax0.plot(df_u.xs(i)['x'], df_u.xs(i)['y'], df_u.xs(i)['z'], color=self.colorPaletteUnstable[i], alpha=plot_alpha, linewidth=line_width)
                ax0.plot(df_s.xs(i)['x'], df_s.xs(i)['y'], df_s.xs(i)['z'], color=self.colorPaletteStable[i], alpha=plot_alpha, linewidth=line_width)
                ax1.plot(df_s.xs(i)['x'], df_s.xs(i)['y'], color=self.colorPaletteStable[i], alpha=plot_alpha, linewidth=line_width)
                ax1.plot(df_u.xs(i)['x'], df_u.xs(i)['y'], color=self.colorPaletteUnstable[i], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(df_s.xs(i)['x'], df_s.xs(i)['z'], color=self.colorPaletteStable[i], alpha=plot_alpha, linewidth=line_width)
                ax2.plot(df_u.xs(i)['x'], df_u.xs(i)['z'], color=self.colorPaletteUnstable[i], alpha=plot_alpha, linewidth=line_width)
                ax3.plot(df_s.xs(i)['y'], df_s.xs(i)['z'], color=self.colorPaletteStable[i], alpha=plot_alpha, linewidth=line_width)
                ax3.plot(df_u.xs(i)['y'], df_u.xs(i)['z'], color=self.colorPaletteUnstable[i], alpha=plot_alpha, linewidth=line_width)

            # Plot near-heteroclinic connection
            try:
                index_near_heteroclinic_s = int(self.minimumImpulse.loc[theta]['tau1'] * self.numberOfOrbitsPerManifold)
                index_near_heteroclinic_u = int(self.minimumImpulse.loc[theta]['tau2'] * self.numberOfOrbitsPerManifold)

                ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax1.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax1.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax2.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax2.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax3.plot(df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax3.plot(df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color=color_near_heteroclinic, alpha=plot_alpha, linewidth=line_width_near_heteroclinic)

                connection_text = '$\Delta \mathbf{V} = $' + str(np.round(self.minimumImpulse.loc[theta]['dv'], 1)) + 'm/s \n' + \
                                  '$\Delta \mathbf{r} = $' + str(np.round(self.minimumImpulse.loc[theta]['dr'], 1)) + 'km'

                ax1.text(0.975, 0.075, connection_text, horizontalalignment='right', verticalalignment='bottom',
                         transform=ax1.transAxes, bbox={'facecolor': 'navy', 'alpha': 0.1, 'pad': 3})
            except KeyError:
                pass

            bodies_df = load_bodies_location()
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
            y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax0.plot_surface(x, y, z, color='black')
            ax1.contourf(x, y, z, colors='black')
            ax2.contourf(x, z, y, colors='black')
            # ax3.contourf(y, z, x, colors='black')

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([max(df_s['x'].max(), df_u['x'].max()) - min(df_s['x'].min(), df_u['x'].min()),
                                  max(df_s['y'].max(), df_u['y'].max()) - min(df_s['y'].min(), df_u['y'].min()),
                                  max(df_s['z'].max(), df_u['z'].max()) - min(df_s['z'].min(), df_u['z'].min())]).max()

            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max(df_s['x'].max(), df_u['x'].max()) + min(df_s['x'].min(), df_u['x'].min()))
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(df_s['y'].max(), df_u['y'].max()) + min(df_s['y'].min(), df_u['y'].min()))
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(df_s['z'].max(), df_u['z'].max()) + min(df_s['z'].min(), df_u['z'].min()))
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax0.plot([xb], [yb], [zb], 'w')

            ax0.set_xlabel('x [-]')
            ax0.set_ylabel('y [-]')
            ax0.set_zlabel('z [-]')

            ax1.set_xlabel('x [-]')
            ax1.set_ylabel('y [-]')

            ax2.set_xlabel('x [-]')
            ax2.set_ylabel('z [-]')

            ax3.set_xlabel('y [-]')
            ax3.set_ylabel('z [-]')

            ax0.grid(True, which='both', ls=':')
            ax1.grid(True, which='both', ls=':')
            ax2.grid(True, which='both', ls=':')
            ax3.grid(True, which='both', ls=':')

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.suptitle('Near-heteroclinic connection for $min(\Delta V) \enskip \\forall \Delta r < 10^{-3}$ (at  $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$, C = 3.1, $\\theta$ = ' + str(theta) + '$^\circ$)',
                         size=self.suptitleSize)
            plt.show()
            # plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/vertical_3.1_heteroclinic_connection_' + str(theta) + '.pdf')
            # plt.close()
        pass

    def plot_poincare_spread(self):
        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)
            fig = plt.figure(figsize=self.figSize)
            ax = fig.gca()

            df_s = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L2_vertical_W_S_min_3.1_' + str(int(theta)) + '_poincare.txt',
                                 delim_whitespace=True, header=None).filter(list(range(8)))
            df_s.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            df_u = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L1_vertical_W_U_plus_3.1_' + str(int(theta)) + '_poincare.txt',
                          delim_whitespace=True, header=None).filter(list(range(8)))
            df_u.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            dr_ls = []
            dv_ls = []
            tau_normalized = [i/self.numberOfOrbitsPerManifold for i in range(0, self.numberOfOrbitsPerManifold+1, 1)]
            colors = matplotlib.colors.ListedColormap(sns.color_palette("YlGnBu"), N=self.numberOfOrbitsPerManifold+1)(tau_normalized)

            sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(sns.color_palette("YlGnBu")),
                                       norm=plt.Normalize(vmin=0, vmax=1))
            # clean the array of the scalar mappable
            sm._A = []

            print('Matrix size: ' + str(len(df_s)) + ' x ' + str(len(df_s.columns)))
            for idx_s, row_s in df_s.iterrows():
                print('s: ' + str(idx_s))
                for idx_u, row_u in df_u.iterrows():
                    dr = np.sqrt((row_s['x'] - row_u['x']) ** 2 + (row_s['y'] - row_u['y']) ** 2 + (row_s['z'] - row_u['z']) ** 2)

                    if dr < (self.maximumPositionDeviation * 10):
                        dv = np.sqrt((row_s['xdot'] - row_u['xdot']) ** 2 + (row_s['ydot'] - row_u['ydot']) ** 2 + (row_s['zdot'] - row_u['zdot']) ** 2)
                        dtheta = abs(row_s['tau'] - row_u['tau'])

                        plot_color_index = int(np.round(dtheta * self.numberOfOrbitsPerManifold))
                        plot_color = colors[plot_color_index]

                        dr_ls.append(dr * self.positionDimensionFactor)
                        dv_ls.append(dv * self.velocityDimensionFactor)
                        ax.scatter(dv * self.velocityDimensionFactor, dr * self.positionDimensionFactor, c=plot_color, alpha=1)
            # ax.scatter(dv_ls, dr_ls, c='navy', alpha=0.5)
            try:
                min_impulse_line = ax.scatter(self.minimumImpulse.loc[theta]['dv'], self.minimumImpulse.loc[theta]['dr'],
                                              c='r', label='$min(\Delta V) \enskip \\forall \Delta r < 10^{-3}$')
                ax.legend(frameon=True, loc='lower right', handles=[min_impulse_line])
            except KeyError:
                pass

            ax.axhline(self.maximumPositionDeviation * self.positionDimensionFactor, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
            ax.set_xlabel('$\Delta \mathbf{V} [m/s]$')
            ax.set_ylabel('$\Delta \mathbf{r} [km]$')
            ax.set_yscale('log')
            ax.set_xlim([0, 100])
            ax.set_ylim([self.maximumPositionDeviation * self.positionDimensionFactor / 100,
                         self.maximumPositionDeviation * self.positionDimensionFactor * 10])
            ax.grid(True, which='both', ls=':')

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            cax, kw = matplotlib.colorbar.make_axes([ax])
            plt.colorbar(sm, cax=cax, label='$|\\tau_1 - \\tau_2| [-]$', **kw)

            plt.suptitle('State vecor discrepancy (at  $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$, C = 3.1, $\\theta$ = '
                         + str(theta) + '$^\circ, \# = $' + str(self.numberOfOrbitsPerManifold) + ')',
                         size=self.suptitleSize)
            plt.show()
            # plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/vertical_3.1_heteroclinic_connection_' + str(theta) + '_poincare_scatter.pdf')
            # plt.close()
        pass

    def compare_number_of_orbits_per_manifold(self, number_of_orbits_per_manifold_ls):
        fig, axarr = plt.subplots(1, 2, figsize=self.figSize)

        blues = sns.color_palette('Blues', len(number_of_orbits_per_manifold_ls))

        for idx, number_of_orbits_per_manifold in enumerate(number_of_orbits_per_manifold_ls):
            df = pd.read_table('../../data/raw/poincare_sections/' + str(number_of_orbits_per_manifold) + '/vertical_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
            df.columns = ['theta', 'tau1', 't1', 'x1', 'y1', 'z1', 'xdot1', 'ydot1', 'zdot1', 'tau2', 't2', 'x2', 'y2', 'z2', 'xdot2', 'ydot2', 'zdot2']
            df['dr'] = np.sqrt((df['x1'] - df['x2']) ** 2 + (df['y1'] - df['y2']) ** 2 + (df['z1'] - df['z2']) ** 2)
            df['dv'] = np.sqrt((df['xdot1'] - df['xdot2']) ** 2 + (df['ydot1'] - df['ydot2']) ** 2 + (df['zdot1'] - df['zdot2']) ** 2)
            df = df.set_index('theta')

            # TODO fix proper constants
            df['dr'] = df['dr'] * self.positionDimensionFactor
            df['dv'] = df['dv'] * self.velocityDimensionFactor
            df = df[df['dv'] != 0]

            axarr[0].plot(df['dv'], color=blues[idx], label=str(number_of_orbits_per_manifold))
            axarr[0].set_xlabel('$ \\theta $')
            axarr[0].set_ylabel('$ m/s $')

            axarr[1].plot(df['dr'], color=blues[idx], label=str(number_of_orbits_per_manifold))
            axarr[1].set_xlabel('$ \\theta $')
            axarr[1].set_ylabel('$ m $')

        for i in range(2):
            axarr[i].grid(True, which='both', ls=':')
            axarr[i].legend(frameon=True, loc='lower right')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.suptitle('Heteroclinic connection $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ (at C = 3.1)',
                     size=self.suptitleSize)
        # plt.show()
        plt.savefig('../../data/figures/poincare_sections/vertical_3.1_heteroclinic_connection_comparison.pdf')
        plt.close()
        pass


if __name__ == '__main__':
    max_position_deviation = {100: 1e-3, 1000: 1e-4, 10000: 1e-5}
    for number_of_orbits_per_manifold in [1000]:
        minimum_impulse_manifold_connections = MinimumImpulseManifoldConnections(number_of_orbits_per_manifold=number_of_orbits_per_manifold,
                                                                                 max_position_dev=max_position_deviation[number_of_orbits_per_manifold])
        # minimum_impulse_manifold_connections.plot_impulse_angle()
        minimum_impulse_manifold_connections.plot_manifolds()
        # minimum_impulse_manifold_connections.plot_poincare_spread()

    # minimum_impulse_manifold_connections = MinimumImpulseManifoldConnections()
    # minimum_impulse_manifold_connections.compare_number_of_orbits_per_manifold([100, 1000, 10000])
