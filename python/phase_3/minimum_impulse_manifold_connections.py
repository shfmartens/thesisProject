import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
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
    load_initial_conditions_incl_M, load_manifold, computeJacobiEnergy, load_manifold_incl_stm, load_manifold_refactored


class MinimumImpulseManifoldConnections:
    def __init__(self, number_of_orbits_per_manifold=100, max_position_dev=1e-3, orbit_type='vertical'):
        self.suptitleSize = 20
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.orbitType = orbit_type
        self.orbitTypeForTitle = orbit_type.capitalize()
        if (self.orbitTypeForTitle == 'Horizontal') or (self.orbitTypeForTitle == 'Vertical'):
            self.orbitTypeForTitle += ' Lyapunov'

        self.numberOfOrbitsPerManifold = number_of_orbits_per_manifold
        self.maximumPositionDeviation = max_position_dev

        df = pd.read_table('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
        df.columns = ['theta', 'tau1', 't1', 'x1', 'y1', 'z1', 'xdot1', 'ydot1', 'zdot1', 'tau2', 't2', 'x2', 'y2', 'z2', 'xdot2', 'ydot2', 'zdot2']
        df['dr'] = np.sqrt((df['x1'] - df['x2']) ** 2 + (df['y1'] - df['y2']) ** 2 + (df['z1'] - df['z2']) ** 2)
        df['dv'] = np.sqrt((df['xdot1'] - df['xdot2']) ** 2 + (df['ydot1'] - df['ydot2']) ** 2 + (df['zdot1'] - df['zdot2']) ** 2)
        df = df.set_index('theta')

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
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors-1], sns.color_palette("viridis", n_colors)[int((n_colors-1)/2)], sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}
        pass

    def plot_impulse_angle(self):
        fig, axarr = plt.subplots(2, 2, figsize=self.figSize)

        axarr[0, 0].semilogy(self.minimumImpulse['dv'], color=self.plottingColors['singleLine'])
        axarr[0, 0].set_xlabel('$ \\theta [^\circ] $')
        axarr[0, 0].set_ylabel('$ \Delta V [-]  \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')
        axarr[0, 0].set_title('Velocity discrepancy')

        axarr[0, 1].semilogy(self.minimumImpulse['dr'], color=self.plottingColors['singleLine'])
        axarr[0, 1].axhline(self.maximumPositionDeviation, c=self.plottingColors['limit'], linewidth=1, linestyle='--')
        axarr[0, 1].set_xlabel('$ \\theta [^\circ]$')
        axarr[0, 1].set_ylabel('$ \Delta r [-]   \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')
        axarr[0, 1].set_title('Position discrepancy')

        scatter_size = 5
        tau1_ls = []
        tau2_ls = []
        theta_ls = []
        for theta in self.thetaRangeList:
            try:
                tau1_ls.append(self.minimumImpulse.loc[theta]['tau1'])
                tau2_ls.append(self.minimumImpulse.loc[theta]['tau2'])
                theta_ls.append(theta)
                pass
            except KeyError:
                pass
        sc1 = axarr[1, 0].scatter(theta_ls, tau1_ls, s=scatter_size, c=self.plottingColors['doubleLine'][0], label='$\\tau_1$')
        sc2 = axarr[1, 0].scatter(theta_ls, tau2_ls, s=scatter_size, c=self.plottingColors['doubleLine'][1], label='$\\tau_2$')

        axarr[1, 0].legend(frameon=True, loc='upper right', handles=[sc1, sc2])

        axarr[1, 0].set_xlabel('$ \\theta [^\circ]$')
        axarr[1, 0].set_ylabel('$ \\tau_i [-]  \quad \\forall \enskip i=1, 2 $')
        sc = axarr[1, 1].scatter(tau1_ls, tau2_ls, c=theta_ls, alpha=0.9, cmap='viridis', vmin=-180, vmax=0, s=scatter_size)
        axarr[1, 1].set_title('Phases on departure/target orbit')

        axarr[1, 1].set_xlabel('$ \\tau_1 [-] $')
        axarr[1, 1].set_ylabel('$ \\tau_2 [-] $')
        axarr[1, 0].set_title('Phases on departure/target orbit')

        for i in range(2):
            for j in range(2):
                axarr[i, j].grid(True, which='both', ls=':')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        cb = plt.colorbar(sc)
        cb.set_label('$\\theta \enskip [^\circ]$')

        plt.suptitle(self.orbitTypeForTitle + ' near-heteroclinic connection $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ (C = 3.1, n = ' +
                     str(self.numberOfOrbitsPerManifold) + ')', size=self.suptitleSize)

        # plt.show()
        plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_' + str(self.numberOfOrbitsPerManifold) + '_heteroclinic_connection_validation.pdf',
                    transparent=True)
        plt.close()
        pass

    def plot_manifolds(self):
        line_width_near_heteroclinic = 2
        color_near_heteroclinic = 'k'

        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)

            df_s = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(int(theta)) + '_full.txt')
            df_u = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(int(theta)) + '_full.txt')

            fig = plt.figure(figsize=self.figSize)
            ax0 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax2 = fig.add_subplot(2, 2, 4)

            # ax0.set_aspect('equal')
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')

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

                connection_text = '$min(\Delta r)  \quad \\forall \enskip \Delta V < 0.5$ \n' + \
                                  '$\Delta V  = \enskip  $' + str(np.round(self.minimumImpulse.loc[theta]['dv'], 1)) + ' \n' + \
                                  '$\Delta r  = \enskip $' + str(np.round(self.minimumImpulse.loc[theta]['dr'], 1))

                ax0.text2D(0.0, 0.8, s=connection_text, horizontalalignment='left', verticalalignment='bottom',
                           transform=ax0.transAxes, bbox=dict(boxstyle="round", fc="w"))
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
            plt.suptitle(self.orbitTypeForTitle + ' near-heteroclinic connection  $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ (C = 3.1, $\\theta$ = ' + str(theta) + '$^\circ$)',
                         size=self.suptitleSize)
            # plt.show()
            plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_heteroclinic_connection_' + str(theta) + '.pdf',
                        transparent=True)
            plt.close()
        pass

    def plot_image_trajectories(self):
        line_width_near_heteroclinic = 1.5

        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)

            df_s = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(int(theta)) + '_full.txt')
            df_u = load_manifold_refactored('../../data/raw/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(int(theta)) + '_full.txt')

            fig = plt.figure(figsize=self.figSize)
            ax0 = fig.add_subplot(111, projection='3d')

            if self.numberOfOrbitsPerManifold > 100:
                # Plot at max 100 lines
                range_step_size = int(self.numberOfOrbitsPerManifold/100)
            else:
                range_step_size = 1

            plot_alpha = 1
            line_width = 0.5

            # Plot near-heteroclinic connection
            try:
                index_near_heteroclinic_s = int(self.minimumImpulse.loc[theta]['tau1'] * self.numberOfOrbitsPerManifold)
                index_near_heteroclinic_u = int(self.minimumImpulse.loc[theta]['tau2'] * self.numberOfOrbitsPerManifold)

                # (T1)
                ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)

                # (T4)
                ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'], df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)
                ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'], df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha, linewidth=line_width_near_heteroclinic)

                if orbit_type != 'horizontal':
                    # (T3)
                    ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], df_s.xs(index_near_heteroclinic_s)['y'],
                             -df_s.xs(index_near_heteroclinic_s)['z'], color='g', alpha=plot_alpha,
                             linewidth=line_width_near_heteroclinic)
                    ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], df_u.xs(index_near_heteroclinic_u)['y'],
                             -df_u.xs(index_near_heteroclinic_u)['z'], color='r', alpha=plot_alpha,
                             linewidth=line_width_near_heteroclinic)
                    # (T2)
                    ax0.plot(df_s.xs(index_near_heteroclinic_s)['x'], -df_s.xs(index_near_heteroclinic_s)['y'],
                             -df_s.xs(index_near_heteroclinic_s)['z'], color='r', alpha=plot_alpha,
                             linewidth=line_width_near_heteroclinic)
                    ax0.plot(df_u.xs(index_near_heteroclinic_u)['x'], -df_u.xs(index_near_heteroclinic_u)['y'],
                             -df_u.xs(index_near_heteroclinic_u)['z'], color='g', alpha=plot_alpha,
                             linewidth=line_width_near_heteroclinic)
                    pass

                connection_text = '$min(\Delta r)  \quad \\forall \enskip \Delta V < 0.5$ \n' + \
                                  '$\Delta V  = \enskip  $' + str(np.round(self.minimumImpulse.loc[theta]['dv'], 1)) + '\n' + \
                                  '$\Delta r  = \enskip $' + str(np.round(self.minimumImpulse.loc[theta]['dr'], 1))

                ax0.text2D(0.0, 0.8, s=connection_text, horizontalalignment='left', verticalalignment='bottom',
                           transform=ax0.transAxes, bbox=dict(boxstyle="round", fc="w"))
            except KeyError:
                pass

            bodies_df = load_bodies_location()
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
            y = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
            z = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax0.plot_surface(x, y, z, color='black')

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

            ax0.grid(True, which='both', ls=':')

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.suptitle(self.orbitTypeForTitle + ' near-heteroclinic cycle  $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ (C = 3.1, $\\theta$ = ' + str(theta) + '$^\circ$)',
                         size=self.suptitleSize)
            # plt.show()
            plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_heteroclinic_cycle_' + str(theta) + '.pdf',
                        transparent=True)
            plt.close()
        pass

    def plot_poincare_spread(self):
        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)
            fig = plt.figure(figsize=self.figSize)
            ax = fig.gca()

            df_s = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(int(theta)) + '_poincare.txt',
                                 delim_whitespace=True, header=None).filter(list(range(8)))
            df_s.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            df_u = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(int(theta)) + '_poincare.txt',
                          delim_whitespace=True, header=None).filter(list(range(8)))
            df_u.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            dr_ls = []
            dv_ls = []
            theta_ls = []

            scatter_size = 15

            print('Matrix size: ' + str(len(df_s)) + ' x ' + str(len(df_s.columns)))
            for idx_s, row_s in df_s.iterrows():
                print('s: ' + str(idx_s))
                for idx_u, row_u in df_u.iterrows():
                    dr = np.sqrt((row_s['x'] - row_u['x']) ** 2 + (row_s['y'] - row_u['y']) ** 2 + (row_s['z'] - row_u['z']) ** 2)

                    if dr < (self.maximumPositionDeviation * 10):
                        dv = np.sqrt((row_s['xdot'] - row_u['xdot']) ** 2 + (row_s['ydot'] - row_u['ydot']) ** 2 + (row_s['zdot'] - row_u['zdot']) ** 2)
                        dtheta = abs(row_s['tau'] - row_u['tau'])
                        dr_ls.append(dr)
                        dv_ls.append(dv)
                        theta_ls.append(dtheta)

            sc = ax.scatter(dv_ls, dr_ls, c=theta_ls, alpha=0.9, cmap='viridis', vmin=0, vmax=1, s=scatter_size)
            cb = plt.colorbar(sc)
            cb.set_label('$|\\tau_s - \\tau_u| \enskip [-]$')
            print(pd.DataFrame({'dtau': theta_ls}).describe())
            try:
                min_impulse_line = ax.scatter(self.minimumImpulse.loc[theta]['dv'], self.minimumImpulse.loc[theta]['dr'],
                                              c='r', label='$min(\Delta r) \quad  \\forall \enskip \Delta V < 0.5$', s=scatter_size)
                ax.legend(frameon=True, loc='lower right', handles=[min_impulse_line])
            except KeyError:
                pass
            # ax.axvline(0.5, c=self.plottingColors['limit'],
            #            linewidth=1, linestyle='--')

            ax.set_xlabel('$\Delta \mathbf{V} \enskip [-]$')
            ax.set_ylabel('$\Delta \mathbf{r} \enskip [-]$')
            ax.set_yscale('log')
            ax.set_xlim([0, 0.5])

            if dr_ls:
                ax.set_ylim([min(min(dr_ls), self.maximumPositionDeviation / 100),
                             self.maximumPositionDeviation * 10])
            else:
                ax.set_ylim([self.maximumPositionDeviation / 100,
                             self.maximumPositionDeviation * 10])

            ax.grid(True, which='both', ls=':')

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

            plt.suptitle(self.orbitTypeForTitle + ' $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ at $\\theta$ = '
                         + str(theta) + '$^\circ (C = 3.1, n = $' + str(self.numberOfOrbitsPerManifold) + ')',
                         size=self.suptitleSize)
            # plt.show()
            plt.savefig('../../data/figures/poincare_sections/' + str(self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_heteroclinic_connection_' + str(theta) + '_poincare_scatter.pdf',
                        transparent=True)
            plt.close()
        pass

    def plot_poincare_spread_zoomed_out(self):
        print('Theta:')
        for theta in self.thetaRangeList:
            print(theta)

            fig = plt.figure(figsize=self.figSize)
            ax = fig.gca()

            df_s = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L2_' + self.orbitType + '_W_S_min_3.1_' + str(
                int(theta)) + '_poincare.txt',
                                 delim_whitespace=True, header=None).filter(list(range(8)))
            df_s.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            df_u = pd.read_table('../../data/raw/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/L1_' + self.orbitType + '_W_U_plus_3.1_' + str(
                int(theta)) + '_poincare.txt',
                                 delim_whitespace=True, header=None).filter(list(range(8)))
            df_u.columns = ['tau', 't', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            dr_ls = []
            dv_ls = []
            theta_ls = []
            scatter_size = 5

            print('Matrix size: ' + str(len(df_s)) + ' x ' + str(len(df_s.columns)))
            for idx_s, row_s in df_s.iterrows():
                print('s: ' + str(idx_s))
                for idx_u, row_u in df_u.iterrows():
                    dr = np.sqrt((row_s['x'] - row_u['x']) ** 2 + (row_s['y'] - row_u['y']) ** 2 + (row_s['z'] - row_u['z']) ** 2)
                    dv = np.sqrt((row_s['xdot'] - row_u['xdot']) ** 2 + (row_s['ydot'] - row_u['ydot']) ** 2 + (row_s['zdot'] - row_u['zdot']) ** 2)
                    dtheta = abs(row_s['tau'] - row_u['tau'])
                    dr_ls.append(dr)
                    dv_ls.append(dv)
                    theta_ls.append(dtheta)

            sc = ax.scatter(dv_ls, dr_ls, c=theta_ls, alpha=0.9, cmap='viridis', vmin=0, vmax=1, s=scatter_size)
            cb = plt.colorbar(sc)
            cb.set_label('$|\\tau_s - \\tau_u| \enskip [-]$')
            print(pd.DataFrame({'dtau': theta_ls}).describe())
            try:
                min_impulse_line = ax.scatter(self.minimumImpulse.loc[theta]['dv'],
                                              self.minimumImpulse.loc[theta]['dr'],
                                              c='r', label='$min(\Delta r) \quad  \\forall \enskip \Delta V < 0.5$',
                                              s=scatter_size)
                ax.legend(frameon=True, loc='lower right', handles=[min_impulse_line])
            except KeyError:
                pass

            ax.axvline(0.5, c=self.plottingColors['limit'],
                       linewidth=1, linestyle='--')
            ax.set_xlabel('$\Delta \mathbf{V} \enskip [-]$')
            ax.set_ylabel('$\Delta \mathbf{r} \enskip [-]$')
            ax.set_yscale('log')
            ax.set_xlim([0, 1])
            ax.grid(True, which='both', ls=':')

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

            plt.suptitle(
                self.orbitTypeForTitle + ' $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ at $\\theta$ = '
                + str(theta) + '$^\circ (C = 3.1 , n = $' + str(self.numberOfOrbitsPerManifold) + ')',
                size=self.suptitleSize)
            # plt.show()
            plt.savefig('../../data/figures/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_heteroclinic_connection_' + str(
                theta) + '_poincare_scatter_zoomed_out.pdf', transparent=True)
            plt.savefig('../../data/figures/poincare_sections/' + str(
                self.numberOfOrbitsPerManifold) + '/' + self.orbitType + '_3.1_heteroclinic_connection_' + str(
                theta) + '_poincare_scatter_zoomed_out.png', dpi=300, transparent=True)
            plt.close()
        pass

    def compare_number_of_orbits_per_manifold(self, number_of_orbits_per_manifold_ls):
        fig, axarr = plt.subplots(2, 1, figsize=self.figSize, sharex=True)

        blues = sns.color_palette('Blues', len(number_of_orbits_per_manifold_ls))
        # linestyles = ['-.', '--', '-']
        linestyles = ['-', '-', '-']
        linewidth = 2
        alpha = 0.7

        for idx, number_of_orbits_per_manifold in enumerate(number_of_orbits_per_manifold_ls):
            df = pd.read_table('../../data/raw/poincare_sections/' + str(number_of_orbits_per_manifold) + '/' + self.orbitType + '_3.1_minimum_impulse_connections.txt', delim_whitespace=True, header=None).filter(list(range(17)))
            df.columns = ['theta', 'tau1', 't1', 'x1', 'y1', 'z1', 'xdot1', 'ydot1', 'zdot1', 'tau2', 't2', 'x2', 'y2', 'z2', 'xdot2', 'ydot2', 'zdot2']
            df['dr'] = np.sqrt((df['x1'] - df['x2']) ** 2 + (df['y1'] - df['y2']) ** 2 + (df['z1'] - df['z2']) ** 2)
            df['dv'] = np.sqrt((df['xdot1'] - df['xdot2']) ** 2 + (df['ydot1'] - df['ydot2']) ** 2 + (df['zdot1'] - df['zdot2']) ** 2)
            df = df.set_index('theta')
            df = df[df['dv'] != 0]

            axarr[0].semilogy(df['dv'], color=self.plottingColors['tripleLine'][idx], label='$n$ = ' + str(number_of_orbits_per_manifold), linestyle=linestyles[idx], linewidth=linewidth, alpha=alpha)

            axarr[1].semilogy(df['dr'], color=self.plottingColors['tripleLine'][idx], label='$n$ = ' + str(number_of_orbits_per_manifold), linestyle=linestyles[idx], linewidth=linewidth, alpha=alpha)

        axarr[0].set_ylabel('$ \Delta V [-] \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')
        axarr[1].set_xlabel('$ \\theta [^\circ]$')
        axarr[1].set_ylabel('$ \Delta r [-] \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')

        axarr[0].set_title('Velocity discrepancy')
        axarr[1].set_title('Position discrepancy')

        for i in range(2):
            axarr[i].grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.875)
        axarr[0].legend(frameon=True, bbox_to_anchor=(1.14, 0))

        plt.suptitle(self.orbitTypeForTitle + ' near-heteroclinic connection $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ - Sampling validation at C = 3.1',
                     size=self.suptitleSize)
        # plt.show()
        plt.savefig('../../data/figures/poincare_sections/' + self.orbitType + '_3.1_heteroclinic_sampling_validation.pdf',
                    transparent=True)
        plt.close()
        pass

    def compare_orbit_types(self, number_of_orbits_per_manifold):
        fig, axarr = plt.subplots(2, 1, figsize=self.figSize, sharex=True)

        linestyles = ['-', '-', '-']
        linewidth = 2
        alpha = 0.7

        orbit_types = ['horizontal', 'vertical', 'halo']
        for idx, orbit_type in enumerate(orbit_types):
            df = pd.read_table('../../data/raw/poincare_sections/' + str(
                number_of_orbits_per_manifold) + '/' + orbit_type + '_3.1_minimum_impulse_connections.txt',
                               delim_whitespace=True, header=None).filter(list(range(17)))
            df.columns = ['theta', 'tau1', 't1', 'x1', 'y1', 'z1', 'xdot1', 'ydot1', 'zdot1', 'tau2', 't2', 'x2', 'y2',
                          'z2', 'xdot2', 'ydot2', 'zdot2']
            df['dr'] = np.sqrt((df['x1'] - df['x2']) ** 2 + (df['y1'] - df['y2']) ** 2 + (df['z1'] - df['z2']) ** 2)
            df['dv'] = np.sqrt(
                (df['xdot1'] - df['xdot2']) ** 2 + (df['ydot1'] - df['ydot2']) ** 2 + (df['zdot1'] - df['zdot2']) ** 2)
            df = df.set_index('theta')
            df = df[df['dv'] != 0]

            orbit_type_for_label = orbit_type.capitalize()
            if (orbit_type_for_label == 'Horizontal') or (orbit_type_for_label == 'Vertical'):
                orbit_type_for_label += ' Lyapunov'

            axarr[0].semilogy(df['dv'], color=self.plottingColors['tripleLine'][idx],
                              label=orbit_type_for_label, linestyle=linestyles[idx], linewidth=linewidth,
                              alpha=alpha)

            axarr[1].semilogy(df['dr'], color=self.plottingColors['tripleLine'][idx],
                              label=orbit_type_for_label, linestyle=linestyles[idx], linewidth=linewidth,
                              alpha=alpha)

        axarr[0].set_ylabel('$ \Delta V [-] \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')
        axarr[1].set_xlabel('$ \\theta [^\circ]$')
        axarr[1].set_ylabel('$ \Delta r [-] \quad \\forall \enskip min(\Delta r), \Delta V < 0.5$')

        axarr[0].set_title('Velocity discrepancy')
        axarr[1].set_title('Position discrepancy')

        for i in range(2):
            axarr[i].grid(True, which='both', ls=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, right=0.825)
        axarr[0].legend(frameon=True, bbox_to_anchor=(1.22, 0))

        plt.suptitle('Near-heteroclinic connection $\mathcal{W}^{U+} \cup \mathcal{W}^{S-}$ - Family overview (C = 3.1, n = ' +
                     str(number_of_orbits_per_manifold) + ')',
            size=self.suptitleSize)
        # plt.show()
        plt.savefig(
            '../../data/figures/poincare_sections/3.1_heteroclinic_family_overview_' + str(number_of_orbits_per_manifold) + '.pdf',
            transparent=True)
        plt.close()
        pass


if __name__ == '__main__':
    max_position_deviation = {100: 1e-3, 1000: 1e-4, 5000: 1e-4}
    orbit_types = ['horizontal', 'vertical', 'halo']
    orbit_types = ['halo']
    
    for orbit_type in orbit_types:
        # minimum_impulse_manifold_connections = MinimumImpulseManifoldConnections(orbit_type=orbit_type)
        # if orbit_type =='horizontal':
        #     minimum_impulse_manifold_connections.compare_number_of_orbits_per_manifold([100, 1000])
        # else:
        #     minimum_impulse_manifold_connections.compare_number_of_orbits_per_manifold([100, 1000, 5000])

        # for number_of_orbits_per_manifold in [100]:
        # for number_of_orbits_per_manifold in [1000]:
        for number_of_orbits_per_manifold in [5000]:

            minimum_impulse_manifold_connections = MinimumImpulseManifoldConnections(number_of_orbits_per_manifold=number_of_orbits_per_manifold,
                                                                                     max_position_dev=max_position_deviation[number_of_orbits_per_manifold],
                                                                                     orbit_type=orbit_type)
            minimum_impulse_manifold_connections.plot_impulse_angle()
            minimum_impulse_manifold_connections.plot_manifolds()
            minimum_impulse_manifold_connections.plot_image_trajectories()
            minimum_impulse_manifold_connections.plot_poincare_spread()
            # minimum_impulse_manifold_connections.plot_poincare_spread_zoomed_out()

    # for number_of_orbits_per_manifold in [100, 1000, 5000]:
    #     minimum_impulse_manifold_connections = MinimumImpulseManifoldConnections()
    #     minimum_impulse_manifold_connections.compare_orbit_types(number_of_orbits_per_manifold)
