import numpy as np
import pandas as pd
import json
import matplotlib
from decimal import *
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib2tikz import save as tikz_save
import seaborn as sns
sns.set_style("whitegrid")
import time
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{gensymb}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)
import sys
sys.path.append('../util')

from load_data import load_orbit, load_bodies_location, load_lagrange_points_location, load_differential_corrections, \
load_initial_conditions_incl_M, load_manifold

from load_data_augmented import compute_hamiltonian_from_list_second_version, load_lagrange_points_location_augmented, load_orbit_augmented, \
compute_hamiltonian_from_state

class VaryingMassAnalysis:
    def __init__(self, lagrange_point_nr,  acceleration_magnitude, alpha, member, specific_impulse, low_dpi):
        self.lagrangePointNr = lagrange_point_nr
        self.accelerationMagnitude = acceleration_magnitude
        self.alpha = alpha
        self.familyMember = member
        self.specificImpulse = specific_impulse
        self.lowDPI = low_dpi

        print('=======================')
        print('L' + str(self.lagrangePointNr) + '_(acc = ' + str(self.accelerationMagnitude) \
              + ',alpha = ' + str(self.alpha), ', member = ' + str(self.familyMember) + ')' + ' Isp = ' + str(self.specificImpulse))
        print('=======================')

        # scale properties
        self.spacingFactor = 1.05
        self.lowDpi = low_dpi
        self.dpi = 150
        self.suptitleSize = 20
        self.scaleDistanceBig = 1

        # label properties
        self.numberOfXTicks = 5

        # Normal figure
        self.figSize = (7 * (1 + np.sqrt(5)) / 2, 7)
        self.figureRatio = (7 * (1 + np.sqrt(5)) / 2) / 7

        # figure with two subplots next to eachother
        self.figSizeWide = (7 * (1 + np.sqrt(5)) / 2, 3.5)
        self.figureRatioWide = (7 * (1 + np.sqrt(5)) / 2) / 3.5

        self.figSizeThird = (7 * (1 + np.sqrt(5)) / 2, 3.5 * 0.75)

        self.figSizeCont = (7 * (1 + np.sqrt(5)) / 2, 7.0 * 0.70)

        # Colour schemes
        n_colors = 3
        n_colors_l = 6
        self.plottingColors = {'lambda1': sns.color_palette("viridis", n_colors_l)[0],
                               'lambda2': sns.color_palette("viridis", n_colors_l)[2],
                               'lambda3': sns.color_palette("viridis", n_colors_l)[4],
                               'lambda4': sns.color_palette("viridis", n_colors_l)[5],
                               'lambda5': sns.color_palette("viridis", n_colors_l)[3],
                               'lambda6': sns.color_palette("viridis", n_colors_l)[1],
                               'singleLine': sns.color_palette("viridis", n_colors)[0],
                               'doubleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'tripleLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                              sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                              sns.color_palette("viridis", n_colors)[0]],
                               'fifthLine': [sns.color_palette("viridis", n_colors)[n_colors - 1],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 1.5)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 2)],
                                             sns.color_palette("viridis", n_colors)[int((n_colors - 1) / 3)],
                                             sns.color_palette("viridis", n_colors)[0]],
                               'limit': 'black'}
        self.plotAlpha = 1
        self.lineWidth = 0.5

        pass
    def verify_howell_reasoning(self):
        xList = [0.8369151483688, 0.4878494189827]
        yList = [0, -0.8660254037844]
        accMag = 0.0
        alphaList = [0.0,0.0]
        natural_energy_range = compute_hamiltonian_from_list_second_version(xList, yList, accMag, alphaList)


        print(' ========== VARYING MASS ANALYSIS ==========')
        print(' ')
        print(' === Checking the natural energy range ==')
        print('Hlt L1: ' +str(natural_energy_range[0][1]))
        print('Hlt L5: ' + str(natural_energy_range[1][1]))
        print('Energy range (Hlt_5 - Hlt_1): ' + str(natural_energy_range[1][1] - natural_energy_range[0][1]))
        print(' ')
        print(' === Checking time derivative of the acceleration ==')
        l_char = 384400*1000
        t_char = 3.751904571432808e5
        grav_0 = 9.80665
        alt_der = self.accelerationMagnitude ** 2 * (l_char)/(self.specificImpulse*grav_0*t_char)
        # print('l_char: ' + str(l_char) + ' [m]')
        # print('t_char: ' + str(t_char) + ' [s]')
        # print('grav_0: ' + str(grav_0) + ' [m/s^2]')
        print('alt: ' + str(self.accelerationMagnitude) + ' [-]')
        print('isp: ' + str(self.specificImpulse) + ' [s]')
        print('alt derivative: ' + str(alt_der) + ' [-]')

    def analyze_varying_mass_effect(self):
        f, arr = plt.subplots(2, 2, figsize=self.figSize)


        arr[0,0].set_xlabel('x [-]')
        arr[0,0].set_ylabel('y [-]')
        arr[0,0].set_title('Planar projection')

        arr[0, 1].set_xlabel('T [-]')
        arr[0, 1].set_ylabel('$H_{lt}$ [-]')
        arr[0, 1].set_title('Hamiltonian evolution')

        arr[1,0].set_xlabel('T [-]')
        arr[1,0].set_ylabel('$m$ [-]')
        arr[1,0].set_title('Mass evolution')

        arr[1, 1].set_xlabel('$m$ [-]')
        arr[1, 1].set_ylabel('$\\Delta H_{lt}$ [-]')
        arr[1, 1].set_title('Rate of change Hamiltonian')

        for i in range(2):
            for j in range(2):
                arr[i, j].grid(True, which='both', ls=':')

        ### ====== Plot GRAPHICAL PROJECTION ========

        lagrange_points_df = load_lagrange_points_location_augmented(self.accelerationMagnitude, self.alpha)
        if self.lagrangePointNr == 1:
            lagrange_point_nrs = ['L1']
        if self.lagrangePointNr == 2:
            lagrange_point_nrs = ['L2']

        for lagrange_point_nr in lagrange_point_nrs:
            arr[0,0].scatter(lagrange_points_df[lagrange_point_nr]['x'], lagrange_points_df[lagrange_point_nr]['y'],
                       color='black', marker='x')

        df_constant = load_orbit_augmented('../../data/raw/varying_mass/L' + str(self.lagrangePointNr) + '_' \
                        + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' \
                        + str("{:12.11f}".format(self.alpha)) + '_' \
                        + str(self.familyMember) + '_' \
                        + str(self.specificImpulse) + '_constantMass.txt')

        df_varying = load_orbit_augmented('../../data/raw/varying_mass/L' + str(self.lagrangePointNr) + '_' \
                                 + str("{:12.11f}".format(self.accelerationMagnitude)) + '_' \
                                 + str("{:12.11f}".format(self.alpha)) + '_' \
                                 + str(self.familyMember) + '_' \
                                 + str(self.specificImpulse) + '_varyingMass.txt')

        arr[0,0].plot(df_constant['x'],df_constant['y'],color=self.plottingColors['doubleLine'][0],label='$\\dot{m} = 0$',linestyle='-')
        arr[0,0].plot(df_varying['x'],df_varying['y'],color=self.plottingColors['doubleLine'][1],label='$\\dot{m} \\neq 0$',linestyle='--')

        minimumX = min(min(df_constant['x']),min(df_varying['x']))
        minimumY = min(min(df_constant['y']), min(df_varying['y']))
        maximumX = max(max(df_constant['x']), max(df_varying['x']))
        maximumY = max(max(df_constant['y']), max(df_varying['y']))

        Xmiddle = minimumX + (maximumX - minimumX) / 2.0
        Ymiddle = minimumY + (maximumY - minimumY) / 2.0

        scaleDistance = max((maximumY - minimumY), (maximumX - minimumX))

        arr[0, 0].set_xlim([(Xmiddle - 0.5 * scaleDistance * self.figureRatio * self.spacingFactor),(Xmiddle + 0.5 * scaleDistance * self.figureRatio * self.spacingFactor)])
        arr[0, 0].set_ylim([Ymiddle - 0.5 * scaleDistance * self.spacingFactor, Ymiddle + 0.5 * scaleDistance * self.spacingFactor])
        arr[0, 0].legend(frameon=True, loc='upper right', markerscale=11)
        # Plot bodies
        bodies_df = load_bodies_location()
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xM = bodies_df['Moon']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Moon']['x']
        yM = bodies_df['Moon']['r'] * np.outer(np.sin(u), np.sin(v))
        zM = bodies_df['Moon']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        xE = bodies_df['Earth']['r'] * np.outer(np.cos(u), np.sin(v)) + bodies_df['Earth']['x']
        yE = bodies_df['Earth']['r'] * np.outer(np.sin(u), np.sin(v))
        zE = bodies_df['Earth']['r'] * np.outer(np.ones(np.size(u)), np.cos(v))

        arr[0,0].contourf(xM, yM, zM, colors='black')
        arr[0,0].contourf(xE, yE, zE, colors='black')


        ### === Compute Deviations at full period and print ==== ###
        init_const = df_constant.head(1).values[0]
        final_const = df_constant.tail(1).values[0]

        init_var = df_varying.head(1).values[0]
        final_var = df_varying.tail(1).values[0]

        DeltaR_constant = np.sqrt((init_const[1]-final_const[1]) ** 2 + \
                                  (init_const[2] - final_const[2]) ** 2 + \
                                  (init_const[3] - final_const[3]) ** 2)
        DeltaV_constant = np.sqrt((init_const[4]-final_const[4]) ** 2 + \
                                  (init_const[5] - final_const[5]) ** 2 + \
                                  (init_const[6] - final_const[6]) ** 2)
        DeltaR_varying = np.sqrt((init_var[1]-final_var[1]) ** 2 + \
                                  (init_var[2] - final_var[2]) ** 2 + \
                                  (init_var[3] - final_var[3]) ** 2)
        DeltaV_varying = np.sqrt((init_var[4]-final_var[4]) ** 2 + \
                                  (init_var[5] - final_var[5]) ** 2 + \
                                  (init_var[6] - final_var[6]) ** 2)
        print(' ')
        print('==== Deviations at full period ====')
        print('DeltaR constant: ' + str(DeltaR_constant) + ', DeltaV constant: ' + str(DeltaV_constant))
        print('DeltaR varying: ' + str(DeltaR_varying) + ', DeltaV constant: ' + str(DeltaV_varying))

        ## compute time, mass and hamiltonian of each orbit
        Hlt_constant = []
        mass_constant = []
        time_constant = []

        Hlt_varying = []
        mass_varying = []
        time_varying = []

        Hlt_diff = []

        test = compute_hamiltonian_from_state(init_const[1],init_const[2],init_const[3],init_const[4],init_const[5],init_const[6],
                                              init_const[7],init_const[8],init_const[10])

        for row in df_constant.iterrows():
            time_constant.append(row[1][0])
            mass_constant.append(row[1][10])
            Hlt_constant.append(compute_hamiltonian_from_state(row[1][1],row[1][2],row[1][3],row[1][4],row[1][5],row[1][6],row[1][7]\
                                                        ,row[1][8],row[1][10]))

        for row in df_varying.iterrows():
            time_varying.append(row[1][0])
            mass_varying.append(row[1][10])
            Hlt_varying.append(compute_hamiltonian_from_state(row[1][1], row[1][2], row[1][3], row[1][4], row[1][5], row[1][6],
                                               row[1][7], row[1][8], row[1][10]))
            Hlt_diff.append(compute_hamiltonian_from_state(row[1][1], row[1][2], row[1][3], row[1][4], row[1][5], row[1][6],
                                               row[1][7], row[1][8], 1.0)-compute_hamiltonian_from_state(row[1][1], row[1][2], row[1][3], row[1][4], row[1][5], row[1][6],
                                               row[1][7], row[1][8], row[1][10]))

        #arr[0, 1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1f'))
        arr[0, 1].plot(time_constant, Hlt_constant, color=self.plottingColors['doubleLine'][0], label='$\\dot{m} = 0$',linestyle='-')
        arr[0, 1].plot(time_varying, Hlt_varying, color=self.plottingColors['doubleLine'][1],label='$\\dot{m} \\neq 0$', linestyle='--')
        arr[0, 1].set_xlim([0, max(max(time_varying), max(time_constant))])

        min_hlt = min(min(Hlt_constant),min(Hlt_varying))
        max_hlt = max(max(Hlt_constant),max(Hlt_varying))
        spacing_hlt = (max_hlt-min_hlt)*0.05

        arr[0, 1].set_ylim([min_hlt-spacing_hlt,max_hlt+spacing_hlt])

        arr[0, 1].legend(frameon=True, loc='lower left', markerscale=11)

        arr[1, 0].plot(time_constant, mass_constant, color=self.plottingColors['doubleLine'][0],label='$\\dot{m} = 0$', linestyle='-')
        arr[1, 0].plot(time_varying, mass_varying, color=self.plottingColors['doubleLine'][1],label='$\\dot{m} \\neq 0$', linestyle='-')
        arr[1, 0].set_xlim([0,max(max(time_varying),max(time_constant))])
        increment_ylim = abs(min(mass_varying)-1.0)*0.05
        arr[1, 0].set_ylim([min(mass_varying),1.0+increment_ylim])
        arr[1, 0].legend(frameon=True, loc='lower left', markerscale=11)

        Delta_Hlt = []
        for i in range(len(Hlt_varying)):
           if i > 0 and i <len(Hlt_varying)-1:
               Delta_Hlt.append(Hlt_varying[i]-Hlt_varying[i-1])

        arr[1,1].plot(mass_varying[1:len(mass_varying)-1],Delta_Hlt)
        arr[1,1].set_xlim([1,min(mass_varying)])
        #arr[1, 1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.1e'))

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)


        hamiltonian = Hlt_constant[0]
        plt.suptitle('$L_' + str(self.lagrangePointNr) + '$ ($a_{lt} = ' + str(
            "{:3.2f}".format(self.accelerationMagnitude)) + '$, $\\alpha = ' + str(
            "{:3.1f}".format(self.alpha)) + '$, $H_{lt} = ' + str(
            "{:7.6f}".format(hamiltonian)) + '$, $I_{sp} = ' + str(self.specificImpulse) + '$) - Varying mass effect',\
                     size=20)

        plt.savefig('../../data/figures/varying_mass/L' + str(
            self.lagrangePointNr) + '_' + str(
            "{:7.6f}".format(self.accelerationMagnitude)) + '_' + str(
            "{:7.6f}".format(self.alpha)) + '_' + str(self.familyMember) + '_' + str(
            self.specificImpulse) + '_periodicity_constraints.png', transparent=True, dpi=300,
                    bbox_inches='tight')

if __name__ == '__main__':

    lagrange_point_nr = 1
    acceleration_magnitude = 0.01
    specific_impulses = [1500,2000,3000]
    alpha = 0.0
    member = 1938
    low_dpi = False

    for specific_impulse in specific_impulses:
        analyze_varying_mass = VaryingMassAnalysis(lagrange_point_nr,  acceleration_magnitude, alpha, member, \
                                                     specific_impulse, low_dpi)

        analyze_varying_mass.verify_howell_reasoning()
        #analyze_varying_mass.analyze_varying_mass_effect()

        del analyze_varying_mass


