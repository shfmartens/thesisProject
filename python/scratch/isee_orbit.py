import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
sns.set_style("whitegrid")
import time
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

# Dynamical system
gamma_1 = 1.001e-2
EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
massParameter = (EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER) / (SUN_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER)
r_earth = 6371
r_sun = 695700
d_earth_sun = 149.6e6
x_loc_l1 = 1 - massParameter - gamma_1
orbit_radius_moon = 384400

print(235*r_earth/d_earth_sun)

# Orbit 1 parameters
e = 0.91
r_p = 1.04 * r_earth / d_earth_sun
r_a = 23 * r_earth / d_earth_sun

a = (r_a + r_p) / 2
b1 = r_a * (1 - e)
b2 = r_p * (1 + e)
b = (b1 + b2) / 2

theta = np.linspace(0, 2*math.pi, 1000)

r_elliptical_orbit = b / (1 + e * np.cos(theta))
x_elliptical_orbit = r_elliptical_orbit * np.cos(theta) + 1 - massParameter
y_elliptical_orbit = r_elliptical_orbit * np.sin(theta)

b_libration = 205000 / d_earth_sun
a_libration = 666670 / d_earth_sun


print((b_libration/a_libration))
print(1 - (b_libration/a_libration)**2)
e_libration = np.sqrt(1 - (b_libration/a_libration)**2)

print(e_libration)
# r_libration = b_libration / (1 + e_libration * np.cos(theta))
# y_libration_orbit = r_libration * np.cos(theta) + a_libration
# x_libration_orbit = r_libration * np.sin(theta) + x_loc_l1

fig = plt.figure(figsize=(7*(1+np.sqrt(5))/2, 7))
ax = fig.gca()

# Plot orbit
ax.plot(x_elliptical_orbit, y_elliptical_orbit, label='ISEE-1, ISEE-2 orbit', c='navy', linestyle='--')

# Plot SE L1
# ax.scatter(x_loc_l1, 0, color='black', marker='x', label='Earth-Sun $L_1$')
ellipse = Ellipse(xy=(x_loc_l1, 0), width=b_libration*2, height=a_libration*2, facecolor='none', edgecolor='navy', label='ISEE-3')
# ax.add_artist(ellipse)

# Plot primaries
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sun = r_sun / d_earth_sun * np.outer(np.cos(u), np.sin(v)) - massParameter
y_sun = r_sun / d_earth_sun * np.outer(np.sin(u), np.sin(v))
z_sun = r_sun / d_earth_sun * np.outer(np.ones(np.size(u)), np.cos(v))
x_earth = r_earth / d_earth_sun * np.outer(np.cos(u), np.sin(v)) + 1 - massParameter
y_earth = r_earth / d_earth_sun * np.outer(np.sin(u), np.sin(v))
z_earth = r_earth / d_earth_sun * np.outer(np.ones(np.size(u)), np.cos(v))

# ax.contourf(x_sun, y_sun, z_sun, colors='black')
ax.contourf(x_sun, y_sun, z_sun, colors='black', label='Earth')
ax.contourf(x_earth, y_earth, z_earth, colors='black', label='Earth')
# ax.legend(frameon=True, loc='lower right')
ax.set_xlim([0.988, 1.002])
ax.set_ylim([-0.005, 0.005])
plt.axis('equal')
plt.axis('off')
# plt.savefig('/Users/koen/Documents/Courses/AE5810 Thesis Space/Meetings/Midterm presentation/isee_3.pdf')
plt.show()
