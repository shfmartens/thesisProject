import pandas as pd
import numpy as np


def load_manifold(file_path):
    pd.options.mode.chained_assignment = None  # Turn off SettingWithCopyWarning

    input_data = pd.read_table(file_path, delim_whitespace=True, header=None,
                               names=['time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot'])
    split_on_index = list(input_data[input_data['time'] == 0].index)

    output_data = []
    for idx, start_index in enumerate(split_on_index):

        if idx != len(split_on_index)-1:
            data_per_orbit = input_data[start_index:split_on_index[idx+1]]
        else:
            data_per_orbit = input_data[start_index:]

        data_per_orbit['orbitNumber'] = idx+1
        output_data.append(data_per_orbit)
        pass

    output_data = pd.concat(output_data).reset_index(drop=True).set_index(['orbitNumber', 'time'])
    return output_data


def load_orbit(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(7)))
    data.columns = ['time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot']
    return data


def load_lagrange_points_location():
    location_lagrange_points = {'L1': [0.8369151483688, 0, 0],
                                'L2': [1.1556821477825, 0, 0],
                                'L3': [-1.003037581609, 0, 0],
                                'L4': [0.4878494189827, 0.8660254037844, 0],
                                'L5': [0.4878494189827, -0.8660254037844, 0]}

    location_lagrange_points = pd.DataFrame.from_dict(location_lagrange_points)
    location_lagrange_points.index = ['x', 'y', 'z']
    return location_lagrange_points


def load_bodies_location():
    location_bodies = {'Earth': [0, 0, 0, 6371 / 384400],
                       'Moon': [1, 0, 0, 1737 / 384400]}
    location_bodies = pd.DataFrame.from_dict(location_bodies)
    location_bodies.index = ['x', 'y', 'z', 'r']
    return location_bodies


def cr3bp_velocity(x_loc, y_loc, c):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    r_1 = np.sqrt((x_loc + massParameter) ** 2 + y_loc ** 2)
    r_2 = np.sqrt((x_loc - 1 + massParameter) ** 2 + y_loc ** 2)
    v = x_loc ** 2 + y_loc ** 2 + 2 * (1 - massParameter) / r_1 + 2 * massParameter / r_2 - c
    return v


if __name__ == "__main__":
    manifold_file_path = "../data/near_vertical_1_W_S_min.txt"
    manifold_df = load_manifold(manifold_file_path)
    print(manifold_df)

    orbit_file_path = "../data/near_vertical_1_final_orbit.txt"
    orbit_df = load_orbit(orbit_file_path)
    print(orbit_df)
