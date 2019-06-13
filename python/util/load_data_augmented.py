import pandas as pd
import numpy as np

def load_orbit_augmented(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot','alt','alpha','beta','m']
    return data

def load_lagrange_points_location_augmented(a_lt,alpha):
    file_path_1 = '../../data/raw/equilibria/L1_' + str("{:7.6f}".format(a_lt)) + "_equilibria.txt"
    file_path_2 = '../../data/raw/equilibria/L2_' + str("{:7.6f}".format(a_lt)) + "_equilibria.txt"

    data_1 = pd.read_table(file_path_1, delim_whitespace=True, header=None).filter(list(range(4)))
    data_1.columns = ['alpha','x','y','iterations']

    data_2 = pd.read_table(file_path_2, delim_whitespace=True, header=None).filter(list(range(4)))
    data_2.columns = ['alpha','x','y','iterations']

    data_1['alpha'] = data_1.alpha*180/np.pi
    index_1 = 10*int(alpha)
    equilibrium_1 = data_1.iloc[index_1]

    data_2['alpha'] = data_2.alpha*180/np.pi
    index_2 = 10*int(alpha)
    equilibrium_2 = data_2.iloc[index_2]


    if a_lt > 0.0:
        location_lagrange_points = {'L1': [equilibrium_1['x'], equilibrium_1['y'], 0.0],
                                    'L2': [equilibrium_2['x'], equilibrium_2['y'], 0.0]}
    else:
        location_lagrange_points = {'L1': [0.8369151483688, 0, 0],
                                    'L2': [1.1556821477825, 0, 0],
                                    'L3': [-1.003037581609, 0, 0],
                                    'L4': [0.4878494189827, 0.8660254037844, 0],
                                    'L5': [0.4878494189827, -0.8660254037844, 0]}

    location_lagrange_points = pd.DataFrame.from_dict(location_lagrange_points)
    location_lagrange_points.index = ['x', 'y', 'z']
    return location_lagrange_points

def load_states_continuation(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(12)))
    data.columns = ['orbitID', 'hlt','x','y','z','xdot','ydot','zdot','alt','alpha','beta','m']
    return data

def load_differential_correction(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['iterations', 'hlt','period','x','y','z','xdot','ydot','zdot','alt','alpha']
    return data


if __name__ == '__main__':
    load_lagrange_points_location(0.001,60.0)