import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def load_manifold_incl_stm(file_path):
    pd.options.mode.chained_assignment = None  # Turn off SettingWithCopyWarning

    input_data = pd.read_table(file_path, delim_whitespace=True, header=None).rename(columns={0: 'time'})
    split_on_index = list(input_data[input_data['time'] == 0].index)

    output_data = []
    for idx, start_index in enumerate(split_on_index):

        if idx != len(split_on_index)-1:
            data_per_orbit = input_data[start_index:split_on_index[idx+1]]
        else:
            data_per_orbit = input_data[start_index:]

        data_per_orbit['orbitNumber'] = idx

        output_data.append(data_per_orbit)
        pass

    output_data = pd.concat(output_data).reset_index(drop=True).set_index(['orbitNumber', 'time'])
    return output_data


# TODO fill in folder path
folder_path = '/Users/koen/tudatBundle/tudatApplications/thesisProject/data/raw/manifold/'

# Load manifolds
L1_W_S_plus = load_manifold_incl_stm(folder_path + 'L1_vertical_1159_W_S_plus.txt')
L1_W_S_min = load_manifold_incl_stm(folder_path + 'L1_vertical_1159_W_S_min.txt')
L1_W_U_plus = load_manifold_incl_stm(folder_path + 'L1_vertical_1159_W_U_plus.txt')
L1_W_U_min = load_manifold_incl_stm(folder_path + 'L1_vertical_1159_W_U_min.txt')
L2_W_S_plus = load_manifold_incl_stm(folder_path + 'L2_vertical_1275_W_S_plus.txt')
L2_W_S_min = load_manifold_incl_stm(folder_path + 'L2_vertical_1275_W_S_min.txt')
L2_W_U_plus = load_manifold_incl_stm(folder_path + 'L2_vertical_1275_W_U_plus.txt')
L2_W_U_min = load_manifold_incl_stm(folder_path + 'L2_vertical_1275_W_U_min.txt')

# Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(0, 100):
    # Plot L1
    ax.plot(L1_W_S_plus.xs(i)[1], L1_W_S_plus.xs(i)[2], L1_W_S_plus.xs(i)[3], c='g')
    ax.plot(L1_W_S_min.xs(i)[1], L1_W_S_min.xs(i)[2], L1_W_S_min.xs(i)[3], c='g')
    ax.plot(L1_W_U_plus.xs(i)[1], L1_W_U_plus.xs(i)[2], L1_W_U_plus.xs(i)[3], c='r')
    ax.plot(L1_W_U_min.xs(i)[1], L1_W_U_min.xs(i)[2], L1_W_U_min.xs(i)[3], c='r')

    # Plot L2
    ax.plot(L2_W_S_plus.xs(i)[1], L2_W_S_plus.xs(i)[2], L2_W_S_plus.xs(i)[3], c='g')
    ax.plot(L2_W_S_min.xs(i)[1], L2_W_S_min.xs(i)[2], L2_W_S_min.xs(i)[3], c='g')
    ax.plot(L2_W_U_plus.xs(i)[1], L2_W_U_plus.xs(i)[2], L2_W_U_plus.xs(i)[3], c='r')
    ax.plot(L2_W_U_min.xs(i)[1], L2_W_U_min.xs(i)[2], L2_W_U_min.xs(i)[3], c='r')

plt.show()
