import pandas as pd
import numpy as np

def load_orbit_augmented(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['time', 'x', 'y', 'z', 'xdot', 'ydot', 'zdot','alt','alpha','beta','m']
    return data

def load_lagrange_points_location_augmented(a_lt,alpha):

    lagrange_point_nrs = [1,2,3,4,5]
    seeds = [0.0,180.0]
    continuations = ['forward','backward']
    counter = 0
    for seed in seeds:
        for cont in continuations:

            counter = counter+1


            file_path_1 = '../../data/raw/equilibria/L1_acceleration_'  \
                                            + str("{:7.6f}".format(a_lt)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + cont +'_equilibria.txt'
            file_path_2 = '../../data/raw/equilibria/L2_acceleration_'  \
                                            + str("{:7.6f}".format(a_lt)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + cont +'_equilibria.txt'
            file_path_3 = '../../data/raw/equilibria/L3_acceleration_'  \
                                            + str("{:7.6f}".format(a_lt)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + cont +'_equilibria.txt'
            file_path_4 = '../../data/raw/equilibria/L4_acceleration_'  \
                                            + str("{:7.6f}".format(a_lt)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + cont +'_equilibria.txt'
            file_path_5 = '../../data/raw/equilibria/L5_acceleration_'  \
                                            + str("{:7.6f}".format(a_lt)) + '_' \
                                            + str("{:7.6f}".format(seed)) + '_' + cont +'_equilibria.txt'

            if counter == 1:
                data_1 = pd.read_table(file_path_1, delim_whitespace=True, header=None).filter(list(range(4)))
                data_1.columns = ['alpha', 'x', 'y', 'iterations']

                data_2 = pd.read_table(file_path_2, delim_whitespace=True, header=None).filter(list(range(4)))
                data_2.columns = ['alpha', 'x', 'y', 'iterations']

                data_3 = pd.read_table(file_path_3, delim_whitespace=True, header=None).filter(list(range(4)))
                data_3.columns = ['alpha', 'x', 'y', 'iterations']

                data_4 = pd.read_table(file_path_4, delim_whitespace=True, header=None).filter(list(range(4)))
                data_4.columns = ['alpha', 'x', 'y', 'iterations']

                data_5 = pd.read_table(file_path_5, delim_whitespace=True, header=None).filter(list(range(4)))
                data_5.columns = ['alpha', 'x', 'y', 'iterations']
            else:
                data_1Temp = pd.read_table(file_path_1, delim_whitespace=True, header=None).filter(list(range(4)))
                data_1Temp.columns = ['alpha', 'x', 'y', 'iterations']

                data_2Temp = pd.read_table(file_path_2, delim_whitespace=True, header=None).filter(list(range(4)))
                data_2Temp.columns = ['alpha', 'x', 'y', 'iterations']

                data_3Temp = pd.read_table(file_path_3, delim_whitespace=True, header=None).filter(list(range(4)))
                data_3Temp.columns = ['alpha', 'x', 'y', 'iterations']

                data_4Temp = pd.read_table(file_path_4, delim_whitespace=True, header=None).filter(list(range(4)))
                data_4Temp.columns = ['alpha', 'x', 'y', 'iterations']

                data_5Temp = pd.read_table(file_path_5, delim_whitespace=True, header=None).filter(list(range(4)))
                data_5Temp.columns = ['alpha', 'x', 'y', 'iterations']

                data_1.append(data_1Temp)
                data_2.append(data_2Temp)
                data_3.append(data_3Temp)
                data_4.append(data_4Temp)
                data_5.append(data_5Temp)



    data_1['alpha'] = data_1.alpha*180/np.pi
    index_1 = 10*int(alpha)
    equilibrium_1 = data_1.iloc[index_1]

    data_2['alpha'] = data_2.alpha*180/np.pi
    index_2 = 10*int(alpha)
    equilibrium_2 = data_2.iloc[index_2]

    data_3['alpha'] = data_3.alpha * 180 / np.pi
    index_3 = 10 * int(alpha)
    equilibrium_3 = data_3.iloc[index_3]

    if len(data_4['alpha']) > 5:
        data_4['alpha'] = data_4.alpha * 180 / np.pi
        index_4 = 10 * int(alpha)
        equilibrium_4 = data_4.iloc[index_4]

    if len(data_4['alpha']) > 5:
        data_5['alpha'] = data_5.alpha * 180 / np.pi
        index_5 = 10 * int(alpha)
        equilibrium_5 = data_5.iloc[index_5]


    if a_lt > 0:
        print
        if len(data_4['alpha']) > 5 and len(data_5['alpha']) > 5:
            location_lagrange_points = {'L1': [equilibrium_1['x'], equilibrium_1['y'], 0.0],
                                        'L2': [equilibrium_2['x'], equilibrium_2['y'], 0.0],
                                        'L3': [equilibrium_3['x'], equilibrium_3['y'], 0.0],
                                        'L4': [equilibrium_4['x'], equilibrium_4['y'], 0.0],
                                        'L5': [equilibrium_5['x'], equilibrium_5['y'], 0.0]}
        if len(data_4['alpha']) > 5 and len(data_5['alpha']) < 5:
            location_lagrange_points = {'L1': [equilibrium_1['x'], equilibrium_1['y'], 0.0],
                                        'L2': [equilibrium_2['x'], equilibrium_2['y'], 0.0],
                                        'L3': [equilibrium_3['x'], equilibrium_3['y'], 0.0],
                                        'L4': [equilibrium_4['x'], equilibrium_4['y'], 0.0],
                                        'L5': [0.0, 0.0, 0.0]}

        if len(data_4['alpha']) < 5 and len(data_5['alpha']) > 5:
            location_lagrange_points = {'L1': [equilibrium_1['x'], equilibrium_1['y'], 0.0],
                                        'L2': [equilibrium_2['x'], equilibrium_2['y'], 0.0],
                                        'L3': [equilibrium_3['x'], equilibrium_3['y'], 0.0],
                                        'L4': [0.0, 0.0, 0.0],
                                        'L5': [equilibrium_5['x'], equilibrium_5['y'], 0.0]}
        if len(data_4['alpha']) < 5 and len(data_5['alpha']) < 5:
            location_lagrange_points = {'L1': [equilibrium_1['x'], equilibrium_1['y'], 0.0],
                                        'L2': [equilibrium_2['x'], equilibrium_2['y'], 0.0],
                                        'L3': [equilibrium_3['x'], equilibrium_3['y'], 0.0],
                                        'L4': [0.0, 0.0, 0.0],
                                        'L5': [0.0, 0.0, 0.0]}

    else:
        location_lagrange_points = {'L1': [0.8369151483688, 0, 0],
                                    'L2': [1.1556821477825, 0, 0],
                                    'L3': [-1.003037581609, 0, 0],
                                    'L4': [0.4878494189827, 0.8660254037844, 0],
                                    'L5': [0.4878494189827, -0.8660254037844, 0]}

    location_lagrange_points = pd.DataFrame.from_dict(location_lagrange_points)
    location_lagrange_points.index = ['x', 'y', 'z']
    return location_lagrange_points

def load_equilibria_acceleration(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(4)))
    data.columns = ['alpha', 'x', 'y', 'iterations']
    return data

def compute_eigenvalue_contour(x_loc,y_loc,desiredType, desiredMode, threshold):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    list = []
    for q in range(len(x_loc)):
        for s in range(len(y_loc)):

            # Compute terms of the state transition matrix
            r1 = np.sqrt((massParameter + x_loc[q]) ** 2 + y_loc[s] ** 2)
            r2 = np.sqrt(( (x_loc[q] - 1 +massParameter) ** 2) + (y_loc[s] ** 2) )

            r1Cubed = r1 ** 3
            r2Cubed = r2 ** 3


            primaryTerm   =  (1-massParameter)/r1Cubed
            secondaryTerm =  massParameter/r2Cubed

            r1Fifth = r1Cubed * r1 * r1
            r2Fifth = r2Cubed * r2 * r2

            primaryTermDer = 3.0 *(1 - massParameter) / r1Fifth
            secondaryTermDer = 3.0 * massParameter / r2Fifth



            # Compute the 4x4 matrix elements
            Uxx = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ( (x_loc[q] + massParameter) ** 2 ) \
            + secondaryTermDer * (( x_loc[q] - 1 + massParameter ) ** 2)
            Uyy = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ( (y_loc[s]) ** 2 ) \
            + secondaryTermDer * ( (y_loc[s]) ** 2)
            Uxy = primaryTermDer * (x_loc[q] + massParameter) * y_loc[s]  \
                  + secondaryTermDer * (x_loc[q] - 1 + massParameter) * y_loc[s]
            Uyx = Uxy

            SPM = [[0,0,1,0],\
                    [0,0,0,1],\
                    [Uxx,Uxy,0,2],\
                    [Uyx,Uyy,-2,0]]

            # SPMTRANSPOSE = [[0, 0, Uxx, Uyx], \
            #                 [0, 0, Uxy, Uyy], \
            #                 [1, 0, 0, -2], \
            #                 [0, 1, 2, 0]]

            eigenValues, eigenVectors = np.linalg.eig(SPM)

            numberOfSaddle = 0
            numberOfCenters = 0
            numberOfMixed = 0

            maxSaddleEigenValue = -1000
            maxCenterEigenValue = -1000
            maxSaddleEigenVector = [0,0,0,0]
            maxCenterEigenVector = [0,0,0,0]


            eigenValueDeviation = 1.0e-3
            for i in range(len(eigenValues)):
                if ( ( np.abs(np.real(eigenValues[i])) > eigenValueDeviation ) and ( np.abs(np.imag(eigenValues[i])) < eigenValueDeviation ) ):
                    numberOfSaddle = numberOfSaddle + 1
                    if desiredType == 1 and desiredMode == 1:
                        if np.real(eigenValues[i]) > maxSaddleEigenValue:
                            maxSaddleEigenValue = np.real(eigenValues[i])
                            maxSaddleEigenVector = np.real(eigenVectors[:,i])


                if ( ( np.abs(np.real(eigenValues[i])) < eigenValueDeviation )and  ( np.abs(np.imag(eigenValues[i])) > eigenValueDeviation ) ):
                    numberOfCenters = numberOfCenters + 1
                    if desiredType < 3 and desiredMode == 2:

                        if np.imag(eigenValues[i]) > maxCenterEigenValue:
                            maxCenterEigenValue = eigenValues[i]
                            maxCenterEigenVector = eigenVectors[:,i]


                if ( ( np.abs(np.real(eigenValues[i])) > eigenValueDeviation ) and ( np.abs(np.imag(eigenValues[i])) > eigenValueDeviation ) ):
                    numberOfMixed = numberOfMixed + 1


            if numberOfSaddle == 2 and numberOfCenters == 2:
                type = 1

            elif numberOfCenters == 4:
                type = 2
            elif numberOfMixed == 4:
                type = 3
            elif numberOfSaddle == 4:
                type = 4

            if type == desiredType:
                if desiredMode == 1:

                    if maxSaddleEigenValue > threshold:
                        maxSaddleEigenValue = 1.2*threshold
                    data = [x_loc[q], y_loc[s], maxSaddleEigenValue, maxSaddleEigenVector[0],maxSaddleEigenVector[1],maxSaddleEigenVector[2],maxSaddleEigenVector[3]]

                if desiredMode == 2:
                    if maxCenterEigenValue > threshold:
                        maxCenterEigenValue = 1.2*threshold
                    data = [x_loc[q], y_loc[s], np.real(maxCenterEigenValue), np.imag(maxCenterEigenValue), \
                            np.real(maxCenterEigenVector[0]), np.imag(maxCenterEigenVector[0]), \
                            np.real(maxCenterEigenVector[1]), np.imag(maxCenterEigenVector[1]), \
                            np.real(maxCenterEigenVector[2]), np.imag(maxCenterEigenVector[2]), \
                            np.real(maxCenterEigenVector[3]), np.imag(maxCenterEigenVector[3]) ]

                    list.append(data)
    if desiredMode == 1:
        df = pd.DataFrame(list, columns=['x','y','maxLambda','maxV1','maxV2','maxV3','maxV4'])

    if desiredMode == 2:
        df = pd.DataFrame(list, columns=['x','y','maxLambdaReal','maxLambdaImag','maxV1Real','maxV1Imag','maxV2Real',\
                                         'maxV2Imag','maxV3Real','maxV3Imag','maxV4Real','maxV4Imag'])
    if desiredType == 1:
        desiredTypeString = 'SxC'
    if desiredType == 2:
        desiredTypeString = 'CxC'
    if desiredMode == 1:
        desiredModeString = 'Saddle'
    if desiredMode == 2:
        desiredModeString = 'Center'

    #np.savetxt('../../data/raw/equilibria/eigenvalue' + desiredTypeString + '_' + desiredModeString + '.txt', df.values, fmt='%13.12f')

    return df

def compute_stability_type(x_loc,y_loc,desiredType):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    list = []
    for q in range(len(x_loc)):
        for s in range(len(y_loc)):

            # Compute terms of the state transition matrix
            r1 = np.sqrt((massParameter + x_loc[q]) ** 2 + y_loc[s] ** 2)
            r2 = np.sqrt(( (x_loc[q] - 1 +massParameter) ** 2) + (y_loc[s] ** 2) )

            r1Cubed = r1 ** 3
            r2Cubed = r2 ** 3


            primaryTerm   =  (1-massParameter)/r1Cubed
            secondaryTerm =  massParameter/r2Cubed

            r1Fifth = r1Cubed * r1 * r1
            r2Fifth = r2Cubed * r2 * r2

            primaryTermDer = 3.0 *(1 - massParameter) / r1Fifth
            secondaryTermDer = 3.0 * massParameter / r2Fifth



            # Compute the 4x4 matrix elements
            Uxx = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ( (x_loc[q] + massParameter) ** 2 ) \
            + secondaryTermDer * (( x_loc[q] - 1 + massParameter ) ** 2)
            Uyy = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ( (y_loc[s]) ** 2 ) \
            + secondaryTermDer * ( (y_loc[s]) ** 2)
            Uxy = primaryTermDer * (x_loc[q] + massParameter) * y_loc[s]  \
                  + secondaryTermDer * (x_loc[q] - 1 + massParameter) * y_loc[s]
            Uyx = Uxy

            SPM = [[0,0,1,0],\
                    [0,0,0,1],\
                    [Uxx,Uxy,0,2],\
                    [Uyx,Uyy,-2,0]]

            print(SPM)

            # SPMTRANSPOSE = [[0, 0, Uxx, Uyx], \
            #                 [0, 0, Uxy, Uyy], \
            #                 [1, 0, 0, -2], \
            #                 [0, 1, 2, 0]]

            eigenValues, eigenVectors = np.linalg.eig(SPM)

            numberOfSaddle = 0
            numberOfCenters = 0
            numberOfMixed = 0

            eigenValueDeviation = 1.0e-3
            for i in range(len(eigenValues)):
                if ( ( np.abs(np.real(eigenValues[i])) > eigenValueDeviation ) and ( np.abs(np.imag(eigenValues[i])) < eigenValueDeviation ) ):
                    numberOfSaddle = numberOfSaddle + 1
                if ( ( np.abs(np.real(eigenValues[i])) < eigenValueDeviation )and  ( np.abs(np.imag(eigenValues[i])) > eigenValueDeviation ) ):
                    numberOfCenters = numberOfCenters + 1
                if ( ( np.abs(np.real(eigenValues[i])) > eigenValueDeviation ) and ( np.abs(np.imag(eigenValues[i])) > eigenValueDeviation ) ):
                    numberOfMixed = numberOfMixed + 1

            if numberOfSaddle == 2 and numberOfCenters == 2:
                type = 1
            elif numberOfCenters == 4:
                type = 2
            elif numberOfMixed == 4:
                type = 3
            elif numberOfSaddle == 4:
                type = 4

            if type == desiredType:
                data = [x_loc[q], y_loc[s], type]
                list.append(data)

    df = pd.DataFrame(list, columns=['x','y','type'])
    #np.savetxt('../../data/raw/equilibria/stability_' + str(desiredType) + '.txt', df.values, fmt='%13.12f')

    return df

def load_stability_data(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(3)))
    data.columns = ['x', 'y','type']
    return data

def load_eigenvalue_data(file_path, mode):
    if mode == 1:
        data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(8)))
        data.columns = ['x','y','maxLambda','maxV1','maxV2','maxV3','maxV4']
    if mode == 2:
        data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(13)))
        data.columns = ['x','y','maxLambdaReal','maxLambdaImag','maxV1Real','maxV1Imag','maxV2Real','maxV2Imag',\
                                                                'maxV3Real','maxV3Imag','maxV4Real','maxV4Imag']

    return data

def cr3bplt_velocity(x_loc, y_loc, acc, alpha, Hlt):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    # Compute the thrust components
    alt_x = acc * np.cos(alpha * np.pi / 180.0)
    alt_y = acc * np.sin(alpha * np.pi / 180.0)


    r_1 = np.sqrt((massParameter + x_loc) ** 2 + y_loc ** 2)
    r_2 = np.sqrt((1 - massParameter - x_loc) ** 2 + y_loc ** 2)


    vSquared = 2*Hlt + 2*alt_x*x_loc + 2*alt_y*y_loc + x_loc ** 2 + y_loc ** 2 + 2 * (1 - massParameter) / r_1 + 2 * massParameter / r_2
    return vSquared

def potential_deviation(acc, alpha, x, y):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)


    r_1 = np.sqrt((massParameter + x) ** 2 + y ** 2)
    r_2 = np.sqrt((1 - massParameter - x) ** 2 + y ** 2)

    r1Cubed = r_1 ** 3
    r2Cubed = r_2 ** 3

    primaryTerm = (1 - massParameter) / r1Cubed
    secondaryTerm = massParameter / r2Cubed

    omegaX = x*(1 - primaryTerm - secondaryTerm) + massParameter*(-primaryTerm-secondaryTerm)+secondaryTerm+acc*np.cos(alpha)
    omegaY = y*(1 - primaryTerm - secondaryTerm) +acc*np.sin(alpha)

    deviationNorm = np.sqrt(omegaX ** 2+ omegaY ** 2)


    return deviationNorm

def load_equilibria_alpha(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(4)))
    data.columns = ['acc', 'x', 'y', 'iterations']
    return data


def load_states_continuation(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(12)))
    data.columns = ['orbitID', 'hlt','x','y','z','xdot','ydot','zdot','alt','alpha','beta','m']
    return data

def load_differential_correction(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['iterations', 'hlt','period','x','y','z','xdot','ydot','zdot','alt','alpha']
    return data

def load_initial_conditions(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['iterations', 'hlt','period','x','y','z','xdot','ydot','zdot','alt','alpha']
    return data

if __name__ == '__main__':
    load_lagrange_points_location(0.001,60.0)