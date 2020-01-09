import pandas as pd
import numpy as np
import math

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

def load_equilibria_acceleration_deviation(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(7)))
    data.columns = ['alpha', 'dx', 'dy', 'dz', 'dxdot', 'dydot', 'dzdot']
    return data

def compute_hamiltonian_from_state(x,y,z,xdot,ydot,zdot,accMag,alpha,mass):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    mass_parameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    r1 = np.sqrt((x+mass_parameter) ** 2 + y ** 2)
    r2 = np.sqrt((x-1.0+mass_parameter) ** 2 + y ** 2 )
    Omega = 0.5*(x ** 2 + y ** 2) + (1.0-mass_parameter)/r1 + (mass_parameter)/r2
    V = np.sqrt(xdot ** 2 + ydot ** 2)
    jacobi = 2*Omega - V ** 2


    inner_product = x * accMag/mass * np.cos(alpha*np.pi/180.0) + y * accMag/mass * np.sin(alpha*np.pi/180.0)
    hamiltonian = -0.5*jacobi - inner_product


    return hamiltonian


def compute_hamiltonian_from_list(xList,yList,accelerationMagnitude,alphaList):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    r1 = np.sqrt((massParameter + x) ** 2 + (y ** 2))
    r2 = np.sqrt(((x - 1 + massParameter) ** 2) + (y ** 2))

    hamiltonianList = []
    for i in range(len(xList)):
        xpos = xList[i]
        ypos = yList[i]
        alpha = alphaList[i]
        alt = accelerationMagnitude

        r1 = np.sqrt((massParameter + xpos) ** 2 + (ypos ** 2))
        r2 = np.sqrt(((xpos - 1 + massParameter) ** 2) + (ypos ** 2))

        primaryTerm = (1 - massParameter) / r1
        secondaryTerm = massParameter / r2

        jacobi_Integral = xpos ** 2 + ypos ** 2 + 2*primaryTerm + 2*secondaryTerm

        inner_product = xpos * alt*np.cos(alpha) + ypos * alt*np.cos(alpha)

        hamiltonian = -0.5*jacobi_Integral - inner_product

        if np.abs(hamiltonian) < 4:
            if alpha > np.pi:
                alphaStore = alpha - 2*np.pi
                hamiltonianList.append([alphaStore,hamiltonian])
            else:
                hamiltonianList.append([alpha,hamiltonian])


    return hamiltonianList

def compute_hamiltonian_from_list_second_version(xList,yList,accelerationMagnitude, alphaList):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    hamiltonianList = []
    for i in range(len(xList)):
        xpos = xList[i]
        ypos = yList[i]
        alpha = alphaList[i]
        alt = accelerationMagnitude

        xpos = xList[i]
        ypos = yList[i]
        alpha = alphaList[i]
        alt = accelerationMagnitude

        r1 = np.sqrt((massParameter + xpos) ** 2 + (ypos ** 2))
        r2 = np.sqrt(((xpos - 1 + massParameter) ** 2) + (ypos ** 2))

        primaryTerm = (1 - massParameter) / r1
        secondaryTerm = massParameter / r2

        jacobi_Integral = xpos ** 2 + ypos ** 2 + 2*primaryTerm + 2*secondaryTerm

        inner_product = xpos * alt * np.cos(alpha) + ypos * alt * np.sin(alpha)

        hamiltonian = -0.5 * jacobi_Integral - inner_product

        if np.abs(hamiltonian) < 4:
            if alpha > 2* np.pi:
                alpha = alpha -2*np.pi
            if alpha < 0* np.pi:
                alpha = alpha +2*np.pi
            hamiltonianList.append([alpha, hamiltonian])

    return hamiltonianList



def compute_potential_from_list(xList,yList,accMag,alphaList):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    potentialList = []
    for i in range(len(xList)):
        xpos = xList[i]
        ypos = yList[i]
        alpha = alphaList[i]
        acc = accMag

        r_1 = np.sqrt((massParameter + xpos) ** 2 + ypos ** 2)
        r_2 = np.sqrt((1 - massParameter - xpos) ** 2 + ypos ** 2)

        r1Cubed = r_1 ** 3
        r2Cubed = r_2 ** 3

        primaryTerm = (1 - massParameter) / r1Cubed
        secondaryTerm = massParameter / r2Cubed

        omegaX = xpos * (1 - primaryTerm - secondaryTerm) + massParameter * (
                    -primaryTerm - secondaryTerm) + secondaryTerm + acc * np.cos(alpha)
        omegaY = ypos * (1 - primaryTerm - secondaryTerm) + acc * np.sin(alpha)

        deviationNorm = np.sqrt(omegaX ** 2 + omegaY ** 2)


        deviationNorm = np.sqrt(omegaX ** 2 + omegaY ** 2)

        potentialList.append([alpha, deviationNorm])


    return potentialList


def compute_stability_type_from_list(alphaList,xList,yList):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    maxSaddleList = []
    for i in range(len(xList)):
        xpos = xList[i]
        ypos = yList[i]
        alpha = alphaList[i]

        r1 = np.sqrt((massParameter + xpos) ** 2 + (ypos ** 2))
        r2 = np.sqrt(((xpos - 1 + massParameter) ** 2) + (ypos ** 2))

        r1Cubed = r1 ** 3
        r2Cubed = r2 ** 3

        primaryTerm = (1 - massParameter) / r1Cubed
        secondaryTerm = massParameter / r2Cubed

        r1Fifth = r1Cubed * r1 * r1
        r2Fifth = r2Cubed * r2 * r2

        primaryTermDer = 3.0 * (1 - massParameter) / r1Fifth
        secondaryTermDer = 3.0 * massParameter / r2Fifth

        # Compute the 4x4 matrix elements
        Uxx = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ((xpos + massParameter) ** 2) \
          + secondaryTermDer * ((xpos - 1 + massParameter) ** 2)
        Uyy = 1.0 - primaryTerm - secondaryTerm + primaryTermDer * ((ypos) ** 2) \
          + secondaryTermDer * ((ypos) ** 2)
        Uxy = primaryTermDer * (xpos + massParameter) * ypos \
          + secondaryTermDer * (xpos - 1 + massParameter) * ypos
        Uyx = Uxy

        SPM = [[0, 0, 1, 0], \
           [0, 0, 0, 1], \
           [Uxx, Uxy, 0, 2], \
           [Uyx, Uyy, -2, 0]]

        eigenValues, eigenVectors = np.linalg.eig(SPM)

        numberOfSaddle = 0
        numberOfCenters = 0
        numberOfMixed = 0

        maxSaddleEigenValue = -1000000000
        maxCenterEigenValue = -1000000000
        minSaddleEigenValue = 1000000000
        minCenterEigenValue = 1000000000
        maxSaddleEigenVector = [0, 0, 0, 0]
        maxCenterEigenVector = [0, 0, 0, 0]

        desiredType = 1
        desiredMode = 1
        eigenValueDeviation = 1.0e-3
        for i in range(len(eigenValues)):
            if ((np.abs(np.real(eigenValues[i])) > eigenValueDeviation) and (
                    np.abs(np.imag(eigenValues[i])) < eigenValueDeviation)):
                numberOfSaddle = numberOfSaddle + 1
                if desiredType == 1 and desiredMode == 1:
                    if np.real(eigenValues[i]) > maxSaddleEigenValue:
                        maxSaddleEigenValue = np.real(eigenValues[i])
                        maxSaddleEigenVector = np.real(eigenVectors[:, i])
                    if np.real(eigenValues[i]) < minSaddleEigenValue:
                        minSaddleEigenValue = np.real(eigenValues[i])
                        minSaddleEigenVector = np.real(eigenVectors[:, i])

            if ((np.abs(np.real(eigenValues[i])) < eigenValueDeviation) and (
                    np.abs(np.imag(eigenValues[i])) > eigenValueDeviation)):
                numberOfCenters = numberOfCenters + 1
                if desiredType < 3 and desiredMode == 2:

                    if np.imag(eigenValues[i]) > maxCenterEigenValue:
                        maxCenterEigenValue = eigenValues[i]
                        maxCenterEigenVector = eigenVectors[:, i]
                    if np.imag(eigenValues[i]) < minCenterEigenValue:
                        minCenterEigenValue = eigenValues[i]
                        minCenterEigenVector = eigenVectors[:, i]

            if ((np.abs(np.real(eigenValues[i])) > eigenValueDeviation) and (
                    np.abs(np.imag(eigenValues[i])) > eigenValueDeviation)):
                numberOfMixed = numberOfMixed + 1

        if numberOfSaddle == 2 and numberOfCenters == 2:
            type = 1

        elif numberOfCenters == 4:
            type = 2
        elif numberOfMixed == 4:
            type = 3
        elif numberOfSaddle == 4:
            type = 4

        if type == 1 and maxSaddleEigenValue < 1000:
            maxSaddleList.append([alpha, maxSaddleEigenValue])

    return maxSaddleList


def compute_eigenvalue_contour(x_loc,y_loc,desiredType, desiredMode, threshold):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    list = []
    for q in range(len(x_loc)):
        print(q)
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

            maxSaddleEigenValue = -1000000000
            maxCenterEigenValue = -1000000000
            minSaddleEigenValue = 1000000000
            minCenterEigenValue = 1000000000
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
                        if np.real(eigenValues[i]) < minSaddleEigenValue:
                            minSaddleEigenValue = np.real(eigenValues[i])
                            minSaddleEigenVector = np.real(eigenVectors[:,i])


                if ( ( np.abs(np.real(eigenValues[i])) < eigenValueDeviation )and  ( np.abs(np.imag(eigenValues[i])) > eigenValueDeviation ) ):
                    numberOfCenters = numberOfCenters + 1
                    if desiredType < 3 and desiredMode == 2:

                        if np.imag(eigenValues[i]) > maxCenterEigenValue:
                            maxCenterEigenValue = eigenValues[i]
                            maxCenterEigenVector = eigenVectors[:,i]
                        if np.imag(eigenValues[i]) < minCenterEigenValue:
                            minCenterEigenValue = eigenValues[i]
                            minCenterEigenVector = eigenVectors[:,i]


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


            if type == desiredType or type == 2:
                if desiredMode == 1:
                    #if maxSaddleEigenValue > threshold:
                        # maxSaddleEigenValue = 1.2*threshold

                    stabIndex = 0.5* (np.abs(maxSaddleEigenValue) + 1.0/np.abs(maxSaddleEigenValue) )
                    data = [x_loc[q], y_loc[s], maxSaddleEigenValue, minSaddleEigenValue, stabIndex, maxSaddleEigenVector[0],maxSaddleEigenVector[1],maxSaddleEigenVector[2],maxSaddleEigenVector[3] \
                        ,minSaddleEigenVector[0], minSaddleEigenVector[1], minSaddleEigenVector[2],minSaddleEigenVector[3]]

                if desiredMode == 2:
                    # if maxCenterEigenValue > threshold:
                    #     maxCenterEigenValue = 1.2*threshold

                    stabIndex = 0.5 * (np.abs(maxCenterEigenValue) + 1.0 / np.abs(maxCenterEigenValue))
                    data = [x_loc[q], y_loc[s], np.real(maxCenterEigenValue), np.imag(maxCenterEigenValue), \
                            np.real(minCenterEigenValue), np.imag(minCenterEigenValue), stabIndex,
                            np.real(maxCenterEigenVector[0]), np.imag(maxCenterEigenVector[0]), \
                            np.real(maxCenterEigenVector[1]), np.imag(maxCenterEigenVector[1]), \
                            np.real(maxCenterEigenVector[2]), np.imag(maxCenterEigenVector[2]), \
                            np.real(maxCenterEigenVector[3]), np.imag(maxCenterEigenVector[3]) ]

                list.append(data)
    if desiredMode == 1:
        df = pd.DataFrame(list, columns=['x','y','maxLambda','minLambda','stabIndex','maxV1','maxV2','maxV3','maxV4','minV1','minV2','minV3','minV4'])

    if desiredMode == 2:
        df = pd.DataFrame(list, columns=['x','y','maxLambdaReal','maxLambdaImag','minLambdaReal','minLambdaImag','stabIndex','maxV1Real','maxV1Imag','maxV2Real',\
                                         'maxV2Imag','maxV3Real','maxV3Imag','maxV4Real','maxV4Imag'])
    if desiredType == 1:
        desiredTypeString = 'SxC'
    if desiredType == 2:
        desiredTypeString = 'CxC'
    if desiredMode == 1:
        desiredModeString = 'Saddle'
    if desiredMode == 2:
        desiredModeString = 'Center'

    np.savetxt('../../data/raw/equilibria/eigenvalue' + desiredTypeString + '_' + desiredModeString + '_incCxC.txt', df.values, fmt='%13.12f')

    return df

def compute_stability_type(x_loc,y_loc,desiredType):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    list = []
    for q in range(len(x_loc)):
        print(q)
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

            #
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
    np.savetxt('../../data/raw/equilibria/stability_' + str(desiredType) + '_ZOOM_2000.txt', df.values, fmt='%13.12f')

    return df

def load_stability_data(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(3)))
    data.columns = ['x', 'y','type']
    return data

def load_eigenvalue_data(file_path, mode):
    if mode == 1:
        data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(13)))
        data.columns = ['x','y','maxLambda','minLambda','stabIndex','maxV1','maxV2','maxV3','maxV4','minV1','minV2','minV3','minV4']
    if mode == 2:
        data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(15)))
        data.columns = ['x','y','maxLambdaReal','maxLambdaImag','minLambdaReal','minLambdaImag','stabIndex','maxV1Real','maxV1Imag','maxV2Real','maxV2Imag',\
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

def compute_phase(x,y, librationPointNr):
    EARTH_GRAVITATIONAL_PARAMETER = 3.986004418E14
    SUN_GRAVITATIONAL_PARAMETER = 1.32712440018e20
    MOON_GRAVITATIONAL_PARAMETER = SUN_GRAVITATIONAL_PARAMETER / (328900.56 * (1.0 + 81.30059))
    massParameter = MOON_GRAVITATIONAL_PARAMETER / (MOON_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER)

    outputVariable = 0.0

    if librationPointNr < 3:
        x_argument = x - (1.0 - massParameter)
        y_argument = y
        phase = math.atan2(y_argument,x_argument)

        if librationPointNr == 1:
            if phase < 0:
                outputVariable = phase + 2*np.pi
            else:
                outputVariable = phase

    return outputVariable

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
    print(file_path)
    #data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(12)))
    #data.columns = ['orbitID', 'hlt','x','y','z','xdot','ydot','zdot','alt','alpha','beta','m']
    input_file = open(file_path)

    data = []
    for i, line in enumerate(input_file):
        lineSplitted = line.split()
        nodes = ((((len(lineSplitted)-3)/11 ) -1 ) /3 )+1
        data.append([float(lineSplitted[0]),float(lineSplitted[1]),float(lineSplitted[2]),float(lineSplitted[3]),float(lineSplitted[4]),
                     float(lineSplitted[5]), float(lineSplitted[6]),float(lineSplitted[7]),float(lineSplitted[8]),float(lineSplitted[9]),
                     float(lineSplitted[10]),float(lineSplitted[11]),float(lineSplitted[12]),nodes])

    outputData = pd.DataFrame(data, columns=['orbitID', 'hlt','T','x','y','z','xdot','ydot','zdot','alt','alpha','beta','m','nodes'])
    input_file.close()
    return outputData

def load_states_continuation_length(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True,header=None)
    return data

def load_differential_correction(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(28)))
    data.columns = ['iterations', 'hlt','period','posDev','velDev','velIntDev','velExtDev','timeDev','x','y','z','xdot','ydot','zdot','alt','alpha','beta','m' \
        , 'xPhaseHalf', 'yPhaseHalf', 'zPhaseHalf', 'xdotPhaseHalf', 'ydotPhaseHalf', 'zdotPhaseHalf', 'altPhaseHalf', 'alphaPhaseHalf', 'betaPhaseHalf', 'mPhaseHalf']
    return data

def load_patch_points(file_path, numberOfPatchPoints):


    arrayVectors = np.loadtxt(file_path)
    arrayVectors = arrayVectors.reshape(numberOfPatchPoints,11)
    data = pd.DataFrame(arrayVectors)
    data.columns = ['x', 'y','z','xdot','ydot','zdot','alt','alpha','beta','m','time']
    return data

def load_propagated_states(file_path, numberOfPatchPoints):
    arrayVectors = np.loadtxt(file_path)
    arrayVectors = arrayVectors.reshape(numberOfPatchPoints, 10)
    data = pd.DataFrame(arrayVectors)
    data.columns = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot', 'alt', 'alpha', 'beta', 'm']
    return data

def load_tlt_stage_properties(file_path):
    arrayVectors = np.loadtxt(file_path)
    arrayVectors = arrayVectors.reshape(1, 7)
    data = pd.DataFrame(arrayVectors)
    data.columns = ['devR', 'devV', 'devVint', 'devVext', 'devT', 'time', 'iterations']
    return data




def load_initial_conditions(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None).filter(list(range(11)))
    data.columns = ['iterations', 'hlt','period','x','y','z','xdot','ydot','zdot','alt','alpha']
    return data

def load_initial_conditions_augmented_incl_M(file_path):
    data = pd.read_table(file_path, delim_whitespace=True, header=None)
    return data

def concanate_alpha_varying_files():
    fileNames =  ['../../data/raw/orbits/augmented/varying_alpha/[111-119]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_initial_conditions.txt', \
                  '../../data/raw/orbits/augmented/varying_alpha/[120-138]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_initial_conditions.txt', \
                  '../../data/raw/orbits/augmented/varying_alpha/[222-240]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_initial_conditions.txt',
                  '../../data/raw/orbits/augmented/varying_alpha/[240-249]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_initial_conditions.txt']


    outFileName = '../../data/raw/orbits/augmented/varying_alpha/L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_initial_conditions.txt'

    with open(outFileName, 'w') as outfile:
        for fname in fileNames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def reverse_alpha_files():
    #initial_conditions
    #differential_correction
    #states_continuation
    fileName = '../../data/raw/orbits/augmented/varying_alpha/[119-111]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_states_continuation.txt'
    fileNameOut = '../../data/raw/orbits/augmented/varying_alpha/[111-119]_L2_horizontal_0.10000000000_0.00000000000_-1.50000000000_states_continuation.txt'

    with open(fileName) as f, open(fileNameOut, 'w') as fout:
        fout.writelines(reversed(f.readlines()))

if __name__ == '__main__':
    concanate_alpha_varying_files()

