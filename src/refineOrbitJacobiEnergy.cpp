#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrection.h"
#include "thesisProject/src/propagateOrbit.h"

Eigen::VectorXd readInitialConditionsFromFile(int librationPointNumber, std::string orbitType, int orbitIdOne, int orbitIdTwo)
{
    std::ifstream textFileInitialConditions("../data/raw/orbits/L" + std::to_string(librationPointNumber) + "_" + orbitType + "_initial_conditions.txt");
    std::vector<std::vector<double>> initialConditions;

    if (textFileInitialConditions) {
        std::string line;

        while (std::getline(textFileInitialConditions, line)) {
            initialConditions.push_back(std::vector<double>());

            // Break down the row into column values
            std::stringstream split(line);
            double value;
            while (split >> value) {
                initialConditions.back().push_back(value);
            }
        }
    }

    double orbitalPeriod1;
    double orbitalPeriod2;
    Eigen::VectorXd initialStateVector1;
    Eigen::VectorXd initialStateVector2;

    orbitalPeriod1         = initialConditions[orbitIdOne][1];
    initialStateVector1    = Eigen::VectorXd::Zero(6);
    initialStateVector1(0) = initialConditions[orbitIdOne][2];
    initialStateVector1(1) = initialConditions[orbitIdOne][3];
    initialStateVector1(2) = initialConditions[orbitIdOne][4];
    initialStateVector1(3) = initialConditions[orbitIdOne][5];
    initialStateVector1(4) = initialConditions[orbitIdOne][6];
    initialStateVector1(5) = initialConditions[orbitIdOne][7];

    orbitalPeriod2         = initialConditions[orbitIdTwo][1];
    initialStateVector2    = Eigen::VectorXd::Zero(6);
    initialStateVector2(0) = initialConditions[orbitIdTwo][2];
    initialStateVector2(1) = initialConditions[orbitIdTwo][3];
    initialStateVector2(2) = initialConditions[orbitIdTwo][4];
    initialStateVector2(3) = initialConditions[orbitIdTwo][5];
    initialStateVector2(4) = initialConditions[orbitIdTwo][6];
    initialStateVector2(5) = initialConditions[orbitIdTwo][7];

    double jacobiEnergy1 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector1);
    double jacobiEnergy2 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector2);

    Eigen::VectorXd selectedInitialConditions(14);
    selectedInitialConditions(0) = orbitalPeriod1;
    selectedInitialConditions.segment(1, 6) = initialStateVector1;
    selectedInitialConditions[7] = orbitalPeriod2;
    selectedInitialConditions.segment(8, 6) = initialStateVector2;

    return selectedInitialConditions;
}


bool checkJacobiOnManifoldOutsideBounds( Eigen::VectorXd currentStateVector, double referenceJacobiEnergy,
                                         double maxJacobiEnergyDeviation = 1.0e-11 )
{
    bool jacobiDeviationOutsideBounds;
    double currentJacobiEnergy = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, currentStateVector);

    if (std::abs(currentJacobiEnergy - referenceJacobiEnergy) < maxJacobiEnergyDeviation)
    {
        jacobiDeviationOutsideBounds = false;
    } else
    {
        jacobiDeviationOutsideBounds = true;
        std::cout << "Jacobi energy deviation on manifold exceeded bounds" << std::endl;
    }

    return jacobiDeviationOutsideBounds;
}


Eigen::MatrixXd computeManifoldStatesAtTheta( Eigen::VectorXd initialStateVector, double orbitalPeriod, int librationPointNumber,
                                              int displacementFromOrbitSign, int integrationTimeDirection,
                                              double thetaStoppingAngle = -90.0, int numberOfManifoldOrbits = 100,
                                              bool writeManifoldToFile = false, int saveEveryNthIntegrationStep = 1000,
                                              double displacementFromOrbit = 1.0e-6,
                                              double maximumIntegrationTimeManifoldOrbits = 50.0,
                                              double maxEigenvalueDeviation = 1.0e-3,
                                              std::string orbitType = "vertical")
{
    double jacobiEnergyOnOrbit = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector);

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    std::cout << "\nInitial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl
              << "\nwith C: " << jacobiEnergyOnOrbit << ", T: " << orbitalPeriod << std::endl;;

    std::vector< Eigen::VectorXd > orbitStateVectors;
    int numberOfPointsOnPeriodicOrbit = 1;  // Initial state

    orbitStateVectors.push_back(initialStateVectorInclSTM);

    // Perform first integration step
    Eigen::VectorXd outputVector               = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0 );
    Eigen::VectorXd stateVectorInclSTM         = outputVector.segment(0,42);
    Eigen::VectorXd previousOutputVector       = outputVector;
    double currentTime                         = outputVector(42);

    // Perform integration steps until end of orbital period
    for (int i = 5; i <= 12; i++) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while (currentTime <= orbitalPeriod) {
            stateVectorInclSTM      = outputVector.segment(0, 42);
            currentTime             = outputVector(42);
            previousOutputVector    = outputVector;

            // Save the STM at every point along the orbit, for the same stepsize
            if (i == 5){
                numberOfPointsOnPeriodicOrbit += 1;
                orbitStateVectors.push_back(stateVectorInclSTM);
            }

            // Propagate to next time step
            outputVector         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, initialStepSize, maximumStepSize);

            if (outputVector(42) > orbitalPeriod) {
                outputVector = previousOutputVector;
                break;
            }
        }
    }

    // Reshape the STM for one period to matrix form.
    Eigen::Map<Eigen::MatrixXd> monodromyMatrix = Eigen::Map<Eigen::MatrixXd>(stateVectorInclSTM.segment(6,36).data(),6,6);
//    std::cout << "\nMonodromy matrix:\n" << monodromyMatrix << "\n" << std::endl;

    // Compute eigenvectors of the monodromy matrix (find minimum eigenvalue, corresponding to stable, and large for unstable)
    Eigen::EigenSolver<Eigen::MatrixXd> eig(monodromyMatrix);
//    std::cout << "Eigenvectors:\n" << eig.eigenvectors() << "\n\n" << "Eigenvalues:\n" << eig.eigenvalues() << "\n" << std::endl;

    int indexMaximumEigenvalue;
    double maximumEigenvalue = 0.0;
    int indexMinimumEigenvalue;
    double minimumEigenvalue = 1000.0;

    for (int i = 0; i <= 5; i++){
        if (eig.eigenvalues().real()(i) > maximumEigenvalue and std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation){
            maximumEigenvalue = eig.eigenvalues().real()(i);
            indexMaximumEigenvalue = i;
        }
        if (std::abs(eig.eigenvalues().real()(i)) < minimumEigenvalue and std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation){
            minimumEigenvalue = std::abs(eig.eigenvalues().real()(i));
            indexMinimumEigenvalue = i;
        }
    }

    Eigen::VectorXd eigenVector1 = eig.eigenvectors().real().col(indexMaximumEigenvalue);
    Eigen::VectorXd eigenVector2 = eig.eigenvectors().real().col(indexMinimumEigenvalue);
    std::cout << "Maximum real eigenvalue of " << maximumEigenvalue << " at " << indexMaximumEigenvalue
              << ", corresponding to eigenvector (unstable manifold): \n" << eigenVector1 << "\n\n"
              << "Minimum absolute real eigenvalue: " << minimumEigenvalue << " at " << indexMinimumEigenvalue
              << ", corresponding to eigenvector (stable manifold): \n" << eigenVector2 << std::endl;

    Eigen::VectorXd manifoldStartingState      = Eigen::VectorXd::Zero(42);
    Eigen::VectorXd localStateVector           = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd localNormalizedEigenvector = Eigen::VectorXd::Zero(6);

    double offsetSign;
    Eigen::VectorXd eigenVector;

    if (integrationTimeDirection == 1.0)
    {
        eigenVector = eigenVector1;
        if (eigenVector1(0) > 0.0){
            offsetSign = displacementFromOrbitSign * 1.0;
        } else {
            offsetSign = displacementFromOrbitSign * -1.0;
        }
    } else
    {
        eigenVector = eigenVector2;
        if (eigenVector2(0) > 0.0){
            offsetSign = displacementFromOrbitSign * 1.0;
        } else {
            offsetSign = displacementFromOrbitSign * -1.0;;
        }
    }

    bool fullManifoldComputed = false;
    double currentAngleOnManifold = 0.0;

    // Initialize matrix of all states at Poincaré section and vector for saving manifold to file
    Eigen::MatrixXd orbitStateVectorsAtPoincareMatrix = Eigen::MatrixXd::Zero(numberOfManifoldOrbits, 8);
    Eigen::VectorXd tempStateVectorsOnManifold        = Eigen::VectorXd::Zero(7);
    std::vector< Eigen::VectorXd > stateVectorsOnManifold;

    std::cout << "\n\nComputing the manifold: \n" << std::endl;

    // Determine the total number of points along the periodic orbit to start the manifolds.
    for (int ii = 0; ii <numberOfManifoldOrbits; ii++) {
        int row_index = floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits);

        // Reshape the STM from vector to a matrix
        Eigen::VectorXd STMvector       = orbitStateVectors.at(row_index).segment(6,36);
        Eigen::Map<Eigen::MatrixXd> STM = Eigen::Map<Eigen::MatrixXd>(STMvector.data(),6,6);

        // Apply displacement epsilon from the periodic orbit at <numberOfManifoldOrbits> locations on the final orbit.
        localStateVector                    = orbitStateVectors.at(floor(ii * numberOfPointsOnPeriodicOrbit / numberOfManifoldOrbits)).segment(0, 6).transpose();
        localNormalizedEigenvector          = (STM*eigenVector).normalized();
        manifoldStartingState.segment(0, 6) = localStateVector + offsetSign * displacementFromOrbit * localNormalizedEigenvector;
        manifoldStartingState.segment(6,36) = identityMatrix;

        // Write first state to file
        if ( writeManifoldToFile )
        {
            tempStateVectorsOnManifold(0) = 0.0;
            for (unsigned int idx = 0; idx < 6; idx++)
            {
                tempStateVectorsOnManifold(idx + 1) = manifoldStartingState(idx);
            }
            stateVectorsOnManifold.push_back(tempStateVectorsOnManifold);
        }

        outputVector       = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationTimeDirection );
        stateVectorInclSTM = outputVector.segment(0, 42);
        currentTime        = outputVector(42);

        std::cout << "Orbit No.: " << ii << std::endl;
        int integrationStepCount = 1;

        while ( (std::abs( currentTime ) <= maximumIntegrationTimeManifoldOrbits) and !fullManifoldComputed ) {

            stateVectorInclSTM = outputVector.segment(0, 42);
            currentTime = outputVector(42);

            // Check whether trajectory still belongs to the same energy level
            fullManifoldComputed = checkJacobiOnManifoldOutsideBounds(outputVector.segment(0, 6), jacobiEnergyOnOrbit);

            // Check whether end condition has been reached
            currentAngleOnManifold = atan2(outputVector(1), outputVector(0) - (1.0 - massParameter)) * 180/tudat::mathematical_constants::PI;

            if (currentAngleOnManifold * integrationTimeDirection > thetaStoppingAngle * integrationTimeDirection and
                    currentAngleOnManifold * thetaStoppingAngle > 0.0 )
            {

                outputVector = previousOutputVector;
                currentAngleOnManifold = atan2(outputVector(1), outputVector(0) - (1.0 - massParameter)) * 180/tudat::mathematical_constants::PI;

                std::cout << "||currentAngle - thetaStoppingAngle|| = "
                          << std::abs(currentAngleOnManifold - thetaStoppingAngle)
                          << ", at start of iterative procedure" << std::endl;

                for (int i = 5; i <= 12; i++) {
                    double initialStepSize = pow(10, (static_cast<float>(-i)));
                    double maximumStepSize = pow(10, (static_cast<float>(-i) + 1.0));

                    while (currentAngleOnManifold * integrationTimeDirection < thetaStoppingAngle * integrationTimeDirection )
                    {
                        stateVectorInclSTM      = outputVector.segment(0, 42);
                        currentTime             = outputVector(42);
                        previousOutputVector    = outputVector;
                        outputVector            = propagateOrbit(stateVectorInclSTM, massParameter, currentTime,
                                                                 integrationTimeDirection, initialStepSize, maximumStepSize);

                        currentAngleOnManifold = atan2(outputVector(1), outputVector(0) - (1.0 - massParameter)) * 180/tudat::mathematical_constants::PI;

                        if (currentAngleOnManifold * integrationTimeDirection > thetaStoppingAngle * integrationTimeDirection)
                        {
                            outputVector           = previousOutputVector;
                            currentTime            = outputVector(42);
                            currentAngleOnManifold = atan2(outputVector(1), outputVector(0) - (1.0 - massParameter)) * 180/tudat::mathematical_constants::PI;
                            break;
                        }
                    }
                }
                std::cout << "||currentAngle - thetaStoppingAngle|| = "
                          << std::abs(currentAngleOnManifold - thetaStoppingAngle)
                          << ", at end of iterative procedure." << std::endl;
                fullManifoldComputed = true;
            } else
            {
                // Propagate to next time step.
                previousOutputVector = outputVector;
                outputVector = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationTimeDirection);
                integrationStepCount += 1;
            }

            // Write every nth integration step to file.
            if ( writeManifoldToFile and
                    ( integrationStepCount % saveEveryNthIntegrationStep == 0 or fullManifoldComputed ) )
            {
                tempStateVectorsOnManifold(0) = currentTime;
                for (unsigned int idx = 0; idx < 6; idx++)
                {
                    tempStateVectorsOnManifold(idx + 1) = outputVector(idx);
                }
                stateVectorsOnManifold.push_back(tempStateVectorsOnManifold);
            }
        }

        // Save phase, time and state at Poincaré section
        orbitStateVectorsAtPoincareMatrix(ii, 0) = (static_cast<double>(ii)) / numberOfManifoldOrbits;
        orbitStateVectorsAtPoincareMatrix(ii, 1) = outputVector(42);
        for (int iCol = 2; iCol < 8; iCol++)
        {
            orbitStateVectorsAtPoincareMatrix(ii, iCol) = outputVector(iCol - 2);
        }

        currentAngleOnManifold = 0.0;
        fullManifoldComputed = false;
    }

    if ( writeManifoldToFile ) {

        // Rounding-off values for file name
        std::string fileNameString;
        std::ostringstream thetaStoppingAngleStr;
        thetaStoppingAngleStr << std::setprecision(4) << thetaStoppingAngle;
        std::ostringstream jacobiEnergyOnOrbitStr;
        jacobiEnergyOnOrbitStr << std::setprecision(4) << jacobiEnergyOnOrbit;

        if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == 1.0)
        {
            fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                              "_W_S_plus_" + jacobiEnergyOnOrbitStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
        } else if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == -1.0)
        {
            fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                              "_W_S_min_" + jacobiEnergyOnOrbitStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
        } else if (integrationTimeDirection == 1.0 and displacementFromOrbitSign == 1.0)
        {
            fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                              "_W_U_plus_" + jacobiEnergyOnOrbitStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
        } else {
            fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                              "_W_U_min_" + jacobiEnergyOnOrbitStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
        }

        remove(fileNameString.c_str());
        std::ofstream textFileStateVectorsOnManifold(fileNameString.c_str());
        textFileStateVectorsOnManifold.precision(std::numeric_limits<double>::digits10);

        for (unsigned int idx = 0; idx < stateVectorsOnManifold.size(); idx++)
        {
            textFileStateVectorsOnManifold << std::left << std::scientific << stateVectorsOnManifold.at(idx).transpose() << std::endl;
        }

        textFileStateVectorsOnManifold.close();
        textFileStateVectorsOnManifold.clear();
    }

    return orbitStateVectorsAtPoincareMatrix;
}


Eigen::VectorXd refineOrbitJacobiEnergy( int librationPointNumber, std::string orbitType, double desiredJacobiEnergy,
                                         Eigen::VectorXd initialStateVector1, double orbitalPeriod1,
                                         Eigen::VectorXd initialStateVector2, double orbitalPeriod2,
                                         double maxPositionDeviationFromPeriodicOrbit = 1.0e-12,
                                         double maxVelocityDeviationFromPeriodicOrbit = 1.0e-12,
                                         double maxJacobiEnergyDeviation = 1.0e-12 )
{
    double jacobiEnergy1 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector1);
    double jacobiEnergy2 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector2);
    double jacobiEnergy3 = 0;

    std::cout << "Refining Jacobi energy from " << jacobiEnergy1 << " and " << jacobiEnergy2
              << " to " << desiredJacobiEnergy << std::endl;

    if (jacobiEnergy2 < jacobiEnergy1) {
        Eigen::VectorXd tempInitialStateVector = initialStateVector1;
        double tempOrbitalPeriod = orbitalPeriod1;
        double tempJacobiEnergy = jacobiEnergy1;

        initialStateVector1 = initialStateVector2;
        orbitalPeriod1 = orbitalPeriod2;
        jacobiEnergy1 = jacobiEnergy2;

        initialStateVector2 = tempInitialStateVector;
        orbitalPeriod2 = tempOrbitalPeriod;
        jacobiEnergy2 = tempJacobiEnergy;
    }

    double jacobiScalingFactor;
    double orbitalPeriod3;
    Eigen::VectorXd initialStateVector3;
    Eigen::VectorXd refineOrbitJacobiEnergyResult;

    while (std::abs(jacobiEnergy3 - desiredJacobiEnergy) > maxJacobiEnergyDeviation) {

        jacobiScalingFactor = (desiredJacobiEnergy - jacobiEnergy1) / (jacobiEnergy2 - jacobiEnergy1);
        orbitalPeriod3 = orbitalPeriod2 * jacobiScalingFactor + orbitalPeriod1 * (1.0 - jacobiScalingFactor);
        initialStateVector3 = initialStateVector2 * jacobiScalingFactor + initialStateVector1 * (1.0 - jacobiScalingFactor);

        // Correct state vector guesses
        refineOrbitJacobiEnergyResult = applyDifferentialCorrection(librationPointNumber, orbitType,
                                                                    initialStateVector3, orbitalPeriod3,
                                                                    massParameter, maxPositionDeviationFromPeriodicOrbit,
                                                                    maxVelocityDeviationFromPeriodicOrbit);
        initialStateVector3 = refineOrbitJacobiEnergyResult.segment(0, 6);
        orbitalPeriod3 = refineOrbitJacobiEnergyResult(6);
        jacobiEnergy3 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVector3);

        if (jacobiEnergy3 < desiredJacobiEnergy) {
            jacobiEnergy1 = jacobiEnergy3;
            orbitalPeriod1 = orbitalPeriod3;
            initialStateVector1 = initialStateVector3;
        } else {
            jacobiEnergy2 = jacobiEnergy3;
            orbitalPeriod2 = orbitalPeriod3;
            initialStateVector2 = initialStateVector3;
        }

        std::cout << "Jacobi energy deviation: " << jacobiEnergy3 - desiredJacobiEnergy << std::endl;
    }

    return refineOrbitJacobiEnergyResult;
}

void writePoincareSectionToFile(Eigen::MatrixXd stateVectorsAtPoincareMatrix, int librationPointNumber,
                                std::string orbitType, double desiredJacobiEnergy, int displacementFromOrbitSign,
                                int integrationTimeDirection, double thetaStoppingAngle)
{
    // Rounding-off values for file name
    std::string fileNameString;
    std::ostringstream thetaStoppingAngleStr;
    thetaStoppingAngleStr << std::setprecision(4) << thetaStoppingAngle;
    std::ostringstream desiredJacobiEnergyStr;
    desiredJacobiEnergyStr << std::setprecision(4) << desiredJacobiEnergy;

    if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                          "_W_S_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == -1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                          "_W_S_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else if (integrationTimeDirection == 1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                          "_W_U_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNumber) + "_" + orbitType +
                          "_W_U_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    }

    remove(fileNameString.c_str());
    std::ofstream textFileStateVectorsAtPoincare(fileNameString.c_str());
    textFileStateVectorsAtPoincare.precision(std::numeric_limits<double>::digits10);
    textFileStateVectorsAtPoincare << std::left << std::scientific << stateVectorsAtPoincareMatrix << std::endl;
    textFileStateVectorsAtPoincare.close();
    textFileStateVectorsAtPoincare.clear();
    return;
}

Eigen::MatrixXd findMinimumImpulseManifoldConnection( Eigen::MatrixXd stableStateVectorsAtPoincareMatrix,
                                                      Eigen::MatrixXd unstableStateVectorsAtPoincareMatrix,
                                                      int numberOfManifoldOrbits,
                                                      double maxPositionDiscrepancy = 1.0e-3 )
{
    double deltaPosition;
    double deltaVelocity;
    double minimumDeltaVelocity                          = 1000.0;
    Eigen::VectorXd stableStateVectorAtPoincare          = Eigen::VectorXd::Zero(8);
    Eigen::VectorXd unstableStateVectorAtPoincare        = Eigen::VectorXd::Zero(8);
    Eigen::MatrixXd minimumImpulseStateVectorsAtPoincare = Eigen::MatrixXd::Zero(2, 8);

    for (int i = 0; i < numberOfManifoldOrbits; i++)
    {
        stableStateVectorAtPoincare = stableStateVectorsAtPoincareMatrix.block(i, 0, 1, 8).transpose();
        for (int j = 0; j < numberOfManifoldOrbits; j++)
        {
            unstableStateVectorAtPoincare = unstableStateVectorsAtPoincareMatrix.block(j, 0, 1, 8).transpose();
            deltaPosition = std::sqrt( (stableStateVectorAtPoincare(2) - unstableStateVectorAtPoincare(2)) * (stableStateVectorAtPoincare(2) - unstableStateVectorAtPoincare(2)) +
                                       (stableStateVectorAtPoincare(3) - unstableStateVectorAtPoincare(3)) * (stableStateVectorAtPoincare(3) - unstableStateVectorAtPoincare(3)) +
                                       (stableStateVectorAtPoincare(4) - unstableStateVectorAtPoincare(4)) * (stableStateVectorAtPoincare(4) - unstableStateVectorAtPoincare(4)) );
            if (deltaPosition < maxPositionDiscrepancy)
            {
                deltaVelocity = std::sqrt( (stableStateVectorAtPoincare(5) - unstableStateVectorAtPoincare(5)) * (stableStateVectorAtPoincare(5) - unstableStateVectorAtPoincare(5)) +
                                           (stableStateVectorAtPoincare(6) - unstableStateVectorAtPoincare(6)) * (stableStateVectorAtPoincare(6) - unstableStateVectorAtPoincare(6)) +
                                           (stableStateVectorAtPoincare(7) - unstableStateVectorAtPoincare(7)) * (stableStateVectorAtPoincare(7) - unstableStateVectorAtPoincare(7)) );
                if (deltaVelocity < minimumDeltaVelocity)
                {
                    minimumDeltaVelocity = deltaVelocity;
                    for (int iCol = 0; iCol < 8; iCol++)
                    {
                        minimumImpulseStateVectorsAtPoincare(0, iCol) = stableStateVectorAtPoincare(iCol);
                        minimumImpulseStateVectorsAtPoincare(1, iCol) = unstableStateVectorAtPoincare(iCol);
                    }
                    std::cout << "New optimum found with DeltaV = " << minimumDeltaVelocity << " at:\n"
                              << minimumImpulseStateVectorsAtPoincare << std::endl;
                }
            }
        }
    }
    return minimumImpulseStateVectorsAtPoincare;
}


Eigen::MatrixXd connectManifolds(std::string orbitType = "vertical", double thetaStoppingAngle = -90.0,
                                 int numberOfManifoldOrbits = 100, double desiredJacobiEnergy = 3.1,
                                 double maxPositionDiscrepancy = 1.0e-3 )
{
    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    // Load orbits in L1 and refine to specific Jacobi energy
    Eigen::VectorXd selectedInitialConditions = readInitialConditionsFromFile(1, orbitType, 1159, 1160);
    Eigen::VectorXd refinedJacobiEnergyResult = refineOrbitJacobiEnergy(1, orbitType, desiredJacobiEnergy,
                                                                        selectedInitialConditions.segment(1, 6),
                                                                        selectedInitialConditions(0),
                                                                        selectedInitialConditions.segment(8, 6),
                                                                        selectedInitialConditions(7));

    Eigen::VectorXd initialStateVectorL1 = refinedJacobiEnergyResult.segment(0, 6);
    double orbitalPeriodL1 = refinedJacobiEnergyResult(6);
    double jacobiEnergyL1 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVectorL1);


    // Calculate state at Poincaré section for exterior unstable manifold departing from L1
    Eigen::MatrixXd unstableStateVectorsAtPoincareMatrix = computeManifoldStatesAtTheta( initialStateVectorL1,
                                                                                         orbitalPeriodL1, 1, 1.0, 1.0,
                                                                                         thetaStoppingAngle,
                                                                                         numberOfManifoldOrbits, true );

    writePoincareSectionToFile(unstableStateVectorsAtPoincareMatrix, 1, orbitType, desiredJacobiEnergy, 1.0, 1.0, thetaStoppingAngle);

    // Load orbits in L2 and refine to specific Jacobi energy
    selectedInitialConditions = readInitialConditionsFromFile(2, orbitType, 1274, 1275);
    refinedJacobiEnergyResult = refineOrbitJacobiEnergy(2, orbitType, desiredJacobiEnergy,
                                                        selectedInitialConditions.segment(1, 6),
                                                        selectedInitialConditions(0),
                                                        selectedInitialConditions.segment(8, 6),
                                                        selectedInitialConditions(7));

    Eigen::VectorXd initialStateVectorL2 = refinedJacobiEnergyResult.segment(0, 6);
    double orbitalPeriodL2 = refinedJacobiEnergyResult(6);
    double jacobiEnergyL2 = tudat::gravitation::circular_restricted_three_body_problem::computeJacobiEnergy(massParameter, initialStateVectorL2);

    // Calculate state at Poincaré section for interior stable manifold departing from L2
    Eigen::MatrixXd stableStateVectorsAtPoincareMatrix = computeManifoldStatesAtTheta( initialStateVectorL2,
                                                                                       orbitalPeriodL2, 2, -1.0, -1.0,
                                                                                       thetaStoppingAngle,
                                                                                       numberOfManifoldOrbits, true );

    writePoincareSectionToFile(stableStateVectorsAtPoincareMatrix, 2, orbitType, desiredJacobiEnergy, -1.0, -1.0, thetaStoppingAngle);

    Eigen::MatrixXd minimumImpulseStateVectorsAtPoincare = findMinimumImpulseManifoldConnection( stableStateVectorsAtPoincareMatrix,
                                                                                                 unstableStateVectorsAtPoincareMatrix,
                                                                                                 numberOfManifoldOrbits,
                                                                                                 maxPositionDiscrepancy);
/*
 * write initial orbits to file
 * per angle: write trajectory to file
 * once: write per angle the minimum impulse and position discrepancy
*/
    return minimumImpulseStateVectorsAtPoincare;
}