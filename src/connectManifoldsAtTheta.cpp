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
#include <map>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "applyDifferentialCorrection.h"
#include "computeDifferentialCorrection.h"
#include "computeManifolds.h"
#include "connectManifoldsAtTheta.h"
#include "propagateOrbit.h"

Eigen::VectorXd readInitialConditionsFromFile(const int librationPointNr, const std::string orbitType,
                                              int orbitIdOne, int orbitIdTwo, const double massParameter)
{
    std::ifstream textFileInitialConditions("../data/raw/orbits/L" + std::to_string(librationPointNr) + "_" + orbitType + "_initial_conditions.txt");
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

    double jacobiEnergy1 = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector1);
    double jacobiEnergy2 = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector2);

    Eigen::VectorXd selectedInitialConditions(14);
    selectedInitialConditions(0) = orbitalPeriod1;
    selectedInitialConditions.segment(1, 6) = initialStateVector1;
    selectedInitialConditions[7] = orbitalPeriod2;
    selectedInitialConditions.segment(8, 6) = initialStateVector2;

    return selectedInitialConditions;
}


bool checkJacobiOnManifoldOutsideBounds( Eigen::VectorXd currentStateVector, const double referenceJacobiEnergy,
                                         const double massParameter, const double maxJacobiEnergyDeviation )
{
    bool jacobiDeviationOutsideBounds;
    double currentJacobiEnergy = tudat::gravitation::computeJacobiEnergy(massParameter, currentStateVector);

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


void computeManifoldStatesAtTheta( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                   Eigen::VectorXd initialStateVector, double orbitalPeriod, int librationPointNr,
                                   const double massParameter, double displacementFromOrbitSign, double integrationTimeDirection,
                                   double thetaStoppingAngle, const int numberOfTrajectoriesPerManifold,
                                   const int saveFrequency, const double eigenvectorDisplacementFromOrbit,
                                   const double maximumIntegrationTimeManifoldTrajectories,
                                   const double maxEigenvalueDeviation, const std::string orbitType )
{
    double jacobiEnergyOnOrbit = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector);
    std::cout << "\nInitial state vector:" << std::endl << initialStateVector       << std::endl
              << "\nwith C: " << jacobiEnergyOnOrbit    << ", T: " << orbitalPeriod << std::endl;;

    // Propagate the initialStateVector for a full period and write output to file.
    std::map< double, Eigen::MatrixXd > stateTransitionMatrixHistory;
    Eigen::MatrixXd stateVectorInclSTM = propagateOrbitWithStateTransitionMatrixToFinalCondition(getFullInitialState( initialStateVector ), massParameter, orbitalPeriod, 1, stateTransitionMatrixHistory, 1, 0.0 ).first;

    const unsigned int numberOfPointsOnPeriodicOrbit = stateTransitionMatrixHistory.size();
    std::cout << "numberOfPointsOnPeriodicOrbit: " << numberOfPointsOnPeriodicOrbit << std::endl;

    // Determine the eigenvector directions of the (un)stable subspace of the monodromy matrix
    Eigen::MatrixXd monodromyMatrix = stateVectorInclSTM.block(0,1,6,6);

    Eigen::Vector6d stableEigenvector;
    Eigen::Vector6d unstableEigenvector;

    try {
        determineStableUnstableEigenvectors( monodromyMatrix, stableEigenvector, unstableEigenvector, maxEigenvalueDeviation );
    }
    catch( const std::exception& ) {
        return;
    }

    // The sign of the x-component of the eigenvector is determined, which is used to determine the eigenvector offset direction (interior/exterior manifold)
    double stableEigenvectorSign   = determineEigenvectorSign( stableEigenvector );
    double unstableEigenvectorSign = determineEigenvectorSign( unstableEigenvector );

    double currentTime;
    double currentAngleOnManifold;
    double offsetSign;
    Eigen::VectorXd monodromyMatrixEigenvector;
    Eigen::Vector6d localStateVector           = Eigen::Vector6d::Zero(6);
    Eigen::Vector6d localNormalizedEigenvector = Eigen::Vector6d::Zero(6);
    Eigen::MatrixXd manifoldStartingState      = Eigen::MatrixXd::Zero(6, 7);
    std::pair< Eigen::MatrixXd, double > stateVectorInclSTMAndTime;
    std::pair< Eigen::MatrixXd, double > previousStateVectorInclSTMAndTime;

    if (integrationTimeDirection == 1.0)
    {
        monodromyMatrixEigenvector = unstableEigenvector;
        if (unstableEigenvectorSign > 0.0){
            offsetSign = displacementFromOrbitSign * 1.0;
        } else {
            offsetSign = displacementFromOrbitSign * -1.0;
        }
    } else
    {
        monodromyMatrixEigenvector = stableEigenvector;
        if (stableEigenvectorSign > 0.0){
            offsetSign = displacementFromOrbitSign * 1.0;
        } else {
            offsetSign = displacementFromOrbitSign * -1.0;;
        }
    }

    bool jacobiOutsideBounds   = false;
    bool fullManifoldComputed  = false;

    // Determine the total number of points along the periodic orbit to start the manifolds.
    for ( int trajectoryOnManifoldNumber = 0; trajectoryOnManifoldNumber < numberOfTrajectoriesPerManifold; trajectoryOnManifoldNumber++ ) {

        int indexCount = 0;
        int stepCounter = 1;
        auto indexOnOrbit = static_cast <int> (std::floor(
                trajectoryOnManifoldNumber * numberOfPointsOnPeriodicOrbit / numberOfTrajectoriesPerManifold));

        Eigen::MatrixXd stateTransitionMatrix;
        for (auto const &it : stateTransitionMatrixHistory) {
            if (indexCount == indexOnOrbit) {
                stateTransitionMatrix = it.second.block(0, 1, 6, 6);
                localStateVector = it.second.block(0, 0, 6, 1);
                break;
            }
            indexCount += 1;
        }

        // Apply displacement epsilon from the periodic orbit at <numberOfTrajectoriesPerManifold> locations on the final orbit.
        localNormalizedEigenvector = (stateTransitionMatrix * monodromyMatrixEigenvector).normalized();
        manifoldStartingState = getFullInitialState(
                localStateVector + offsetSign * eigenvectorDisplacementFromOrbit * localNormalizedEigenvector);

        if (saveFrequency >= 0) {
            manifoldStateHistory[trajectoryOnManifoldNumber][0.0] = manifoldStartingState.block(0, 0, 6, 1);
        }

        stateVectorInclSTMAndTime = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationTimeDirection);
        previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;  // set first value of this parameter
        stateVectorInclSTM = stateVectorInclSTMAndTime.first;
        currentTime = stateVectorInclSTMAndTime.second;

        std::cout << "Trajectory on manifold number: " << trajectoryOnManifoldNumber << std::endl;
        while ((std::abs(currentTime) <= maximumIntegrationTimeManifoldTrajectories) and !fullManifoldComputed) {

            // Check whether trajectory still belongs to the same energy level
            jacobiOutsideBounds  = checkJacobiOnManifoldOutsideBounds(stateVectorInclSTM, jacobiEnergyOnOrbit,
                                                                      massParameter);
            fullManifoldComputed = jacobiOutsideBounds;

            // Check whether end condition has been reached
            currentAngleOnManifold = atan2(stateVectorInclSTM(1, 0), stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * 180.0 / tudat::mathematical_constants::PI;

            if (currentAngleOnManifold * integrationTimeDirection > thetaStoppingAngle * integrationTimeDirection and
                currentAngleOnManifold * thetaStoppingAngle > 0.0) {

                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
                currentTime               = stateVectorInclSTMAndTime.second;

                currentAngleOnManifold = atan2(stateVectorInclSTM(1, 0), stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * 180.0 / tudat::mathematical_constants::PI;

                std::cout << "||currentAngle - thetaStoppingAngle|| = "
                          << std::abs(currentAngleOnManifold - thetaStoppingAngle)
                          << ", at start of iterative procedure" << std::endl;

                for (int i = 6; i <= 12; i++) {
                    double initialStepSize = pow(10, (static_cast<float>(-i)));
                    double maximumStepSize = pow(10, (static_cast<float>(-i) + 1.0));

                    while (currentAngleOnManifold * integrationTimeDirection <
                           thetaStoppingAngle * integrationTimeDirection) {

                        previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
                        stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
                        currentTime                       = stateVectorInclSTMAndTime.second;
                        stateVectorInclSTMAndTime         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime,
                                                                           integrationTimeDirection, initialStepSize, maximumStepSize);

                        currentAngleOnManifold = atan2(stateVectorInclSTMAndTime.first(1, 0), stateVectorInclSTMAndTime.first(0, 0) - (1.0 - massParameter)) * 180.0 / tudat::mathematical_constants::PI;

                        if (currentAngleOnManifold * integrationTimeDirection >
                            thetaStoppingAngle * integrationTimeDirection) {

                            stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                            stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
                            currentTime               = stateVectorInclSTMAndTime.second;
                            currentAngleOnManifold    = atan2(stateVectorInclSTM(1, 0), stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * 180.0 / tudat::mathematical_constants::PI;

                            break;
                        }
                    }
                }
                std::cout << "||currentAngle - thetaStoppingAngle|| = "
                          << std::abs(currentAngleOnManifold - thetaStoppingAngle)
                          << ", at end of iterative procedure." << std::endl;
                fullManifoldComputed = true;
            } else {
                // Propagate to next time step.
                previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
                stateVectorInclSTMAndTime         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationTimeDirection);
                stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
                currentTime                       = stateVectorInclSTMAndTime.second;
                stepCounter++;
            }
            // Write every nth integration step to file.
            if (saveFrequency > 0 && ((stepCounter % saveFrequency == 0) || fullManifoldComputed) && !jacobiOutsideBounds) {
                manifoldStateHistory[trajectoryOnManifoldNumber][currentTime] = stateVectorInclSTM.block(0, 0, 6, 1);
            }
        }
        jacobiOutsideBounds    = false;
        fullManifoldComputed   = false;
        currentAngleOnManifold = 0.0;
    }
}


Eigen::VectorXd refineOrbitJacobiEnergy( const int librationPointNr, const std::string orbitType, const double desiredJacobiEnergy,
                                         Eigen::VectorXd initialStateVector1, double orbitalPeriod1,
                                         Eigen::VectorXd initialStateVector2, double orbitalPeriod2,
                                         const double massParameter,
                                         const double maxPositionDeviationFromPeriodicOrbit,
                                         const double maxVelocityDeviationFromPeriodicOrbit,
                                         const double maxJacobiEnergyDeviation )
{
    double jacobiEnergy1 = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector1);
    double jacobiEnergy2 = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector2);
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
        refineOrbitJacobiEnergyResult = applyDifferentialCorrection(librationPointNr, orbitType,
                                                                    initialStateVector3, orbitalPeriod3,
                                                                    massParameter, maxPositionDeviationFromPeriodicOrbit,
                                                                    maxVelocityDeviationFromPeriodicOrbit);

        initialStateVector3 = refineOrbitJacobiEnergyResult.segment(0, 6);
        orbitalPeriod3      = refineOrbitJacobiEnergyResult(6);
        jacobiEnergy3       = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVector3);

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

void writePoincareSectionToFile( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                 int librationPointNr, std::string orbitType, double desiredJacobiEnergy,
                                 double displacementFromOrbitSign, double integrationTimeDirection, double thetaStoppingAngle,
                                 int numberOfTrajectoriesPerManifold )
{
    // Rounding-off values for file name
    std::string fileNameString;
    std::ostringstream thetaStoppingAngleStr;
    thetaStoppingAngleStr << std::setprecision(4) << thetaStoppingAngle;
    std::ostringstream desiredJacobiEnergyStr;
    desiredJacobiEnergyStr << std::setprecision(4) << desiredJacobiEnergy;

    if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_S_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == -1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_S_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else if (integrationTimeDirection == 1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_U_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    } else {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_U_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_poincare.txt");
    }

    remove(fileNameString.c_str());
    std::ofstream textFileStateVectorsAtPoincare(fileNameString.c_str());
    textFileStateVectorsAtPoincare.precision(14);

    double phase;

    // For all numberOfTrajectoriesPerManifold
    for( auto const &ent1 : manifoldStateHistory  ) {
        phase = ent1.first / (double) numberOfTrajectoriesPerManifold;

        // For last state on manifold trajectory
        if (integrationTimeDirection > 0 ) {
            for (auto ent2 = ent1.second.rbegin(); ent2 != ent1.second.rend(); ++ent2) {
                textFileStateVectorsAtPoincare << std::left << std::scientific << std::setw(25) << phase
                                               << std::setw(25) << ent2->first
                                               << std::setw(25) << ent2->second(0) << std::setw(25) << ent2->second(1)
                                               << std::setw(25) << ent2->second(2) << std::setw(25) << ent2->second(3)
                                               << std::setw(25) << ent2->second(4) << std::setw(25) << ent2->second(5)
                                               << std::endl;
                break;
            }
        }
        else {
            for (auto ent2 = ent1.second.begin(); ent2 != ent1.second.end(); ++ent2) {
                textFileStateVectorsAtPoincare << std::left << std::scientific << std::setw(25) << phase
                                               << std::setw(25) << ent2->first
                                               << std::setw(25) << ent2->second(0) << std::setw(25) << ent2->second(1)
                                               << std::setw(25) << ent2->second(2) << std::setw(25) << ent2->second(3)
                                               << std::setw(25) << ent2->second(4) << std::setw(25) << ent2->second(5)
                                               << std::endl;
                break;
            }
        }
    }

    textFileStateVectorsAtPoincare.close();
    textFileStateVectorsAtPoincare.clear();
}

Eigen::MatrixXd findMinimumImpulseManifoldConnection( std::map< int, std::map< double, Eigen::Vector6d > >& stableManifoldStateHistoryAtTheta,
                                                      std::map< int, std::map< double, Eigen::Vector6d > >& unstableManifoldStateHistoryAtTheta,
                                                      const int numberOfTrajectoriesPerManifold, const double maximumVelocityDiscrepancy )
{
    double deltaPosition;
    double deltaVelocity;
    double minimumDeltaPosition                          = 1000.0;
    Eigen::VectorXd stableStateVectorAtPoincare          = Eigen::VectorXd::Zero(8);
    Eigen::VectorXd unstableStateVectorAtPoincare        = Eigen::VectorXd::Zero(8);
    Eigen::MatrixXd minimumImpulseStateVectorsAtPoincare = Eigen::MatrixXd::Zero(2, 8);

    for (int stableTrajectoryNumber = 0; stableTrajectoryNumber < numberOfTrajectoriesPerManifold; stableTrajectoryNumber++)
    {
        stableStateVectorAtPoincare(0) = static_cast<double>(stableTrajectoryNumber) / static_cast<double>(numberOfTrajectoriesPerManifold);
        stableStateVectorAtPoincare(1) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->first;
        stableStateVectorAtPoincare(2) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(0);
        stableStateVectorAtPoincare(3) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(1);
        stableStateVectorAtPoincare(4) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(2);
        stableStateVectorAtPoincare(5) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(3);
        stableStateVectorAtPoincare(6) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(4);
        stableStateVectorAtPoincare(7) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second(5);

//        stableStateVectorAtPoincare.segment(2, 8) = stableManifoldStateHistoryAtTheta.at(stableTrajectoryNumber).begin()->second;

        for (int unstableTrajectoryNumber = 0; unstableTrajectoryNumber < numberOfTrajectoriesPerManifold; unstableTrajectoryNumber++) {
//            unstableStateVectorAtPoincare(0)            = static_cast<double>(unstableTrajectoryNumber) / static_cast<double>(numberOfTrajectoriesPerManifold);
//            unstableStateVectorAtPoincare(1)            = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->first;
//            unstableStateVectorAtPoincare.segment(2, 8) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second;
            unstableStateVectorAtPoincare(0) = static_cast<double>(unstableTrajectoryNumber) / static_cast<double>(numberOfTrajectoriesPerManifold);
            unstableStateVectorAtPoincare(1) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->first;
            unstableStateVectorAtPoincare(2) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(0);
            unstableStateVectorAtPoincare(3) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(1);
            unstableStateVectorAtPoincare(4) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(2);
            unstableStateVectorAtPoincare(5) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(3);
            unstableStateVectorAtPoincare(6) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(4);
            unstableStateVectorAtPoincare(7) = unstableManifoldStateHistoryAtTheta.at(unstableTrajectoryNumber).rbegin()->second(5);

            deltaVelocity = std::sqrt( (stableStateVectorAtPoincare(5) - unstableStateVectorAtPoincare(5)) * (stableStateVectorAtPoincare(5) - unstableStateVectorAtPoincare(5)) +
                                       (stableStateVectorAtPoincare(6) - unstableStateVectorAtPoincare(6)) * (stableStateVectorAtPoincare(6) - unstableStateVectorAtPoincare(6)) +
                                       (stableStateVectorAtPoincare(7) - unstableStateVectorAtPoincare(7)) * (stableStateVectorAtPoincare(7) - unstableStateVectorAtPoincare(7)) );

            if (deltaVelocity < maximumVelocityDiscrepancy)
            {
                deltaPosition = std::sqrt( (stableStateVectorAtPoincare(2) - unstableStateVectorAtPoincare(2)) * (stableStateVectorAtPoincare(2) - unstableStateVectorAtPoincare(2)) +
                                           (stableStateVectorAtPoincare(3) - unstableStateVectorAtPoincare(3)) * (stableStateVectorAtPoincare(3) - unstableStateVectorAtPoincare(3)) +
                                           (stableStateVectorAtPoincare(4) - unstableStateVectorAtPoincare(4)) * (stableStateVectorAtPoincare(4) - unstableStateVectorAtPoincare(4)) );
                if (deltaPosition < minimumDeltaPosition)
                {
                    minimumDeltaPosition = deltaPosition;
                    for (int iCol = 0; iCol < 8; iCol++)
                    {
                        minimumImpulseStateVectorsAtPoincare(0, iCol) = stableStateVectorAtPoincare(iCol);
                        minimumImpulseStateVectorsAtPoincare(1, iCol) = unstableStateVectorAtPoincare(iCol);
                    }
                    std::cout << "New optimum found with deltaR = " << minimumDeltaPosition << " (deltaV = " << deltaVelocity << ") at:\n"
                              << minimumImpulseStateVectorsAtPoincare << std::endl;
                }
            }
        }
    }
    return minimumImpulseStateVectorsAtPoincare;
}

void writeManifoldStateHistoryAtThetaToFile( std::map< int, std::map< double, Eigen::Vector6d > >& manifoldStateHistory,
                                             int librationPointNr, std::string orbitType, double desiredJacobiEnergy,
                                             double displacementFromOrbitSign, double integrationTimeDirection, double thetaStoppingAngle)
{
    // Rounding-off values for file name
    std::string fileNameString;
    std::ostringstream thetaStoppingAngleStr;
    thetaStoppingAngleStr << std::setprecision(4) << thetaStoppingAngle;
    std::ostringstream desiredJacobiEnergyStr;
    desiredJacobiEnergyStr << std::setprecision(4) << desiredJacobiEnergy;

    if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_S_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
    } else if (integrationTimeDirection == -1.0 and displacementFromOrbitSign == -1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_S_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
    } else if (integrationTimeDirection == 1.0 and displacementFromOrbitSign == 1.0)
    {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_U_plus_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
    } else {
        fileNameString = ("../data/raw/poincare_sections/L" + std::to_string(librationPointNr) + "_" + orbitType +
                          "_W_U_min_" + desiredJacobiEnergyStr.str() + "_" + (thetaStoppingAngleStr).str() + "_full.txt");
    }

    remove(fileNameString.c_str());
    std::ofstream textFileStateVectors(fileNameString.c_str());
    textFileStateVectors.precision(14);

    // For all numberOfTrajectoriesPerManifold
    for( auto const &ent1 : manifoldStateHistory  ) {
        // For all states on manifold trajectory
        for( auto const &ent2 : ent1.second ) {
            textFileStateVectors << std::left    << std::scientific << std::setw(25) << ent2.first
                                 << std::setw(25) << ent2.second(0) << std::setw(25) << ent2.second(1)
                                 << std::setw(25) << ent2.second(2) << std::setw(25) << ent2.second(3)
                                 << std::setw(25) << ent2.second(4) << std::setw(25) << ent2.second(5) << std::endl;
        }
    }

    textFileStateVectors.close();
    textFileStateVectors.clear();
}

Eigen::MatrixXd connectManifoldsAtTheta( const std::string orbitType, const double thetaStoppingAngle,
                                         const int numberOfTrajectoriesPerManifold, const double desiredJacobiEnergy,
                                         const int saveFrequency, const double massParameter )
{
    // Set output maximum precision
    std::cout.precision(std::numeric_limits<double>::digits10);

    int orbitOneL1;
    int orbitTwoL1;
    int orbitOneL2;
    int orbitTwoL2;

    if ( orbitType == "horizontal" ){
        orbitOneL1 = 577;
        orbitTwoL1 = 578;
        orbitOneL2 = 759;
        orbitTwoL2 = 760;
    } if ( orbitType == "vertical" ){
        orbitOneL1 = 1159;
        orbitTwoL1 = 1160;
        orbitOneL2 = 1274;
        orbitTwoL2 = 1275;
    } if ( orbitType == "halo" ){
        orbitOneL1 = 836;
        orbitTwoL1 = 837;
        orbitOneL2 = 650;
        orbitTwoL2 = 651;
    }

    // Load orbits in L1 and refine to specific Jacobi energy
    Eigen::VectorXd selectedInitialConditions = readInitialConditionsFromFile(1, orbitType, orbitOneL1, orbitTwoL1, massParameter);
    Eigen::VectorXd refinedJacobiEnergyResult = refineOrbitJacobiEnergy(1, orbitType, desiredJacobiEnergy,
                                                                        selectedInitialConditions.segment(1, 6),
                                                                        selectedInitialConditions(0),
                                                                        selectedInitialConditions.segment(8, 6),
                                                                        selectedInitialConditions(7), massParameter);

    Eigen::VectorXd initialStateVectorL1 = refinedJacobiEnergyResult.segment(0, 6);
    double orbitalPeriodL1               = refinedJacobiEnergyResult(6);
    double jacobiEnergyL1                = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVectorL1);

    // Calculate state at Poincaré section for exterior unstable manifold departing from L1
    std::map< int, std::map< double, Eigen::Vector6d > > unstableManifoldStateHistoryAtTheta;  // 1. per trajectory 2. per time-step
    computeManifoldStatesAtTheta( unstableManifoldStateHistoryAtTheta, initialStateVectorL1, orbitalPeriodL1, 1, massParameter, 1.0, 1.0, thetaStoppingAngle, numberOfTrajectoriesPerManifold );

    if( saveFrequency >= 0 ) {
        writeManifoldStateHistoryAtThetaToFile( unstableManifoldStateHistoryAtTheta, 1, orbitType, desiredJacobiEnergy, 1.0, 1.0, thetaStoppingAngle );
        writePoincareSectionToFile( unstableManifoldStateHistoryAtTheta, 1, orbitType, desiredJacobiEnergy, 1.0, 1.0, thetaStoppingAngle, numberOfTrajectoriesPerManifold );
    }

    // Load orbits in L2 and refine to specific Jacobi energy
    selectedInitialConditions = readInitialConditionsFromFile(2, orbitType, orbitOneL2, orbitTwoL2, massParameter);
    refinedJacobiEnergyResult = refineOrbitJacobiEnergy(2, orbitType, desiredJacobiEnergy,
                                                        selectedInitialConditions.segment(1, 6),
                                                        selectedInitialConditions(0),
                                                        selectedInitialConditions.segment(8, 6),
                                                        selectedInitialConditions(7), massParameter);

    Eigen::VectorXd initialStateVectorL2 = refinedJacobiEnergyResult.segment(0, 6);
    double orbitalPeriodL2 = refinedJacobiEnergyResult(6);
    double jacobiEnergyL2 = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVectorL2);

    // Calculate state at Poincaré section for interior stable manifold departing from L2
    std::map< int, std::map< double, Eigen::Vector6d > > stableManifoldStateHistoryAtTheta;  // 1. per trajectory 2. per time-step
    computeManifoldStatesAtTheta( stableManifoldStateHistoryAtTheta, initialStateVectorL2, orbitalPeriodL2, 2, massParameter, -1.0, -1.0, thetaStoppingAngle, numberOfTrajectoriesPerManifold );

    if( saveFrequency >= 0 ) {
        writeManifoldStateHistoryAtThetaToFile( stableManifoldStateHistoryAtTheta, 2, orbitType, desiredJacobiEnergy, -1.0, -1.0, thetaStoppingAngle );
        writePoincareSectionToFile( stableManifoldStateHistoryAtTheta, 2, orbitType, desiredJacobiEnergy, -1.0, -1.0, thetaStoppingAngle, numberOfTrajectoriesPerManifold);
    }


    Eigen::MatrixXd minimumImpulseStateVectorsAtPoincare = findMinimumImpulseManifoldConnection( stableManifoldStateHistoryAtTheta,
                                                                                                 unstableManifoldStateHistoryAtTheta,
                                                                                                 numberOfTrajectoriesPerManifold );
/*
 * write initial orbits to file
 * per angle: write trajectory to file
 * once: write per angle the minimum impulse and position discrepancy
*/
    return minimumImpulseStateVectorsAtPoincare;
}
