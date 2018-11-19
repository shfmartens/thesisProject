#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <exception>

#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"

#include "propagateOrbit.h"
#include "computeManifolds.h"


void determineStableUnstableEigenvectors( Eigen::MatrixXd& monodromyMatrix, Eigen::Vector6d& stableEigenvector,
                                          Eigen::Vector6d& unstableEigenvector, const double maxEigenvalueDeviation )
{
    int indexMaximumEigenvalue;
    int indexMinimumEigenvalue;
    double maximumEigenvalue = 0.0;
    double minimumEigenvalue = 1000.0;

    // Compute eigenvectors of the monodromy matrix
    Eigen::EigenSolver< Eigen::MatrixXd > eig(monodromyMatrix);

    // Find the minimum (maximum) eigenvalue, corresponding to the stable (unstable) subspace
    for ( int i = 0; i <= 5; i++ ) {
        if ( eig.eigenvalues().real()(i) > maximumEigenvalue && std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation ) {
            maximumEigenvalue      = eig.eigenvalues().real()(i);
            indexMaximumEigenvalue = i;
        }
        if ( std::abs(eig.eigenvalues().real()(i)) < minimumEigenvalue && std::abs(eig.eigenvalues().imag()(i)) < maxEigenvalueDeviation ) {
            minimumEigenvalue      = std::abs(eig.eigenvalues().real()(i));
            indexMinimumEigenvalue = i;
        }
    }

    unstableEigenvector = eig.eigenvectors().real().col(indexMaximumEigenvalue);
    stableEigenvector   = eig.eigenvectors().real().col(indexMinimumEigenvalue);

    // Check whether the two selected eigenvalues belong to the same reciprocal pair
    if ( (1.0 / minimumEigenvalue - maximumEigenvalue) > maxEigenvalueDeviation ) {
        std::cout << "\n\n\nERROR - EIGENVALUES MIGHT NOT BELONG TO SAME RECIPROCAL PAIR" << std::endl;
        throw std::exception();
    }
}

double determineEigenvectorSign( Eigen::Vector6d& eigenvector )
{
    double eigenvectorSign;

    if ( eigenvector(0) > 0.0 ) {
        eigenvectorSign = 1.0;
    } else {
        eigenvectorSign = -1.0;
    }

    return eigenvectorSign;
}

bool checkJacobiOnManifoldOutsideBounds( Eigen::MatrixXd& stateVectorInclSTM, double& referenceJacobiEnergy,
                                         const double massParameter, const double maxJacobiEnergyDeviation )
{
    bool jacobiDeviationOutsideBounds;
    double currentJacobiEnergy = tudat::gravitation::computeJacobiEnergy(massParameter, stateVectorInclSTM.block(0,0,6,1));

    if ( std::abs(currentJacobiEnergy - referenceJacobiEnergy) < maxJacobiEnergyDeviation ) {
        jacobiDeviationOutsideBounds = false;
    } else {
        jacobiDeviationOutsideBounds = true;
        std::cout << "Jacobi energy deviation on manifold exceeded bounds" << std::endl;
    }

    return jacobiDeviationOutsideBounds;
}

void reduceOvershootAtPoincareSectionU1U4( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& ySign,
                                           int& integrationDirection, const double& massParameter )
{
    // TODO join together with reduceOvershootAtPoincareSectionU2U3
    stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
    stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
    currentTime               = stateVectorInclSTMAndTime.second;
    std::cout << "||y|| = " << stateVectorInclSTM(1, 0) << ", at start of iterative procedure" << std::endl;

    for ( int i = 5; i <= 12; i++ ) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while ( stateVectorInclSTMAndTime.first(1, 0) * ySign > 0 ) {
            stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
            currentTime                       = stateVectorInclSTMAndTime.second;
            previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
            stateVectorInclSTMAndTime         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime,
                                                               integrationDirection, initialStepSize, maximumStepSize);

            if ( stateVectorInclSTMAndTime.first(1, 0) * ySign < 0 ) {
                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                break;
            }
        }
    }
    std::cout << "||y|| = " << stateVectorInclSTM(1, 0) << ", at end of iterative procedure" << std::endl;
}

void reduceOvershootAtPoincareSectionU2U3( std::pair< Eigen::MatrixXd, double >& stateVectorInclSTMAndTime,
                                           std::pair< Eigen::MatrixXd, double >& previousStateVectorInclSTMAndTime,
                                           Eigen::MatrixXd& stateVectorInclSTM, double& currentTime, double& xDiffSign,
                                           int& integrationDirection, const double& massParameter )
{
    stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
    stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
    currentTime               = stateVectorInclSTMAndTime.second;
    std::cout << "||x - (1-mu)|| = "                 << (stateVectorInclSTM(0, 0) - (1.0 - massParameter))
              << ", at start of iterative procedure" << std::endl;

    for ( int i = 5; i <= 12; i++ ) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while ( (stateVectorInclSTMAndTime.first(0, 0) - (1.0 - massParameter)) * xDiffSign > 0 ) {
            stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
            currentTime                       = stateVectorInclSTMAndTime.second;
            previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
            stateVectorInclSTMAndTime         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime,
                                                               integrationDirection, initialStepSize, maximumStepSize);

            if ( (stateVectorInclSTMAndTime.first(0, 0) - (1.0 - massParameter)) * xDiffSign < 0 ) {
                stateVectorInclSTMAndTime = previousStateVectorInclSTMAndTime;
                break;
            }
        }
    }
    std::cout << "||x - (1-mu)|| = "               << (stateVectorInclSTM(0, 0) - 1.0 + massParameter)
              << ", at end of iterative procedure" << std::endl;
}

void writeManifoldStateHistoryToFile( std::map< int, std::map< int, std::map< double, Eigen::Vector6d > > >& manifoldStateHistory,
                                      const int& orbitNumber, const int& librationPointNr, const std::string& orbitType )
{
    std::string fileNameStateVector;
    std::ofstream textFileStateVectors;
    std::vector<std::string> fileNamesStateVectors = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_plus.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_min.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_plus.txt",
                                                      "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_min.txt"};

    // For all four manifolds
    for( auto const &ent1 : manifoldStateHistory ) {
        fileNameStateVector = fileNamesStateVectors.at(ent1.first);

        remove(("../data/raw/manifolds/" + fileNameStateVector).c_str());
        textFileStateVectors.open(("../data/raw/manifolds/" + fileNameStateVector));
        textFileStateVectors.precision(14);

        // For all numberOfTrajectoriesPerManifold
        for( auto const &ent2 : ent1.second ) {
            // For all states on manifold trajectory
            for( auto const &ent3 : ent2.second ) {
                textFileStateVectors << std::left    << std::scientific << std::setw(25) << ent3.first
                                     << std::setw(25) << ent3.second(0) << std::setw(25) << ent3.second(1)
                                     << std::setw(25) << ent3.second(2) << std::setw(25) << ent3.second(3)
                                     << std::setw(25) << ent3.second(4) << std::setw(25) << ent3.second(5) << std::endl;
            }
        }
        textFileStateVectors.close();
        textFileStateVectors.clear();
    }
}

void writeEigenvectorStateHistoryToFile( std::map< int, std::map< int, std::pair< Eigen::Vector6d, Eigen::Vector6d > > >& eigenvectorStateHistory,
                                         const int& orbitNumber, const int& librationPointNr,
                                         const std::string& orbitType )
{
    std::string fileNameEigenvectorDirection;
    std::string fileNameEigenvectorLocation;
    std::ofstream textFileEigenvectorDirections;
    std::ofstream textFileEigenvectorLocations;

    std::vector<std::string> fileNamesEigenvectorDirections = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_plus_eigenvector.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_min_eigenvector.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_plus_eigenvector.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_min_eigenvector.txt"};
    std::vector<std::string> fileNamesEigenvectorLocations  = {"L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_plus_eigenvector_location.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_S_min_eigenvector_location.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_plus_eigenvector_location.txt",
                                                               "L" + std::to_string(librationPointNr) + "_" + orbitType + "_" + std::to_string(orbitNumber) + "_W_U_min_eigenvector_location.txt"};

    // For all four manifolds
    for( auto const &ent1 : eigenvectorStateHistory ) {
        fileNameEigenvectorDirection = fileNamesEigenvectorDirections.at(ent1.first);
        fileNameEigenvectorLocation    = fileNamesEigenvectorLocations.at(ent1.first);

        remove(("../data/raw/manifolds/" + fileNameEigenvectorDirection).c_str());
        remove(("../data/raw/manifolds/" + fileNameEigenvectorLocation).c_str());
        textFileEigenvectorDirections.open(("../data/raw/manifolds/" + fileNameEigenvectorDirection));
        textFileEigenvectorLocations.open(("../data/raw/manifolds/" + fileNameEigenvectorLocation));
        textFileEigenvectorDirections.precision(14);
        textFileEigenvectorLocations.precision(14);

        // For all numberOfTrajectoriesPerManifold
        for( auto const &ent2 : ent1.second ) {
            textFileEigenvectorDirections << std::left << std::scientific << std::setw(25)
                                          << ent2.second.first(0) << std::setw(25) << ent2.second.first(1) << std::setw(25)
                                          << ent2.second.first(2) << std::setw(25) << ent2.second.first(3) << std::setw(25)
                                          << ent2.second.first(4) << std::setw(25) << ent2.second.first(5) << std::endl;

            textFileEigenvectorLocations << std::left << std::scientific << std::setw(25)
                                         << ent2.second.second(0) << std::setw(25) << ent2.second.second(1) << std::setw(25)
                                         << ent2.second.second(2) << std::setw(25) << ent2.second.second(3) << std::setw(25)
                                         << ent2.second.second(4) << std::setw(25) << ent2.second.second(5) << std::endl;
        }
        textFileEigenvectorDirections.close();
        textFileEigenvectorDirections.clear();
        textFileEigenvectorLocations.close();
        textFileEigenvectorLocations.clear();
    }
}

void computeManifolds( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
                       const int librationPointNr, const std::string orbitType, const double massParameter,
                       const double eigenvectorDisplacementFromOrbit, const int numberOfTrajectoriesPerManifold,
                       const int saveFrequency, const bool saveEigenvectors,
                       const double maximumIntegrationTimeManifoldTrajectories, const double maxEigenvalueDeviation )
{
    // Set output maximum precision

    std::cout.precision(std::numeric_limits<double>::digits10);

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
    std::cout << "EIGEN VECTORS COMPUTED: " <<  std::endl;
    // The sign of the x-component of the eigenvector is determined, which is used to determine the eigenvector offset direction (interior/exterior manifold)
    double stableEigenvectorSign   = determineEigenvectorSign( stableEigenvector );
    double unstableEigenvectorSign = determineEigenvectorSign( unstableEigenvector );

    int integrationDirection;
    double currentTime;
    double offsetSign;
    Eigen::VectorXd monodromyMatrixEigenvector;
    Eigen::Vector6d localStateVector           = Eigen::Vector6d::Zero(6);
    Eigen::Vector6d localNormalizedEigenvector = Eigen::Vector6d::Zero(6);
    Eigen::MatrixXd manifoldStartingState      = Eigen::MatrixXd::Zero(6, 7);
    std::vector<double> offsetSigns            = {1.0 * stableEigenvectorSign, -1.0 * stableEigenvectorSign, 1.0 * unstableEigenvectorSign, -1.0 * unstableEigenvectorSign};
    std::vector<Eigen::VectorXd> eigenVectors  = {stableEigenvector, stableEigenvector, unstableEigenvector, unstableEigenvector};
    std::vector<int> integrationDirections     = {-1, -1, 1, 1};
    std::pair< Eigen::MatrixXd, double >                                            stateVectorInclSTMAndTime;
    std::pair< Eigen::MatrixXd, double >                                            previousStateVectorInclSTMAndTime;
    std::map< int, std::map< int, std::map< double, Eigen::Vector6d > > >           manifoldStateHistory;  // 1. per manifold 2. per trajectory 3. per time-step
    std::map< int, std::map< int, std::pair< Eigen::Vector6d, Eigen::Vector6d > > > eigenvectorStateHistory;  // 1. per manifold 2. per trajectory 3. direction and location

    std::cout << "START MANIFOLD INITIAL CONDITION AND INTEGRATION COMPUTATION: " <<  std::endl;

    for ( int manifoldNumber = 0; manifoldNumber < 4; manifoldNumber++ ) {

        bool fullManifoldComputed       = false;
        bool jacobiEnergyOutsideBounds  = false;
        bool ySignSet                   = false;
        bool xDiffSignSet               = false;
        double ySign                    = 0.0;
        double xDiffSign                = 0.0;

        offsetSign                 = offsetSigns.at(manifoldNumber);
        monodromyMatrixEigenvector = eigenVectors.at(manifoldNumber);
        integrationDirection       = integrationDirections.at(manifoldNumber);

        // TODO replace with text (like interior/exterior unstable/stable)
        std::cout << "\n\nManifold: " << manifoldNumber << "\n" << std::endl;

        // Determine the total number of points along the periodic orbit to start the manifolds.
        for ( int trajectoryOnManifoldNumber = 0; trajectoryOnManifoldNumber < numberOfTrajectoriesPerManifold; trajectoryOnManifoldNumber++ ) {

            int indexCount    = 0;
            int stepCounter   = 1;
            auto indexOnOrbit = static_cast <int> (std::floor(trajectoryOnManifoldNumber * numberOfPointsOnPeriodicOrbit / numberOfTrajectoriesPerManifold));

            Eigen::MatrixXd stateTransitionMatrix;
            for ( auto const& it : stateTransitionMatrixHistory ) {
                if ( indexCount == indexOnOrbit ) {
                    stateTransitionMatrix = it.second.block(0, 1, 6, 6);
                    localStateVector = it.second.block(0, 0, 6, 1);
                    break;
                }
                indexCount += 1;
            }

            std::cout << "APPLY DISPLACEMENTS: " <<  std::endl;

            // Apply displacement epsilon from the periodic orbit at <numberOfTrajectoriesPerManifold> locations on the final orbit.
            localNormalizedEigenvector = (stateTransitionMatrix * monodromyMatrixEigenvector).normalized();
            manifoldStartingState      = getFullInitialState( localStateVector + offsetSign * eigenvectorDisplacementFromOrbit * localNormalizedEigenvector );

            if ( saveEigenvectors ) {
                eigenvectorStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ] = std::make_pair(localNormalizedEigenvector, localStateVector);
            }
            if ( saveFrequency >= 0 ) {
                manifoldStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ][ 0.0 ] = manifoldStartingState.block( 0, 0, 6, 1 );
            }

            stateVectorInclSTMAndTime = propagateOrbit(manifoldStartingState, massParameter, 0.0, integrationDirection );
            stateVectorInclSTM        = stateVectorInclSTMAndTime.first;
            currentTime               = stateVectorInclSTMAndTime.second;

            std::cout << "Trajectory on manifold number: " << trajectoryOnManifoldNumber << std::endl;

            while ( (std::abs( currentTime ) <= maximumIntegrationTimeManifoldTrajectories) && !fullManifoldComputed ) {

                // Check whether trajectory still belongs to the same energy level
                jacobiEnergyOutsideBounds = checkJacobiOnManifoldOutsideBounds(stateVectorInclSTM, jacobiEnergyOnOrbit, massParameter);
                fullManifoldComputed      = jacobiEnergyOutsideBounds;

                // Determine sign of y when crossing x = 0  (U1, U4)
                if ( (stateVectorInclSTM(0, 0) < 0) && !ySignSet ) {
                    if ( stateVectorInclSTM(1, 0) < 0 ){
                        ySign = -1.0;
                    }
                    if ( stateVectorInclSTM(1, 0) > 0 ) {
                        ySign = 1.0;
                    }
                    ySignSet = true;
                }

                // Determine whether the trajectory approaches U2, U3 from the right or left (U2, U3)
                if ( !xDiffSignSet ) {
                    if ( (stateVectorInclSTM(0, 0) - (1.0 - massParameter)) < 0 ) {
                        xDiffSign = -1.0;
                    }
                    if ( (stateVectorInclSTM(0, 0) - (1.0 - massParameter)) > 0 ) {
                        xDiffSign = 1.0;
                    }
                    xDiffSignSet = true;
                }

                // Determine when the manifold crosses the x-axis again (U1, U4)
                if ( (stateVectorInclSTM(1, 0) * ySign < 0) && ySignSet ) {
                    reduceOvershootAtPoincareSectionU1U4(stateVectorInclSTMAndTime, previousStateVectorInclSTMAndTime,
                                                         stateVectorInclSTM, currentTime, ySign, integrationDirection,
                                                         massParameter);
                    fullManifoldComputed = true;
                }

                // Determine when the manifold crosses the Poincare section near the second primary (U2, U3)
                if ( ((stateVectorInclSTM(0, 0) - (1.0 - massParameter)) * xDiffSign < 0) &&
                        ((librationPointNr == 1 && ( manifoldNumber == 0 || manifoldNumber == 2)) ||
                         (librationPointNr == 2 && ( manifoldNumber == 1 || manifoldNumber == 3))) ) {
                    reduceOvershootAtPoincareSectionU2U3(stateVectorInclSTMAndTime,
                                                         previousStateVectorInclSTMAndTime,
                                                         stateVectorInclSTM, currentTime, xDiffSign,
                                                         integrationDirection, massParameter);
                    fullManifoldComputed = true;
                }

                // Write every nth integration step to file.

                if ( saveFrequency > 0 && ((stepCounter % saveFrequency == 0 || fullManifoldComputed) && !jacobiEnergyOutsideBounds ) ) {
                    std::cout << "MANIFOLD STATE IS BEING COMPUTED: " <<  std::endl;
                    manifoldStateHistory[ manifoldNumber ][ trajectoryOnManifoldNumber ][ currentTime ] = stateVectorInclSTM.block( 0, 0, 6, 1 );
                }

                if ( !fullManifoldComputed ){
                    // Propagate to next time step
                    std::cout << "PROPAGATE TO NEXT TIME STEP: " <<  std::endl;
                    previousStateVectorInclSTMAndTime = stateVectorInclSTMAndTime;
                    stateVectorInclSTMAndTime         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, integrationDirection);
                    stateVectorInclSTM                = stateVectorInclSTMAndTime.first;
                    currentTime                       = stateVectorInclSTMAndTime.second;
                    stepCounter++;
                }
            }
            ySignSet             = false;
            xDiffSignSet         = false;
            fullManifoldComputed = false;
        }

    }
    std::cout << "The saveFrequency value is: " << saveFrequency  << std::endl;
    if( saveFrequency >= 0 ) {
        writeManifoldStateHistoryToFile( manifoldStateHistory, orbitNumber, librationPointNr, orbitType );
    }
    if ( saveEigenvectors ) {
        writeEigenvectorStateHistoryToFile( eigenvectorStateHistory, orbitNumber, librationPointNr, orbitType );
    }

    std::cout << std::endl
              << "================================================"                             << std::endl
              << "                          "   << orbitNumber    << "                        " << std::endl
              << "Mass parameter: "             << massParameter                                << std::endl
              << "C at initial conditions: "    << jacobiEnergyOnOrbit                          << std::endl
              << "C at end of manifold orbit: " << tudat::gravitation::computeJacobiEnergy(massParameter, stateVectorInclSTM.block(0,0,6,1)) << std::endl
              << "T: " << orbitalPeriod                                                         << std::endl
              << "================================================"                             << std::endl;
}
