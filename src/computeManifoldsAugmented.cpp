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
#include "computeManifoldsAugmented.h"

void computeManifoldsAugmented( const Eigen::Vector6d initialStateVector, const double orbitalPeriod, const int orbitNumber,
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
}
