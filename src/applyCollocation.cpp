#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <map>

#include <chrono>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"
#include "Tudat/InputOutput/basicInputOutput.h"

#include "createLowThrustInitialConditions.h"
#include "applyCollocation.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

Eigen::VectorXd applyCollocation(const Eigen::VectorXd initialCollocationGuess, const double massParameter, const int numberOfCollocationPoints, Eigen::VectorXd& collocatedGuess,
                                                         const double maxPositionDeviationFromPeriodicOrbit, const double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit, const int maxNumberOfIterations)
{
    // initialize Variables
    Eigen::VectorXd outputVector = Eigen::VectorXd(25);
    int numberOfCorrections = 0;

    std::cout << "collocation Reached " << std::endl;

    // Compute variables for outputVector and collocatedGuess

    collocatedGuess = initialCollocationGuess;

    Eigen::VectorXd  initialCondition = collocatedGuess.segment(0,10);
    Eigen::VectorXd  finalCondition = collocatedGuess.segment(11*(numberOfCollocationPoints-1),10);

    double orbitalPeriod = collocatedGuess(11*numberOfCollocationPoints+10) - collocatedGuess(10);

    double hamiltonianInitialCondition  = computeHamiltonian( massParameter, initialCondition);
    double hamiltonianEndState          = computeHamiltonian( massParameter, finalCondition  );

    outputVector.segment(0,10) = initialCondition;
    outputVector(10) = orbitalPeriod;
    outputVector(11) = hamiltonianInitialCondition;
    outputVector.segment(12,10) = finalCondition;
    outputVector(22) = collocatedGuess(11*(numberOfCollocationPoints-1) + 10);
    outputVector(23) = hamiltonianEndState;
    outputVector(24) = numberOfCorrections;

    return outputVector;

}

