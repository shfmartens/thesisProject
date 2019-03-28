#include <cmath>
#include <iostream>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "applyDifferentialCorrectionAugmented.h"
#include "createLowThrustInitialConditions.h"
#include "computeDifferentialCorrectionAugmented.h"
#include "propagateOrbitAugmented.h"

Eigen::VectorXd computeDeviationVector (const Eigen::VectorXd& initialStateVector, const double initialPeriod,
                                        const Eigen::VectorXd& targetStateVector, const double targetPeriod) {
    Eigen::VectorXd deviationVector = Eigen::VectorXd::Zero(11);
    deviationVector.block(0,0,10,1) = targetStateVector - initialStateVector;
    deviationVector(11) = targetPeriod - initialPeriod;
    return deviationVector;

}

Eigen::VectorXd applyDifferentialCorrectionAugmented(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            double orbitalPeriod, const double massParameter,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nApply differential correction:" << std::endl;

    Eigen::MatrixXd initialStateVectorInclSTM = Eigen::MatrixXd::Zero( 10, 11 );

    initialStateVectorInclSTM.block( 0, 0, 10, 1 ) = initialStateVector;
    initialStateVectorInclSTM.block( 0, 1, 10, 10 ).setIdentity( );

    std::map< double, Eigen::VectorXd > stateHistory;
    std::pair< Eigen::MatrixXd, double > fullPeriodState = propagateOrbitAugmentedToFinalCondition(
                initialStateVectorInclSTM, massParameter, orbitalPeriod, 1.0, stateHistory, -1, 0.0 );
    Eigen::MatrixXd stateVectorInclSTM      = fullPeriodState.first;
    double currentTime             = fullPeriodState.second;
    Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

    // Initialize variables
    Eigen::VectorXd differentialCorrection(11);
    Eigen::VectorXd outputVector(23);
    Eigen::VectorXd deviationVector(11);
    double positionDeviationFromPeriodicOrbit;
    double velocityDeviationFromPeriodicOrbit;

    // Compute deviations
    deviationVector = computeDeviationVector(initialStateVectorInclSTM.block( 0, 0, 10, 1 ), orbitalPeriod, stateVectorInclSTM.block( 0, 0, 10, 1 ), currentTime);
    positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(0), 2)+pow(deviationVector(1), 2) + +pow(deviationVector(2), 2)) ;
    velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2)+pow(deviationVector(4), 2) + +pow(deviationVector(5), 2)) ;

    bool deviationFromPeriodicOrbitRelaxed = false;

    int numberOfIterations = 0;

    std::cout <<" STATE VECTOR AFTER PROPAGATION: \n" << stateVectorOnly << "\n" << std::endl;

    std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "numberOfIterations: " << numberOfIterations  << std::endl
              << "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              << "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << "\n" << std::endl;

    // Apply differential correction and propagate to full-period point until converged.
    while ( positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
            velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit ) {

        // If the maximum number of iterations has been reached, return a zero vector to stop the numerical continuation
        if ( numberOfIterations > maxNumberOfIterations and deviationFromPeriodicOrbitRelaxed == false )
        {
            // Relax the periodicity constraints after exceeding the maximum number of iterations instead of termination
            maxPositionDeviationFromPeriodicOrbit = 10.0 * maxPositionDeviationFromPeriodicOrbit;
            maxVelocityDeviationFromPeriodicOrbit = 10.0 * maxVelocityDeviationFromPeriodicOrbit;
            //deviationFromPeriodicOrbitRelaxed = true;
            //return Eigen::VectorXd::Zero(15)
    }
        // Relax the maximum deviation requirements to compute the horizontal Lyapunov family in L2
                    if (deviationFromPeriodicOrbitRelaxed == false and numberOfIterations > 10 and librationPointNr == 2)
        {

           maxPositionDeviationFromPeriodicOrbit = 10.0 * maxPositionDeviationFromPeriodicOrbit;
            maxVelocityDeviationFromPeriodicOrbit = 10.0 * maxVelocityDeviationFromPeriodicOrbit;
            //deviationFromPeriodicOrbitRelaxed = true;
       }

        differentialCorrection = computeDifferentialCorrectionAugmented(stateVectorInclSTM, deviationVector);
        //differentialCorrection = Eigen::VectorXd::Zero(11);

       std::cout<<"APPLYING DIFF. CORR: "<<differentialCorrection<<std::endl;

        initialStateVectorInclSTM.block( 0, 0, 10, 1 ) += differentialCorrection.segment( 0, 10 ) / 1.0;
        orbitalPeriod  = orbitalPeriod + 1.0 * differentialCorrection( 10 ) / 1.0;

       std::pair< Eigen::MatrixXd, double > fullPeriodState = propagateOrbitAugmentedToFinalCondition(
                    initialStateVectorInclSTM, massParameter, orbitalPeriod, 1.0, stateHistory, -1, 0.0 );
        fullPeriodState = propagateOrbitAugmentedToFinalCondition(
                    initialStateVectorInclSTM, massParameter, orbitalPeriod, 1.0, stateHistory, -1, 0.0 );
        stateVectorInclSTM      = fullPeriodState.first;
        currentTime             = fullPeriodState.second;
        stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );


        numberOfIterations += 1;

        deviationVector = computeDeviationVector(initialStateVectorInclSTM.block( 0, 0, 10, 1 ), orbitalPeriod, stateVectorInclSTM.block( 0, 0, 10, 1 ), currentTime);
        positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(0), 2)+pow(deviationVector(1), 2) + +pow(deviationVector(2), 2)) ;
        velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2)+pow(deviationVector(4), 2) + +pow(deviationVector(5), 2)) ;

        std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
                  << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << std::endl
                  << "numberOfIterations: " << numberOfIterations << std::endl
                  << "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
                  << "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << "\n" << std::endl;

    }

        double hamiltonianFullPeriod       = computeHamiltonian(massParameter, stateVectorOnly );
        double hamiltonianInitialCondition = computeHamiltonian(massParameter, initialStateVectorInclSTM.block(0,0,10,1) );


        std::cout << "\nCorrected initial state vector:" << std::endl << initialStateVectorInclSTM.block( 0, 0, 10, 1 )        << std::endl
                  << "\nwith orbital period: "           << orbitalPeriod                                              << std::endl
                  << "||J(0) - J(T)|| = "               << std::abs(hamiltonianInitialCondition - hamiltonianFullPeriod) << std::endl
                  << "||T - t|| = "                    << std::abs(orbitalPeriod - currentTime) << "\n"               << std::endl;

//        // The output vector consists of:
//        // 1. Corrected initial state vector, including orbital period
//        // 2. Half period state vector, including currentTime of integration
//       // 3. numberOfIterations

        outputVector.segment(0,10)    = initialStateVectorInclSTM.block( 0, 0, 10, 1 );
        outputVector(10)              = orbitalPeriod;
       outputVector.segment(11,10)    = stateVectorOnly;
        outputVector(21)             = currentTime;
        outputVector(22)             = numberOfIterations;
        outputVector = Eigen::VectorXd::Zero(23);
        return outputVector;


   // return initialStateVectorInclSTM;
}
