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
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"


Eigen::VectorXd computeDeviationVector (const Eigen::VectorXd& initialStateVector, const double targetPeriod,
                                        const Eigen::VectorXd& targetStateVector, const double currentPeriod) {
    Eigen::VectorXd deviationVector = Eigen::VectorXd::Zero(11);
    deviationVector.block(0,0,10,1) = initialStateVector-targetStateVector;
    deviationVector(10) = targetPeriod - currentPeriod;
    return deviationVector;

}

Eigen::VectorXd applyDifferentialCorrectionAugmented(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            double orbitalPeriod, const double massParameter,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nApply differential correction:" << std::endl;

    Eigen::MatrixXd initialStateVectorInclSTM = Eigen::MatrixXd::Zero( 10, 11 );
    initialStateVectorInclSTM.block( 0, 0, 10, 1 ) = initialStateVector;
    initialStateVectorInclSTM.block( 0, 1, 10, 10 ).setIdentity( );

    double targetOrbitalPeriod = orbitalPeriod;

    double multiplierPeriod = 2.0;


    std::map< double, Eigen::VectorXd > stateHistory;
    std::pair< Eigen::MatrixXd, double > fullPeriodState = propagateOrbitAugmentedToFinalCondition(
                initialStateVectorInclSTM, massParameter, orbitalPeriod / multiplierPeriod, 1.0, stateHistory, -1, 0.0 );
    Eigen::MatrixXd stateVectorInclSTM      = fullPeriodState.first;
    double currentTime             = fullPeriodState.second;
    Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );


    // Initialize variables
    Eigen::VectorXd differentialCorrection(11);
    Eigen::VectorXd outputVector(23);
    double positionDeviationFromPeriodicOrbit;
    double velocityDeviationFromPeriodicOrbit;
    double periodDeviationFromPeriodicOrbit;

    // Compute Deviation Vector
    Eigen::VectorXd deviationVector(11);
    deviationVector= computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), orbitalPeriod, stateVectorInclSTM.block(0,0,10,1), multiplierPeriod * currentTime  );

    // Full Period Deviation
    //positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(0), 2) + pow(deviationVector(1), 2) );
    //velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2) + pow(deviationVector(4), 2) );
    //periodDeviationFromPeriodicOrbit = sqrt( pow( deviationVector(10), 2) );

    positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(1), 2) + pow(deviationVector(2), 2) );
    velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2) + pow(deviationVector(5), 2) );
    periodDeviationFromPeriodicOrbit = sqrt( pow( deviationVector(10), 2) );



    bool deviationFromPeriodicOrbitRelaxed = false;

    int numberOfIterations = 0;

    std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "periodDeviationFromPeriodicOrbit: " << periodDeviationFromPeriodicOrbit << std::endl
              << "numberOfIterations: " << numberOfIterations  << std::endl
              << "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              << "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
              << "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << "\n" << std::endl;

    // Apply differential correction and propagate to full-period point until converged.
    while ( positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
            velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit or periodDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit ) {

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

//         std::cout << "deviationVector: \n " << deviationVector << std::endl;
//         std::cout << "currentTime: \n " << currentTime << std::endl;

        differentialCorrection = computeDifferentialCorrectionAugmented( stateVectorInclSTM, deviationVector );

       std::cout<<"APPLYING DIFF. CORR: "<<differentialCorrection<<std::endl;

        initialStateVectorInclSTM.block( 0, 0, 10, 1 ) = initialStateVectorInclSTM.block( 0, 0, 10, 1 ) + differentialCorrection.segment( 0, 10 ) / 1.0;
        orbitalPeriod  = orbitalPeriod + differentialCorrection( 10 ) / 1.0;
       std::pair< Eigen::MatrixXd, double > fullPeriodState = propagateOrbitAugmentedToFinalCondition(
                   initialStateVectorInclSTM, massParameter, orbitalPeriod / multiplierPeriod, 1.0, stateHistory, -1, 0.0 );

        stateVectorInclSTM      = fullPeriodState.first;
        currentTime             = fullPeriodState.second;
        stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

        numberOfIterations += 1;

        deviationVector = computeDeviationVector( initialStateVectorInclSTM.block(0, 0, 10, 1), targetOrbitalPeriod, stateVectorInclSTM.block(0, 0, 10, 1), multiplierPeriod * currentTime);

          // Full period Deviation
//        positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(0), 2) + pow(deviationVector(1), 2) );
//        velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2) + pow(deviationVector(4), 2) );
//        periodDeviationFromPeriodicOrbit = sqrt( pow( deviationVector(10), 2) );

        positionDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(1), 2) + pow(deviationVector(2), 2) );
        velocityDeviationFromPeriodicOrbit = sqrt(pow(deviationVector(3), 2) + pow(deviationVector(5), 2) );
        periodDeviationFromPeriodicOrbit = sqrt( pow( deviationVector(10), 2) );


        std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
                  << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << std::endl
                  << "periodDeviationFromPeriodicOrbit: " << periodDeviationFromPeriodicOrbit << std::endl
                  << "numberOfIterations: " << numberOfIterations << std::endl
                  << "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
                  << "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
                  << "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << "\n" << std::endl;

    }

        double hamiltonianFullPeriod       = computeHamiltonian(massParameter, stateVectorInclSTM.block(0,0,10,1) );
        double hamiltonianInitialCondition = computeHamiltonian(massParameter, initialStateVector );


        std::cout << "\nCorrected initial state vector:" << std::endl << initialStateVectorInclSTM.block( 0, 0, 10, 1 )        << std::endl
                  << "\nwith orbital period: "           << currentTime * 2.0                                             << std::endl
                  << "||H(0) - H(T/2)|| = "               << std::abs(hamiltonianInitialCondition - hamiltonianFullPeriod) << std::endl
                  << "||T - t|| = "                    << std::abs(targetOrbitalPeriod - 2.0 * currentTime) << "\n"               << std::endl;

       // The output vector consists of:
       // 1. Corrected initial state vector, including orbital period
       // 2. Full period state vector, including currentTime of integration
      // 3. numberOfIterations

        outputVector.segment(0,10)    = initialStateVectorInclSTM.block( 0, 0, 10, 1 );
        outputVector(10)              = orbitalPeriod;
       outputVector.segment(11,10)    = stateVectorOnly;
        outputVector(21)             = currentTime;
        outputVector(22)             = numberOfIterations;
        //outputVector = Eigen::VectorXd::Zero(23);
        return outputVector;


   // return initialStateVectorInclSTM;
}

