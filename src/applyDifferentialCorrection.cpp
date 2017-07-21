#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeDifferentialCorrectionNearVertical.h"
#include "thesisProject/src/propagateOrbit.h"


Eigen::VectorXd applyDifferentialCorrection( Eigen::VectorXd initialStateVector, std::string orbitType, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit)
{
    std::cout << "\nApply differential correction:" << std::endl;

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);

    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    Eigen::VectorXd halfPeriodState = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbitType);
    double orbitalPeriod = 2.0 * halfPeriodState( 42 );

    // Initialize variables
    Eigen::VectorXd differentialCorrection( 6 );
    Eigen::VectorXd outputVector( 43 );
    double positionDeviationFromPeriodicOrbit;
    double velocityDeviationFromPeriodicOrbit;

    if (orbitType == "vertical") {
        positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1),2) + pow(halfPeriodState(2),2));
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2));
    } else {
        positionDeviationFromPeriodicOrbit = halfPeriodState(1);
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
    }

    std::cout << "\nInitial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl
              << "\nPosition deviation from periodic orbit: " << positionDeviationFromPeriodicOrbit
              << "\nVelocity deviation from periodic orbit: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "\nDifferential correction:" << std::endl;

    // Apply differential correction and propagate to half-period point until converged.
    while ( //positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
           velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit) {

        // Apply differential correction.
        if (orbitType == "vertical") {
            differentialCorrection = computeDifferentialCorrectionNearVertical(halfPeriodState);
        } else {
            differentialCorrection = computeDifferentialCorrectionHalo( halfPeriodState );
        }

        initialStateVectorInclSTM( 0 ) = initialStateVectorInclSTM( 0 ) + differentialCorrection( 0 )/1.0;
        initialStateVectorInclSTM( 1 ) = initialStateVectorInclSTM( 1 ) + differentialCorrection( 1 )/1.0;
        initialStateVectorInclSTM( 2 ) = initialStateVectorInclSTM( 2 ) + differentialCorrection( 2 )/1.0;
        initialStateVectorInclSTM( 3 ) = initialStateVectorInclSTM( 3 ) + differentialCorrection( 3 )/1.0;
        initialStateVectorInclSTM( 4 ) = initialStateVectorInclSTM( 4 ) + differentialCorrection( 4 )/1.0;
        initialStateVectorInclSTM( 5 ) = initialStateVectorInclSTM( 5 ) + differentialCorrection( 5 )/1.0;

        // Propagate new state forward to half-period point.
        outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbitType);
        halfPeriodState = outputVector.segment( 0, 42 );
        orbitalPeriod = 2.0 * outputVector( 42 );

        // Calculate deviation from periodic orbit.
        if (orbitType == "vertical") {
            positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1),2) + pow(halfPeriodState(2),2));
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2));
        } else {
            positionDeviationFromPeriodicOrbit = halfPeriodState(1);
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
        }

        std::cout << positionDeviationFromPeriodicOrbit << std::endl
                  << velocityDeviationFromPeriodicOrbit << "\n" << std::endl;
    }

    std::cout << "\nCorrected initial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl;
    initialStateVectorInclSTM(6) = orbitalPeriod;

    return initialStateVectorInclSTM.segment( 0, 7 );
}