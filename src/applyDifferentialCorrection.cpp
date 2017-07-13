#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"


Eigen::VectorXd applyDifferentialCorrection( Eigen::VectorXd initialStateVector, string orbitType, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit, double maxVelocityDeviationFromPeriodicOrbit)
{
    cout << "\nApply differential correction:" << endl;

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix = Eigen::MatrixXd::Identity(6, 6);
    identityMatrix.resize(36, 1);

    initialStateVectorInclSTM.segment(0,6) = initialStateVector;
    initialStateVectorInclSTM.segment(6,36) = identityMatrix;

    Eigen::VectorXd halfPeriodState = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbitType);
    double orbitalPeriod = 2.0 * halfPeriodState( 42 );
    Eigen::VectorXd differentialCorrection( 6 );
    Eigen::VectorXd outputVector( 43 );

    double positionDeviationFromPeriodicOrbit = halfPeriodState(1);
    double velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));

    cout << "\nInitial state vector:" << endl << initialStateVectorInclSTM.segment(0,6) << endl
         << "\nPosition deviation from periodic orbit: " << positionDeviationFromPeriodicOrbit
         << "\nVelocity deviation from periodic orbit: " << velocityDeviationFromPeriodicOrbit << endl
         << "\nDifferential correction:" << endl;

    // Apply differential correction and propagate to half-period point until converged.
    while (positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
           velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit) {

        // Apply differential correction.
        differentialCorrection = computeDifferentialCorrectionHalo( halfPeriodState );

        initialStateVectorInclSTM( 0 ) = initialStateVectorInclSTM( 0 ) + differentialCorrection( 0 )/1.0;
        initialStateVectorInclSTM( 2 ) = initialStateVectorInclSTM( 2 ) + differentialCorrection( 2 )/1.0;
        initialStateVectorInclSTM( 4 ) = initialStateVectorInclSTM( 4 ) + differentialCorrection( 4 )/1.0;

        // Propagate new state forward to half-period point.
        outputVector = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.5, 1.0, orbitType);
        halfPeriodState = outputVector.segment( 0, 42 );
        orbitalPeriod = 2.0 * outputVector( 42 );

        // Calculate deviation from periodic orbit.
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));
        positionDeviationFromPeriodicOrbit = halfPeriodState(1);
        cout << velocityDeviationFromPeriodicOrbit << endl;
    }

    cout << "\nCorrected initial state vector:" << endl << initialStateVectorInclSTM.segment(0,6) << endl;
    initialStateVectorInclSTM(6) = orbitalPeriod;

    return initialStateVectorInclSTM.segment( 0, 7 );
}