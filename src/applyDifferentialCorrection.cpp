#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "thesisProject/src/applyDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrection.h"
#include "thesisProject/src/computeDifferentialCorrectionHalo.h"
#include "thesisProject/src/computeDifferentialCorrectionNearVertical.h"
#include "thesisProject/src/propagateOrbit.h"


Eigen::VectorXd applyDifferentialCorrection( std::string orbitType, Eigen::VectorXd initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit)
{
    std::cout << "\nApply differential correction:" << std::endl;

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix            = Eigen::MatrixXd::Identity(6,6);
    identityMatrix.resize(36,1);

    initialStateVectorInclSTM.segment(0,6)    = initialStateVector;
    initialStateVectorInclSTM.segment(6,36)   = identityMatrix;

    // Perform first integration step
    Eigen::VectorXd previousHalfPeriodState;
    Eigen::VectorXd halfPeriodState    = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0);
    Eigen::VectorXd stateVectorInclSTM = halfPeriodState.segment(0,42);
    double currentTime                 = halfPeriodState(42);

    // Perform integration steps until end of half orbital period
    for (int i = 4; i <= 12; i++) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

//        cout << "maximumStepSize: " << maximumStepSize << endl;
//        cout << "initialStepSize: " << initialStepSize << endl;

        while (currentTime <= (orbitalPeriod / 2.0)) {
            stateVectorInclSTM      = halfPeriodState.segment(0, 42);
            currentTime             = halfPeriodState(42);
            previousHalfPeriodState = halfPeriodState;
            halfPeriodState         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, initialStepSize, maximumStepSize);

            if (halfPeriodState(42) > (orbitalPeriod / 2.0)) {
                halfPeriodState = previousHalfPeriodState;
                break;
            }
        }
//        cout << "orbitalPeriod/2 - currentTime: " << (orbitalPeriod/2.0 - currentTime) << endl;
    }

    // Initialize variables
    Eigen::VectorXd differentialCorrection(7);
    Eigen::VectorXd outputVector(43);
    double positionDeviationFromPeriodicOrbit;
    double velocityDeviationFromPeriodicOrbit;

    positionDeviationFromPeriodicOrbit = abs(halfPeriodState(1));
    velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));

    std::cout << "\nInitial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl
              << "\nPosition deviation from periodic orbit: " << positionDeviationFromPeriodicOrbit
              << "\nVelocity deviation from periodic orbit: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "\nDifferential correction:" << std::endl;

    // Apply differential correction and propagate to half-period point until converged.
    while ( positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
            velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit) {

        // Apply differential correction
//        differentialCorrection = computeDifferentialCorrectionHalo( halfPeriodState );
//        cout << "\n\n\nCorrection by halo " << endl << differentialCorrection << endl;
//        differentialCorrection = computeDifferentialCorrectionNearVertical(halfPeriodState);
//        cout << "Correction by nearvertical " << endl << differentialCorrection << endl;
        differentialCorrection = computeDifferentialCorrection( halfPeriodState );
//        cout << "Correction by  " << endl << differentialCorrection  << "\n\n\n" << endl;

        initialStateVectorInclSTM(0) = initialStateVectorInclSTM(0) + differentialCorrection(0)/1.0;
        initialStateVectorInclSTM(1) = initialStateVectorInclSTM(1) + differentialCorrection(1)/1.0;
        initialStateVectorInclSTM(2) = initialStateVectorInclSTM(2) + differentialCorrection(2)/1.0;
        initialStateVectorInclSTM(3) = initialStateVectorInclSTM(3) + differentialCorrection(3)/1.0;
        initialStateVectorInclSTM(4) = initialStateVectorInclSTM(4) + differentialCorrection(4)/1.0;
        initialStateVectorInclSTM(5) = initialStateVectorInclSTM(5) + differentialCorrection(5)/1.0;
        orbitalPeriod                = orbitalPeriod + 2.0 * differentialCorrection(6) / 1.0;

//        // Propagate new state forward to half-period point.
//        halfPeriodState     = propagateOrbit(initialStateVectorInclSTM, massParameter, 0.0, 1.0);
//        stateVectorInclSTM  = halfPeriodState.segment(0,42);
//        currentTime         = halfPeriodState(42);
//        while (currentTime <= (orbitalPeriod / 2.0)) {
//            stateVectorInclSTM = halfPeriodState.segment(0,42);
//            currentTime = halfPeriodState(42);
//            halfPeriodState = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0);
//        }



        // Perform first integration step
        halfPeriodState    = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1.0);
        stateVectorInclSTM = halfPeriodState.segment(0,42);
        currentTime        = halfPeriodState(42);

        // Perform integration steps until end of half orbital period
        for (int i = 4; i <= 12; i++) {

            double initialStepSize = pow(10,(static_cast<float>(-i)));
            double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

            while (currentTime <= (orbitalPeriod / 2.0)) {
                stateVectorInclSTM      = halfPeriodState.segment(0, 42);
                currentTime             = halfPeriodState(42);
                previousHalfPeriodState = halfPeriodState;
                halfPeriodState         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1.0, initialStepSize, maximumStepSize);

                if (halfPeriodState(42) > (orbitalPeriod / 2.0)) {
                    halfPeriodState = previousHalfPeriodState;
                    break;
                }
            }
//            cout << "orbitalPeriod/2 - currentTime: " << (orbitalPeriod/2.0 - currentTime) << endl;
        }

        positionDeviationFromPeriodicOrbit = abs(halfPeriodState(1));
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3),2) + pow(halfPeriodState(5),2));

        std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
                  << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << "\n" << std::endl;
    }

    std::cout << "\nCorrected initial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6) << std::endl;
    std::cout << "\nwith orbital period: " << orbitalPeriod << std::endl;
    initialStateVectorInclSTM(6) = orbitalPeriod;

    return initialStateVectorInclSTM.segment(0,7);
}