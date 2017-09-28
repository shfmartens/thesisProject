#include <cmath>
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"

#include "applyDifferentialCorrection.h"
#include "computeDifferentialCorrection.h"
#include "propagateOrbit.h"



Eigen::VectorXd applyDifferentialCorrection( int librationPointNr, std::string orbitType,
                                             Eigen::VectorXd initialStateVector,
                                             double orbitalPeriod, const double massParameter,
                                             double maxPositionDeviationFromPeriodicOrbit,
                                             double maxVelocityDeviationFromPeriodicOrbit,
                                             int maxNumberOfIterations = 1000 )
{
    std::cout << "\nApply differential correction:" << std::endl;

    Eigen::VectorXd initialStateVectorInclSTM = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd identityMatrix            = Eigen::MatrixXd::Identity(6,6);
    identityMatrix.resize(36,1);

    initialStateVectorInclSTM.segment(0,6)    = initialStateVector;
    initialStateVectorInclSTM.segment(6,36)   = identityMatrix;

    // Perform first integration step
    Eigen::VectorXd previousHalfPeriodState;
    Eigen::VectorXd halfPeriodState     = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1);
    Eigen::VectorXd stateVectorInclSTM  = halfPeriodState.segment(0,42);
    double currentTime                  = halfPeriodState(42);
    int numberOfIterations              = 0;

    // Perform integration steps until end of half orbital period
    for (int i = 5; i <= 12; i++) {

        double initialStepSize = pow(10,(static_cast<float>(-i)));
        double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

        while (currentTime <= (orbitalPeriod / 2.0)) {
            stateVectorInclSTM      = halfPeriodState.segment(0, 42);
            currentTime             = halfPeriodState(42);
            previousHalfPeriodState = halfPeriodState;
            halfPeriodState         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

            if (halfPeriodState(42) > (orbitalPeriod / 2.0)) {
                halfPeriodState = previousHalfPeriodState;
                break;
            }
        }
    }

    // Initialize variables
    Eigen::VectorXd differentialCorrection(7);
    Eigen::VectorXd outputVector(15);
    double positionDeviationFromPeriodicOrbit;
    double velocityDeviationFromPeriodicOrbit;

    if (orbitType == "axial"){
        // Initial condition for axial family should be [x, 0, 0, 0, ydot, zdot]
        positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1), 2) + pow(halfPeriodState(2), 2));
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3), 2));
    } else {
        // Initial condition for other families should be [x, 0, y, 0, ydot, 0]
        positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1), 2));
        velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3), 2) + pow(halfPeriodState(5), 2));
    }

    std::cout << "\nInitial state vector:\n"                  << initialStateVectorInclSTM.segment(0,6)
              << "\nPosition deviation from periodic orbit: " << positionDeviationFromPeriodicOrbit
              << "\nVelocity deviation from periodic orbit: " << velocityDeviationFromPeriodicOrbit
              << "\n\nDifferential correction:"               << std::endl;

    bool deviationFromPeriodicOrbitRelaxed = false;

    // Apply differential correction and propagate to half-period point until converged.
    while ( positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or
            velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit ){

        // If the maximum number of iterations has been reached, return a zero vector to stop the numerical continuation
        if (numberOfIterations > maxNumberOfIterations){
           return Eigen::VectorXd::Zero(15);
        }

        // Relax the maximum deviation requirements to compute the horizontal Lyapunov family in L2
        if (deviationFromPeriodicOrbitRelaxed == false and numberOfIterations > 10 and
            orbitType == "horizontal" and librationPointNr == 2){

            maxPositionDeviationFromPeriodicOrbit = 10.0 * maxPositionDeviationFromPeriodicOrbit;
            maxVelocityDeviationFromPeriodicOrbit = 10.0 * maxVelocityDeviationFromPeriodicOrbit;
            deviationFromPeriodicOrbitRelaxed = true;
        }

        // Apply differential correction
        if (numberOfIterations > 10 and orbitType == "axial" and librationPointNr == 2){
            // To compute the full L2 axial family, fix x position after not finding a fully periodic solution after 10 iterations
            differentialCorrection = computeDifferentialCorrection( librationPointNr, orbitType, halfPeriodState, true );
        } else{
            differentialCorrection = computeDifferentialCorrection( librationPointNr, orbitType, halfPeriodState );
        }

        initialStateVectorInclSTM(0) = initialStateVectorInclSTM(0) + differentialCorrection(0)/1.0;
        initialStateVectorInclSTM(1) = initialStateVectorInclSTM(1) + differentialCorrection(1)/1.0;
        initialStateVectorInclSTM(2) = initialStateVectorInclSTM(2) + differentialCorrection(2)/1.0;
        initialStateVectorInclSTM(3) = initialStateVectorInclSTM(3) + differentialCorrection(3)/1.0;
        initialStateVectorInclSTM(4) = initialStateVectorInclSTM(4) + differentialCorrection(4)/1.0;
        initialStateVectorInclSTM(5) = initialStateVectorInclSTM(5) + differentialCorrection(5)/1.0;
        orbitalPeriod                = orbitalPeriod + 2.0 * differentialCorrection(6) / 1.0;

        // Perform first integration step
        halfPeriodState    = propagateOrbit( initialStateVectorInclSTM, massParameter, 0.0, 1);
        stateVectorInclSTM = halfPeriodState.segment(0,42);
        currentTime        = halfPeriodState(42);

        // Perform integration steps until end of half orbital period
        for (int i = 5; i <= 12; i++) {

            double initialStepSize = pow(10,(static_cast<float>(-i)));
            double maximumStepSize = pow(10,(static_cast<float>(-i) + 1.0));

            while (currentTime <= (orbitalPeriod / 2.0)) {
                stateVectorInclSTM      = halfPeriodState.segment(0, 42);
                currentTime             = halfPeriodState(42);
                previousHalfPeriodState = halfPeriodState;
                halfPeriodState         = propagateOrbit(stateVectorInclSTM, massParameter, currentTime, 1, initialStepSize, maximumStepSize);

                if (halfPeriodState(42) > (orbitalPeriod / 2.0)) {
                    halfPeriodState = previousHalfPeriodState;
                    break;
                }
            }
        }

        if (orbitType == "axial"){
            // Initial condition for axial family should be [x, 0, 0, 0, ydot, zdot]
            positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1), 2) + pow(halfPeriodState(2), 2));
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3), 2));
        } else {
            // Initial condition for other families should be [x, 0, y, 0, ydot, 0]
            positionDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(1), 2));
            velocityDeviationFromPeriodicOrbit = sqrt(pow(halfPeriodState(3), 2) + pow(halfPeriodState(5), 2));
        }
        numberOfIterations += 1;

        std::cout << "positionDeviationFromPeriodicOrbit: " << positionDeviationFromPeriodicOrbit << std::endl
                  << "velocityDeviationFromPeriodicOrbit: " << velocityDeviationFromPeriodicOrbit << "\n" << std::endl;
    }

    double jacobiEnergyHalfPeriod       = tudat::gravitation::computeJacobiEnergy(massParameter, halfPeriodState.segment(0,6));
    double jacobiEnergyInitialCondition = tudat::gravitation::computeJacobiEnergy(massParameter, initialStateVectorInclSTM.segment(0,6));

    std::cout << "\nCorrected initial state vector:" << std::endl << initialStateVectorInclSTM.segment(0,6)        << std::endl
              << "\nwith orbital period: "           << orbitalPeriod                                              << std::endl
              << "||J(0) - J(T/2|| = "               << std::abs(jacobiEnergyInitialCondition - jacobiEnergyHalfPeriod) << std::endl
              << "||T/2 - t|| = "                    << std::abs(orbitalPeriod/2.0 - currentTime) << "\n"               << std::endl;

    // The output vector consists of:
    // 1. Corrected initial state vector, including orbital period
    // 2. Half period state vector, including currentTime of integration
    // 3. numberOfIterations
    outputVector.segment(0,6)    = initialStateVectorInclSTM.segment(0,6);
    outputVector(6)              = orbitalPeriod;
    outputVector.segment(7,6)    = halfPeriodState.segment(0,6);
    outputVector(13)             = currentTime;
    outputVector(14)             = numberOfIterations;

    return outputVector;
}
