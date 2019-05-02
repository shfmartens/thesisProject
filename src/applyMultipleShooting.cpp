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

Eigen::VectorXd computeDeviationsFromPeriodicOrbit(const Eigen::VectorXd deviationVector, const int numberOfPatchPoints)
{
        Eigen::VectorXd outputVector(3);
        Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd periodDeviations(numberOfPatchPoints-1);

        for(int i = 0; i <= (numberOfPatchPoints - 2); i++){
            positionDeviations = deviationVector.segment(11*i,3);
            velocityDeviations = deviationVector.segment(11*i+3,3);
            periodDeviations = deviationVector.segment(11*i+10,1);
        }

        outputVector(0) = positionDeviations.norm();
        outputVector(1) = velocityDeviations.norm();
        outputVector(2) = periodDeviations.norm();

        return outputVector;

}

Eigen::VectorXd applyMultipleShooting(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nApply Multiple Shooting:" << std::endl;

    // Declare and/or initialize variables variables and matrices
    Eigen::VectorXd initialStateVectors(numberOfPatchPoints*11);
    initialStateVectors = initialStateVector;
    Eigen::MatrixXd initialStateVectorInclSTM = Eigen::MatrixXd::Zero(10,11);
    double finalTime;
    double initialTime;
    Eigen::VectorXd deviationVector = Eigen::VectorXd::Zero(11*(numberOfPatchPoints-1));
    Eigen::VectorXd outputVector(23);
    Eigen::MatrixXd propagatedStatesInclSTM((numberOfPatchPoints-1)*10,11);
    Eigen::VectorXd multipleShooting(11*numberOfPatchPoints);
    Eigen::VectorXd correctionVector(10*(numberOfPatchPoints-1));



    // seed the for loop by extracting time and state from first patch point
    initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(0,10);
    initialStateVectorInclSTM.block(0,1,10,10).setIdentity();
    initialTime = initialStateVectors( 10 );
    finalTime = initialStateVectors( 21 );

    for (int i = 0; i <= (numberOfPatchPoints -2); i++) {

        initialTime = initialStateVectors((i+1)*10 + (i));
        finalTime = initialStateVectors((i+2)*10 + (i+1) );

        std::map< double, Eigen::VectorXd > stateHistory;
        std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition(
                    initialStateVectorInclSTM, massParameter, finalTime, 1.0, stateHistory, -1, initialTime );

        Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
        double currentTime             = finalTimeState.second;
        Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

        // compute the state, STM and time the next patch point and set as initial conditions for next loop
        initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(11*(i+1),10);
        initialStateVectorInclSTM.block(0,1,10,10).setIdentity();

        // compute deviations at current patch points
        Eigen::VectorXd deviationAtCurrentPatchPoint(11);
        Eigen::VectorXd stateAtNextPatchPoint =
        deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );

        deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
        propagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

    }

    // compute deviations at the patch points
    Eigen::VectorXd deviationsFromPeriodicOrbit(3);
    deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

    double positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
    double velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
    double periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);

    std::cout << "\npositionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl
              << "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              << "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
              << "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << std::endl;

    int numberOfIterations = 0;

    while ( positionDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit
            or velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit
            or periodDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit)
    {
        if( numberOfIterations > maxNumberOfIterations )
        {
            return outputVector = Eigen::VectorXd::Zero(23);
        }

        // compute the corrections obtained via multiple shooting and apply them
        std::cout << "APPLYING MULTIPLE SHOOTING"<< std::endl;
        correctionVector.setZero();



        // propagate the sub-arcs to final time

        // compute deviations

        numberOfIterations += 1;

    }


    outputVector = Eigen::VectorXd::Zero(23);
    outputVector.segment(0,11) = initialStateVector.segment(0,11);
    return outputVector;



}

