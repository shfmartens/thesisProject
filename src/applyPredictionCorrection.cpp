#include <cmath>
#include <iostream>


#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "applyDifferentialCorrectionAugmented.h"
#include "createLowThrustInitialConditions.h"
#include "computeLevel1Correction.h"
#include "computeLevel2Correction.h"
#include "propagateOrbitAugmented.h"
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

Eigen::VectorXd computeDeviationsFromPeriodicOrbit(const Eigen::VectorXd deviationVector, const int numberOfPatchPoints)
{
        Eigen::VectorXd outputVector(4);
        Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd velocityInteriorDeviations(3*(numberOfPatchPoints-2));
        Eigen::VectorXd periodDeviations(numberOfPatchPoints-1);

        for(int i = 0; i < (numberOfPatchPoints - 1); i++){
            positionDeviations.segment(i*3,3) = deviationVector.segment(11*i,3);
            velocityDeviations.segment(i*3,3) = deviationVector.segment(11*i+3,3);
            if (i < numberOfPatchPoints -2) {
            velocityInteriorDeviations.segment(i*3,3) = deviationVector.segment(11*i+3,3);
            }
            periodDeviations(i) = deviationVector(11*i+10);
        }

        // construct the velocityInteriorDeviations measure by setting last 3 entries to zero

//        std::cout << "positionDeviations: " << positionDeviations <<std::endl
//                  << "velocityDeviations: " << velocityDeviations <<std::endl
//                  << "velocityInteriorDeviations: " << velocityInteriorDeviations << std::endl
//                  << "periodDeviations: " << periodDeviations << std::endl;

        outputVector(0) = positionDeviations.norm();
        outputVector(1) = velocityDeviations.norm();
        outputVector(2) = periodDeviations.norm();
        outputVector(3) = velocityInteriorDeviations.norm();


        return outputVector;

}

Eigen::VectorXd applyPredictionCorrection(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nApply Prediction Correction:" << std::endl;
    std::cout << "Initial guess from linearized dynamics: \n" << initialStateVector << std::endl;

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
    Eigen::VectorXd correctionVectorLevel1(11*numberOfPatchPoints);
    Eigen::VectorXd correctionVectorLevel2(11*numberOfPatchPoints);

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

        // compute deviations at current patch points  and store stateVector and STM and end of propagation
        Eigen::VectorXd deviationAtCurrentPatchPoint(11);
        deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );

        deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
        propagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

//        std::cout.precision(8);
//        std::cout << "initialTime: " << initialTime <<std::endl;
//        std::cout << "finalTime: " << finalTime <<std::endl;
//        std::cout << "initialStateVector: " << initialStateVectors.segment(11*(i+1),10) <<std::endl;

//        std::cout << "stateVectorInclSTM: " << stateVectorInclSTM <<std::endl;
    }

//        std::cout << "deviationVector: \n" << deviationVector <<std::endl;


    // compute deviations at the patch points
    Eigen::VectorXd deviationsFromPeriodicOrbit(3);
    deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

    double positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
    double velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
    double periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
    double velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);

    std::cout << "\npositionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "velocityInteriorDeviations: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
              << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl;
              //<< "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              //<< "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
              //<< "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << std::endl;

    int numberOfIterations = 0;

    while ( positionDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit
            or velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit
            or periodDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit)
    {
        if( numberOfIterations > maxNumberOfIterations )
        {
            std::cout << "Predictor Corrector did not converge within maxNumberOfIterations" << std::endl;
            return outputVector = Eigen::VectorXd::Zero(23);
        }

        int numberOfIterationsLevel1 = 0;

        // ==== Start of the Level 1 corrector ==== //
        while (positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit) {

            if( numberOfIterationsLevel1 > maxNumberOfIterations )
            {
                std::cout << "Level I dit not converger within maxNumberOfIterations" << std::endl;
                return outputVector = Eigen::VectorXd::Zero(23);
            }

            // compute the Level 1 corrections and apply them
            std::cout << "\nAPPLYING LEVEL I CORRECTION"<< std::endl;
            correctionVectorLevel1 = computeLevel1Correction(deviationVector, propagatedStatesInclSTM, numberOfPatchPoints );
            initialStateVectors = initialStateVectors + correctionVectorLevel1;

            // Propagate the updated guess and compute deviations and STM's
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
                deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );

                deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
                propagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

            }

            // compute deviations at the patch points
            deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

            positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
            velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
            periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
            velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);

            std::cout << "\nnumberOfIterationsLevelI: " << numberOfIterationsLevel1 << std::endl
                      << "positionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
                      << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
                      << "velocityInteriorDeviations: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
                      << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl;

            numberOfIterationsLevel1++;
        }
        //std::cout << "deviationVector after level 1 convergence: \n" << deviationVector << std::endl;

        // compute the Level 2 corrections and apply them
        std::cout << "\nAPPLYING LEVEL II CORRECTION"<< std::endl;
        correctionVectorLevel2 = computeLevel2Correction(deviationVector, propagatedStatesInclSTM, initialStateVectors, numberOfPatchPoints );
        initialStateVectors = initialStateVectors + correctionVectorLevel2;

        std::cout << "result LII: \n" << initialStateVectors << std::endl;

        // Propagate the updated guess and compute deviations and STM's
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
            deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );
            deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
            propagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

            }



            //std::cout << "deviationVector After LII CORRECTOR: \n" << deviationVector << std::endl;

            // compute deviations at the patch points
            deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

            positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
            velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
            periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
            velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);


            std::cout << "positionDeviationsAfterLII: " << positionDeviationFromPeriodicOrbit << std::endl
                      << "velocityDeviationsAfterLII: " << velocityDeviationFromPeriodicOrbit << std::endl
                      << "velocityInteriorDeviationsAfterLII: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl

                      << "timeDeviationsAfterLII: " << periodDeviationFromPeriodicOrbit << std::endl;


        numberOfIterations += 1;

    }


    outputVector = Eigen::VectorXd::Zero(23);
    outputVector.segment(0,11) = initialStateVector.segment(0,11);
    return outputVector;



}

