#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>


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
        Eigen::VectorXd outputVector(5);
        Eigen::VectorXd positionDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd velocityDeviations(3*(numberOfPatchPoints-1));
        Eigen::VectorXd velocityInteriorDeviations(3*(numberOfPatchPoints-2));
        Eigen::VectorXd velocityExteriorDeviations(3);
        Eigen::VectorXd periodDeviations(numberOfPatchPoints-1);

        outputVector.setZero();
        positionDeviations.setZero();
        velocityDeviations.setZero();
        velocityInteriorDeviations.setZero();
        velocityExteriorDeviations.setZero();
        periodDeviations.setZero();

        for(int i = 0; i < (numberOfPatchPoints - 1); i++ ){

            positionDeviations.segment(i*3,3) = deviationVector.segment((11*i),3);
            velocityDeviations.segment(i*3,3) = deviationVector.segment((11*i+3),3);

            if (i < numberOfPatchPoints -2) {

                velocityInteriorDeviations.segment(i*3,3) = deviationVector.segment((11*i+3),3);
            }

            if ( i == (numberOfPatchPoints - 2)) {

                velocityExteriorDeviations = deviationVector.segment((11*i+3),3);

            }

            periodDeviations(i) = deviationVector(11*i+10);
        }

        // construct the velocityInteriorDeviations measure by setting last 3 entries to zero

        outputVector(0) = positionDeviations.norm();
        outputVector(1) = velocityDeviations.norm();
        outputVector(2) = periodDeviations.norm();
        outputVector(3) = velocityInteriorDeviations.norm();
        outputVector(4) = velocityExteriorDeviations.norm();


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
    //std::cout << "Initial guess from linearized dynamics: \n" << initialStateVector << std::endl;

    // Declare and/or initialize variables variables and matrices
    Eigen::VectorXd initialStateVectors(numberOfPatchPoints*11);
    Eigen::VectorXd initialStateVectorsBeforeCorrection(numberOfPatchPoints*11);
    initialStateVectors = initialStateVector;
    Eigen::MatrixXd initialStateVectorInclSTM = Eigen::MatrixXd::Zero(10,11);
    Eigen::MatrixXd finalStateVectorInclSTM = Eigen::MatrixXd::Zero(10,11);
    double finalTime;
    double initialTime;
    double lineSearchIterationNumber;
    double attenuationFactor;


    Eigen::VectorXd deviationVector = Eigen::VectorXd::Zero(11*(numberOfPatchPoints-1));
    Eigen::VectorXd deviationVectorBackward = Eigen::VectorXd::Zero(11*(numberOfPatchPoints-1));
    Eigen::VectorXd outputVector(23);
    Eigen::MatrixXd forwardPropagatedStatesInclSTM((numberOfPatchPoints-1)*10,11);
    Eigen::MatrixXd backwardPropagatedStatesInclSTM((numberOfPatchPoints-1)*10,11);
    Eigen::VectorXd multipleShooting(11*numberOfPatchPoints);
    Eigen::VectorXd correctionVectorLevel1(11*numberOfPatchPoints);
    Eigen::VectorXd correctionVectorLevel2(11*numberOfPatchPoints);

    // ========= PROPAGATE THE INITIAL GUESS IN THE NONLINEAR MODEL ======= //

    // seed the for loop by extracting time and state from first patch point
    initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(0,10);
    initialStateVectorInclSTM.block(0,1,10,10).setIdentity();
    initialTime = initialStateVectors( 10 );
    finalTime = initialStateVectors( 21 );

    std::map< double, Eigen::VectorXd > stateHistory;

    for (int i = 0; i <= (numberOfPatchPoints -2); i++) {

        initialTime = initialStateVectors((i+1)*10 + (i));
        finalTime = initialStateVectors((i+2)*10 + (i+1) );

        std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition(
                    initialStateVectorInclSTM, massParameter, finalTime, 1.0, stateHistory, 2000, initialTime );

        Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
        double currentTime             = finalTimeState.second;
        Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

        // compute the state, STM and time the next patch point and set as initial conditions for next loop
        initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(11*(i+1),10);
        initialStateVectorInclSTM.block(0,1,10,10).setIdentity();

        // compute deviations at current patch points  and store stateVector and STM and end of propagation
        Eigen::VectorXd deviationAtCurrentPatchPoint(11);
        if (i < (numberOfPatchPoints -2))
        {

          deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );

        } else
        {
            deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectors.segment(0,10), finalTime, stateVectorOnly, currentTime );

        }

        deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
        forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

    }

    //std::cout << "Test the deviaiton Vector: " << deviationVector << std::endl;

    // compute deviations at the patch points
    Eigen::VectorXd deviationsFromPeriodicOrbit(5);
    deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

    double positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
    double velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
    double periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
    double velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
    double velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);

    double benchmarkPositionDeviation = positionDeviationFromPeriodicOrbit;
    double benchmarkVelocityDeviation = velocityDeviationFromPeriodicOrbit;
    double benchmarkInteriorVelocityDeviation = velocityInteriorDeviationFromPeriodicOrbit;
    double benchmarkExteriorVelocityDeviation = velocityExteriorDeviationFromPeriodicOrbit;

    std::cout << "\npositionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "velocityInteriorDeviations: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
              << "velocityExteriorDeviations: " << velocityExteriorDeviationFromPeriodicOrbit << std::endl
              << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl;
              //<< "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              //<< "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
              //<< "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << std::endl;

    int numberOfIterations = 0;

    writeStateHistoryAndStateVectorsToFile( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 0);

    // ==== LEVEL I CORRECTION ======//
    stateHistory.clear();

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
        bool applyLevel1Correction = true;
        // ==== Start of the Level 1 corrector ==== //
        while (positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or applyLevel1Correction) {

            if( numberOfIterationsLevel1 > maxNumberOfIterations )
            {
                std::cout << "Level I dit not converger within maxNumberOfIterations" << std::endl;
                return outputVector = Eigen::VectorXd::Zero(23);
            }

            // compute the Level 1 corrections and apply them to obtain an updatedEquation
            std::cout << "\nAPPLYING LEVEL I CORRECTION"<< std::endl;
            correctionVectorLevel1 = computeLevel1Correction(deviationVector, forwardPropagatedStatesInclSTM, initialStateVectors, numberOfPatchPoints );

            initialStateVectorsBeforeCorrection = initialStateVectors;
            lineSearchIterationNumber = 0.0;
            attenuationFactor = 0.0;
            int blockAttenuation = 0;

            while ( benchmarkPositionDeviation <= positionDeviationFromPeriodicOrbit and blockAttenuation == 0 )
            {
                // Reset input to values before correction
                initialStateVectors = initialStateVectorsBeforeCorrection;

                // Empty the stateHistory Vector
                stateHistory.clear();

                // Apply the line search attenuated correction
                attenuationFactor = pow(0.8, lineSearchIterationNumber);
                initialStateVectors = initialStateVectors + attenuationFactor * correctionVectorLevel1;

                // Propagate guess in the nonlinear model
                // Propagate the updated guess and compute deviations and STM's
                initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(0,10);
                initialStateVectorInclSTM.block(0,1,10,10).setIdentity();
                initialTime = initialStateVectors( 10 );
                finalTime = initialStateVectors( 21 );

                for (int i = 0; i <= (numberOfPatchPoints -2); i++) {

                    initialTime = initialStateVectors((i+1)*10 + (i));
                    finalTime = initialStateVectors((i+2)*10 + (i+1) );


                    std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition(
                                initialStateVectorInclSTM, massParameter, finalTime, 1.0, stateHistory, 2000, initialTime );

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

                    // Fill the PropagatedStatesInclSTM matrix
                    forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

                }

                // compute deviations at the patch points
                deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

                positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
                velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
                periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
                velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
                velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);

                std::cout << "=== Check line search attenuation loop ==="<< std::endl
                          << " attenuation Factor: " << attenuationFactor << std::endl
                          << " positionDeviation: " << positionDeviationFromPeriodicOrbit << std::endl
                          << " benchmarkDeviation: " << benchmarkPositionDeviation << std::endl;

                lineSearchIterationNumber = lineSearchIterationNumber + 1.0;
                blockAttenuation = 1;

            }

            benchmarkPositionDeviation = positionDeviationFromPeriodicOrbit;
            benchmarkVelocityDeviation = velocityDeviationFromPeriodicOrbit;
            benchmarkInteriorVelocityDeviation = velocityInteriorDeviationFromPeriodicOrbit;
            benchmarkExteriorVelocityDeviation = velocityExteriorDeviationFromPeriodicOrbit;

            std::cout << "\nnumberOfIterationsLevelI: " << numberOfIterationsLevel1 << std::endl
                      << "positionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
                      << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
                      << "velocityInteriorDeviations: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
                      << "velocityExteriorDeviations: " << velocityExteriorDeviationFromPeriodicOrbit << std::endl
                      << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl;

            if (positionDeviationFromPeriodicOrbit < maxPositionDeviationFromPeriodicOrbit)
            {
                applyLevel1Correction = false;
            }

            //std::cout.precision();
//            std::cout << "OUTPUT AFTER LI PROPAGATION TESTT: " << std::endl
//                      << "initialStateVectors: \n" << initialStateVectors << std::endl;
//                      << "propagatedStatesInclSTM: \n" << forwardPropagatedStatesInclSTM << std::endl
//                   std::cout   << "-deviationVector: \n" << -deviationVector << std::endl;
//                      << "numberOfPatchPoints: " << numberOfPatchPoints << std::endl;

            numberOfIterationsLevel1++;


        }

        writeStateHistoryAndStateVectorsToFile( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 1);

//        // ====== BACKWARD PROPAGATION TO OBTAIN THE STMS BACKWARD IN TIME and deviations //

//        // Seed backwards propagation by selecting final state and appropriate times
//        finalStateVectorInclSTM.block(0,0,10,1) = forwardPropagatedStatesInclSTM.block(10*(numberOfPatchPoints-1),0,10,1);
//        finalStateVectorInclSTM.block(0,1,10,10).setIdentity();
//        initialTime = initialStateVectors( 11*(numberOfPatchPoints-1) + 10 ) - deviationVector(11*(numberOfPatchPoints-1) + 10);
//        finalTime = initialStateVectors( 11*(numberOfPatchPoints-1) -1 );

//        // compute the State Transition Matrices backwards in time
//        for ( int i = ( numberOfPatchPoints-1 ); i > 0; i--)
//        {

//            initialTime = initialStateVectors((i)*11 + 10) - deviationVector(11*(i) + 10);
//            finalTime = initialStateVectors((i)*11 - 1 );

//            std::map< double, Eigen::VectorXd > stateHistory;
//            std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition(
//                        finalStateVectorInclSTM, massParameter, finalTime, -1, stateHistory, -1, initialTime );

//            Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
//            double currentTime             = finalTimeState.second;
//            Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

//            // compute the state, STM and time the next patch point and set as initial conditions for next loop
//            finalStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(11*(i-1),10);
//            finalStateVectorInclSTM.block(0,1,10,10).setIdentity();

//            // compute deviations at current patch points
//            Eigen::VectorXd deviationAtCurrentPatchPoint(11);
//            deviationAtCurrentPatchPoint = computeDeviationVector( finalStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );
//            deviationVectorBackward.segment((i-1)*11,11) = deviationAtCurrentPatchPoint;


//            // fill backwardPropagatedStatesInclSTM backwards
//            backwardPropagatedStatesInclSTM.block((i-1)*10,0,10,11) = stateVectorInclSTM;

//        }

//std::cout << "Backward Propagation STM's from - to plus: \n" << backwardPropagatedStatesInclSTM << std::endl;

        // ============ LEVEL II CORRECTION ============= //

        std::cout << "\nAPPLYING LEVEL II CORRECTION"<< std::endl;
        correctionVectorLevel2 = computeLevel2Correction( deviationVector, forwardPropagatedStatesInclSTM, initialStateVectors, numberOfPatchPoints );


        initialStateVectorsBeforeCorrection = initialStateVectors;
        lineSearchIterationNumber = 0.0;
        attenuationFactor = 1.0;

        // Currently disabled, remove lineSearchIterationNumberCondition
        while ( velocityInteriorDeviationFromPeriodicOrbit >= benchmarkInteriorVelocityDeviation and lineSearchIterationNumber == 0.0 )
        {
            // Reset input to values before correction
            initialStateVectors = initialStateVectorsBeforeCorrection;

            // Reset stateHistory
            stateHistory.clear();

            // Apply the line search attenuated correction
            attenuationFactor = pow(0.8, lineSearchIterationNumber);
            initialStateVectors = initialStateVectors + attenuationFactor * correctionVectorLevel2;

            // Propagate the updated guess and compute deviations and STM's
            initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(0,10);
            initialStateVectorInclSTM.block(0,1,10,10).setIdentity();
            initialTime = initialStateVectors( 10 );
            finalTime = initialStateVectors( 21 );

            //std::cout << "\nCorrection applied" << std::endl;
            //std::cout << "initialStateVectors after correction: " << initialStateVectors << std::endl;
            for (int i = 0; i <= (numberOfPatchPoints -2); i++) {

                initialTime = initialStateVectors((i+1)*10 + (i));
                finalTime = initialStateVectors((i+2)*10 + (i+1) );

                std::pair< Eigen::MatrixXd, double > finalTimeState = propagateOrbitAugmentedToFinalCondition(
                           initialStateVectorInclSTM, massParameter, finalTime, 1.0, stateHistory, 2000, initialTime );

                Eigen::MatrixXd stateVectorInclSTM      = finalTimeState.first;
                double currentTime             = finalTimeState.second;
                Eigen::VectorXd stateVectorOnly = stateVectorInclSTM.block( 0, 0, 10, 1 );

                // compute the state, STM and time the next patch point and set as initial conditions for next loop
                initialStateVectorInclSTM.block(0,0,10,1) = initialStateVectors.segment(11*(i+1),10);
                initialStateVectorInclSTM.block(0,1,10,10).setIdentity();

                // compute deviations at current patch points
                Eigen::VectorXd deviationAtCurrentPatchPoint(11);
                if (i < (numberOfPatchPoints -2))
                {

                  deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );

                } else
                {
                    deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectors.segment(0,10), finalTime, stateVectorOnly, currentTime );

                }


                deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;

                forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

                }

            // compute deviations at the patch points
            deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

            positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
            velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
            periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
            velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
            velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);


            std::cout << "=== Check LII line search attenuation loop ==="<< std::endl
                      << " attenuation Factor: " << attenuationFactor << std::endl
                      << " interiorVelocityDeviation: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
                      << " benchmarkInteriorVelocityDeviation: " << benchmarkInteriorVelocityDeviation  << std::endl;

            lineSearchIterationNumber = lineSearchIterationNumber + 1.0;

        }

        // Set new benchmarks for grid search attenuation
        benchmarkPositionDeviation = positionDeviationFromPeriodicOrbit;
        benchmarkVelocityDeviation = velocityDeviationFromPeriodicOrbit;
        benchmarkInteriorVelocityDeviation = velocityInteriorDeviationFromPeriodicOrbit;
        benchmarkExteriorVelocityDeviation = velocityExteriorDeviationFromPeriodicOrbit;

            std::cout << "\npositionDeviationsAfterLII: " << positionDeviationFromPeriodicOrbit << std::endl
                      << "velocityDeviationsAfterLII: " << velocityDeviationFromPeriodicOrbit << std::endl
                      << "velocityInteriorDeviationsAfterLII: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
                      << "velocityExteriorDeviationsAfterLII: " << velocityExteriorDeviationFromPeriodicOrbit << std::endl
                      << "timeDeviationsAfterLII: " << periodDeviationFromPeriodicOrbit << std::endl;

        applyLevel1Correction = true;
        //std::cout << "deviationVector: "<< deviationVector << std::endl;
        //std::cout << "Level II OUTPUT TESTT: "<< initialStateVectors << std::endl;

        writeStateHistoryAndStateVectorsToFile( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 2);

        numberOfIterations += 1;

    }


    outputVector = Eigen::VectorXd::Zero(23);
    outputVector.segment(0,11) = initialStateVector.segment(0,11);
    return outputVector;



}

