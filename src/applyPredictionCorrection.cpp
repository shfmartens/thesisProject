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
                                            const double targetHamiltonian,
                                            const double massParameter, const int numberOfPatchPoints,
                                            const bool hamiltonianConstraint,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nApply Prediction Correction:" << std::endl;
    //std::cout << "Initial guess from linearized dynamics: \n" << initialStateVector << std::endl;

    //std::cout << "TESTING TARGETHAMILTONIAN: " << targetHamiltonian << std::endl;
    //std::cout << "TESTING HAMILTONIANCONSTRAINT: " << hamiltonianConstraint << std::endl;


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
    int blockAttenuation;
    bool convergenceReached = false;
    double finalPropagatedTime = 0.0;


    Eigen::VectorXd deviationVector = Eigen::VectorXd::Zero(11*(numberOfPatchPoints-1));
    Eigen::VectorXd hamiltonianDeviationVector = Eigen::VectorXd::Zero(numberOfPatchPoints);

    Eigen::VectorXd outputVector(23);
    Eigen::MatrixXd forwardPropagatedStatesInclSTM((numberOfPatchPoints-1)*10,11);
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
        double hamiltonianDeviationAtCurrentPatchPoint;
        if (i < (numberOfPatchPoints -2))
        {

          deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );
          hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));

        } else
        {
            deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectors.segment(0,10), finalTime, stateVectorOnly, currentTime );
            hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));

        }

        deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
        forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;
        hamiltonianDeviationVector(i) = hamiltonianDeviationAtCurrentPatchPoint;

        if ( i == (numberOfPatchPoints -2))
        {
            hamiltonianDeviationVector(i+1) = targetHamiltonian - computeHamiltonian(massParameter, stateVectorOnly);

        }

        finalPropagatedTime = currentTime;

    }

    //std::cout << "Test the hamiltonianDeviation Vector: " << hamiltonianDeviationVector << std::endl;

    // compute deviations at the patch points
    Eigen::VectorXd deviationsFromPeriodicOrbit(5);
    deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

    double positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
    double velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
    double periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
    double velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
    double velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);
    double hamiltonianDeviationFromDesiredValue = hamiltonianDeviationVector.norm();

    double benchmarkPositionDeviation = positionDeviationFromPeriodicOrbit;
    double benchmarkVelocityDeviation = velocityDeviationFromPeriodicOrbit;
    double benchmarkInteriorVelocityDeviation = velocityInteriorDeviationFromPeriodicOrbit;
    double benchmarkExteriorVelocityDeviation = velocityExteriorDeviationFromPeriodicOrbit;

    std::cout << "\npositionDeviations: " << positionDeviationFromPeriodicOrbit << std::endl
              << "velocityDeviations: " << velocityDeviationFromPeriodicOrbit << std::endl
              << "velocityInteriorDeviations: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
              << "velocityExteriorDeviations: " << velocityExteriorDeviationFromPeriodicOrbit << std::endl
              << "timeDeviations: " << periodDeviationFromPeriodicOrbit << std::endl;
    if (hamiltonianConstraint == false )
    {
        std::cout << "hamiltonianDeviation: " << "Orbit not refined for Hamiltonian value." << std::endl;
    } else
    {
        std::cout << "hamiltonianDeviation: " << std::abs(hamiltonianDeviationFromDesiredValue) << std::endl;

    }
              //<< "maxPositionDeviationFromPeriodicOrbit: " << maxPositionDeviationFromPeriodicOrbit << std::endl
              //<< "maxVelocityDeviationFromPeriodicOrbit: " << maxVelocityDeviationFromPeriodicOrbit << std::endl
              //<< "maxPeriodDeviationFromPeriodicOrbit: " << maxPeriodDeviationFromPeriodicOrbit << std::endl;

    int numberOfIterations = 0;

    //writeStateHistoryAndStateVectorsToFile( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 0);

    // LINE ATTENUATION IS CURRENTLY BLOCKED FOR LEVEL 1

    // ==== LEVEL I CORRECTION ======//
    stateHistory.clear();

    while ( positionDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit
            or velocityDeviationFromPeriodicOrbit > maxVelocityDeviationFromPeriodicOrbit
            or periodDeviationFromPeriodicOrbit > maxPeriodDeviationFromPeriodicOrbit)
    {
        if( numberOfIterations > maxNumberOfIterations )
        {
            std::cout << "Predictor Corrector did not converge within maxNumberOfIterations" << std::endl;
            return outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
        }

        int numberOfIterationsLevel1 = 0;
        bool applyLevel1Correction = true;
        // ==== Start of the Level 1 corrector ==== //
        while (positionDeviationFromPeriodicOrbit > maxPositionDeviationFromPeriodicOrbit or applyLevel1Correction) {

            if( numberOfIterationsLevel1 > maxNumberOfIterations )
            {
                std::cout << "Level I dit not converger within maxNumberOfIterations" << std::endl;
                return outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
            }

            // compute the Level 1 corrections and apply them to obtain an updatedEquation
            std::cout << "\nAPPLYING LEVEL I CORRECTION"<< std::endl;
            correctionVectorLevel1 = computeLevel1Correction(deviationVector, forwardPropagatedStatesInclSTM, initialStateVectors, numberOfPatchPoints );

            initialStateVectorsBeforeCorrection = initialStateVectors;           

            // Line search attenuation parameters, Currently
            lineSearchIterationNumber = 0.0;
            attenuationFactor = 0.0;
            blockAttenuation = 0;

            while ( benchmarkPositionDeviation <= positionDeviationFromPeriodicOrbit and blockAttenuation == 0 )
            {
                // Reset input to values before correction
                initialStateVectors = initialStateVectorsBeforeCorrection;

                //std::cout << "initialStateVectors after LI correction: " << initialStateVectors << std::endl;

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
                    double hamiltonianDeviationAtCurrentPatchPoint;

                    if (i < (numberOfPatchPoints -2))
                    {

                      deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );
                      hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));


                    } else
                    {
                        deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectors.segment(0,10), finalTime, stateVectorOnly, currentTime );
                        hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));

                    }

                    deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
                    hamiltonianDeviationVector(i) = hamiltonianDeviationAtCurrentPatchPoint;

                    if ( i == (numberOfPatchPoints -2))
                    {
                        hamiltonianDeviationVector(i+1) = targetHamiltonian - computeHamiltonian(massParameter, stateVectorOnly);

                    }

                    // Fill the PropagatedStatesInclSTM matrix
                    forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

                    finalPropagatedTime = currentTime;

                }

                // compute deviations at the patch points
                deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

                positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
                velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
                periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
                velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
                velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);
                hamiltonianDeviationFromDesiredValue = hamiltonianDeviationVector.norm();


                //std::cout << "=== Check line search attenuation loop ==="<< std::endl
                //          << " attenuation Factor: " << attenuationFactor << std::endl
                //          << " positionDeviation: " << positionDeviationFromPeriodicOrbit << std::endl
                //          << " benchmarkDeviation: " << benchmarkPositionDeviation << std::endl;

                // Line search attenuation currently disabled, to active change blockattenuation below to 0
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
            if (hamiltonianConstraint == false )
            {
                std::cout << "hamiltonianDeviation: " << "Orbit not refined for Hamiltonian value." << std::endl;
            } else
            {
                std::cout << "hamiltonianDeviation: " << std::abs(hamiltonianDeviationFromDesiredValue) << std::endl;

            }

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

            //std::cout << "Test the hamiltonianDeviation Vector: " << hamiltonianDeviationVector << std::endl;


        }

        //( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 1);

        // ========= CHECK IF LI OUTPUT MEETS THE DESIRED CRITERIA ==== //
        if (positionDeviationFromPeriodicOrbit < maxPeriodDeviationFromPeriodicOrbit
            and velocityDeviationFromPeriodicOrbit < maxVelocityDeviationFromPeriodicOrbit
            and periodDeviationFromPeriodicOrbit < maxPeriodDeviationFromPeriodicOrbit)
        {
            convergenceReached = true;
            std::cout << "Convergence is reached after LI for numberOfIterations: " << numberOfIterations << std::endl;
        }

        //std::cout << "Hamiltonian Deviation Vector: \n" << hamiltonianDeviationVector << std::endl;

        // ============ LEVEL II CORRECTION ============= //

        if (convergenceReached == false)
        {

            std::cout << "\nAPPLYING LEVEL II CORRECTION"<< std::endl;
            correctionVectorLevel2 = computeLevel2Correction( deviationVector, forwardPropagatedStatesInclSTM, initialStateVectors, numberOfPatchPoints, massParameter, hamiltonianConstraint, hamiltonianDeviationVector );

            // Line attenuation is currently blocked since later on, blockattenuation is set to 1
            initialStateVectorsBeforeCorrection = initialStateVectors;
            lineSearchIterationNumber = 0.0;
            attenuationFactor = 1.0;
            blockAttenuation = 0;


            // Currently disabled, remove lineSearchIterationNumberCondition
            while ( velocityInteriorDeviationFromPeriodicOrbit >= benchmarkInteriorVelocityDeviation and blockAttenuation == 0 )
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
                    double hamiltonianDeviationAtCurrentPatchPoint;

                    if (i < (numberOfPatchPoints -2))
                    {

                      deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectorInclSTM.block(0,0,10,1), finalTime, stateVectorOnly, currentTime );
                      hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));


                    } else
                    {
                        deviationAtCurrentPatchPoint = computeDeviationVector( initialStateVectors.segment(0,10), finalTime, stateVectorOnly, currentTime );
                        hamiltonianDeviationAtCurrentPatchPoint = targetHamiltonian - computeHamiltonian(massParameter, initialStateVectors.segment(i*11,10));

                    }


                    deviationVector.segment(i*11,11) = deviationAtCurrentPatchPoint;
                    hamiltonianDeviationVector(i) = hamiltonianDeviationAtCurrentPatchPoint;

                    if ( i == (numberOfPatchPoints -2))
                    {
                        hamiltonianDeviationVector(i+1) = targetHamiltonian - computeHamiltonian(massParameter, stateVectorOnly);

                    }

                    forwardPropagatedStatesInclSTM.block(i*10,0,10,11) = stateVectorInclSTM;

                    }

                // compute deviations at the patch points
                deviationsFromPeriodicOrbit = computeDeviationsFromPeriodicOrbit(deviationVector, numberOfPatchPoints);

                positionDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(0);
                velocityDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(1);
                periodDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(2);
                velocityInteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(3);
                velocityExteriorDeviationFromPeriodicOrbit = deviationsFromPeriodicOrbit(4);
                hamiltonianDeviationFromDesiredValue = hamiltonianDeviationVector.norm();


                // Line attenuation is currently blocked
                //std::cout << "=== Check LII line search attenuation loop ==="<< std::endl
                //          << " attenuation Factor: " << attenuationFactor << std::endl
                //          << " interiorVelocityDeviation: " << velocityInteriorDeviationFromPeriodicOrbit << std::endl
                //          << " benchmarkInteriorVelocityDeviation: " << benchmarkInteriorVelocityDeviation  << std::endl;

                lineSearchIterationNumber = lineSearchIterationNumber + 1.0;
                blockAttenuation = 1;

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
                if (hamiltonianConstraint == false )
                {
                    std::cout << "hamiltonianDeviationAfterLII: " << "Orbit not refined for Hamiltonian value." << std::endl;
                } else
                {
                    std::cout << "hamiltonianDeviationAfterLII: " << std::abs(hamiltonianDeviationFromDesiredValue) << std::endl;

                }

                //std::cout << "Test the hamiltonianDeviation Vector: " << hamiltonianDeviationVector << std::endl;


            applyLevel1Correction = true;

            //writeStateHistoryAndStateVectorsToFile( stateHistory, initialStateVectors, deviationsFromPeriodicOrbit, deviationVector, numberOfIterations, 2);

            numberOfIterations += 1;

        }

    }

    Eigen::VectorXd initialCondition = initialStateVectors.segment(0,10);
    Eigen::VectorXd finalCondition   = forwardPropagatedStatesInclSTM.block(10*(numberOfPatchPoints-2),0,10,1);
    double orbitalPeriod = finalPropagatedTime - initialStateVectors(10);

    double hamiltonianInitialCondition  = computeHamiltonian( massParameter, initialCondition);
    double hamiltonianEndState          = computeHamiltonian( massParameter, finalCondition  );

    // The output vector consists of:
    // 1. Corrected initial state vector, including orbital period and energy
    // 2. Full period state vector, including currentTime of integration and energy
    // 3. numberOfIterations

    outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
    outputVector.segment(0,10) = initialCondition;
    outputVector(10) = orbitalPeriod;
    outputVector(11) = hamiltonianInitialCondition;
    outputVector.segment(12,10) = finalCondition;
    outputVector(22) = finalPropagatedTime;
    outputVector(23) = hamiltonianEndState;
    outputVector(24) = numberOfIterations;
    outputVector.segment(25,11*numberOfPatchPoints) = initialStateVectors;

    return outputVector;



}

