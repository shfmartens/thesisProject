#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <chrono>

#include <boost/function.hpp>

#include "Tudat/Astrodynamics/BasicAstrodynamics/celestialBodyConstants.h"
#include "Tudat/Astrodynamics/Gravitation/librationPoint.h"
#include "Tudat/Astrodynamics/Gravitation/jacobiEnergy.h"

#include "createLowThrustInitialConditions.h"
#include "computeLevel1Correction.h"
#include "computeLevel2Correction.h"
#include "propagateOrbitAugmented.h"
#include "propagateOrbit.h"
#include "stateDerivativeModel.h"
#include "stateDerivativeModelAugmented.h"

Eigen::VectorXd computeDeviationNorms (const Eigen::VectorXd defectVector, const int numberOfPatchPoints )
{

    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(5);
    Eigen::VectorXd defectVectorAtPatchPoint = Eigen::VectorXd::Zero(11);
    double positionDeviationTotal = 0.0;
    double velocityDeviationTotal = 0.0;
    double velocityInteriorDeviationTotal = 0.0;
    double velocityExteriorDeviationTotal = 0.0;
    double timeDeviationTotal = 0.0;

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ ){

        defectVectorAtPatchPoint = defectVector.segment(i*11,11);

        positionDeviationTotal = positionDeviationTotal + ((defectVectorAtPatchPoint.segment(0,3)).norm());
        velocityDeviationTotal = velocityDeviationTotal + ((defectVectorAtPatchPoint.segment(3,3)).norm());
        timeDeviationTotal = timeDeviationTotal + defectVectorAtPatchPoint(10);

        if (i < (numberOfPatchPoints - 2) ){
                    velocityInteriorDeviationTotal = velocityInteriorDeviationTotal + ((defectVectorAtPatchPoint.segment(3,3)).norm());
           } else{

                    velocityExteriorDeviationTotal = velocityExteriorDeviationTotal + ((defectVectorAtPatchPoint.segment(3,3)).norm());

                }

    }



    outputVector(0) = positionDeviationTotal;
    outputVector(1) = velocityDeviationTotal;
    outputVector(2) = velocityInteriorDeviationTotal;
    outputVector(3) = velocityExteriorDeviationTotal;
    outputVector(4) = timeDeviationTotal;

    return outputVector;

}

void computeOrbitDeviations(Eigen::VectorXd inputStateVector, const int numberOfPatchPoints, Eigen::MatrixXd& propagatedStatesInclSTM, Eigen::VectorXd& defectVector, const double massParameter  )
{

    Eigen::MatrixXd initialStateAndSTM = Eigen::MatrixXd::Zero(10,11);
    initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(0,10);
    initialStateAndSTM.block(0,1,10,10).setIdentity();

    for (int i = 0; i < (numberOfPatchPoints - 1); i++ )
    {
        double initialTime = inputStateVector(i+(i+1)*10);
        double finalTime = inputStateVector((i+1)+(i+2)*10);

        std::map< double, Eigen::VectorXd > segmentStateHistory;
        segmentStateHistory.clear();


        std::pair< Eigen::MatrixXd, double > endStateAndSTMAndTime = propagateOrbitAugmentedToFinalCondition(
                    initialStateAndSTM, massParameter, finalTime, 1, segmentStateHistory, -1, initialTime);

        Eigen::MatrixXd endStateAndSTM      = endStateAndSTMAndTime.first;
        double endTime                  = endStateAndSTMAndTime.second;
        Eigen::VectorXd stateVectorOnly = endStateAndSTM.block( 0, 0, 10, 1 );


        // Select the begin state of the next segment
        initialStateAndSTM.setZero();
        initialStateAndSTM.block(0,0,10,1) = inputStateVector.segment(11*(i+1),10);
        initialStateAndSTM.block(0,1,10,10).setIdentity();

        // Compute defects by computing difference between initial state of next segment and end state of current segment
        if(i < numberOfPatchPoints - 2)
        {
            defectVector.segment(i*11,10) = inputStateVector.segment(11*(i+1),10) - stateVectorOnly; // state difference
            defectVector(i*11+10) = finalTime - endTime;
        } else
        {

            std::cout  <<  "comparing to final state on trajectory: \n "<< inputStateVector.segment(11*(i+1),10) - stateVectorOnly << std::endl; // state difference
            std::cout  <<  "comparing to initial state on trajectory: \n "<< inputStateVector.segment(0,10) - stateVectorOnly << std::endl; // state difference

            // Compare the final state of propagated trajectory to initial patch point to have periodicity!
            defectVector.segment(i*11,10) = inputStateVector.segment(0,10) - stateVectorOnly;
            defectVector(i*11+10) = finalTime - endTime;
        }



        // Store the state and STM at the end of propagation in the propagatedStatesInclSTM matrix
        propagatedStatesInclSTM.block(10*i,0,10,11) = endStateAndSTM;

       }

}

Eigen::VectorXd applyPredictionCorrection(const int librationPointNr,
                                            const Eigen::VectorXd& initialStateVector,
                                            const double massParameter, const int numberOfPatchPoints,
                                            double maxPositionDeviationFromPeriodicOrbit,
                                            double maxVelocityDeviationFromPeriodicOrbit, const double maxPeriodDeviationFromPeriodicOrbit,
                                            const int maxNumberOfIterations )
{
    std::cout << "\nAPPLY PREDICTION CORRECTION\n" << std::endl;

    // == Define the relevant variables == //
    Eigen::VectorXd outputVector = Eigen::VectorXd::Zero(25 + (11 * numberOfPatchPoints));
    Eigen::VectorXd currentTrajectoryGuess =  Eigen::VectorXd::Zero(11*numberOfPatchPoints);
    Eigen::MatrixXd propagatedStatesInclSTM = Eigen::MatrixXd::Zero(10*numberOfPatchPoints,11);
    Eigen::VectorXd defectVector =  Eigen::VectorXd::Zero(11* (numberOfPatchPoints - 1) );
    Eigen::VectorXd deviationNorms = Eigen::VectorXd::Zero(5);

    currentTrajectoryGuess = initialStateVector;

    // ========= Compute current defects deviations by propagating the initialStateVector in CR3BPLT  ======= //
    computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

    std::cout << "defectVector new" << defectVector << std::endl;

    deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

    double positionDeviationNorm = deviationNorms(0);
    double velocityTotalDeviationNorm = deviationNorms(1);
    double velocityInteriorDeviationNorm = deviationNorms(2);
    double velocityExteriorDeviationNorm = deviationNorms(3);
    double timeDeviationNorm = deviationNorms(4);

    std::cout << "==== CHECKING DEVIATION NORMS =====" << std::endl
              << "positionDeviationNorm: " << positionDeviationNorm << std::endl
              << "velocityTotalDeviationNorm: " << velocityTotalDeviationNorm << std::endl
              << "velocityInteriorDeviationNorm: " << velocityInteriorDeviationNorm << std::endl
              << "velocityExteriorDeviationNorm: " << velocityExteriorDeviationNorm << std::endl
              << "timeDeviationNorm: " << timeDeviationNorm << std::endl
              << "===================================" << std::endl;


    int numberOfIterations = 0;
    while (positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
           velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
           timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit ){

        if( numberOfIterations > maxNumberOfIterations ){

            std::cout << "Predictor Corrector did not converge within maxNumberOfIterations" << std::endl;
            return outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
        }

        // ==== Apply Level 1 Correction ==== //

        int numberOfIterationsLevel1 = 0;
        bool applyLevel1Correction = false;

        while (positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or applyLevel1Correction){

            Eigen::VectorXd correctionVectorLevel1 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel1 = computeLevel1Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints );

            //std::cout << "correctionVectorLevel1: " << correctionVectorLevel1 << std::endl;

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel1;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

            deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

            positionDeviationNorm = deviationNorms(0);
            velocityTotalDeviationNorm = deviationNorms(1);
            velocityInteriorDeviationNorm = deviationNorms(2);
            velocityExteriorDeviationNorm = deviationNorms(3);
            timeDeviationNorm = deviationNorms(4);

            std::cout << "==== Level I Correction Status Update =====" << std::endl
                      << "numberOfIterationsLevel1: " << numberOfIterationsLevel1 << std::endl
                      << "positionDeviationNorm: " << positionDeviationNorm << std::endl
                      << "velocityTotalDeviationNorm: " << velocityTotalDeviationNorm << std::endl
                      << "velocityInteriorDeviationNorm: " << velocityInteriorDeviationNorm << std::endl
                      << "velocityExteriorDeviationNorm: " << velocityExteriorDeviationNorm << std::endl
                      << "timeDeviationNorm: " << timeDeviationNorm << std::endl
                      << "===================================" << std::endl;

            numberOfIterationsLevel1++;
            applyLevel1Correction = false;

        }

        std::cout << "Level I Converged after" << numberOfIterationsLevel1 << " iterations" << std::endl;

        // ========= Apply Level II correction if all constraints are not met  ======= //

        if( positionDeviationNorm > maxPositionDeviationFromPeriodicOrbit or
                   velocityTotalDeviationNorm > maxVelocityDeviationFromPeriodicOrbit or
                timeDeviationNorm > maxPeriodDeviationFromPeriodicOrbit )
        {

            Eigen::VectorXd correctionVectorLevel2 = Eigen::VectorXd::Zero(11*numberOfPatchPoints);
            correctionVectorLevel2 = computeLevel2Correction(defectVector, propagatedStatesInclSTM, currentTrajectoryGuess, numberOfPatchPoints, massParameter );

            currentTrajectoryGuess = currentTrajectoryGuess + correctionVectorLevel2;

            // ========= Compute the defects of the corrected trajectory  ======= //
            defectVector.setZero();
            propagatedStatesInclSTM.setZero();
            computeOrbitDeviations(currentTrajectoryGuess, numberOfPatchPoints, propagatedStatesInclSTM, defectVector, massParameter);

            deviationNorms = computeDeviationNorms(defectVector, numberOfPatchPoints);

            positionDeviationNorm = deviationNorms(0);
            velocityTotalDeviationNorm = deviationNorms(1);
            velocityInteriorDeviationNorm = deviationNorms(2);
            velocityExteriorDeviationNorm = deviationNorms(3);
            timeDeviationNorm = deviationNorms(4);

            std::cout << "==== Level II Status Update =====" << std::endl
                      << "positionDeviationNorm: " << positionDeviationNorm << std::endl
                      << "velocityTotalDeviationNorm: " << velocityTotalDeviationNorm << std::endl
                      << "velocityInteriorDeviationNorm: " << velocityInteriorDeviationNorm << std::endl
                      << "velocityExteriorDeviationNorm: " << velocityExteriorDeviationNorm << std::endl
                      << "timeDeviationNorm: " << timeDeviationNorm << std::endl
                      << "===================================" << std::endl;

            if (positionDeviationNorm < maxPositionDeviationFromPeriodicOrbit)
            {
                applyLevel1Correction = true;
            }
        }

        numberOfIterations++;
    }

    std::cout << "trajectory converged after " << numberOfIterations << " cycles" << std::endl;




    outputVector = Eigen::VectorXd::Zero(25+11*numberOfPatchPoints);
    return outputVector;



}

